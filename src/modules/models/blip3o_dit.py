"""
BLIP3-o Patch-Level DiT Model - Aligned with BLIP3-o Paper
src/modules/models/blip3o_patch_dit.py

This implementation follows the BLIP3-o approach:
1. DiT takes noisy CLIP patch embeddings (256 tokens, 1024-dim) as input
2. Conditioned on EVA-CLIP patch embeddings (256 tokens, 4096-dim)  
3. Outputs denoised CLIP patch embeddings
4. Uses flow matching training objective
5. Evaluation via image-to-text recall metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np
from transformers import PreTrainedModel

try:
    from ..config.blip3o_config import BLIP3oDiTConfig
except ImportError:
    # Fallback for direct usage
    from dataclasses import dataclass

    @dataclass
    class BLIP3oDiTConfig:
        # Model dimensions
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        intermediate_size: int = 3072
        
        # Input/output dimensions
        eva_embedding_size: int = 4096  # EVA-CLIP dimension
        clip_embedding_size: int = 1024  # CLIP patch dimension
        input_size: int = 16  # 16x16 = 256 patches
        
        # Training configuration
        max_position_embeddings: int = 256
        dropout_prob: float = 0.1
        
        # Flow matching parameters
        prediction_type: str = "velocity"
        sigma_min: float = 1e-4
        sigma_max: float = 1.0


class RotaryPositionalEmbedding3D(nn.Module):
    """3D Rotary Position Embedding for spatial-temporal structure (BLIP3-o style)"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 256):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # Create frequency for 3D RoPE (spatial + temporal)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply 3D rotary position embedding"""
        batch_size, seq_len, hidden_size = x.shape
        
        if position_ids is None:
            # Create 2D position ids for 16x16 grid
            grid_size = int(math.sqrt(seq_len))
            pos_y, pos_x = torch.meshgrid(
                torch.arange(grid_size, device=x.device),
                torch.arange(grid_size, device=x.device),
                indexing='ij'
            )
            position_ids = torch.stack([pos_x.flatten(), pos_y.flatten()], dim=1)  # [256, 2]
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 256, 2]
        
        # Apply rotary embedding
        freqs = torch.einsum('bij,k->bijk', position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        # Apply to hidden states
        x_rot = self._apply_rotary_pos_emb(x, cos_emb, sin_emb)
        return x_rot
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding to input tensor"""
        # Split x into pairs
        x1, x2 = x[..., 0::2], x[..., 1::2]
        cos_expanded = cos[..., :x1.shape[-1]]
        sin_expanded = sin[..., :x1.shape[-1]]
        
        # Apply rotation
        x_rotated = torch.stack([
            x1 * cos_expanded - x2 * sin_expanded,
            x1 * sin_expanded + x2 * cos_expanded
        ], dim=-1).flatten(-2)
        
        return x_rotated


class TimestepEmbedder(nn.Module):
    """Timestep embedding for flow matching"""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MultiHeadAttention(nn.Module):
    """Multi-head attention with grouped query attention (BLIP3-o style)"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Project to q, k, v
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.o_proj(attn_output)


class BLIP3oDiTBlock(nn.Module):
    """DiT block with adaptive layer norm and cross-attention conditioning"""
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Layer normalization (sandwich normalization for stability)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            config.dropout_prob
        )
        
        # Cross-attention with EVA features
        self.cross_attn = MultiHeadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            config.dropout_prob
        )
        
        # EVA feature projection
        self.eva_proj = nn.Linear(config.eva_embedding_size, config.hidden_size)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout_prob)
        )
        
        # Adaptive Layer Norm modulation (AdaLN-Zero)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size)
        )
        
        # 3D Rotary Position Embedding
        self.rope = RotaryPositionalEmbedding3D(config.hidden_size)

    def forward(self, 
                hidden_states: torch.Tensor,
                eva_features: torch.Tensor,
                timestep_emb: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, 256, hidden_size] - Noisy CLIP patch embeddings
            eva_features: [B, 256, eva_size] - EVA-CLIP conditioning
            timestep_emb: [B, hidden_size] - Timestep embeddings
            attention_mask: Optional attention mask
        """
        
        # Project EVA features to hidden size
        eva_projected = self.eva_proj(eva_features)  # [B, 256, hidden_size]
        
        # AdaLN modulation parameters
        shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross = \
            self.adaLN_modulation(timestep_emb).chunk(6, dim=1)
        
        # Self-attention with AdaLN
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # Apply 3D RoPE
        norm_hidden = self.rope(norm_hidden)
        
        # Self-attention
        attn_output = self.self_attn(norm_hidden, norm_hidden, norm_hidden, attention_mask)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        
        # Cross-attention with EVA features
        norm_hidden = self.norm2(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_cross.unsqueeze(1)) + shift_cross.unsqueeze(1)
        
        cross_attn_output = self.cross_attn(norm_hidden, eva_projected, eva_projected, attention_mask)
        hidden_states = hidden_states + gate_cross.unsqueeze(1) * cross_attn_output
        
        # Feed-forward network
        norm_hidden = self.norm3(hidden_states)
        mlp_output = self.mlp(norm_hidden)
        hidden_states = hidden_states + mlp_output
        
        return hidden_states


class BLIP3oPatchDiTModel(PreTrainedModel):
    """
    BLIP3-o Patch-Level DiT Model for Image-to-Text Recall
    
    This model follows the BLIP3-o architecture:
    - Takes noisy CLIP patch embeddings (256 tokens, 1024-dim) as input
    - Conditioned on EVA-CLIP patch embeddings (256 tokens, 4096-dim)
    - Outputs denoised CLIP patch embeddings for flow matching training
    - Designed for image-to-text recall evaluation
    """
    
    config_class = BLIP3oDiTConfig
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__(config)
        self.config = config
        
        # Input projection from CLIP patches to hidden size
        self.input_proj = nn.Linear(config.clip_embedding_size, config.hidden_size)
        
        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        
        # Position embedding for 256 patches (16x16 grid)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            BLIP3oDiTBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output projection back to CLIP dimension
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=1e-6),
            nn.Linear(config.hidden_size, config.clip_embedding_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Zero-out output layer for better initial training
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)
    
    def forward(self,
                hidden_states: torch.Tensor,  # [B, 256, 1024] - Noisy CLIP patches
                timestep: torch.Tensor,       # [B] - Timesteps
                encoder_hidden_states: torch.Tensor,  # [B, 256, 4096] - EVA conditioning
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for BLIP3-o patch-level DiT
        
        Args:
            hidden_states: Noisy CLIP patch embeddings [B, 256, 1024]
            timestep: Flow matching timesteps [B]
            encoder_hidden_states: EVA-CLIP conditioning [B, 256, 4096]
            attention_mask: Optional attention mask
            return_dict: Whether to return a dict
            
        Returns:
            Predicted velocity for flow matching [B, 256, 1024]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Validate inputs
        assert seq_len == self.config.max_position_embeddings, \
            f"Expected {self.config.max_position_embeddings} tokens, got {seq_len}"
        assert hidden_states.shape[2] == self.config.clip_embedding_size, \
            f"Expected CLIP dim {self.config.clip_embedding_size}, got {hidden_states.shape[2]}"
        assert encoder_hidden_states.shape[2] == self.config.eva_embedding_size, \
            f"Expected EVA dim {self.config.eva_embedding_size}, got {encoder_hidden_states.shape[2]}"
        
        # Project inputs to hidden size
        x = self.input_proj(hidden_states)  # [B, 256, hidden_size]
        
        # Add positional embeddings
        x = x + self.pos_embed  # [B, 256, hidden_size]
        
        # Timestep embedding
        timestep_emb = self.timestep_embedder(timestep)  # [B, hidden_size]
        
        # Pass through DiT blocks
        for block in self.blocks:
            x = block(x, encoder_hidden_states, timestep_emb, attention_mask)
        
        # Project back to CLIP dimension
        velocity_pred = self.output_proj(x)  # [B, 256, 1024]
        
        if return_dict:
            return {
                "velocity_prediction": velocity_pred,
                "hidden_states": x,
                "timestep_embeddings": timestep_emb
            }
        
        return velocity_pred
    
    @torch.no_grad()
    def generate(self,
                 eva_features: torch.Tensor,  # [B, 256, 4096]
                 num_inference_steps: int = 50,
                 generator: Optional[torch.Generator] = None,
                 return_intermediate: bool = False) -> torch.Tensor:
        """
        Generate CLIP patch embeddings using flow matching sampling
        
        Args:
            eva_features: EVA-CLIP conditioning [B, 256, 4096]
            num_inference_steps: Number of sampling steps
            generator: Random number generator
            return_intermediate: Whether to return intermediate states
            
        Returns:
            Generated CLIP patch embeddings [B, 256, 1024]
        """
        device = eva_features.device
        batch_size = eva_features.shape[0]
        
        # Start from noise
        x = torch.randn(
            batch_size, 256, self.config.clip_embedding_size,
            device=device,
            generator=generator,
            dtype=eva_features.dtype
        )
        
        # Flow matching sampling (Euler method)
        dt = 1.0 / num_inference_steps
        intermediate_states = [x.clone()] if return_intermediate else None
        
        for step in range(num_inference_steps):
            t = step * dt
            timestep = torch.full((batch_size,), t, device=device, dtype=eva_features.dtype)
            
            # Predict velocity
            velocity = self.forward(
                hidden_states=x,
                timestep=timestep,
                encoder_hidden_states=eva_features,
                return_dict=False
            )
            
            # Euler step
            x = x + dt * velocity
            
            if return_intermediate:
                intermediate_states.append(x.clone())
        
        if return_intermediate:
            return x, intermediate_states
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_blip3o_patch_dit_model(
    config: Optional[BLIP3oDiTConfig] = None,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    **kwargs
) -> BLIP3oPatchDiTModel:
    """
    Create BLIP3-o patch-level DiT model
    
    Args:
        config: Model configuration
        hidden_size: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        **kwargs: Additional config parameters
        
    Returns:
        BLIP3oPatchDiTModel instance
    """
    if config is None:
        config = BLIP3oDiTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            **kwargs
        )
    
    model = BLIP3oPatchDiTModel(config)
    
    print(f"✅ BLIP3-o Patch DiT model created")
    print(f"   Parameters: {model.get_num_parameters():,}")
    print(f"   Architecture: Patch-level DiT for image-to-text recall")
    print(f"   Input: 256 CLIP patches (1024-dim)")
    print(f"   Conditioning: 256 EVA-CLIP patches (4096-dim)")
    print(f"   Output: 256 denoised CLIP patches (1024-dim)")
    
    return model