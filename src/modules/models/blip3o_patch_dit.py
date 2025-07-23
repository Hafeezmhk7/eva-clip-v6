"""
Flexible BLIP3-o Patch-Level DiT Model - Support for 256/257 tokens
src/modules/models/blip3o_patch_dit.py

CHANGES:
1. Support both 256 (patch only) and 257 (CLS + patch) token modes
2. Flexible position embeddings
3. Updated validation for both modes
4. Training mode configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np
from transformers import PreTrainedModel
import logging

logger = logging.getLogger(__name__)

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
        
        # Input/output dimensions - FLEXIBLE
        eva_embedding_size: int = 4096  # EVA-CLIP dimension
        clip_embedding_size: int = 1024  # CLIP dimension
        num_tokens: int = 257  # 257 for CLS+patch, 256 for patch only
        
        # Training configuration
        max_position_embeddings: int = 257  # Max tokens supported
        dropout_prob: float = 0.1
        training_mode: str = "cls_patch"  # "cls_patch" or "patch_only"
        
        # Flow matching parameters
        prediction_type: str = "velocity"
        sigma_min: float = 1e-4
        sigma_max: float = 1.0


class RotaryPositionalEmbedding3D(nn.Module):
    """3D Rotary Position Embedding with flexible token support"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 257):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # Create frequency for RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 4, 2).float() / (dim // 4)))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply 3D rotary position embedding with flexible token count"""
        batch_size, seq_len, hidden_size = x.shape
        
        try:
            # Handle both 256 and 257 token cases
            if seq_len == 257:
                # CLS + patches: CLS gets position (0,0), patches get spatial positions
                if position_ids is None:
                    # Create positions: [CLS: (0,0)] + [patches: 16x16 grid]
                    grid_size = 16  # 16x16 = 256 patches
                    
                    # CLS token position
                    cls_pos_x = torch.zeros(1, device=x.device, dtype=torch.float32)
                    cls_pos_y = torch.zeros(1, device=x.device, dtype=torch.float32)
                    
                    # Patch positions
                    pos_x = torch.arange(grid_size, device=x.device, dtype=torch.float32)
                    pos_y = torch.arange(grid_size, device=x.device, dtype=torch.float32)
                    pos_grid = torch.meshgrid(pos_x, pos_y, indexing='ij')
                    patch_pos_x = pos_grid[0].flatten()  # [256]
                    patch_pos_y = pos_grid[1].flatten()  # [256]
                    
                    # Combine CLS + patch positions
                    pos_x_all = torch.cat([cls_pos_x, patch_pos_x])  # [257]
                    pos_y_all = torch.cat([cls_pos_y, patch_pos_y])  # [257]
                    
                    # Repeat for batch
                    pos_x_flat = pos_x_all.unsqueeze(0).repeat(batch_size, 1)  # [B, 257]
                    pos_y_flat = pos_y_all.unsqueeze(0).repeat(batch_size, 1)  # [B, 257]
                
            elif seq_len == 256:
                # Patches only: 16x16 spatial grid
                if position_ids is None:
                    grid_size = 16
                    pos_x = torch.arange(grid_size, device=x.device, dtype=torch.float32)
                    pos_y = torch.arange(grid_size, device=x.device, dtype=torch.float32)
                    pos_grid = torch.meshgrid(pos_x, pos_y, indexing='ij')
                    
                    pos_x_flat = pos_grid[0].flatten().unsqueeze(0).repeat(batch_size, 1)  # [B, 256]
                    pos_y_flat = pos_grid[1].flatten().unsqueeze(0).repeat(batch_size, 1)  # [B, 256]
            
            else:
                # Fallback for other sequence lengths
                logger.debug(f"Unsupported sequence length {seq_len}, using linear positions")
                pos_x_flat = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
                pos_y_flat = torch.zeros_like(pos_x_flat)
            
            # Apply rotary embedding to spatial dimensions
            rope_dim = min(hidden_size, len(self.inv_freq) * 4)
            
            if rope_dim >= 4:
                x_rope = x[..., :rope_dim]
                x_pass = x[..., rope_dim:]
                
                x_rope_rotated = self._apply_2d_rotary_pos_emb(x_rope, pos_x_flat, pos_y_flat)
                x = torch.cat([x_rope_rotated, x_pass], dim=-1)
            
            return x
            
        except Exception as e:
            logger.debug(f"3D RoPE failed, using input unchanged: {e}")
            return x
    
    def _apply_2d_rotary_pos_emb(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        """Apply 2D rotary position embedding"""
        try:
            batch_size, seq_len, rope_dim = x.shape
            
            if rope_dim % 4 != 0:
                return x
            
            quarter_dim = rope_dim // 4
            
            # Split into 4 parts for x and y dimensions
            x1 = x[..., :quarter_dim]
            x2 = x[..., quarter_dim:2*quarter_dim]
            x3 = x[..., 2*quarter_dim:3*quarter_dim]
            x4 = x[..., 3*quarter_dim:]
            
            # Compute frequencies
            freqs_x = torch.einsum('bi,j->bij', pos_x, self.inv_freq[:quarter_dim])
            freqs_y = torch.einsum('bi,j->bij', pos_y, self.inv_freq[:quarter_dim])
            
            cos_x, sin_x = freqs_x.cos(), freqs_x.sin()
            cos_y, sin_y = freqs_y.cos(), freqs_y.sin()
            
            # Apply 2D rotation
            x_rot = torch.cat([
                x1 * cos_x - x2 * sin_x,
                x1 * sin_x + x2 * cos_x,
                x3 * cos_y - x4 * sin_y,
                x3 * sin_y + x4 * cos_y,
            ], dim=-1)
            
            return x_rot
            
        except Exception as e:
            logger.debug(f"2D rotary position embedding failed: {e}")
            return x


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
    """Multi-head attention with support for flexible sequence lengths"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size
        
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
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.o_proj(attn_output)


class BLIP3oDiTBlock(nn.Module):
    """DiT block with flexible token support"""
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        
        # RMSNorm for stability
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.norm3 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        
        # Attention layers
        self.self_attn = MultiHeadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            config.dropout_prob
        )
        
        self.cross_attn = MultiHeadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            config.dropout_prob
        )
        
        # EVA feature projection
        self.eva_proj = nn.Linear(config.eva_embedding_size, config.hidden_size)
        
        # Feed-forward network (SwiGLU)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Adaptive Layer Norm modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )
        
        # 3D Rotary Position Embedding
        self.rope = RotaryPositionalEmbedding3D(config.hidden_size)

    def forward(self,
                hidden_states: torch.Tensor,        # [B, N, hidden_size] where N=256 or 257
                encoder_hidden_states: torch.Tensor, # [B, N, 4096] - EVA features
                timestep_emb: torch.Tensor,         # [B, hidden_size] - Timestep embedding
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with flexible token support"""
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project EVA features to hidden dimension
        eva_features = self.eva_proj(encoder_hidden_states)  # [B, N, hidden_size]
        
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(timestep_emb).chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # Apply 3D RoPE (handles both 256 and 257 tokens)
        norm_hidden = self.rope(norm_hidden)
        
        self_attn_output = self.self_attn(norm_hidden, norm_hidden, norm_hidden, attention_mask)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * self_attn_output
        
        # Cross-attention with EVA features
        norm_hidden = self.norm2(hidden_states)
        cross_attn_output = self.cross_attn(norm_hidden, eva_features, eva_features, attention_mask)
        hidden_states = hidden_states + cross_attn_output
        
        # Feed-forward with SwiGLU and AdaLN
        norm_hidden = self.norm3(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        # SwiGLU activation
        gate = self.gate_proj(norm_hidden)
        up = self.up_proj(norm_hidden)
        mlp_output = self.down_proj(F.silu(gate) * up)
        mlp_output = self.dropout(mlp_output)
        
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * mlp_output
        
        return hidden_states


class BLIP3oPatchDiTModel(PreTrainedModel):
    """
    Flexible BLIP3-o Patch-Level DiT Model
    
    Supports both:
    - 256 tokens (patch only mode)
    - 257 tokens (CLS + patch mode)
    """
    
    config_class = BLIP3oDiTConfig
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__(config)
        self.config = config
        
        # Flexible token support
        self.num_tokens = getattr(config, 'num_tokens', 257)
        self.training_mode = getattr(config, 'training_mode', 'cls_patch')
        
        # Input projection from CLIP to hidden size
        self.input_proj = nn.Linear(config.clip_embedding_size, config.hidden_size, bias=True)
        
        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        
        # Flexible position embedding (supports both 256 and 257 tokens)
        max_tokens = getattr(config, 'max_position_embeddings', 257)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, config.hidden_size))
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            BLIP3oDiTBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output projection back to CLIP dimension
        self.output_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.output_proj = nn.Linear(config.hidden_size, config.clip_embedding_size, bias=True)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Zero-out output layer for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        logger.info(f"✅ Flexible BLIP3-o model initialized")
        logger.info(f"   Training mode: {self.training_mode}")
        logger.info(f"   Token count: {self.num_tokens}")
        logger.info(f"   Max position embeddings: {max_tokens}")
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def is_gradient_checkpointing(self):
        """Check if gradient checkpointing is enabled."""
        return getattr(self, '_gradient_checkpointing', False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with flexible token support
        
        Args:
            hidden_states: Noisy CLIP embeddings [B, N, 1024] where N=256 or 257
            timestep: Flow matching timesteps [B]
            encoder_hidden_states: EVA features [B, N, 4096]
            attention_mask: Optional attention mask
            return_dict: Whether to return dictionary
            
        Returns:
            Velocity prediction [B, N, 1024] or dict with additional info
        """
        
        # Validate inputs
        batch_size, seq_len, input_dim = hidden_states.shape
        
        # Flexible validation
        if seq_len not in [256, 257]:
            raise ValueError(f"Expected 256 or 257 tokens, got {seq_len}")
        
        if input_dim != self.config.clip_embedding_size:
            raise ValueError(f"Expected CLIP dim {self.config.clip_embedding_size}, got {input_dim}")
        
        if encoder_hidden_states.shape[1] != seq_len:
            raise ValueError(f"EVA and CLIP token count mismatch: {encoder_hidden_states.shape[1]} vs {seq_len}")
        
        if encoder_hidden_states.shape[2] != self.config.eva_embedding_size:
            raise ValueError(f"Expected EVA dim {self.config.eva_embedding_size}, got {encoder_hidden_states.shape[2]}")
        
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Update current token count (for inference flexibility)
        current_tokens = seq_len
        
        # Project inputs to hidden size
        x = self.input_proj(hidden_states)  # [B, N, hidden_size]
        
        # Add positional embeddings (flexible)
        if current_tokens <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :current_tokens, :]
        else:
            # Handle case where input is longer than position embeddings
            logger.warning(f"Input tokens ({current_tokens}) > position embeddings ({self.pos_embed.shape[1]})")
            # Use last position embedding for extra tokens
            pos_emb = self.pos_embed
            if current_tokens > self.pos_embed.shape[1]:
                extra_pos = self.pos_embed[:, -1:, :].repeat(1, current_tokens - self.pos_embed.shape[1], 1)
                pos_emb = torch.cat([pos_emb, extra_pos], dim=1)
            x = x + pos_emb[:, :current_tokens, :]
        
        # Timestep embedding
        timestep_emb = self.timestep_embedder(timestep)  # [B, hidden_size]
        
        # Pass through DiT blocks
        for block in self.blocks:
            if self.is_gradient_checkpointing() and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, encoder_hidden_states, timestep_emb, attention_mask,
                    use_reentrant=False
                )
            else:
                x = block(x, encoder_hidden_states, timestep_emb, attention_mask)
        
        # Output projection
        x = self.output_norm(x)
        velocity_pred = self.output_proj(x)  # [B, N, 1024]
        
        if return_dict:
            return {
                "velocity_prediction": velocity_pred,
                "hidden_states": x,
                "timestep_embeddings": timestep_emb,
                "num_tokens": current_tokens,
                "training_mode": self.training_mode,
            }
        
        return velocity_pred
    
    @torch.no_grad()
    def generate(self,
                 eva_features: torch.Tensor,  # [B, N, 4096] where N=256 or 257
                 num_inference_steps: int = 50,
                 generator: Optional[torch.Generator] = None,
                 return_intermediate: bool = False) -> torch.Tensor:
        """
        Generate CLIP embeddings using flow matching sampling
        """
        device = eva_features.device
        batch_size, num_tokens, eva_dim = eva_features.shape
        
        # Start from noise
        x = torch.randn(
            batch_size, num_tokens, self.config.clip_embedding_size,
            device=device,
            generator=generator,
            dtype=eva_features.dtype
        )
        
        # Flow matching sampling
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
    
    def set_training_mode(self, mode: str):
        """Set training mode for token handling"""
        if mode not in ["cls_patch", "patch_only"]:
            raise ValueError(f"Unknown training mode: {mode}")
        
        self.training_mode = mode
        self.num_tokens = 257 if mode == "cls_patch" else 256
        
        logger.info(f"Training mode set to: {mode} ({self.num_tokens} tokens)")


def create_blip3o_patch_dit_model(
    config: Optional[BLIP3oDiTConfig] = None,
    training_mode: str = "cls_patch",
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    **kwargs
) -> BLIP3oPatchDiTModel:
    """
    Create flexible BLIP3-o patch-level DiT model
    
    Args:
        config: Model configuration
        training_mode: "cls_patch" (257 tokens) or "patch_only" (256 tokens)
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        **kwargs: Additional config parameters
        
    Returns:
        BLIP3oPatchDiTModel instance
    """
    if config is None:
        num_tokens = 257 if training_mode == "cls_patch" else 256
        max_pos_emb = 257  # Always support up to 257 for flexibility
        
        config = BLIP3oDiTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_tokens=num_tokens,
            max_position_embeddings=max_pos_emb,
            training_mode=training_mode,
            **kwargs
        )
    
    model = BLIP3oPatchDiTModel(config)
    
    logger.info(f"✅ Flexible BLIP3-o Patch DiT model created")
    logger.info(f"   Parameters: {model.get_num_parameters():,}")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Token support: {model.num_tokens}")
    logger.info(f"   Max tokens: {config.max_position_embeddings}")
    
    return model