"""
src/modules/models/blip3o_patch_dit.py
Flexible BLIP3-o Patch-Level DiT Model - Support for 256/257 tokens
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np
import logging
from transformers import PreTrainedModel
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

class BLIP3oDiTConfig(PretrainedConfig):
    """Configuration class for BLIP3-o DiT model"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        eva_embedding_size: int = 4096,
        clip_embedding_size: int = 1024,
        num_tokens: int = 257,
        max_position_embeddings: int = 257,
        dropout_prob: float = 0.1,
        training_mode: str = "cls_patch"
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.eva_embedding_size = eva_embedding_size
        self.clip_embedding_size = clip_embedding_size
        self.num_tokens = num_tokens
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        self.training_mode = training_mode

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class RotaryPositionalEmbedding3D(nn.Module):
    """3D Rotary Position Embedding with flexible token support"""
    def __init__(self, dim: int, max_position_embeddings: int = 257):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 4, 2).float() / (dim // 4)))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        if seq_len not in [256, 257]:
            return x
        
        # Create position IDs based on token count
        if seq_len == 257:
            # CLS token at (0,0), patches in 16x16 grid
            cls_pos = torch.zeros(1, device=x.device)
            patch_x = torch.arange(16, device=x.device).repeat(16)
            patch_y = torch.arange(16, device=x.device).repeat_interleave(16)
            pos_x = torch.cat([cls_pos, patch_x])
            pos_y = torch.cat([cls_pos, patch_y])
        else:
            # Only patches in 16x16 grid
            patch_x = torch.arange(16, device=x.device).repeat(16)
            patch_y = torch.arange(16, device=x.device).repeat_interleave(16)
            pos_x = patch_x
            pos_y = patch_y
        
        # Apply rotary embedding to spatial dimensions
        rope_dim = min(hidden_size, len(self.inv_freq) * 4)
        if rope_dim < 4:
            return x
        
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        
        # Apply 2D rotary position embedding
        x_rot = self._apply_2d_rotary_pos_emb(x_rope, pos_x, pos_y)
        return torch.cat([x_rot, x_pass], dim=-1)
    
    def _apply_2d_rotary_pos_emb(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, rope_dim = x.shape
        quarter_dim = rope_dim // 4
        
        # Split into 4 parts for x and y dimensions
        x1, x2, x3, x4 = torch.split(x, quarter_dim, dim=-1)
        
        # Compute frequencies
        freqs_x = torch.einsum('i,j->ij', pos_x, self.inv_freq[:quarter_dim])
        freqs_y = torch.einsum('i,j->ij', pos_y, self.inv_freq[:quarter_dim])
        
        cos_x, sin_x = freqs_x.cos(), freqs_x.sin()
        cos_y, sin_y = freqs_y.cos(), freqs_y.sin()
        
        # Apply 2D rotation
        x1_rot = x1 * cos_x - x2 * sin_x
        x2_rot = x1 * sin_x + x2 * cos_x
        x3_rot = x3 * cos_y - x4 * sin_y
        x4_rot = x3 * sin_y + x4 * cos_y
        
        return torch.cat([x1_rot, x2_rot, x3_rot, x4_rot], dim=-1)

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
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

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
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Project to q, k, v
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

class BLIP3oDiTBlock(nn.Module):
    """DiT block with flexible token support"""
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Attention layers
        self.self_attn = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.dropout_prob)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        
        # EVA feature projection
        self.eva_proj = nn.Linear(config.eva_embedding_size, config.hidden_size)
        
        # 3D Rotary Position Embedding
        self.rope = RotaryPositionalEmbedding3D(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention with RoPE
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = self.rope(norm_hidden)
        attn_output = self.self_attn(norm_hidden, norm_hidden, norm_hidden)
        hidden_states = hidden_states + attn_output
        
        # Cross-attention with EVA features
        norm_hidden = self.norm2(hidden_states)
        eva_features = self.eva_proj(encoder_hidden_states)
        cross_attn_output = self.self_attn(norm_hidden, eva_features, eva_features)
        hidden_states = hidden_states + cross_attn_output
        
        # Feed-forward network
        ffn_output = self.ffn(hidden_states)
        return hidden_states + ffn_output

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
        
        # Input projection from CLIP to hidden size
        self.input_proj = nn.Linear(config.clip_embedding_size, config.hidden_size, bias=True)
        
        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            BLIP3oDiTBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output projection back to CLIP dimension
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.output_proj = nn.Linear(config.hidden_size, config.clip_embedding_size, bias=True)
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        logger.info(f"✅ Flexible BLIP3-o model initialized")
        logger.info(f"   Training mode: {config.training_mode}")
        logger.info(f"   Token count: {config.num_tokens}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Validate inputs
        batch_size, seq_len, input_dim = hidden_states.shape
        if seq_len not in [256, 257]:
            raise ValueError(f"Expected 256 or 257 tokens, got {seq_len}")
        
        # Project inputs to hidden size
        x = self.input_proj(hidden_states)
        
        # Add positional embeddings
        if seq_len <= self.config.max_position_embeddings:
            x = x + self.pos_embed[:, :seq_len, :]
        else:
            raise ValueError(f"Sequence length {seq_len} exceeds max position embeddings")
        
        # Timestep embedding
        timestep_emb = self.timestep_embedder(timestep)
        
        # Pass through DiT blocks
        for block in self.blocks:
            x = block(x, encoder_hidden_states)
        
        # Output projection
        x = self.output_norm(x)
        velocity_pred = self.output_proj(x)
        
        if return_dict:
            return {
                "velocity_prediction": velocity_pred,
                "hidden_states": x,
                "timestep_embeddings": timestep_emb,
            }
        return velocity_pred
    
    @torch.no_grad()
    def generate(
        self,
        eva_features: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = eva_features.device
        batch_size, num_tokens, _ = eva_features.shape
        
        # Start from noise
        x = torch.randn(
            batch_size, num_tokens, self.config.clip_embedding_size,
            device=device,
            generator=generator,
            dtype=eva_features.dtype
        )
        
        # Flow matching sampling
        dt = 1.0 / num_inference_steps
        
        for step in range(num_inference_steps):
            t = torch.full((batch_size,), step * dt, device=device, dtype=eva_features.dtype)
            velocity = self.forward(
                hidden_states=x,
                timestep=t,
                encoder_hidden_states=eva_features,
                return_dict=False
            )
            x = x + dt * velocity
        
        return x
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_blip3o_patch_dit_model(
    config: Optional[BLIP3oDiTConfig] = None,
    training_mode: str = "cls_patch",
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
) -> BLIP3oPatchDiTModel:
    if config is None:
        num_tokens = 257 if training_mode == "cls_patch" else 256
        config = BLIP3oDiTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_tokens=num_tokens,
            training_mode=training_mode,
        )
    return BLIP3oPatchDiTModel(config)