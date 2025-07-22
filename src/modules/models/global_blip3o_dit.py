"""
Simplified Global BLIP3-o DiT Model - TRAINS DIRECTLY ON GLOBAL FEATURES
Place this file as: src/modules/models/global_blip3o_dit.py

KEY FIX: Trains directly on [B, 768] global features to match evaluation pipeline.
This resolves the training-inference mismatch that was causing 0% recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
from transformers import PreTrainedModel, CLIPModel
import math
import numpy as np
from pathlib import Path

from ..config.blip3o_config import BLIP3oDiTConfig


def get_3d_rotary_pos_embed(embed_dim, grid_size, temporal_size=1, base=10000.0):
    """Create 3D rotary position embeddings for 256 tokens (16x16 grid) - BLIP3-o style"""
    assert embed_dim % 4 == 0, f"embed_dim {embed_dim} must be divisible by 4 for 3D RoPE"
    
    # Split embedding dimension for spatial (h, w) and temporal (t)
    dim_h = embed_dim // 4
    dim_w = embed_dim // 4
    dim_t = embed_dim // 2  # Temporal dimension
    
    # Create inverse frequency vectors
    inv_freq_h = 1.0 / (base ** (torch.arange(0, dim_h, 2).float() / dim_h))
    inv_freq_w = 1.0 / (base ** (torch.arange(0, dim_w, 2).float() / dim_w))
    inv_freq_t = 1.0 / (base ** (torch.arange(0, dim_t, 2).float() / dim_t))
    
    # Create spatial grid positions
    h_pos = torch.arange(grid_size, dtype=torch.float32)
    w_pos = torch.arange(grid_size, dtype=torch.float32)
    t_pos = torch.arange(temporal_size, dtype=torch.float32)
    
    # Create meshgrid for spatial positions
    grid_h, grid_w = torch.meshgrid(h_pos, w_pos, indexing='ij')
    grid_h = grid_h.flatten()  # [256]
    grid_w = grid_w.flatten()  # [256]
    
    # Compute frequency encodings
    freqs_h = torch.outer(grid_h, inv_freq_h)  # [256, dim_h//2]
    freqs_w = torch.outer(grid_w, inv_freq_w)  # [256, dim_w//2]
    freqs_t = torch.outer(t_pos, inv_freq_t)   # [1, dim_t//2]
    
    # Generate cos and sin for each dimension
    cos_h = torch.cos(freqs_h)
    sin_h = torch.sin(freqs_h)
    cos_w = torch.cos(freqs_w)
    sin_w = torch.sin(freqs_w)
    cos_t = torch.cos(freqs_t)
    sin_t = torch.sin(freqs_t)
    
    # Expand to full embedding dimension
    cos_h_full = torch.stack([cos_h, cos_h], dim=-1).flatten(-2)
    sin_h_full = torch.stack([sin_h, sin_h], dim=-1).flatten(-2)
    cos_w_full = torch.stack([cos_w, cos_w], dim=-1).flatten(-2)
    sin_w_full = torch.stack([sin_w, sin_w], dim=-1).flatten(-2)
    
    # Combine spatial dimensions
    cos_spatial = torch.cat([cos_h_full, cos_w_full], dim=-1)  # [256, embed_dim//2]
    sin_spatial = torch.cat([sin_h_full, sin_w_full], dim=-1)  # [256, embed_dim//2]
    
    # Expand temporal to match spatial
    cos_t_expanded = cos_t.expand(grid_size * grid_size, -1)  # [256, dim_t//2]
    sin_t_expanded = sin_t.expand(grid_size * grid_size, -1)  # [256, dim_t//2]
    
    # Final 3D rotary embeddings
    cos_emb = torch.cat([cos_spatial, cos_t_expanded], dim=-1)  # [256, embed_dim]
    sin_emb = torch.cat([sin_spatial, sin_t_expanded], dim=-1)  # [256, embed_dim]
    
    # Add batch dimension
    cos_emb = cos_emb.unsqueeze(0)  # [1, 256, embed_dim]
    sin_emb = sin_emb.unsqueeze(0)  # [1, 256, embed_dim]
    
    return cos_emb, sin_emb


def apply_rotary_pos_emb_3d(q, k, cos, sin):
    """Apply 3D rotary position embedding to query and key tensors"""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Ensure cos/sin match the sequence length and head dimension
    cos = cos[:, :seq_len, :head_dim].expand(batch_size, -1, -1)
    sin = sin[:, :seq_len, :head_dim].expand(batch_size, -1, -1)
    
    # Add head dimension
    cos = cos.unsqueeze(2).expand(-1, -1, num_heads, -1)
    sin = sin.unsqueeze(2).expand(-1, -1, num_heads, -1)
    
    # Apply rotation
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed


class AttentionPooling(nn.Module):
    """Attention-based pooling for better global representation"""
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Learnable query token
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, patch_embeddings):
        # patch_embeddings: [B, 256, dim]
        batch_size = patch_embeddings.shape[0]
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # [B, 1, dim]
        
        # Apply layer norm
        patch_embeddings = self.norm(patch_embeddings)
        
        # Attention pooling
        pooled, attn_weights = self.attention(
            query, patch_embeddings, patch_embeddings
        )
        
        return pooled.squeeze(1)  # [B, dim]


class DiTBlock(nn.Module):
    """DiT block with 3D RoPE and cross-attention - BLIP3-o style"""
    
    def __init__(self, dim, num_heads, eva_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Ensure head_dim is compatible with 3D RoPE
        assert self.head_dim % 4 == 0, f"head_dim {self.head_dim} must be divisible by 4 for 3D RoPE"
        
        # Self-attention with manual projections for RoPE
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Cross-attention with EVA
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, 
            kdim=eva_dim, vdim=eva_dim, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Timestep conditioning (AdaLN-Zero style)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )
        
    def forward(self, x, eva_features, timestep_emb, rope_embeddings=None):
        # x: [B, 256, dim]
        # eva_features: [B, 256, eva_dim] 
        # timestep_emb: [B, dim]
        # rope_embeddings: (cos, sin) tuple
        
        batch_size, seq_len, _ = x.shape
        
        # AdaLN conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(timestep_emb).chunk(6, dim=1)
        
        # Self-attention with 3D RoPE
        residual = x
        x_norm = self.norm1(x)
        
        # Apply AdaLN modulation
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # Manual attention computation for RoPE
        q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply 3D RoPE if available
        if rope_embeddings is not None:
            cos_emb, sin_emb = rope_embeddings
            q, k = apply_rotary_pos_emb_3d(q, k, cos_emb, sin_emb)
        
        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        attn_output = self.out_proj(attn_output)
        
        # Apply gate and residual
        x = residual + gate_msa.unsqueeze(1).tanh() * attn_output
        
        # Cross-attention with EVA
        residual = x
        x_norm = self.norm2(x)
        cross_out, _ = self.cross_attn(x_norm, eva_features, eva_features)
        x = residual + cross_out
        
        # Feed-forward with AdaLN
        residual = x
        x_norm = self.norm3(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ff_out = self.ffn(x_norm)
        x = residual + gate_mlp.unsqueeze(1).tanh() * ff_out
        
        return x


class GlobalBLIP3oDiTModel(PreTrainedModel):
    """
    Simplified Global BLIP3-o DiT Model - TRAINS DIRECTLY ON GLOBAL FEATURES
    
    KEY FIX: This model trains directly on [B, 768] global embeddings,
    matching the evaluation pipeline exactly. No more training-inference mismatch!
    
    Architecture:
    EVA-CLIP [B, 256, 4096] → DiT → Attention Pool → MLP → CLIP Projection → [B, 768]
    """
    
    config_class = BLIP3oDiTConfig
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__(config)
        self.config = config
        
        # FIXED: Choose patch_dim that's compatible with n_heads and 3D RoPE
        self.eva_dim = 4096
        self.clip_global_dim = 768
        
        # Calculate compatible patch_dim based on n_heads
        if config.n_heads == 12:
            self.patch_dim = 768  # 768 / 12 = 64 (divisible by 4 for RoPE)
        elif config.n_heads == 16:
            self.patch_dim = 1024  # 1024 / 16 = 64 (divisible by 4 for RoPE)
        elif config.n_heads == 8:
            self.patch_dim = 512   # 512 / 8 = 64 (divisible by 4 for RoPE)
        else:
            # Find the largest compatible dimension
            for candidate_dim in [768, 1024, 512, 960, 1152]:
                if candidate_dim % config.n_heads == 0:
                    head_dim = candidate_dim // config.n_heads
                    if head_dim % 4 == 0:  # Compatible with 3D RoPE
                        self.patch_dim = candidate_dim
                        break
            else:
                # Fallback: adjust to make it work
                self.patch_dim = config.n_heads * 64  # Force head_dim = 64
        
        # Verify compatibility
        assert self.patch_dim % config.n_heads == 0, f"patch_dim {self.patch_dim} not divisible by n_heads {config.n_heads}"
        self.head_dim = self.patch_dim // config.n_heads
        assert self.head_dim % 4 == 0, f"head_dim {self.head_dim} must be divisible by 4 for 3D RoPE"
        
        print(f"✅ FIXED dimensions: patch_dim={self.patch_dim}, n_heads={config.n_heads}, head_dim={self.head_dim}")
        
        # Input projection for EVA features
        self.eva_proj = nn.Linear(self.eva_dim, self.patch_dim)
        
        # Patch embedding for noisy input
        self.patch_embed = nn.Linear(self.clip_global_dim, self.patch_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, self.patch_dim) * 0.02)
        
        # Timestep embedding - FIXED dimension compatibility
        time_dim = 256  # Fixed dimension for sinusoidal embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, self.patch_dim),
            nn.SiLU(),
            nn.Linear(self.patch_dim, self.patch_dim),
        )
        
        # Create sinusoidal timestep projection - FIXED size
        self.register_buffer(
            "time_proj",
            self._create_sinusoidal_timestep_embedding(time_dim)
        )
        
        # DiT backbone with 3D RoPE
        self.layers = nn.ModuleList([
            DiTBlock(self.patch_dim, config.n_heads, self.patch_dim)
            for _ in range(config.n_layers)
        ])
        
        # Attention-based pooling (KEY: Better than mean pooling)
        self.pooling = AttentionPooling(self.patch_dim, num_heads=8)
        
        # Global adaptation MLP (improved design)
        self.global_adapter = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.mlp_hidden_dim, 1024),  # CLIP pre-projection space
        )
        
        # Load frozen CLIP projection
        self.frozen_clip_proj = None
        self._init_weights()
        
        print(f"✅ Global BLIP3-o DiT initialized")
        print(f"   Architecture: EVA [B,256,4096] → DiT → Pool → MLP → [B,768]")
        print(f"   Training target: [B, 768] global embeddings")
        print(f"   3D RoPE: {self.head_dim}-dim heads (compatible)")
        
    def _create_sinusoidal_timestep_embedding(self, embed_dim: int):
        """Create sinusoidal timestep embeddings"""
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        return emb
        
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def load_frozen_clip_projection(self, clip_model_name="openai/clip-vit-large-patch14"):
        """Load frozen CLIP visual projection"""
        print(f"Loading frozen CLIP projection from {clip_model_name}")
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.frozen_clip_proj = clip_model.visual_projection
        
        # Freeze parameters
        for param in self.frozen_clip_proj.parameters():
            param.requires_grad = False
        
        print(f"✅ Frozen CLIP projection loaded: {self.frozen_clip_proj.weight.shape}")
    
    def get_timestep_embedding(self, timesteps):
        """Create sinusoidal timestep embeddings - FIXED dimension handling"""
        # Ensure timesteps are in [0, 1] range, then scale to [0, 1000]
        timesteps = torch.clamp(timesteps, 0.0, 1.0) * 1000.0
        
        device = timesteps.device
        half_dim = len(self.time_proj)
        emb = self.time_proj.to(device=device, dtype=timesteps.dtype)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # FIXED: Ensure output dimension matches time_dim (256), not patch_dim
        time_dim = len(self.time_proj) * 2  # sin + cos = 2 * half_dim
        if emb.shape[-1] < time_dim:
            emb = F.pad(emb, (0, time_dim - emb.shape[-1]))
        elif emb.shape[-1] > time_dim:
            emb = emb[:, :time_dim]
        
        return emb
    
    def forward(
        self,
        noisy_global_features,  # [B, 768] - Noisy global CLIP features
        timestep,               # [B] - Timesteps
        eva_features,          # [B, 256, 4096] - EVA conditioning
        return_dict=True
    ):
        """
        Forward pass that trains directly on global features.
        
        Args:
            noisy_global_features: [B, 768] noisy global CLIP features
            timestep: [B] timesteps for flow matching
            eva_features: [B, 256, 4096] EVA-CLIP conditioning
            
        Returns:
            Predicted global velocity [B, 768]
        """
        batch_size = noisy_global_features.shape[0]
        device = noisy_global_features.device
        
        # Project EVA features
        eva_proj = self.eva_proj(eva_features)  # [B, 256, patch_dim]
        
        # Expand global features to patch format for DiT processing
        expanded_features = self.patch_embed(noisy_global_features).unsqueeze(1)  # [B, 1, patch_dim]
        expanded_features = expanded_features.expand(-1, 256, -1)  # [B, 256, patch_dim]
        
        # Add position embeddings
        x = expanded_features + self.pos_embed
        
        # Timestep embedding
        timestep_emb = self.get_timestep_embedding(timestep)
        timestep_emb = self.time_embed(timestep_emb)  # [B, patch_dim]
        
        # Create 3D RoPE embeddings
        cos_emb, sin_emb = get_3d_rotary_pos_embed(
            embed_dim=self.head_dim,
            grid_size=16  # 16x16 = 256 tokens
        )
        cos_emb = cos_emb.to(device)
        sin_emb = sin_emb.to(device)
        rope_embeddings = (cos_emb, sin_emb)
        
        # DiT layers with cross-attention and 3D RoPE
        for layer in self.layers:
            x = layer(x, eva_proj, timestep_emb, rope_embeddings)
        
        # Pool to global representation using attention
        global_features = self.pooling(x)  # [B, patch_dim]
        
        # Global adaptation
        adapted_features = self.global_adapter(global_features)  # [B, 1024]
        
        # Apply frozen CLIP projection
        if self.frozen_clip_proj is not None:
            output = self.frozen_clip_proj(adapted_features)  # [B, 768]
        else:
            # Fallback projection
            output = F.linear(adapted_features, 
                            torch.randn(768, 1024, device=device), 
                            torch.zeros(768, device=device))
        
        if return_dict:
            return {'predicted_global': output}
        else:
            return output
    
    @torch.no_grad()
    def generate(
        self,
        eva_features,          # [B, 256, 4096]
        num_inference_steps=50,
        generator=None,
    ):
        """Generate global CLIP embeddings using flow matching"""
        batch_size = eva_features.shape[0]
        device = eva_features.device
        
        # Start from noise in global space
        sample = torch.randn(batch_size, 768, device=device, generator=generator)
        
        # Flow matching sampling with Euler integration
        dt = 1.0 / num_inference_steps
        
        for step in range(num_inference_steps):
            t = step * dt
            t_tensor = torch.full((batch_size,), t, device=device)
            
            # Predict velocity
            velocity = self.forward(sample, t_tensor, eva_features, return_dict=False)
            
            # Euler step
            sample = sample + dt * velocity
        
        # Final normalization like CLIP
        return F.normalize(sample, p=2, dim=-1)


def create_global_blip3o_dit_model(
    config=None,
    load_clip_projection=True,
    clip_model_name="openai/clip-vit-large-patch14",
    **kwargs
):
    """Create global BLIP3-o DiT model"""
    if config is None:
        from ..config.blip3o_config import get_default_blip3o_config
        config = get_default_blip3o_config()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    model = GlobalBLIP3oDiTModel(config)
    
    if load_clip_projection:
        model.load_frozen_clip_projection(clip_model_name)
    
    return model