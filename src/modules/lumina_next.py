# lumina_next.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TimestepEmbedder(nn.Module):
    """Embeds timesteps into vector representations"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)

class LuminaDiTBlock(nn.Module):
    """Transformer block with self-attention and cross-attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, cross_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Cross-attention layer
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            kdim=cross_dim or dim,
            vdim=cross_dim or dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, cond):
        # Self-attention
        x_norm1 = self.norm1(x)
        x = x + self.self_attn(x_norm1, x_norm1, x_norm1)[0]
        
        # Cross-attention with conditioning
        x_norm2 = self.norm2(x)
        x = x + self.cross_attn(
            query=x_norm2,
            key=cond,
            value=cond
        )[0]
        
        # MLP
        x_norm3 = self.norm3(x)
        x = x + self.mlp(x_norm3)
        return x

class LuminaDiT(nn.Module):
    """Flow Matching DiT for EVA→CLIP embedding translation with cross-attention"""
    def __init__(self, 
        input_dim=768,        # CLIP embedding size
        cond_dim=768,         # EVA embedding size
        dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0
    ):
        super().__init__()
        # Input projection
        self.input_proj = nn.Linear(input_dim, dim)
        
        # Conditioning projection
        self.cond_proj = nn.Linear(cond_dim, dim)
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            LuminaDiTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cross_dim=dim
            ) for _ in range(depth)
        ])
        
        # Output layers
        self.norm_out = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x, t, cond):
        """
        x: Noisy CLIP embeddings [B, D]
        t: Timesteps [B]
        cond: EVA embeddings [B, D_cond]
        """
        # Ensure consistent dtype with model parameters
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        t = t.to(dtype)
        cond = cond.to(dtype)
        
        # Project inputs
        x = self.input_proj(x).unsqueeze(1)  # [B, 1, dim]
        
        # Project conditioning and add timestep embedding
        cond = self.cond_proj(cond) + self.t_embedder(t)
        cond = cond.unsqueeze(1)  # [B, 1, dim]
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, cond)
            
        # Output projection
        x = self.norm_out(x)
        return self.output_proj(x.squeeze(1))  # [B, input_dim]