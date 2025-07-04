import torch
import torch.nn as nn
import torch.nn.functional as F

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

class LuminaDiT(nn.Module):
    """Flow Matching DiT for EVA→CLIP embedding translation"""
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
            self._build_block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        # Output layers
        self.norm_out = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_block(self, dim, num_heads, mlp_ratio):
        return nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )

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
        x = self.input_proj(x)
        c = self.cond_proj(cond) + self.t_embedder(t)
        
        # Combine conditioning
        x = x + c.unsqueeze(1)  # Add as sequence element
        
        # Process through transformer
        for block in self.blocks:
            x = block(x)
            
        # Output projection
        x = self.norm_out(x)
        return self.output_proj(x.mean(dim=1))  # Pool sequence