# src/modules/lumina_next.py
"""
Enhanced Lumina-Next Implementation with BLIP3-o Improvements
Contains both Enhanced (Time RoPE) and Original LuminaDiT models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class TimeRoPE(nn.Module):
    """Time-only Rotary Position Embedding for temporal modeling (BLIP3-o aligned)"""
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Compute frequency basis for rotary embeddings
        self.register_buffer('freqs', self._compute_freqs())
    
    def _compute_freqs(self) -> torch.Tensor:
        """Compute frequency basis for rotary embeddings"""
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        return freqs
    
    def _apply_rotary_emb(self, x: torch.Tensor, freqs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to input tensor"""
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)  # [1, seq_len] or [1, batch]
        
        # Handle different position tensor shapes
        if pos.shape[0] == 1 and x.shape[0] > 1:
            pos = pos.expand(x.shape[0], -1)  # Expand to batch size
        elif pos.shape[1] == 1 and x.shape[1] > 1:
            pos = pos.expand(-1, x.shape[1])  # Expand to sequence length
            
        # Compute angles: pos * freqs
        angles = pos.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # [batch, seq_len, dim//2]
        
        # Split x into pairs for rotation
        x1, x2 = x.chunk(2, dim=-1)  # Each: [batch, seq_len, dim//2]
        
        # Apply rotation
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        x_rotated = torch.cat([
            x1 * cos_angles - x2 * sin_angles,
            x1 * sin_angles + x2 * cos_angles
        ], dim=-1)
        
        return x_rotated
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply Time RoPE to input tensor"""
        batch_size, seq_len, dim = x.shape
        
        # Handle time dimension - use timestep for temporal encoding
        if t is not None:
            if t.dim() == 1:  # [batch] -> expand to [batch, seq_len]
                pos_t = t.unsqueeze(1).expand(batch_size, seq_len)
            else:  # [batch, seq_len] 
                pos_t = t
        else:
            # Use sequence positions as time if not provided
            pos_t = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(batch_size, -1)
        
        # Apply rotary embeddings for time dimension only
        return self._apply_rotary_emb(x, self.freqs, pos_t)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for better efficiency"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class KQNorm(nn.Module):
    """Key-Query Normalization to prevent attention collapse (BLIP3-o improvement)"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize queries and keys to prevent attention collapse
        q = self.norm_q(q)
        k = self.norm_k(k)
        # Value doesn't need normalization
        return q, k, v

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with KQ-Norm for stability"""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: Optional[int] = None, use_kq_norm: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = dim // num_heads
        self.use_kq_norm = use_kq_norm
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.group_size = num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # KQ-Norm for stability
        if self.use_kq_norm:
            self.kq_norm = KQNorm(self.head_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        kv_seq_len = key.shape[1]
        
        # Project to q, k, v
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply KQ-Norm if enabled
        if self.use_kq_norm:
            # Reshape for norm application
            q_flat = q.view(-1, self.head_dim)
            k_flat = k.view(-1, self.head_dim)
            v_flat = v.view(-1, self.head_dim)
            
            q_flat, k_flat, v_flat = self.kq_norm(q_flat, k_flat, v_flat)
            
            # Reshape back
            q = q_flat.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k_flat.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)
            v = v_flat.view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads, kv_seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_heads, kv_seq_len, head_dim]
        
        # Expand k, v to match number of query heads
        k = k.repeat_interleave(self.group_size, dim=1)  # [batch, num_heads, kv_seq_len, head_dim]
        v = v.repeat_interleave(self.group_size, dim=1)  # [batch, num_heads, kv_seq_len, head_dim]
        
        # Attention computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(out)

class SwiGLU(nn.Module):
    """SwiGLU activation function used in modern transformers"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, linear = x.chunk(2, dim=-1)
        return F.silu(gate) * linear


class TimestepEmbedder(nn.Module):
    """Embeds timesteps into vector representations (fallback for non-Time RoPE)"""
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

# ENHANCED VERSION - Use this for tasks that benefit from temporal modeling
class EnhancedDiTBlock(nn.Module):
    """Enhanced Transformer block with Time RoPE, sandwich norm, KQ-norm"""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: Optional[int] = None, 
                 mlp_ratio: float = 4.0, cross_dim: Optional[int] = None, use_kq_norm: bool = True):
        super().__init__()
        self.dim = dim
        
        # Sandwich normalization layers (BLIP3-o style)
        self.norm1_pre = RMSNorm(dim)
        self.norm1_post = RMSNorm(dim)
        self.norm2_pre = RMSNorm(dim)
        self.norm2_post = RMSNorm(dim)
        self.norm3_pre = RMSNorm(dim)
        self.norm3_post = RMSNorm(dim)
        
        # Self-attention with grouped query attention and KQ-norm
        self.self_attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, use_kq_norm)
        
        # Cross-attention with grouped query attention and KQ-norm
        self.cross_attn = GroupedQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_kq_norm=use_kq_norm
        )
        
        # MLP with SwiGLU activation
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim * 2, bias=False),  # Gate and linear projections
            SwiGLU(),
            nn.Linear(mlp_hidden_dim, dim, bias=False),
        )
        
        # Time RoPE for positional encoding
        self.time_rope = TimeRoPE(dim)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, 
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with sandwich normalization and Time RoPE"""
        # Apply Time RoPE to input
        x_rope = self.time_rope(x, t)
        
        # Sandwich normalization structure
        # First sandwich: Self-attention
        x_norm1_pre = self.norm1_pre(x_rope)
        attn_out = self.self_attn(x_norm1_pre, x_norm1_pre, x_norm1_pre)
        x = x + self.norm1_post(attn_out)
        
        # Second sandwich: Cross-attention  
        x_norm2_pre = self.norm2_pre(x)
        cross_out = self.cross_attn(x_norm2_pre, cond, cond)
        x = x + self.norm2_post(cross_out)
        
        # Third sandwich: MLP
        x_norm3_pre = self.norm3_pre(x)
        mlp_out = self.mlp(x_norm3_pre)
        x = x + self.norm3_post(mlp_out)
        
        return x
    
class EnhancedLuminaDiT(nn.Module):
    """Enhanced Flow Matching DiT using EnhancedDiTBlock components"""
    def __init__(self, 
        input_dim: int = 768,        # CLIP embedding size
        cond_dim: int = 768,         # EVA embedding size
        dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        num_kv_heads: Optional[int] = None,  # For grouped query attention
        mlp_ratio: float = 4.0,
        max_seq_len: int = 8192,
        use_kq_norm: bool = True
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, dim)
        
        # Conditioning projection  
        self.cond_proj = nn.Linear(cond_dim, dim)
        
        # Time RoPE for time embedding (replaces 3D RoPE)
        self.time_rope = TimeRoPE(dim, max_seq_len)
        
        # Enhanced transformer blocks
        self.blocks = nn.ModuleList([
            EnhancedDiTBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                cross_dim=dim,
                use_kq_norm=use_kq_norm
            ) for _ in range(depth)
        ])
        
        # Output layers with RMSNorm
        self.norm_out = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f" Enhanced LuminaDiT initialized:")
        print(f"    Architecture: {depth} blocks, {num_heads} heads, {dim} dim")
        print(f"    Time RoPE: Enabled for temporal modeling")  
        print(f"    Sandwich norm: Enabled for training stability")
        print(f"    Grouped QA: {num_heads}/{num_kv_heads or num_heads} heads")
        print(f"    KQ-Norm: {'Enabled' if use_kq_norm else 'Disabled'}")
        
    def _init_weights(self, module):
        """Initialize weights with proper scaling"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, RMSNorm):
            nn.init.constant_(module.weight, 1.0)
                
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced architecture"""
        # Ensure consistent dtype
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        t = t.to(dtype)
        cond = cond.to(dtype)
        
        # Project inputs
        x = self.input_proj(x).unsqueeze(1)  # [B, 1, dim]
        
        # Project conditioning with time embedding using Time RoPE
        cond_projected = self.cond_proj(cond).unsqueeze(1)  # [B, 1, dim]
        
        # Apply time encoding using Time RoPE instead of 3D spatial splitting
        time_emb = self.time_rope(cond_projected, t=t)
        cond = cond_projected + time_emb
        
        # Process through enhanced transformer blocks
        for block in self.blocks:
            x = block(x, cond, t=t)
            
        # Output projection with RMSNorm
        x = self.norm_out(x)
        return self.output_proj(x.squeeze(1))  # [B, input_dim]

# ORIGINAL VERSION - Recommended for embedding translation tasks
class DiTBlock(nn.Module):
    """Original transformer block with self-attention and cross-attention"""
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
    """Original Flow Matching DiT using DiTBlock components"""
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
            DiTBlock(
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
        
        print(f" Original LuminaDiT initialized:")
        print(f"    Architecture: {depth} blocks, {num_heads} heads, {dim} dim")
        print(f"    Optimized for embedding translation tasks")
        print(f"    No spatial assumptions, clean semantic mapping")
        print(f"    Proven stable for EVA-CLIP → CLIP L-14 alignment")
        
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