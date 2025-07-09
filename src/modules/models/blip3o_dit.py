"""
BLIP3-o DiT Model Implementation - FIXED VERSION
Exact implementation of BLIP3-o diffusion transformer architecture using NextDiT backbone.
Fixed with proper 3D Rotary Position Embedding (3D RoPE) implementation following Lumina-Next.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union
from transformers import PreTrainedModel
import math

from ..config.blip3o_config import BLIP3oDiTConfig


def get_3d_rotary_pos_embed(embed_dim, grid_size, temporal_size=1, base=10000.0):
    """
    Create 3D rotary position embeddings following Lumina-Next implementation.
    
    Args:
        embed_dim: Embedding dimension (must be divisible by 4 for 2D spatial + 1D temporal)
        grid_size: Spatial grid size (e.g., 8 for 8x8 = 64 tokens)
        temporal_size: Temporal dimension (1 for image generation)
        base: Base frequency for RoPE
        
    Returns:
        cos_emb, sin_emb: Cosine and sine embeddings for 3D RoPE
    """
    # Ensure embed_dim is divisible by 4 (2 for height, 2 for width)
    assert embed_dim % 4 == 0, f"embed_dim {embed_dim} must be divisible by 4 for 3D RoPE"
    
    # Divide embedding dimension into spatial (height, width) components
    # Following Lumina-Next: height gets embed_dim//4, width gets embed_dim//4
    dim_h = embed_dim // 4
    dim_w = embed_dim // 4
    
    # Create frequency vectors for each dimension
    inv_freq_h = 1.0 / (base ** (torch.arange(0, dim_h, 2).float() / dim_h))
    inv_freq_w = 1.0 / (base ** (torch.arange(0, dim_w, 2).float() / dim_w))
    
    # Create spatial position grids
    h_pos = torch.arange(grid_size, dtype=torch.float32)  # [grid_size]
    w_pos = torch.arange(grid_size, dtype=torch.float32)  # [grid_size]
    
    # Create 2D grid positions for 8x8 -> 64 tokens
    grid_h, grid_w = torch.meshgrid(h_pos, w_pos, indexing='ij')  # [grid_size, grid_size]
    grid_h = grid_h.flatten()  # [64]
    grid_w = grid_w.flatten()  # [64]
    
    # Compute frequency interactions
    # Height frequencies: [64, dim_h//2]
    freqs_h = torch.outer(grid_h, inv_freq_h)
    # Width frequencies: [64, dim_w//2] 
    freqs_w = torch.outer(grid_w, inv_freq_w)
    
    # Create cosine and sine embeddings
    cos_h = torch.cos(freqs_h)  # [64, dim_h//2]
    sin_h = torch.sin(freqs_h)  # [64, dim_h//2]
    cos_w = torch.cos(freqs_w)  # [64, dim_w//2]
    sin_w = torch.sin(freqs_w)  # [64, dim_w//2]
    
    # Interleave cos and sin for proper rotation
    # Height component: [64, dim_h]
    cos_h_full = torch.stack([cos_h, cos_h], dim=-1).flatten(-2)
    sin_h_full = torch.stack([sin_h, sin_h], dim=-1).flatten(-2)
    
    # Width component: [64, dim_w]  
    cos_w_full = torch.stack([cos_w, cos_w], dim=-1).flatten(-2)
    sin_w_full = torch.stack([sin_w, sin_w], dim=-1).flatten(-2)
    
    # Concatenate height and width components
    # Total: [64, dim_h + dim_w] = [64, embed_dim//2]
    cos_emb = torch.cat([cos_h_full, cos_w_full], dim=-1)
    sin_emb = torch.cat([sin_h_full, sin_w_full], dim=-1)
    
    # Add batch and head dimensions: [1, 64, embed_dim//2]
    cos_emb = cos_emb.unsqueeze(0)
    sin_emb = sin_emb.unsqueeze(0)
    
    return cos_emb, sin_emb


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor [batch_size, seq_len, num_heads, head_dim] 
        cos: Cosine embedding [1, seq_len, head_dim//2]
        sin: Sine embedding [1, seq_len, head_dim//2]
        
    Returns:
        Rotated query and key tensors
    """
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Get dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # cos, sin should be [1, seq_len, head_dim//2]
    # We need to expand them to [batch_size, seq_len, num_heads, head_dim//2]
    
    # First, expand batch dimension: [batch_size, seq_len, head_dim//2]
    cos = cos.expand(batch_size, -1, -1)
    sin = sin.expand(batch_size, -1, -1)
    
    # Then, add head dimension: [batch_size, seq_len, 1, head_dim//2]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    
    # Expand to all heads: [batch_size, seq_len, num_heads, head_dim//2]
    cos = cos.expand(-1, -1, num_heads, -1)
    sin = sin.expand(-1, -1, num_heads, -1)
    
    # Apply rotation to each half of the head_dim
    # Split q and k into two halves
    q1 = q[..., : head_dim // 2]  # [batch_size, seq_len, num_heads, head_dim//2]
    q2 = q[..., head_dim // 2 :]  # [batch_size, seq_len, num_heads, head_dim//2]
    
    k1 = k[..., : head_dim // 2]  # [batch_size, seq_len, num_heads, head_dim//2]
    k2 = k[..., head_dim // 2 :]  # [batch_size, seq_len, num_heads, head_dim//2]
    
    # Apply rotation: [cos, -sin; sin, cos] * [q1; q2]
    q_rot1 = q1 * cos - q2 * sin
    q_rot2 = q1 * sin + q2 * cos
    
    k_rot1 = k1 * cos - k2 * sin
    k_rot2 = k1 * sin + k2 * cos
    
    # Concatenate back
    q_embed = torch.cat([q_rot1, q_rot2], dim=-1)
    k_embed = torch.cat([k_rot1, k_rot2], dim=-1)
    
    return q_embed, k_embed


class SimpleTokenEmbedder(nn.Module):
    """
    Simple embedding layer for pre-tokenized BLIP3-o features.
    Since we're working with pre-extracted 64-token embeddings, we don't need 
    complex patch embedding - just a linear transformation.
    """
    
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Simple linear transformation for pre-tokenized features
        self.proj = nn.Linear(in_channels, embed_dim, bias=True)
        
        # Position embeddings for 8x8 grid (64 tokens) - optional, since we use RoPE
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor, image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass for token embedding.
        
        Args:
            x: Input tokens [B, 64, in_channels]
            image_rotary_emb: Tuple of (cos_emb, sin_emb) for 3D RoPE (not used here)
            
        Returns:
            Tuple of (embedded_tokens, attention_mask, image_size, rotary_emb)
        """
        batch_size, num_tokens, in_channels = x.shape
        
        # Validate input
        assert num_tokens == 64, f"Expected 64 tokens, got {num_tokens}"
        assert in_channels == self.in_channels, f"Expected {self.in_channels} channels, got {in_channels}"
        
        # Linear projection
        embedded = self.proj(x)  # [B, 64, embed_dim]
        
        # Add position embeddings (optional with RoPE)
        embedded = embedded + self.pos_embed
        
        # Create attention mask (all tokens are valid)
        attention_mask = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        
        # Image size (8x8 for 64 tokens)
        img_size = [(8, 8)] * batch_size
        
        # Pass through the provided rotary embeddings (will be created in main model)
        return embedded, attention_mask, img_size, image_rotary_emb


class BLIP3oAttentionBlock(nn.Module):
    """
    Fixed DiT block for BLIP3-o that works with pre-tokenized embeddings and proper 3D RoPE.
    Following the exact Lumina-Next architecture.
    """
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_attention_heads,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_attention_heads,
            kdim=cross_attention_dim,
            vdim=cross_attention_dim,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm3 = nn.LayerNorm(dim, eps=norm_eps)
        self.cross_norm = nn.LayerNorm(cross_attention_dim, eps=norm_eps)
        
        # Timestep conditioning
        self.time_proj = nn.Linear(dim, dim * 6)  # For various gates and scales
        
        # Manual query/key/value projections for RoPE application
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        
    def forward(
        self,
        hidden_states: torch.Tensor,              # [B, 64, dim]
        encoder_hidden_states: torch.Tensor,     # [B, 64, cross_attention_dim]
        timestep_emb: torch.Tensor,              # [B, dim]
        attention_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through BLIP3-o DiT block with proper 3D RoPE."""
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get timestep conditioning
        time_cond = self.time_proj(timestep_emb)  # [B, dim * 6]
        time_chunks = time_cond.chunk(6, dim=-1)  # 6 chunks of [B, dim]
        
        scale_msa, gate_msa, scale_mlp, gate_mlp, scale_cross, gate_cross = time_chunks
        
        # Self-attention with timestep conditioning and 3D RoPE
        residual = hidden_states
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa.unsqueeze(1))
        
        # Apply 3D RoPE to self-attention if available
        if image_rotary_emb is not None:
            cos_emb, sin_emb = image_rotary_emb
            
            # Manual Q, K, V computation for RoPE
            q = self.q_proj(norm_hidden)  # [B, 64, dim]
            k = self.k_proj(norm_hidden)  # [B, 64, dim]
            v = self.v_proj(norm_hidden)  # [B, 64, dim]
            
            # Reshape for multi-head attention: [B, 64, num_heads, head_dim]
            q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            
            # Apply RoPE
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
            
            # Reshape back for attention: [B, 64, dim]
            q_rot = q_rot.view(batch_size, seq_len, self.dim)
            k_rot = k_rot.view(batch_size, seq_len, self.dim)
            v = v.view(batch_size, seq_len, self.dim)
            
            # Compute attention manually with proper tensor handling
            # Transpose and reshape with contiguous() to avoid stride issues
            q_for_attn = q_rot.transpose(1, 2).contiguous().view(batch_size * self.num_attention_heads, seq_len, self.head_dim)
            k_for_attn = k_rot.transpose(1, 2).contiguous().view(batch_size * self.num_attention_heads, seq_len, self.head_dim)
            v_for_attn = v.transpose(1, 2).contiguous().view(batch_size * self.num_attention_heads, seq_len, self.head_dim)
            
            # Don't pass attention_mask since all tokens are valid and mask would need reshaping
            attn_output = F.scaled_dot_product_attention(
                q_for_attn,
                k_for_attn,
                v_for_attn,
                attn_mask=None,  # Skip mask since all tokens are valid
                dropout_p=0.0,
                is_causal=False
            )
            
            # Reshape and project output
            attn_output = attn_output.view(batch_size, self.num_attention_heads, seq_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
            attn_output = self.out_proj(attn_output)
            
        else:
            # Fallback to standard attention without RoPE
            attn_output, _ = self.self_attn(
                norm_hidden, norm_hidden, norm_hidden,
                attn_mask=attention_mask,
                need_weights=False
            )
        
        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * attn_output
        
        # Cross-attention with EVA-CLIP conditioning
        residual = hidden_states
        norm_hidden = self.norm2(hidden_states)
        norm_encoder = self.cross_norm(encoder_hidden_states)
        norm_hidden = norm_hidden * (1 + scale_cross.unsqueeze(1))
        
        cross_attn_output, _ = self.cross_attn(
            norm_hidden, norm_encoder, norm_encoder,
            key_padding_mask=~encoder_mask if encoder_mask is not None else None,
            need_weights=False
        )
        
        hidden_states = residual + gate_cross.unsqueeze(1).tanh() * cross_attn_output
        
        # Feed-forward with timestep conditioning
        residual = hidden_states
        norm_hidden = self.norm3(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp.unsqueeze(1))
        
        ffn_output = self.ffn(norm_hidden)
        hidden_states = residual + gate_mlp.unsqueeze(1).tanh() * ffn_output
        
        return hidden_states


class BLIP3oDiTModel(PreTrainedModel):
    """
    BLIP3-o Diffusion Transformer Model with FIXED 3D RoPE implementation.
    
    This model implements the exact BLIP3-o architecture for generating CLIP embeddings
    from EVA-CLIP conditioning using flow matching with proper 3D Rotary Position Embedding.
    
    Architecture:
    - Input: Noisy CLIP features [B, 64, 1024] + EVA-CLIP conditioning [B, 64, 4096]
    - Backbone: DiT transformer with 3D RoPE and cross-attention
    - Output: Velocity field predictions [B, 64, 1024] for flow matching
    """
    
    config_class = BLIP3oDiTConfig
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__(config)
        
        # Store configuration
        self.config = config
        self._gradient_checkpointing = config._gradient_checkpointing
        
        # Validate configuration for BLIP3-o
        self._validate_blip3o_config(config)
        
        # Ensure head_dim is compatible with 3D RoPE
        self.head_dim = config.dim // config.n_heads
        
        # Check if head_dim is compatible with 3D RoPE (must be divisible by 4)
        if self.head_dim % 4 != 0:
            print(f"⚠️  Head dimension {self.head_dim} not divisible by 4, adjusting for 3D RoPE compatibility...")
            
            # Find compatible dimensions
            compatible_found = False
            
            # Strategy 1: Adjust num_heads to make head_dim divisible by 4
            for candidate_heads in range(1, config.dim + 1):
                if config.dim % candidate_heads == 0:  # Must be divisible for MultiheadAttention
                    candidate_head_dim = config.dim // candidate_heads
                    if candidate_head_dim % 4 == 0:  # Must be divisible by 4 for 3D RoPE
                        config.n_heads = candidate_heads
                        config.n_kv_heads = candidate_heads  
                        self.head_dim = candidate_head_dim
                        compatible_found = True
                        print(f"✅ Adjusted num_heads to {candidate_heads} (head_dim={candidate_head_dim})")
                        break
            
            # Strategy 2: If no compatible num_heads found, adjust dim
            if not compatible_found:
                # Find the largest dim <= original_dim that works
                original_heads = config.n_heads
                for candidate_dim in range(config.dim, 0, -4):  # Step by 4 for RoPE compatibility
                    if candidate_dim % original_heads == 0:  # Must work with original heads
                        candidate_head_dim = candidate_dim // original_heads
                        if candidate_head_dim % 4 == 0:
                            config.dim = candidate_dim
                            self.head_dim = candidate_head_dim
                            print(f"✅ Adjusted dim to {candidate_dim} (head_dim={candidate_head_dim})")
                            compatible_found = True
                            break
                
                # Strategy 3: Use safe default
                if not compatible_found:
                    config.dim = 512  # Safe default
                    config.n_heads = 8
                    config.n_kv_heads = 8
                    self.head_dim = 64
                    print(f"✅ Using safe default: dim=512, heads=8, head_dim=64")
        else:
            print(f"✅ Head dimension {self.head_dim} is compatible with 3D RoPE")
        
        # Final validation
        assert config.dim % config.n_heads == 0, f"dim {config.dim} must be divisible by num_heads {config.n_heads}"
        assert self.head_dim % 4 == 0, f"head_dim {self.head_dim} must be divisible by 4 for 3D RoPE"
        
        # Token embedder for pre-tokenized inputs
        self.token_embedder = SimpleTokenEmbedder(
            in_channels=config.in_channels,
            embed_dim=config.dim
        )
        
        # Timestep embedding
        time_embed_dim = min(config.dim, 1024)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        
        # Timestep frequency embedding
        self.time_proj = self._create_sinusoidal_timestep_embedding(time_embed_dim)
        
        # EVA-CLIP conditioning projection
        self.eva_proj = nn.Linear(config.eva_embedding_size, config.dim)
        
        # Transformer layers with 3D RoPE
        self.layers = nn.ModuleList([
            BLIP3oAttentionBlock(
                dim=config.dim,
                num_attention_heads=config.n_heads,
                cross_attention_dim=config.dim,  # After EVA projection
                norm_eps=config.norm_eps,
                qk_norm=config.qk_norm,
            )
            for _ in range(config.n_layers)
        ])
        
        # Output layer
        self.norm_out = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.proj_out = nn.Linear(config.dim, config.in_channels, bias=True)
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ BLIP3-o DiT model with FIXED 3D RoPE initialized")
        print(f"   Parameters: {self.get_num_parameters():,}")
        print(f"   Final dimensions: dim={config.dim}, heads={config.n_heads}, head_dim={self.head_dim}")
        print(f"   3D RoPE compatible: head_dim % 4 = {self.head_dim % 4}")
    
    def _create_sinusoidal_timestep_embedding(self, embed_dim: int):
        """Create sinusoidal timestep embedding layer."""
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        return emb
    
    def _validate_blip3o_config(self, config: BLIP3oDiTConfig):
        """Validate configuration specific to BLIP3-o requirements."""
        assert config.learn_sigma is False, "BLIP3-o uses flow matching, sigma learning must be False"
        assert config.in_channels > 0, "CLIP embedding dimension must be positive"
        assert config.eva_embedding_size > 0, "EVA-CLIP conditioning dimension must be positive"
        assert config.input_size == 8, "BLIP3-o uses 8x8 = 64 token format"
        assert config.patch_size == 1, "Features are pre-tokenized, patch_size must be 1"
        
        # Log the dimensions being used (helpful for debugging)
        print(f"🔧 Model configured for:")
        print(f"   CLIP dimension: {config.in_channels}")
        print(f"   EVA-CLIP dimension: {config.eva_embedding_size}")
        print(f"   Hidden dimension: {config.dim}")
        print(f"   Tokens: {config.input_size}x{config.input_size} = {config.input_size * config.input_size}")
    
    def _init_weights(self):
        """Initialize model weights following BLIP3-o methodology."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True
        print("✅ Gradient checkpointing enabled")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        print("✅ Gradient checkpointing disabled")
    
    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get sinusoidal timestep embeddings."""
        device = timesteps.device
        dtype = timesteps.dtype
        
        # Ensure timesteps are in [0, 1] range for flow matching
        timesteps = torch.clamp(timesteps, 0.0, 1.0)
        
        # Scale to a larger range for better embedding
        timesteps = timesteps * 1000.0
        
        # Create sinusoidal embeddings
        half_dim = len(self.time_proj)
        emb = self.time_proj.to(device=device, dtype=dtype)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, 64, 1024] - Noisy CLIP features
        timestep: torch.Tensor,                # [B] - Flow matching timesteps
        encoder_hidden_states: torch.Tensor,  # [B, 64, 4096] - EVA-CLIP conditioning
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, 64] - Attention mask
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass of BLIP3-o DiT model with FIXED 3D RoPE.
        
        Args:
            hidden_states: Noisy CLIP features [batch_size, 64, 1024]
            timestep: Flow matching timesteps [batch_size] or scalar
            encoder_hidden_states: EVA-CLIP conditioning [batch_size, 64, 4096]
            encoder_attention_mask: Optional attention mask [batch_size, 64]
            cross_attention_kwargs: Additional cross-attention arguments
            return_dict: Whether to return ModelOutput object
            
        Returns:
            Predicted velocity field [batch_size, 64, 1024] for flow matching
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # Validate input shapes
        self._validate_forward_inputs(hidden_states, timestep, encoder_hidden_states)
        
        # Handle timestep format
        if timestep.dim() == 0:  # Scalar timestep
            timestep = timestep.unsqueeze(0).expand(batch_size)
        elif timestep.shape[0] != batch_size:
            raise ValueError(f"Timestep batch size {timestep.shape[0]} != hidden states batch size {batch_size}")
        
        # Create encoder attention mask if not provided
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                (batch_size, encoder_hidden_states.shape[1]),
                device=device,
                dtype=torch.bool
            )
        
        # Embed input tokens (without creating RoPE here)
        hidden_states, attention_mask, img_size, _ = self.token_embedder(hidden_states)
        
        # Create 3D RoPE embeddings with correct head dimension
        cos_emb, sin_emb = get_3d_rotary_pos_embed(
            embed_dim=self.head_dim,  # Use head_dim for RoPE
            grid_size=8  # 8x8 = 64 tokens
        )
        cos_emb = cos_emb.to(device)
        sin_emb = sin_emb.to(device)
        image_rotary_emb = (cos_emb, sin_emb)
        
        # Get timestep embeddings
        timestep_emb = self.get_timestep_embedding(timestep)
        timestep_emb = self.time_embed(timestep_emb)  # [B, dim]
        
        # Project EVA-CLIP conditioning
        encoder_hidden_states = self.eva_proj(encoder_hidden_states)  # [B, 64, dim]
        
        # Pass through transformer layers with 3D RoPE
        for layer in self.layers:
            if self.training and self._gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    encoder_hidden_states,
                    timestep_emb,
                    attention_mask,
                    encoder_attention_mask,
                    image_rotary_emb,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_emb=timestep_emb,
                    attention_mask=attention_mask,
                    encoder_mask=encoder_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
        
        # Output projection
        hidden_states = self.norm_out(hidden_states)
        output = self.proj_out(hidden_states)  # [B, 64, 1024]
        
        if return_dict:
            from transformers.modeling_outputs import BaseModelOutput
            return BaseModelOutput(last_hidden_state=output)
        else:
            return output
    
    def _validate_forward_inputs(
        self, 
        hidden_states: torch.Tensor, 
        timestep: torch.Tensor, 
        encoder_hidden_states: torch.Tensor
    ):
        """Validate forward pass inputs."""
        # Check hidden_states shape [B, 64, clip_dim]
        if hidden_states.shape[1] != 64:
            raise ValueError(f"Expected 64 tokens, got {hidden_states.shape[1]}")
        if hidden_states.shape[2] != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels}-dim CLIP features, got {hidden_states.shape[2]}")
        
        # Check encoder_hidden_states shape [B, 64, eva_dim]
        if encoder_hidden_states.shape[1] != 64:
            raise ValueError(f"Expected 64 conditioning tokens, got {encoder_hidden_states.shape[1]}")
        if encoder_hidden_states.shape[2] != self.config.eva_embedding_size:
            raise ValueError(f"Expected {self.config.eva_embedding_size}-dim EVA-CLIP features, got {encoder_hidden_states.shape[2]}")
        
        # Check batch size consistency
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError(f"Batch size mismatch: {hidden_states.shape[0]} vs {encoder_hidden_states.shape[0]}")
    
    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,  # [B, 64, 4096] - EVA-CLIP conditioning
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,  # DDIM parameter
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate CLIP embeddings using flow matching sampling with FIXED 3D RoPE.
        
        Args:
            encoder_hidden_states: EVA-CLIP conditioning [batch_size, 64, 4096]
            num_inference_steps: Number of sampling steps
            guidance_scale: Guidance scale (not used in current flow matching)
            generator: Random number generator for reproducibility
            eta: DDIM parameter for stochasticity
            return_intermediate: Whether to return intermediate states
            
        Returns:
            Generated CLIP embeddings [batch_size, 64, 1024]
        """
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        dtype = encoder_hidden_states.dtype
        
        # Initialize from random noise (source distribution)
        sample = torch.randn(
            (batch_size, 64, self.config.in_channels),
            device=device,
            dtype=dtype,
            generator=generator
        )
        
        # Flow matching sampling with Euler integration
        dt = 1.0 / num_inference_steps
        intermediate_samples = [] if return_intermediate else None
        
        self.eval()
        
        for step in range(num_inference_steps):
            # Current time in [0, 1]
            t = step * dt
            t_tensor = torch.full((batch_size,), t, device=device, dtype=dtype)
            
            # Predict velocity field
            velocity = self.forward(
                hidden_states=sample,
                timestep=t_tensor,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )
            
            # Euler integration step: x_{t+dt} = x_t + dt * v_t
            sample = sample + dt * velocity
            
            if return_intermediate:
                intermediate_samples.append(sample.clone())
        
        if return_intermediate:
            return sample, intermediate_samples
        else:
            return sample
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Get the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_memory_footprint(self) -> str:
        """Get approximate memory footprint of the model."""
        num_params = self.get_num_parameters(trainable_only=False)
        # Assuming float32 parameters (4 bytes each)
        memory_mb = num_params * 4 / (1024 * 1024)
        return f"{memory_mb:.1f} MB"


def create_blip3o_dit_model(
    config: Optional[BLIP3oDiTConfig] = None,
    **kwargs
) -> BLIP3oDiTModel:
    """
    Factory function to create a BLIP3-o DiT model.
    
    Args:
        config: Model configuration. If None, uses default configuration.
        **kwargs: Additional configuration parameters to override
        
    Returns:
        BLIP3oDiTModel instance
    """
    if config is None:
        from ..config.blip3o_config import get_default_blip3o_config
        config = get_default_blip3o_config()
    
    # Override config with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}' ignored")
    
    return BLIP3oDiTModel(config)


def load_blip3o_dit_model(
    model_path: str,
    device: str = "auto",
    torch_dtype: Optional[torch.dtype] = None
) -> BLIP3oDiTModel:
    """
    Load a pre-trained BLIP3-o DiT model.
    
    Args:
        model_path: Path to the model directory or checkpoint
        device: Device to load the model on
        torch_dtype: Data type for the model
        
    Returns:
        Loaded BLIP3oDiTModel instance
    """
    from transformers import AutoModel
    
    # Load the model using transformers
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device if device != "auto" else None,
    )
    
    return model