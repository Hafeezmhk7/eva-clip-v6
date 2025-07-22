"""
FIXED Global BLIP3-o DiT Model - Compatible Parameter Names
Place this file as: src/modules/models/global_blip3o_dit.py

KEY FIXES:
1. Updated forward() method to use standard transformer parameter names
2. Added parameter compatibility layer
3. Proper handling of both patch and global input formats
4. Enhanced error handling and input validation
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
    """
    FIXED: Create 3D rotary position embeddings for 256 tokens (16x16 grid)
    
    Key fix: Proper dimension allocation to ensure head_dim compatibility
    """
    assert embed_dim % 4 == 0, f"embed_dim {embed_dim} must be divisible by 4 for 3D RoPE"
    
    # FIXED: Allocate dimensions properly for 3D RoPE
    # Split into 3 components: x, y, temporal
    dim_per_component = embed_dim // 3
    
    # Ensure each component is even for sin/cos pairs
    if dim_per_component % 2 != 0:
        dim_per_component = (dim_per_component // 2) * 2
    
    dim_spatial = dim_per_component  # For both x and y
    dim_temporal = embed_dim - (2 * dim_spatial)  # Remainder for temporal
    
    # Ensure temporal dimension is even
    if dim_temporal % 2 != 0:
        dim_temporal -= 1
        dim_spatial = (embed_dim - dim_temporal) // 2
    
    # print(f"✅ 3D RoPE dimensions: spatial={dim_spatial}, temporal={dim_temporal}, total={2*dim_spatial + dim_temporal}")
    
    # Create inverse frequency vectors
    inv_freq_x = 1.0 / (base ** (torch.arange(0, dim_spatial, 2).float() / dim_spatial))
    inv_freq_y = 1.0 / (base ** (torch.arange(0, dim_spatial, 2).float() / dim_spatial))
    inv_freq_t = 1.0 / (base ** (torch.arange(0, dim_temporal, 2).float() / dim_temporal))
    
    # Create position grids
    x_pos = torch.arange(grid_size, dtype=torch.float32)
    y_pos = torch.arange(grid_size, dtype=torch.float32)
    t_pos = torch.arange(temporal_size, dtype=torch.float32)
    
    # Create meshgrid for spatial positions
    grid_x, grid_y = torch.meshgrid(x_pos, y_pos, indexing='ij')
    grid_x = grid_x.flatten()  # [256]
    grid_y = grid_y.flatten()  # [256]
    
    # Compute frequency encodings
    freqs_x = torch.outer(grid_x, inv_freq_x)  # [256, dim_spatial//2]
    freqs_y = torch.outer(grid_y, inv_freq_y)  # [256, dim_spatial//2]
    freqs_t = torch.outer(t_pos, inv_freq_t)   # [1, dim_temporal//2]
    
    # Generate cos and sin for each dimension
    cos_x = torch.cos(freqs_x)
    sin_x = torch.sin(freqs_x)
    cos_y = torch.cos(freqs_y)
    sin_y = torch.sin(freqs_y)
    cos_t = torch.cos(freqs_t)
    sin_t = torch.sin(freqs_t)
    
    # Expand to full dimensions
    cos_x_full = torch.stack([cos_x, cos_x], dim=-1).flatten(-2)  # [256, dim_spatial]
    sin_x_full = torch.stack([sin_x, sin_x], dim=-1).flatten(-2)
    cos_y_full = torch.stack([cos_y, cos_y], dim=-1).flatten(-2)
    sin_y_full = torch.stack([sin_y, sin_y], dim=-1).flatten(-2)
    
    # Expand temporal to match spatial
    cos_t_expanded = cos_t.expand(grid_size * grid_size, -1)  # [256, dim_temporal//2]
    sin_t_expanded = sin_t.expand(grid_size * grid_size, -1)
    cos_t_full = torch.stack([cos_t_expanded, cos_t_expanded], dim=-1).flatten(-2)  # [256, dim_temporal]
    sin_t_full = torch.stack([sin_t_expanded, sin_t_expanded], dim=-1).flatten(-2)
    
    # Combine all components
    cos_emb = torch.cat([cos_x_full, cos_y_full, cos_t_full], dim=-1)  # [256, embed_dim]
    sin_emb = torch.cat([sin_x_full, sin_y_full, sin_t_full], dim=-1)  # [256, embed_dim]
    
    # Add batch dimension
    cos_emb = cos_emb.unsqueeze(0)  # [1, 256, embed_dim]
    sin_emb = sin_emb.unsqueeze(0)  # [1, 256, embed_dim]
    
    # FIXED: Ensure exact dimension match
    assert cos_emb.shape[-1] == embed_dim, f"Dimension mismatch: {cos_emb.shape[-1]} != {embed_dim}"
    assert sin_emb.shape[-1] == embed_dim, f"Dimension mismatch: {sin_emb.shape[-1]} != {embed_dim}"
    
    return cos_emb, sin_emb


def apply_rotary_pos_emb_3d(q, k, cos, sin):
    """
    FIXED: Apply 3D rotary position embedding to query and key tensors
    
    Args:
        q: query tensor [batch_size, seq_len, num_heads, head_dim]
        k: key tensor [batch_size, seq_len, num_heads, head_dim]  
        cos: cosine embeddings [1, seq_len, head_dim]
        sin: sine embeddings [1, seq_len, head_dim]
    """
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # FIXED: Handle cos/sin tensor dimensions properly
    # cos and sin should be [1, seq_len, embed_dim] from get_3d_rotary_pos_embed
    
    # Ensure cos/sin match the sequence length and head dimension
    if cos.dim() == 3:  # [1, seq_len, embed_dim]
        cos = cos[:, :seq_len, :head_dim].to(q.device)  # [1, seq_len, head_dim]
        sin = sin[:, :seq_len, :head_dim].to(q.device)  # [1, seq_len, head_dim]
    else:
        # Handle other dimension cases
        cos = cos[:seq_len, :head_dim].to(q.device)
        sin = sin[:seq_len, :head_dim].to(q.device)
        cos = cos.unsqueeze(0)  # Add batch dim: [1, seq_len, head_dim]
        sin = sin.unsqueeze(0)  # Add batch dim: [1, seq_len, head_dim]
    
    # FIXED: Proper expansion to match q and k dimensions
    # We need: [batch_size, seq_len, num_heads, head_dim]
    
    # Expand batch dimension: [1, seq_len, head_dim] -> [batch_size, seq_len, head_dim]
    cos = cos.expand(batch_size, -1, -1)
    sin = sin.expand(batch_size, -1, -1)
    
    # Add head dimension: [batch_size, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
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
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, patch_embeddings):
        # patch_embeddings: [B, 256, dim]
        batch_size = patch_embeddings.shape[0]
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # [B, 1, dim]
        
        # Apply layer norm
        patch_embeddings = self.norm(patch_embeddings)
        
        # Attention pooling
        pooled, _ = self.attention(query, patch_embeddings, patch_embeddings)
        
        return pooled.squeeze(1)  # [B, dim]


class GlobalDiTBlock(nn.Module):
    """DiT block optimized for global training with fixed 3D RoPE"""
    
    def __init__(self, dim, num_heads, eva_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # FIXED: Ensure head_dim compatibility
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
    FIXED Global BLIP3-o DiT Model - Compatible Parameter Names
    
    Architecture: EVA-CLIP [B, 256, 4096] → DiT → Attention Pool → MLP → [B, 768]
    Training Target: [B, 768] global embeddings (matches evaluation)
    
    FIXED: Uses standard transformer parameter names for compatibility
    """
    
    config_class = BLIP3oDiTConfig
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__(config)
        self.config = config
        
        # Fixed dimensions
        self.eva_dim = 4096
        self.clip_global_dim = 768
        
        # FIXED: Calculate compatible patch_dim
        self.patch_dim = self._calculate_compatible_patch_dim(config.n_heads)
        self.head_dim = self.patch_dim // config.n_heads
        
        print(f"✅ FIXED Global Model Dimensions:")
        print(f"   patch_dim: {self.patch_dim}")
        print(f"   n_heads: {config.n_heads}")
        print(f"   head_dim: {self.head_dim}")
        print(f"   RoPE compatible: {self.head_dim % 4 == 0}")
        
        # Input projections
        self.eva_proj = nn.Linear(self.eva_dim, self.patch_dim)
        self.patch_embed = nn.Linear(self.clip_global_dim, self.patch_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, self.patch_dim) * 0.02)
        
        # FIXED: Timestep embedding with proper dimensions
        self.time_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, self.patch_dim),
            nn.SiLU(),
            nn.Linear(self.patch_dim, self.patch_dim),
        )
        
        # Create sinusoidal timestep projection
        self.register_buffer(
            "time_proj_weights",
            self._create_sinusoidal_weights(self.time_dim // 2)
        )
        
        # DiT backbone
        self.layers = nn.ModuleList([
            GlobalDiTBlock(self.patch_dim, config.n_heads, self.patch_dim, dropout=0.1)
            for _ in range(config.n_layers)
        ])
        
        # Attention-based pooling
        self.pooling = AttentionPooling(self.patch_dim, num_heads=min(8, config.n_heads))
        
        # Global adaptation MLP
        self.global_adapter = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.mlp_hidden_dim, config.mlp_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.mlp_hidden_dim // 2, 1024),  # CLIP space
            nn.LayerNorm(1024),
        )
        
        # Load frozen CLIP projection
        self.frozen_clip_proj = None
        self._init_weights()
        
        print(f"✅ Global BLIP3-o DiT initialized successfully")
    
    def _calculate_compatible_patch_dim(self, n_heads):
        """Calculate patch_dim that's compatible with n_heads and 3D RoPE"""
        # Common head dimensions that work well with RoPE
        good_head_dims = [32, 48, 64, 80, 96, 128]
        
        for head_dim in good_head_dims:
            if head_dim % 4 == 0:  # RoPE compatibility
                patch_dim = n_heads * head_dim
                if 256 <= patch_dim <= 1536:  # Reasonable range
                    return patch_dim
        
        # Fallback: force head_dim = 64
        return n_heads * 64
    
    def _create_sinusoidal_weights(self, half_dim):
        """Create sinusoidal timestep embedding weights"""
        emb = math.log(10000) / (half_dim - 1)
        return torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        
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
        try:
            clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.frozen_clip_proj = clip_model.visual_projection
            
            # Freeze parameters
            for param in self.frozen_clip_proj.parameters():
                param.requires_grad = False
            
            print(f"✅ Frozen CLIP projection loaded: {self.frozen_clip_proj.weight.shape}")
        except Exception as e:
            print(f"⚠️ Failed to load CLIP projection: {e}")
            # Fallback projection
            self.frozen_clip_proj = nn.Linear(1024, 768, bias=False)
            nn.init.xavier_uniform_(self.frozen_clip_proj.weight)
            self.frozen_clip_proj.requires_grad_(False)
            print(f"⚠️ Using fallback projection")
    
    def get_timestep_embedding(self, timesteps):
        """Create sinusoidal timestep embeddings"""
        # Clamp and scale timesteps
        timesteps = torch.clamp(timesteps, 0.0, 1.0) * 1000.0
        
        device = timesteps.device
        half_dim = len(self.time_proj_weights)
        emb = self.time_proj_weights.to(device=device, dtype=timesteps.dtype)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Ensure correct dimension
        if emb.shape[-1] != self.time_dim:
            emb = F.pad(emb, (0, self.time_dim - emb.shape[-1]))
        
        return emb
    
    def _convert_patch_to_global(self, patch_features):
        """
        Convert patch features to global features for input processing
        
        Args:
            patch_features: [B, 256, 1024] CLIP patch features
            
        Returns:
            global_features: [B, 768] global features
        """
        # Pool patches to global representation
        pooled = patch_features.mean(dim=1)  # [B, 1024]
        
        # Apply CLIP projection to get [B, 768]
        if self.frozen_clip_proj is not None:
            # Move to correct device if needed
            if self.frozen_clip_proj.weight.device != pooled.device:
                self.frozen_clip_proj = self.frozen_clip_proj.to(pooled.device)
            global_features = self.frozen_clip_proj(pooled)  # [B, 768]
        else:
            # Fallback: create projection on the fly
            global_features = F.linear(
                pooled, 
                torch.randn(768, 1024, device=pooled.device, dtype=pooled.dtype) * 0.02
            )
        
        return global_features
    
    def forward(
        self,
        hidden_states=None,        # [B, 768] - Noisy global features OR [B, 256, 1024] patches
        timestep=None,             # [B] - Timesteps  
        encoder_hidden_states=None, # [B, 256, 4096] - EVA conditioning
        return_dict=True,
        
        # Legacy parameter names for backward compatibility
        noisy_global_features=None,
        eva_features=None,
        **kwargs
    ):
        """
        FIXED Forward pass with compatible parameter names
        
        Args:
            hidden_states: Input features (can be global [B, 768] or patches [B, 256, 1024])
            timestep: Timesteps [B]
            encoder_hidden_states: EVA conditioning [B, 256, 4096]
            return_dict: Whether to return dict
            
        Returns:
            Predicted global velocity [B, 768]
        """
        
        # FIXED: Parameter compatibility layer
        if hidden_states is None and noisy_global_features is not None:
            hidden_states = noisy_global_features
        
        if encoder_hidden_states is None and eva_features is not None:
            encoder_hidden_states = eva_features
        
        # Validate inputs
        if hidden_states is None:
            raise ValueError("hidden_states (or noisy_global_features) is required")
        if timestep is None:
            raise ValueError("timestep is required")
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states (or eva_features) is required")
        
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # FIXED: Handle both global and patch inputs
        if hidden_states.dim() == 3 and hidden_states.shape[1] == 256:
            # Input is patch features [B, 256, 1024] - convert to global
            print(f"🔄 Converting patch features to global: {hidden_states.shape}")
            global_features = self._convert_patch_to_global(hidden_states)
        elif hidden_states.dim() == 2 and hidden_states.shape[1] == 768:
            # Input is already global features [B, 768]
            global_features = hidden_states
        else:
            raise ValueError(f"Unsupported hidden_states shape: {hidden_states.shape}. "
                           f"Expected [B, 768] or [B, 256, 1024]")
        
        # Validate EVA features
        if encoder_hidden_states.shape != (batch_size, 256, 4096):
            raise ValueError(f"encoder_hidden_states must be [B, 256, 4096], got {encoder_hidden_states.shape}")
        
        # Project EVA features
        eva_proj = self.eva_proj(encoder_hidden_states)  # [B, 256, patch_dim]
        
        # Expand global features to patch format for DiT processing
        expanded_features = self.patch_embed(global_features).unsqueeze(1)  # [B, 1, patch_dim]
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
        rope_embeddings = (cos_emb.to(device), sin_emb.to(device))
        
        # DiT layers
        for layer in self.layers:
            x = layer(x, eva_proj, timestep_emb, rope_embeddings)
        
        # Pool to global representation
        global_features_processed = self.pooling(x)  # [B, patch_dim]
        
        # Global adaptation
        adapted_features = self.global_adapter(global_features_processed)  # [B, 1024]
        
        # Apply CLIP projection
        if self.frozen_clip_proj is not None:
            output = self.frozen_clip_proj(adapted_features)  # [B, 768]
        else:
            # Fallback: create projection on the fly
            output = F.linear(
                adapted_features, 
                torch.randn(768, 1024, device=device, dtype=adapted_features.dtype) * 0.02
            )
        
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
        sample = torch.randn(
            batch_size, 768, 
            device=device, 
            generator=generator,
            dtype=eva_features.dtype
        )
        
        # Flow matching sampling
        dt = 1.0 / num_inference_steps
        
        for step in range(num_inference_steps):
            t = step * dt
            t_tensor = torch.full((batch_size,), t, device=device)
            
            # Predict velocity using standard parameter names
            velocity = self.forward(
                hidden_states=sample, 
                timestep=t_tensor, 
                encoder_hidden_states=eva_features, 
                return_dict=False
            )
            
            # Euler step
            sample = sample + dt * velocity
        
        # Final normalization
        return F.normalize(sample, p=2, dim=-1)


def create_global_blip3o_dit_model(
    config=None,
    load_clip_projection=True,
    clip_model_name="openai/clip-vit-large-patch14",
    **kwargs
):
    """Create FIXED global BLIP3-o DiT model"""
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
    
    print(f"✅ Global BLIP3-o model created successfully")
    return model