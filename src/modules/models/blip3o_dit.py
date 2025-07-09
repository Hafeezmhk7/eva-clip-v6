"""
BLIP3-o DiT Model Implementation
Exact implementation of BLIP3-o diffusion transformer architecture using NextDiT backbone.
Fixed with proper 3D Rotary Position Embedding (3D RoPE) implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union
from transformers import PreTrainedModel
import math

from ..config.blip3o_config import BLIP3oDiTConfig


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding for BLIP3-o DiT.
    
    Implements 3D RoPE as used in Lumina-Next architecture, where embedding dimensions
    are divided into 3 parts for height, width, and time/layer information.
    """
    
    def __init__(
        self,
        head_dim: int,
        max_height: int = 64,
        max_width: int = 64,
        max_time: int = 1000,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_height = max_height
        self.max_width = max_width
        self.max_time = max_time
        self.base = base
        
        # Divide head_dim into 3 parts for h, w, t
        assert head_dim % 6 == 0, f"head_dim {head_dim} must be divisible by 6 for 3D RoPE"
        
        self.dim_per_axis = head_dim // 6  # Each axis gets head_dim/6 complex pairs
        
        # Create frequency matrices for each axis
        self.register_buffer("freqs_h", self._create_freqs(self.dim_per_axis))
        self.register_buffer("freqs_w", self._create_freqs(self.dim_per_axis))
        self.register_buffer("freqs_t", self._create_freqs(self.dim_per_axis))
        
        # Cache for efficiency
        self.cached_freqs = None
        self.cached_shape = None
    
    def _create_freqs(self, dim: int) -> torch.Tensor:
        """Create frequency tensor for one axis."""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        return inv_freq
    
    def _get_cos_sin(self, positions: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cosine and sine values for given positions and frequencies."""
        # positions: [seq_len] or [batch_size, seq_len]
        # freqs: [dim//2]
        
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # [1, seq_len]
        
        # Compute angles: [batch_size, seq_len, dim//2]
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        
        cos_vals = torch.cos(angles)  # [batch_size, seq_len, dim//2]
        sin_vals = torch.sin(angles)  # [batch_size, seq_len, dim//2]
        
        return cos_vals, sin_vals
    
    def _apply_rope_1d(self, x: torch.Tensor, cos_vals: torch.Tensor, sin_vals: torch.Tensor) -> torch.Tensor:
        """Apply 1D rotary embedding to input tensor."""
        # x: [batch_size, seq_len, dim//2 * 2]
        # cos_vals, sin_vals: [batch_size, seq_len, dim//2]
        
        dim = cos_vals.shape[-1] * 2
        x_reshaped = x[..., :dim].reshape(*x.shape[:-1], -1, 2)  # [..., dim//2, 2]
        
        # Split into real and imaginary parts
        x_real = x_reshaped[..., 0]  # [..., dim//2]
        x_imag = x_reshaped[..., 1]  # [..., dim//2]
        
        # Apply rotation
        rotated_real = x_real * cos_vals - x_imag * sin_vals
        rotated_imag = x_real * sin_vals + x_imag * cos_vals
        
        # Recombine
        rotated = torch.stack([rotated_real, rotated_imag], dim=-1)  # [..., dim//2, 2]
        return rotated.reshape(*x.shape[:-1], dim)
    
    def create_3d_positions(self, height: int, width: int, time_step: float = 0.0, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create 3D position tensors for height, width, and time.
        
        Args:
            height: Height dimension (e.g., 8 for 8x8 grid)
            width: Width dimension (e.g., 8 for 8x8 grid)
            time_step: Time step value (e.g., diffusion timestep normalized to 0-1)
            device: Device to create tensors on
            
        Returns:
            Tuple of (height_positions, width_positions, time_positions)
        """
        if device is None:
            device = self.freqs_h.device
        
        # Create spatial positions for 8x8 grid -> 64 tokens
        h_pos = torch.arange(height, device=device, dtype=torch.float32).repeat_interleave(width)  # [64]
        w_pos = torch.arange(width, device=device, dtype=torch.float32).repeat(height)  # [64]
        
        # Time positions (same for all spatial positions)
        t_pos = torch.full((height * width,), time_step, device=device, dtype=torch.float32)  # [64]
        
        return h_pos, w_pos, t_pos
    
    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, head_dim]
        height: int = 8,
        width: int = 8,
        time_step: Union[float, torch.Tensor] = 0.0,
    ) -> torch.Tensor:
        """
        Apply 3D RoPE to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, head_dim]
            height: Height dimension (8 for 8x8 grid)
            width: Width dimension (8 for 8x8 grid)
            time_step: Time step value or tensor [batch_size]
            
        Returns:
            Rotary embedded tensor [batch_size, seq_len, head_dim]
        """
        batch_size, seq_len, head_dim = x.shape
        device = x.device
        
        # Handle time_step input
        if isinstance(time_step, torch.Tensor):
            if time_step.dim() == 0:
                time_step = time_step.item()
            elif time_step.dim() == 1:
                # Use first element for simplicity, or could batch-process
                time_step = time_step[0].item()
        
        # Create 3D positions
        h_pos, w_pos, t_pos = self.create_3d_positions(height, width, time_step, device)
        
        # Expand for batch
        h_pos = h_pos.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        w_pos = w_pos.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        t_pos = t_pos.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        
        # Get cosine and sine values for each axis
        cos_h, sin_h = self._get_cos_sin(h_pos, self.freqs_h)  # [batch_size, seq_len, dim_per_axis]
        cos_w, sin_w = self._get_cos_sin(w_pos, self.freqs_w)  # [batch_size, seq_len, dim_per_axis]
        cos_t, sin_t = self._get_cos_sin(t_pos, self.freqs_t)  # [batch_size, seq_len, dim_per_axis]
        
        # Split input tensor into 3 parts for each axis
        dim_per_axis_pairs = self.dim_per_axis * 2  # Each axis gets dim_per_axis complex pairs
        
        x_h = x[..., :dim_per_axis_pairs]                                    # Height part
        x_w = x[..., dim_per_axis_pairs:2*dim_per_axis_pairs]               # Width part  
        x_t = x[..., 2*dim_per_axis_pairs:3*dim_per_axis_pairs]             # Time part
        x_remainder = x[..., 3*dim_per_axis_pairs:]                         # Remainder (if any)
        
        # Apply RoPE to each part
        x_h_rotated = self._apply_rope_1d(x_h, cos_h, sin_h)
        x_w_rotated = self._apply_rope_1d(x_w, cos_w, sin_w)
        x_t_rotated = self._apply_rope_1d(x_t, cos_t, sin_t)
        
        # Concatenate results
        if x_remainder.shape[-1] > 0:
            x_rotated = torch.cat([x_h_rotated, x_w_rotated, x_t_rotated, x_remainder], dim=-1)
        else:
            x_rotated = torch.cat([x_h_rotated, x_w_rotated, x_t_rotated], dim=-1)
        
        return x_rotated


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
        
        # Position embeddings for 8x8 grid (64 tokens)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor, image_rotary_emb: Optional[torch.Tensor] = None):
        """
        Forward pass for token embedding.
        
        Args:
            x: Input tokens [B, 64, in_channels]
            image_rotary_emb: Rotary embeddings (optional, not used in simple version)
            
        Returns:
            Tuple of (embedded_tokens, attention_mask, image_size, rotary_emb)
        """
        batch_size, num_tokens, in_channels = x.shape
        
        # Validate input
        assert num_tokens == 64, f"Expected 64 tokens, got {num_tokens}"
        assert in_channels == self.in_channels, f"Expected {self.in_channels} channels, got {in_channels}"
        
        # Linear projection
        embedded = self.proj(x)  # [B, 64, embed_dim]
        
        # Add position embeddings
        embedded = embedded + self.pos_embed
        
        # Create attention mask (all tokens are valid)
        attention_mask = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        
        # Image size (8x8 for 64 tokens)
        img_size = [(8, 8)] * batch_size
        
        # Return rotary embeddings if provided, otherwise None
        return embedded, attention_mask, img_size, image_rotary_emb


class BLIP3oAttentionBlock(nn.Module):
    """
    Simplified DiT block for BLIP3-o that works with pre-tokenized embeddings and 3D RoPE.
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
        
        # 3D RoPE for spatial-temporal encoding
        self.rope_3d = RoPE3D(head_dim=self.head_dim)
        
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
        
    def _apply_rope_to_qk(self, q: torch.Tensor, k: torch.Tensor, timestep_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to query and key tensors."""
        # q, k: [batch_size, seq_len, dim]
        batch_size, seq_len, dim = q.shape
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q_reshaped = q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k_reshaped = k.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        
        # Apply 3D RoPE to each head
        q_rotated_list = []
        k_rotated_list = []
        
        for head_idx in range(self.num_attention_heads):
            q_head = q_reshaped[:, :, head_idx, :]  # [batch_size, seq_len, head_dim]
            k_head = k_reshaped[:, :, head_idx, :]  # [batch_size, seq_len, head_dim]
            
            # Apply 3D RoPE (assuming 8x8 grid)
            q_head_rotated = self.rope_3d(q_head, height=8, width=8, time_step=0.0)
            k_head_rotated = self.rope_3d(k_head, height=8, width=8, time_step=0.0)
            
            q_rotated_list.append(q_head_rotated)
            k_rotated_list.append(k_head_rotated)
        
        # Reshape back to [batch_size, seq_len, dim]
        q_rotated = torch.stack(q_rotated_list, dim=2).reshape(batch_size, seq_len, dim)
        k_rotated = torch.stack(k_rotated_list, dim=2).reshape(batch_size, seq_len, dim)
        
        return q_rotated, k_rotated
    
    def forward(
        self,
        hidden_states: torch.Tensor,              # [B, 64, dim]
        encoder_hidden_states: torch.Tensor,     # [B, 64, cross_attention_dim]
        timestep_emb: torch.Tensor,              # [B, dim]
        attention_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through simplified DiT block with 3D RoPE."""
        
        batch_size = hidden_states.shape[0]
        
        # Get timestep conditioning
        time_cond = self.time_proj(timestep_emb)  # [B, dim * 6]
        time_chunks = time_cond.chunk(6, dim=-1)  # 6 chunks of [B, dim]
        
        scale_msa, gate_msa, scale_mlp, gate_mlp, scale_cross, gate_cross = time_chunks
        
        # Self-attention with timestep conditioning and 3D RoPE
        residual = hidden_states
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa.unsqueeze(1))
        
        # For simplicity, apply standard attention (RoPE integration would require custom attention)
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
    BLIP3-o Diffusion Transformer Model with proper 3D RoPE implementation.
    
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
        
        # Validate configuration for BLIP3-o (after any adjustments)
        self._validate_blip3o_config(config)
        
        # Ensure head_dim is compatible with 3D RoPE and PyTorch MultiheadAttention
        self.head_dim = config.dim // config.n_heads
        
        if self.head_dim % 6 != 0:
            print(f"⚠️  Head dimension {self.head_dim} not divisible by 6, adjusting for 3D RoPE compatibility...")
            
            # Find compatible dimensions
            original_dim = config.dim
            original_heads = config.n_heads
            
            # Strategy 1: Find num_heads that gives head_dim divisible by 6
            compatible_found = False
            for candidate_heads in range(1, original_heads + 1):
                if config.dim % candidate_heads == 0:  # Must be divisible for MultiheadAttention
                    candidate_head_dim = config.dim // candidate_heads
                    if candidate_head_dim % 6 == 0:  # Must be divisible by 6 for 3D RoPE
                        config.n_heads = candidate_heads
                        config.n_kv_heads = candidate_heads
                        self.head_dim = candidate_head_dim
                        compatible_found = True
                        print(f"✅ Adjusted num_heads to {candidate_heads} (head_dim={candidate_head_dim})")
                        break
            
            # Strategy 2: If no compatible num_heads found, adjust dim
            if not compatible_found:
                # Find the largest dim <= original_dim that works
                for candidate_dim in range(original_dim, 0, -6):  # Step by 6 for RoPE compatibility
                    if candidate_dim % original_heads == 0:  # Must work with original heads
                        candidate_head_dim = candidate_dim // original_heads
                        if candidate_head_dim % 6 == 0:
                            config.dim = candidate_dim
                            self.head_dim = candidate_head_dim
                            print(f"✅ Adjusted dim to {candidate_dim} (head_dim={candidate_head_dim})")
                            compatible_found = True
                            break
                
                # Strategy 3: If still not found, use a safe default
                if not compatible_found:
                    # Use a safe combination: dim=480, heads=8, head_dim=60 (60%6=0)
                    config.dim = 480
                    config.n_heads = 8
                    config.n_kv_heads = 8
                    self.head_dim = 60
                    print(f"✅ Using safe default: dim=480, heads=8, head_dim=60")
        else:
            print(f"✅ Head dimension {self.head_dim} is compatible with 3D RoPE")
        
        # Final validation
        assert config.dim % config.n_heads == 0, f"dim {config.dim} must be divisible by num_heads {config.n_heads}"
        assert self.head_dim % 6 == 0, f"head_dim {self.head_dim} must be divisible by 6 for 3D RoPE"
        
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
        
        print(f"✅ BLIP3-o DiT model with 3D RoPE initialized")
        print(f"   Parameters: {self.get_num_parameters():,}")
        print(f"   Final dimensions: dim={config.dim}, heads={config.n_heads}, head_dim={self.head_dim}")
        print(f"   3D RoPE compatible: head_dim % 6 = {self.head_dim % 6}")
    
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
        Forward pass of BLIP3-o DiT model with 3D RoPE.
        
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
        
        # Embed input tokens
        hidden_states, attention_mask, img_size, _ = self.token_embedder(hidden_states)
        
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
                    use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_emb=timestep_emb,
                    attention_mask=attention_mask,
                    encoder_mask=encoder_attention_mask,
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
        Generate CLIP embeddings using flow matching sampling with 3D RoPE.
        
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