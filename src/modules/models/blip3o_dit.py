"""
UPDATED BLIP3-o DiT Model with Dual Supervision Architecture
Key Changes:
1. Added average pooling after DiT output
2. Added custom MLP layers for domain adaptation  
3. Added frozen CLIP visual projection layer
4. Dual output: patch-level + global embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union
from transformers import PreTrainedModel, CLIPModel
import math

from ..config.blip3o_config import BLIP3oDiTConfig


def get_3d_rotary_pos_embed(embed_dim, grid_size, temporal_size=1, base=10000.0):
    """Create 3D rotary position embeddings for 256 tokens (16x16 grid)"""
    assert embed_dim % 4 == 0, f"embed_dim {embed_dim} must be divisible by 4 for 3D RoPE"
    dim_h = embed_dim // 4
    dim_w = embed_dim // 4
    
    inv_freq_h = 1.0 / (base ** (torch.arange(0, dim_h, 2).float() / dim_h))
    inv_freq_w = 1.0 / (base ** (torch.arange(0, dim_w, 2).float() / dim_w))
    
    h_pos = torch.arange(grid_size, dtype=torch.float32)
    w_pos = torch.arange(grid_size, dtype=torch.float32)
    
    grid_h, grid_w = torch.meshgrid(h_pos, w_pos, indexing='ij')
    grid_h = grid_h.flatten()
    grid_w = grid_w.flatten()
    
    freqs_h = torch.outer(grid_h, inv_freq_h)
    freqs_w = torch.outer(grid_w, inv_freq_w)
    
    cos_h = torch.cos(freqs_h)
    sin_h = torch.sin(freqs_h)
    cos_w = torch.cos(freqs_w)
    sin_w = torch.sin(freqs_w)
    
    cos_h_full = torch.stack([cos_h, cos_h], dim=-1).flatten(-2)
    sin_h_full = torch.stack([sin_h, sin_h], dim=-1).flatten(-2)
    cos_w_full = torch.stack([cos_w, cos_w], dim=-1).flatten(-2)
    sin_w_full = torch.stack([sin_w, sin_w], dim=-1).flatten(-2)
    
    cos_emb = torch.cat([cos_h_full, cos_w_full], dim=-1)
    sin_emb = torch.cat([sin_h_full, sin_w_full], dim=-1)
    
    cos_emb = cos_emb.unsqueeze(0)
    sin_emb = sin_emb.unsqueeze(0)
    
    return cos_emb, sin_emb


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    cos = cos.expand(batch_size, -1, -1)
    sin = sin.expand(batch_size, -1, -1)
    cos = cos.unsqueeze(2).expand(-1, -1, num_heads, -1)
    sin = sin.unsqueeze(2).expand(-1, -1, num_heads, -1)
    
    q1 = q[..., : head_dim // 2]
    q2 = q[..., head_dim // 2 :]
    k1 = k[..., : head_dim // 2]
    k2 = k[..., head_dim // 2 :]
    
    q_rot1 = q1 * cos - q2 * sin
    q_rot2 = q1 * sin + q2 * cos
    k_rot1 = k1 * cos - k2 * sin
    k_rot2 = k1 * sin + k2 * cos
    
    q_embed = torch.cat([q_rot1, q_rot2], dim=-1)
    k_embed = torch.cat([k_rot1, k_rot2], dim=-1)
    
    return q_embed, k_embed


class SimpleTokenEmbedder(nn.Module):
    """Simple embedding layer for pre-tokenized BLIP3-o features (256 tokens)"""
    
    def __init__(self, in_channels: int, embed_dim: int, num_tokens: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        self.proj = nn.Linear(in_channels, embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        
    def forward(self, x: torch.Tensor, image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        batch_size, num_tokens, in_channels = x.shape
        assert num_tokens == self.num_tokens, f"Expected {self.num_tokens} tokens, got {num_tokens}"
        assert in_channels == self.in_channels, f"Expected {self.in_channels} channels, got {in_channels}"
        
        embedded = self.proj(x)
        embedded = embedded + self.pos_embed
        attention_mask = torch.ones(batch_size, num_tokens, device=x.device, dtype=torch.bool)
        img_size = [(16, 16)] * batch_size
        
        return embedded, attention_mask, img_size, image_rotary_emb


class BLIP3oAttentionBlock(nn.Module):
    """DiT block with DDP compatibility for dual supervision"""
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        
        # Self-attention projections
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        
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
        self.time_proj = nn.Linear(dim, dim * 6)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        assert seq_len == 256, f"Expected 256 tokens, got {seq_len}"
        
        # Get timestep conditioning
        time_cond = self.time_proj(timestep_emb)
        scale_msa, gate_msa, scale_mlp, gate_mlp, scale_cross, gate_cross = time_cond.chunk(6, dim=-1)
        
        # Self-attention with manual projections
        residual = hidden_states
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_msa.unsqueeze(1))
        
        q = self.q_proj(norm_hidden)
        k = self.k_proj(norm_hidden)
        v = self.v_proj(norm_hidden)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        
        # Apply RoPE if available
        if image_rotary_emb is not None:
            cos_emb, sin_emb = image_rotary_emb
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
        else:
            q_rot, k_rot = q, k
        
        # Prepare for attention computation
        q_for_attn = q_rot.transpose(1, 2).contiguous().view(batch_size * self.num_attention_heads, seq_len, self.head_dim)
        k_for_attn = k_rot.transpose(1, 2).contiguous().view(batch_size * self.num_attention_heads, seq_len, self.head_dim)
        v_for_attn = v.transpose(1, 2).contiguous().view(batch_size * self.num_attention_heads, seq_len, self.head_dim)
        
        # Handle attention mask
        attn_mask = None
        if attention_mask is not None:
            additive_mask = torch.zeros_like(attention_mask, dtype=q_for_attn.dtype)
            additive_mask = additive_mask.masked_fill(~attention_mask, torch.finfo(q_for_attn.dtype).min)
            additive_mask = additive_mask.view(batch_size, 1, seq_len)
            additive_mask = additive_mask.expand(-1, self.num_attention_heads, -1)
            attn_mask = additive_mask.reshape(batch_size * self.num_attention_heads, 1, seq_len)
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            q_for_attn,
            k_for_attn,
            v_for_attn,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # Reshape and project output
        attn_output = attn_output.view(batch_size, self.num_attention_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        attn_output = self.out_proj(attn_output)
        
        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * attn_output
        
        # Cross-attention
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
        
        # Feed-forward
        residual = hidden_states
        norm_hidden = self.norm3(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp.unsqueeze(1))
        
        ffn_output = self.ffn(norm_hidden)
        hidden_states = residual + gate_mlp.unsqueeze(1).tanh() * ffn_output
        
        return hidden_states


class GlobalAdaptationMLP(nn.Module):
    """
    NEW: Custom MLP for domain adaptation after average pooling
    Maps DiT features [1024] → CLIP space [1024] before visual projection
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        output_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create MLP layers
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adaptation MLP
        
        Args:
            x: Input tensor [B, input_dim]
            
        Returns:
            Output tensor [B, output_dim]
        """
        return self.mlp(x)


class BLIP3oDiTModel(PreTrainedModel):
    """
    UPDATED: BLIP3-o DiT Model with Dual Supervision Architecture
    
    New Architecture:
    EVA [B,256,4096] → DiT → [B,256,1024] → {
        Patch Output: [B,256,1024] (for patch loss)
        Global Path: Avg Pool → MLP → Frozen CLIP Proj → [B,768]
    }
    """
    
    config_class = BLIP3oDiTConfig
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__(config)
        self.config = config
        self._gradient_checkpointing = config._gradient_checkpointing
        self._validate_blip3o_config(config)
        
        self.num_tokens = config.input_size * config.input_size  # 256
        self.head_dim = config.dim // config.n_heads
        
        # Adjust dimensions for RoPE compatibility
        if self.head_dim % 4 != 0:
            for candidate_heads in range(1, config.dim + 1):
                if config.dim % candidate_heads == 0:
                    candidate_head_dim = config.dim // candidate_heads
                    if candidate_head_dim % 4 == 0:
                        config.n_heads = candidate_heads
                        config.n_kv_heads = candidate_heads
                        self.head_dim = candidate_head_dim
                        break
            if self.head_dim % 4 != 0:
                config.dim = 512
                config.n_heads = 8
                config.n_kv_heads = 8
                self.head_dim = 64
        
        assert config.dim % config.n_heads == 0, f"dim {config.dim} must be divisible by num_heads {config.n_heads}"
        assert self.head_dim % 4 == 0, f"head_dim {self.head_dim} must be divisible by 4 for 3D RoPE"
        
        # Token embedder
        self.token_embedder = SimpleTokenEmbedder(
            in_channels=config.in_channels,
            embed_dim=config.dim,
            num_tokens=self.num_tokens,
        )
        
        # Timestep embedding
        time_embed_dim = min(config.dim, 1024)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        self.time_proj = self._create_sinusoidal_timestep_embedding(time_embed_dim)
        
        # EVA-CLIP projection
        self.eva_proj = nn.Linear(config.eva_embedding_size, config.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BLIP3oAttentionBlock(
                dim=config.dim,
                num_attention_heads=config.n_heads,
                cross_attention_dim=config.dim,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_layers)
        ])
        
        # Output layer for patch-level supervision
        self.norm_out = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.proj_out = nn.Linear(config.dim, config.in_channels, bias=True)
        
        # NEW: Global adaptation pipeline
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool over token dimension
        self.global_adaptation_mlp = GlobalAdaptationMLP(
            input_dim=config.in_channels,  # 1024 after proj_out
            hidden_dim=getattr(config, 'mlp_hidden_dim', 2048),
            output_dim=config.in_channels,  # 1024 to match CLIP space
            num_layers=getattr(config, 'mlp_num_layers', 3),
            dropout=getattr(config, 'mlp_dropout', 0.1),
        )
        
        # NEW: Frozen CLIP visual projection layer (will be loaded separately)
        self.frozen_clip_visual_proj = None
        self.clip_output_dim = 768  # CLIP text embedding dimension
        
        self._init_weights()
        
        print(f"✅ DUAL SUPERVISION BLIP3-o DiT model initialized for {self.num_tokens} tokens")
        print(f"   Patch output: [B, {self.num_tokens}, {config.in_channels}]")
        print(f"   Global output: [B, {self.clip_output_dim}]")

    def _create_sinusoidal_timestep_embedding(self, embed_dim: int):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        return emb
    
    def _validate_blip3o_config(self, config: BLIP3oDiTConfig):
        assert config.learn_sigma is False, "BLIP3-o uses flow matching, sigma learning must be False"
        assert config.in_channels > 0, "CLIP embedding dimension must be positive"
        assert config.eva_embedding_size > 0, "EVA-CLIP conditioning dimension must be positive"
        assert config.patch_size == 1, "Features are pre-tokenized, patch_size must be 1"
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def load_frozen_clip_projection(self, clip_model_name: str = "openai/clip-vit-large-patch14"):
        """
        Load and freeze CLIP visual projection layer
        
        Args:
            clip_model_name: Name of CLIP model to load
        """
        print(f"🔒 Loading frozen CLIP visual projection from {clip_model_name}")
        
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        # Extract visual projection layer
        visual_proj = clip_model.visual_projection
        
        # Move to same device and dtype as model
        if hasattr(self, 'device'):
            visual_proj = visual_proj.to(self.device)
        if hasattr(self, 'dtype'):
            visual_proj = visual_proj.to(self.dtype)
        
        # Freeze parameters
        for param in visual_proj.parameters():
            param.requires_grad = False
        
        self.frozen_clip_visual_proj = visual_proj
        
        print(f"✅ Frozen CLIP visual projection loaded: {visual_proj.weight.shape}")
        print(f"   Input dim: {visual_proj.in_features}, Output dim: {visual_proj.out_features}")
    
    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False
    
    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        dtype = timesteps.dtype
        timesteps = torch.clamp(timesteps, 0.0, 1.0) * 1000.0
        half_dim = len(self.time_proj)
        emb = self.time_proj.to(device=device, dtype=dtype)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        UPDATED: Forward pass with dual outputs
        
        Returns:
            Dict containing:
            - patch_output: [B, 256, 1024] for patch-level supervision
            - global_output: [B, 768] for global supervision  
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        self._validate_forward_inputs(hidden_states, timestep, encoder_hidden_states)
        
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(batch_size)
        elif timestep.shape[0] != batch_size:
            raise ValueError(f"Timestep batch size mismatch")
        
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                (batch_size, encoder_hidden_states.shape[1]),
                device=device,
                dtype=torch.bool
            )
        
        hidden_states, attention_mask, img_size, _ = self.token_embedder(hidden_states)
        
        # Create 3D RoPE embeddings
        cos_emb, sin_emb = get_3d_rotary_pos_embed(
            embed_dim=self.head_dim,
            grid_size=self.config.input_size
        )
        cos_emb = cos_emb.to(device)
        sin_emb = sin_emb.to(device)
        image_rotary_emb = (cos_emb, sin_emb)
        
        # Timestep embedding
        timestep_emb = self.get_timestep_embedding(timestep)
        timestep_emb = self.time_embed(timestep_emb)
        
        # Project EVA-CLIP
        encoder_hidden_states = self.eva_proj(encoder_hidden_states)
        
        # Transformer layers
        for layer in self.layers:
            if self.training and self._gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
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
        patch_output = self.proj_out(hidden_states)  # [B, 256, 1024]
        
        # NEW: Global adaptation pipeline
        # 1. Average pooling over token dimension
        pooled_features = patch_output.mean(dim=1)  # [B, 1024]
        
        # 2. Custom MLP for domain adaptation  
        adapted_features = self.global_adaptation_mlp(pooled_features)  # [B, 1024]
        
        # 3. Frozen CLIP visual projection
        global_output = None
        if self.frozen_clip_visual_proj is not None:
            global_output = self.frozen_clip_visual_proj(adapted_features)  # [B, 768]
            # Normalize to unit norm (like CLIP)
            global_output = F.normalize(global_output, p=2, dim=-1)
        
        if return_dict:
            return {
                'patch_output': patch_output,      # [B, 256, 1024] for patch loss
                'global_output': global_output,    # [B, 768] for global loss
                'pooled_features': pooled_features,  # [B, 1024] intermediate
                'adapted_features': adapted_features, # [B, 1024] after MLP
            }
        else:
            return patch_output, global_output
    
    def _validate_forward_inputs(self, hidden_states, timestep, encoder_hidden_states):
        actual_tokens = hidden_states.shape[1]
        if actual_tokens != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {actual_tokens}")
        if hidden_states.shape[2] != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels}-dim CLIP features")
        if encoder_hidden_states.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} conditioning tokens")
        if encoder_hidden_states.shape[2] != self.config.eva_embedding_size:
            raise ValueError(f"Expected {self.config.eva_embedding_size}-dim EVA-CLIP features")
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError(f"Batch size mismatch")
    
    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        return_intermediate: bool = False,
        return_global_only: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        UPDATED: Generation with dual outputs
        
        Args:
            encoder_hidden_states: EVA-CLIP conditioning [batch_size, num_tokens, 4096]
            num_inference_steps: Number of sampling steps
            guidance_scale: Guidance scale
            generator: Random number generator
            eta: DDIM parameter
            return_intermediate: Whether to return intermediate states
            return_global_only: If True, return only global embeddings [B, 768]
            
        Returns:
            Generated embeddings (global [B, 768] or patch [B, 256, 1024])
        """
        batch_size = encoder_hidden_states.shape[0]
        num_tokens = encoder_hidden_states.shape[1]
        device = encoder_hidden_states.device
        dtype = encoder_hidden_states.dtype
        
        if num_tokens != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} conditioning tokens, got {num_tokens}")
        
        # Initialize from random noise
        sample = torch.randn(
            (batch_size, num_tokens, self.config.in_channels),
            device=device,
            dtype=dtype,
            generator=generator
        )
        
        # Flow matching sampling
        dt = 1.0 / num_inference_steps
        intermediate_samples = [] if return_intermediate else None
        
        self.eval()
        
        for step in range(num_inference_steps):
            t = step * dt
            t_tensor = torch.full((batch_size,), t, device=device, dtype=dtype)
            
            # Forward pass
            outputs = self.forward(
                hidden_states=sample,
                timestep=t_tensor,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True
            )
            
            # Use patch output for velocity
            velocity = outputs['patch_output']
            
            # Euler integration
            sample = sample + dt * velocity
            
            if return_intermediate:
                intermediate_samples.append(sample.clone())
        
        # Final forward pass to get both outputs
        final_outputs = self.forward(
            hidden_states=sample,
            timestep=torch.zeros(batch_size, device=device, dtype=dtype),
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        
        # Return appropriate output
        if return_global_only and final_outputs['global_output'] is not None:
            result = final_outputs['global_output']  # [B, 768]
        else:
            result = final_outputs['patch_output']   # [B, 256, 1024]
        
        if return_intermediate:
            return result, intermediate_samples
        else:
            return result
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Get the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_memory_footprint(self) -> str:
        """Get approximate memory footprint of the model."""
        num_params = self.get_num_parameters(trainable_only=False)
        memory_mb = num_params * 4 / (1024 * 1024)
        return f"{memory_mb:.1f} MB"


def create_blip3o_dit_model(
    config: Optional[BLIP3oDiTConfig] = None,
    load_clip_projection: bool = True,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    **kwargs
) -> BLIP3oDiTModel:
    """
    Factory function to create a BLIP3-o DiT model with dual supervision.
    
    Args:
        config: Model configuration
        load_clip_projection: Whether to load frozen CLIP projection
        clip_model_name: CLIP model to load projection from
        **kwargs: Additional configuration parameters
        
    Returns:
        BLIP3oDiTModel instance
    """
    if config is None:
        from ..config.blip3o_config import get_default_blip3o_config
        config = get_default_blip3o_config()
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}' ignored")
    
    model = BLIP3oDiTModel(config)
    
    # Load frozen CLIP projection if requested
    if load_clip_projection:
        model.load_frozen_clip_projection(clip_model_name)
    
    return model


def load_blip3o_dit_model(
    model_path: str,
    device: str = "auto",
    torch_dtype: Optional[torch.dtype] = None
) -> BLIP3oDiTModel:
    """Load a pre-trained BLIP3-o DiT model with dual supervision."""
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device if device != "auto" else None,
    )
    
    return model