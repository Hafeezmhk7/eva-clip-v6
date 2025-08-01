#!/usr/bin/env python3
"""
FIXED: BLIP3-o DiT Model with Scale-Aware Generation - Target Norm Bug Fix
Key fixes:
1. Better target_norm validation and type checking
2. Prevent target_norm from being overwritten with tensors
3. Enhanced error handling and debugging
4. Safer tensor-to-scalar conversions
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import logging
from transformers import PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)


class BLIP3oCLIPDiTConfig(PretrainedConfig):
    """Configuration for BLIP3-o CLIP DiT model"""
    model_type = "blip3o_clip_dit"
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        intermediate_size: int = 3072,
        eva_embedding_size: int = 4096,
        clip_embedding_size: int = 1024,
        num_tokens: int = 256,
        max_position_embeddings: int = 256,
        # 3D RoPE parameters
        use_3d_rope: bool = True,
        rope_theta: float = 10000.0,
        image_size: int = 224,
        patch_size: int = 14,
        # Sandwich normalization
        use_sandwich_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        dropout_prob: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_gradient_checkpointing: bool = False,
        training_mode: str = "patch_only",
        # FIXED: Scale-aware generation parameters - ensure they're always scalars
        typical_clip_norm: float = 26.0,
        velocity_explosion_threshold: float = 100.0,
        norm_guidance_strength: float = 0.1,
        norm_guidance_frequency: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.eva_embedding_size = eva_embedding_size
        self.clip_embedding_size = clip_embedding_size
        self.num_tokens = num_tokens
        self.max_position_embeddings = max_position_embeddings
        
        # 3D RoPE
        self.use_3d_rope = use_3d_rope
        self.rope_theta = rope_theta
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Sandwich normalization
        self.use_sandwich_norm = use_sandwich_norm
        self.rms_norm_eps = rms_norm_eps
        
        self.dropout_prob = dropout_prob
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        
        # FIXED: Ensure scale-aware parameters are always Python floats/ints
        self.typical_clip_norm = float(typical_clip_norm)
        self.velocity_explosion_threshold = float(velocity_explosion_threshold)
        self.norm_guidance_strength = float(norm_guidance_strength)
        self.norm_guidance_frequency = int(norm_guidance_frequency)
        
        # Calculate grid size for 3D RoPE
        self.grid_size = image_size // patch_size  # 224 // 14 = 16


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Rotary3DEmbedding(nn.Module):
    """3D Rotary Position Embedding for BLIP3-o"""
    def __init__(
        self, 
        dim: int, 
        grid_size: int = 16,
        max_position_embeddings: int = 256, 
        base: float = 10000,
        use_3d: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.use_3d = use_3d
        
        if use_3d:
            assert dim % 4 == 0, "Dimension must be divisible by 4 for 3D RoPE"
            self.dim_h = dim // 4
            self.dim_w = dim // 4
            self.dim_d = dim // 2
        else:
            self.dim_h = dim // 2
            self.dim_w = dim // 2
            self.dim_d = 0
        
        self._create_frequency_tensors()

    def _create_frequency_tensors(self):
        """Create frequency tensors for each spatial dimension"""
        if self.use_3d:
            inv_freq_h = 1.0 / (self.base ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
            self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
            
            inv_freq_w = 1.0 / (self.base ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))
            self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)
            
            inv_freq_d = 1.0 / (self.base ** (torch.arange(0, self.dim_d, 2).float() / self.dim_d))
            self.register_buffer("inv_freq_d", inv_freq_d, persistent=False)
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        
        device = x.device
        
        if not self.use_3d:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        
        batch_size = x.shape[0]
        has_cls = seq_len == self.grid_size * self.grid_size + 1
        start_idx = 1 if has_cls else 0
        spatial_len = seq_len - start_idx
        
        pos_embeddings = []
        
        if has_cls:
            cls_emb = torch.zeros(1, self.dim, device=device)
            pos_embeddings.append(cls_emb)
        
        grid_h, grid_w = int(math.sqrt(spatial_len)), int(math.sqrt(spatial_len))
        
        pos_h = torch.arange(grid_h, device=device, dtype=torch.float32)
        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq_h)
        
        pos_w = torch.arange(grid_w, device=device, dtype=torch.float32)
        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq_w)
        
        depth_scale = torch.zeros(1, device=device, dtype=torch.float32)
        freqs_d = torch.einsum("i,j->ij", depth_scale, self.inv_freq_d)
        
        for h in range(grid_h):
            for w in range(grid_w):
                h_emb = torch.cat((freqs_h[h], freqs_h[h]), dim=-1)
                w_emb = torch.cat((freqs_w[w], freqs_w[w]), dim=-1)
                d_emb = torch.cat((freqs_d[0], freqs_d[0]), dim=-1)
                
                combined_emb = torch.cat([h_emb, w_emb, d_emb], dim=0)
                pos_embeddings.append(combined_emb)
        
        all_pos_emb = torch.stack(pos_embeddings, dim=0)
        cos_emb = all_pos_emb.cos().unsqueeze(0)
        sin_emb = all_pos_emb.sin().unsqueeze(0)
        
        return cos_emb, sin_emb


def apply_rotary_pos_emb_3d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply 3D rotary position embedding"""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimestepEmbedder(nn.Module):
    """Enhanced timestep embedding for BLIP3-o"""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings"""
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


class Attention3D(nn.Module):
    """Multi-head attention with 3D RoPE for BLIP3-o"""
    def __init__(self, config: BLIP3oCLIPDiTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        assert self.hidden_size % self.num_heads == 0
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = Rotary3DEmbedding(
            self.head_dim,
            grid_size=config.grid_size,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            use_3d=config.use_3d_rope,
        )
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights"""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1.0 / math.sqrt(self.config.num_hidden_layers))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        
        if key_value_states is not None:
            kv_seq_len = key_value_states.shape[1]
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            kv_seq_len = q_len
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if key_value_states is None:
            cos, sin = self.rotary_emb(hidden_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb_3d(query_states, key_states, cos, sin)
        
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """Enhanced MLP with better initialization"""
    def __init__(self, config: BLIP3oCLIPDiTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout_prob)
        
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1.0 / math.sqrt(config.num_hidden_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for timestep conditioning"""
    def __init__(self, hidden_size: int, conditioning_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_size, 2 * hidden_size, bias=True)
        )
        
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1)
        
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        
        if shift.shape[1] == 1 and x.shape[1] > 1:
            shift = shift.expand(-1, x.shape[1], -1)
            scale = scale.expand(-1, x.shape[1], -1)
        
        normalized = self.norm(x)
        return normalized * (1 + scale) + shift


class DiTBlock3D(nn.Module):
    """BLIP3-o DiT transformer block with 3D RoPE and Sandwich Normalization"""
    def __init__(self, config: BLIP3oCLIPDiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_sandwich_norm = config.use_sandwich_norm
        
        self.self_attn = Attention3D(config)
        self.cross_attn = Attention3D(config)
        self.mlp = MLP(config)
        
        self.eva_proj = nn.Linear(config.eva_embedding_size, config.hidden_size, bias=True)
        nn.init.xavier_uniform_(self.eva_proj.weight)
        nn.init.zeros_(self.eva_proj.bias)
        
        if config.use_sandwich_norm:
            self.self_attn_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn_ada_ln_pre = AdaLN(config.hidden_size, config.hidden_size)
            self.self_attn_ada_ln_post = AdaLN(config.hidden_size, config.hidden_size)
            
            self.cross_attn_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.cross_attn_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.cross_attn_ada_ln_pre = AdaLN(config.hidden_size, config.hidden_size)
            self.cross_attn_ada_ln_post = AdaLN(config.hidden_size, config.hidden_size)
            
            self.mlp_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_ada_ln_pre = AdaLN(config.hidden_size, config.hidden_size)
            self.mlp_ada_ln_post = AdaLN(config.hidden_size, config.hidden_size)
        else:
            self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norm3 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.ada_ln1 = AdaLN(config.hidden_size, config.hidden_size)
            self.ada_ln2 = AdaLN(config.hidden_size, config.hidden_size)
            self.ada_ln3 = AdaLN(config.hidden_size, config.hidden_size)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor
    ) -> torch.Tensor:
        
        if self.use_sandwich_norm:
            # Self-attention with sandwich norm
            residual = hidden_states
            hidden_states = self.self_attn_pre_norm(hidden_states)
            hidden_states = self.self_attn_ada_ln_pre(hidden_states, timestep_emb)
            hidden_states = self.self_attn(hidden_states)
            hidden_states = self.self_attn_post_norm(hidden_states)
            hidden_states = self.self_attn_ada_ln_post(hidden_states, timestep_emb)
            hidden_states = residual + hidden_states
            
            # Cross-attention with sandwich norm
            residual = hidden_states
            hidden_states = self.cross_attn_pre_norm(hidden_states)
            hidden_states = self.cross_attn_ada_ln_pre(hidden_states, timestep_emb)
            eva_features = self.eva_proj(encoder_hidden_states)
            hidden_states = self.cross_attn(hidden_states, key_value_states=eva_features)
            hidden_states = self.cross_attn_post_norm(hidden_states)
            hidden_states = self.cross_attn_ada_ln_post(hidden_states, timestep_emb)
            hidden_states = residual + hidden_states
            
            # MLP with sandwich norm
            residual = hidden_states
            hidden_states = self.mlp_pre_norm(hidden_states)
            hidden_states = self.mlp_ada_ln_pre(hidden_states, timestep_emb)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.mlp_post_norm(hidden_states)
            hidden_states = self.mlp_ada_ln_post(hidden_states, timestep_emb)
            hidden_states = residual + hidden_states
            
        else:
            # Standard pre-norm pattern
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
            hidden_states = self.ada_ln1(hidden_states, timestep_emb)
            hidden_states = self.self_attn(hidden_states)
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            hidden_states = self.ada_ln2(hidden_states, timestep_emb)
            eva_features = self.eva_proj(encoder_hidden_states)
            hidden_states = self.cross_attn(hidden_states, key_value_states=eva_features)
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = self.norm3(hidden_states)
            hidden_states = self.ada_ln3(hidden_states, timestep_emb)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        
        return hidden_states


class BLIP3oCLIPDiTModel(PreTrainedModel):
    """FIXED: BLIP3-o DiT Model with Scale-Aware Generation - Target Norm Bug Fix"""
    
    config_class = BLIP3oCLIPDiTConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: BLIP3oCLIPDiTConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        
        self.input_proj = nn.Linear(config.clip_embedding_size, config.hidden_size, bias=True)
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        
        self.blocks = nn.ModuleList([
            DiTBlock3D(config) for _ in range(config.num_hidden_layers)
        ])
        
        if config.use_sandwich_norm:
            self.output_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_adaln_pre = AdaLN(config.hidden_size, config.hidden_size)
        else:
            self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.output_adaln = AdaLN(config.hidden_size, config.hidden_size)
        
        self.output_proj = nn.Linear(config.hidden_size, config.clip_embedding_size, bias=True)
        
        self._init_weights()
        
        logger.info(f"FIXED BLIP3-o CLIP DiT model initialized with {self.get_num_parameters():,} parameters")
        logger.info(f"  3D RoPE: {config.use_3d_rope}")
        logger.info(f"  Sandwich Normalization: {config.use_sandwich_norm}")
        logger.info(f"  🎯 Typical CLIP norm: {config.typical_clip_norm} (type: {type(config.typical_clip_norm).__name__})")

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass"""
        batch_size, seq_len, _ = hidden_states.shape
        
        x = self.input_proj(hidden_states)
        
        if not self.config.use_3d_rope and seq_len <= self.config.max_position_embeddings:
            x = x + self.pos_embed[:, :seq_len, :]
        
        timestep_emb = self.timestep_embedder(timestep)
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, encoder_hidden_states, timestep_emb, use_reentrant=False
                )
            else:
                x = block(x, encoder_hidden_states, timestep_emb)
        
        if self.config.use_sandwich_norm:
            x = self.output_pre_norm(x)
            x = self.output_adaln_pre(x, timestep_emb)
            velocity_pred = self.output_proj(x)
        else:
            x = self.output_norm(x)
            x = self.output_adaln(x, timestep_emb)
            velocity_pred = self.output_proj(x)
        
        if return_dict:
            return {"velocity_prediction": velocity_pred, "hidden_states": x}
        return velocity_pred
    
    def _create_lognormal_timestep_schedule(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        """Create log-normal timestep schedule for better sampling"""
        # Start with linear schedule
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)[:-1]
        
        # Transform to log-normal distribution
        timesteps = torch.exp(-2 * timesteps)
        timesteps = timesteps / (1 + timesteps)
        
        return timesteps
    
    def _safe_estimate_target_norm_from_eva(self, eva_features: torch.Tensor) -> float:
        """
        FIXED: Safely estimate target CLIP norm from EVA features with proper error handling
        """
        try:
            # Ensure eva_features is on the right device and is a proper tensor
            if not torch.is_tensor(eva_features):
                logger.error(f"❌ eva_features is not a tensor: {type(eva_features)}")
                return 26.0
            
            if eva_features.numel() == 0:
                logger.warning("⚠️ eva_features is empty, using default norm")
                return 26.0
            
            # Compute norm safely
            eva_norm_tensor = torch.norm(eva_features, dim=-1)  # [B, N]
            eva_norm_mean_tensor = eva_norm_tensor.mean()       # [1]
            
            # Convert to Python float safely
            if eva_norm_mean_tensor.numel() != 1:
                logger.error(f"❌ eva_norm_mean is not a scalar: shape={eva_norm_mean_tensor.shape}")
                return 26.0
            
            eva_norm_mean = float(eva_norm_mean_tensor.item())
            
            # Empirical relationship between EVA and CLIP norms
            estimated_clip_norm = eva_norm_mean * 0.6
            
            # Clamp to reasonable range
            result = max(20.0, min(35.0, estimated_clip_norm))
            
            # Validate result
            if not isinstance(result, (int, float)):
                logger.error(f"❌ estimated norm is not a scalar: {type(result)}")
                return 26.0
            
            return float(result)
            
        except Exception as e:
            logger.error(f"❌ Error in _safe_estimate_target_norm_from_eva: {e}")
            logger.error(f"eva_features shape: {eva_features.shape if torch.is_tensor(eva_features) else 'not tensor'}")
            return 26.0
    
    def _safe_convert_to_scalar(self, value, param_name: str, default_value: float = 26.0) -> float:
        """
        FIXED: Safely convert any value to a Python float scalar
        """
        try:
            if torch.is_tensor(value):
                if value.numel() == 1:
                    # Single element tensor
                    return float(value.item())
                elif value.numel() == 0:
                    # Empty tensor
                    logger.warning(f"⚠️ {param_name} is empty tensor, using default: {default_value}")
                    return default_value
                else:
                    # Multi-element tensor - this is the error we're trying to fix
                    logger.error(f"❌ {param_name} is a multi-element tensor with {value.numel()} elements!")
                    logger.error(f"   Shape: {value.shape}, dtype: {value.dtype}")
                    logger.error(f"   This should not happen! Using default value: {default_value}")
                    return default_value
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                else:
                    logger.error(f"❌ {param_name} is a multi-element numpy array!")
                    return default_value
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                logger.error(f"❌ {param_name} has unexpected type: {type(value)}")
                return default_value
        except Exception as e:
            logger.error(f"❌ Error converting {param_name} to scalar: {e}")
            logger.error(f"   Value type: {type(value)}")
            if torch.is_tensor(value):
                logger.error(f"   Tensor shape: {value.shape}, numel: {value.numel()}")
            return default_value

    @torch.no_grad()
    def generate(
        self,
        eva_features: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        normalize_output: bool = False,
        # FIXED: Scale-aware generation parameters with better type checking
        target_norm: Optional[Union[float, torch.Tensor]] = None,
        use_lognormal_schedule: bool = True,
        velocity_explosion_threshold: Optional[float] = None,
        norm_guidance_strength: Optional[float] = None,
        norm_guidance_frequency: Optional[int] = None,
        debug_generation: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        FIXED: Scale-aware generation with robust target_norm handling
        """
        device = eva_features.device
        batch_size, num_tokens, _ = eva_features.shape
        
        if debug_generation:
            logger.info(f"🚀 Starting scale-aware generation")
            logger.info(f"   eva_features shape: {eva_features.shape}")
            logger.info(f"   target_norm input: {target_norm} (type: {type(target_norm)})")
        
        # FIXED: Get config parameters safely
        velocity_explosion_threshold = self._safe_convert_to_scalar(
            velocity_explosion_threshold if velocity_explosion_threshold is not None 
            else self.config.velocity_explosion_threshold,
            "velocity_explosion_threshold", 100.0
        )
        
        norm_guidance_strength = self._safe_convert_to_scalar(
            norm_guidance_strength if norm_guidance_strength is not None 
            else self.config.norm_guidance_strength,
            "norm_guidance_strength", 0.1
        )
        
        norm_guidance_frequency = int(
            norm_guidance_frequency if norm_guidance_frequency is not None 
            else self.config.norm_guidance_frequency
        )
        
        # FIXED: Handle target_norm with comprehensive error checking
        if target_norm is None:
            # Try to get from config first
            config_norm = getattr(self.config, 'typical_clip_norm', None)
            if config_norm is not None:
                target_norm_scalar = self._safe_convert_to_scalar(config_norm, "config.typical_clip_norm", 26.0)
                if debug_generation:
                    logger.info(f"   Using config typical_clip_norm: {target_norm_scalar}")
            else:
                # Estimate from EVA features
                target_norm_scalar = self._safe_estimate_target_norm_from_eva(eva_features)
                if debug_generation:
                    logger.info(f"   Estimated from EVA features: {target_norm_scalar}")
        else:
            # Convert provided target_norm to scalar safely
            target_norm_scalar = self._safe_convert_to_scalar(target_norm, "target_norm", 26.0)
            if debug_generation:
                logger.info(f"   Using provided target_norm: {target_norm_scalar}")
        
        # FIXED: Validate target_norm_scalar is reasonable and is definitely a Python float
        if not isinstance(target_norm_scalar, (int, float)):
            logger.error(f"❌ target_norm_scalar is not a number: {type(target_norm_scalar)}")
            target_norm_scalar = 26.0
        
        target_norm_scalar = float(target_norm_scalar)  # Ensure it's a Python float
        
        if not (10.0 <= target_norm_scalar <= 100.0):
            logger.warning(f"⚠️ target_norm {target_norm_scalar:.3f} is outside reasonable range [10, 100], clamping")
            target_norm_scalar = max(10.0, min(100.0, target_norm_scalar))
        
        if debug_generation:
            logger.info(f"🎯 Final target_norm_scalar: {target_norm_scalar} (type: {type(target_norm_scalar).__name__})")
            logger.info(f"   Scale-aware parameters:")
            logger.info(f"     Velocity explosion threshold: {velocity_explosion_threshold}")
            logger.info(f"     Norm guidance strength: {norm_guidance_strength}")
            logger.info(f"     Norm guidance frequency: {norm_guidance_frequency}")
        
        # Start from properly scaled noise
        x = torch.randn(
            batch_size, num_tokens, self.config.clip_embedding_size,
            device=device, generator=generator, dtype=eva_features.dtype
        )
        
        if debug_generation:
            initial_norm = torch.norm(x, dim=-1).mean().item()
            logger.info(f"📊 Initial noise norm: {initial_norm:.3f}")
        
        # Create timestep schedule
        if use_lognormal_schedule:
            timesteps = self._create_lognormal_timestep_schedule(num_inference_steps, device)
            if debug_generation:
                logger.info(f"📅 Using log-normal timestep schedule")
        else:
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)[:-1]
            if debug_generation:
                logger.info(f"📅 Using linear timestep schedule")
        
        # Forward ODE integration with scale-aware improvements
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t.item(), device=device, dtype=eva_features.dtype)
            
            # Get velocity prediction
            velocity = self.forward(
                hidden_states=x,
                timestep=t_batch,
                encoder_hidden_states=eva_features,
                return_dict=False
            )
            
            # IMPROVEMENT 1: Prevent velocity explosion
            velocity_norm = torch.norm(velocity, dim=-1).mean()
            if velocity_norm > velocity_explosion_threshold:
                scale_factor = (velocity_explosion_threshold * 0.5) / velocity_norm.item()
                velocity = velocity * scale_factor
                if debug_generation and i % 20 == 0:
                    logger.info(f"  Step {i}: Scaled velocity by {scale_factor:.3f} (norm was {velocity_norm:.1f})")
            
            # Compute step size
            if i < len(timesteps) - 1:
                dt = timesteps[i] - timesteps[i + 1]
            else:
                dt = timesteps[i]
            
            # Euler step
            x = x + dt * velocity
            
            # IMPROVEMENT 2: Periodic norm guidance
            if i % norm_guidance_frequency == 0 and i > 0:
                current_norm = torch.norm(x, dim=-1, keepdim=True)  # [B, N, 1]
                
                # FIXED: Create target_norm_tensor safely - target_norm_scalar is guaranteed to be a Python float
                try:
                    target_norm_tensor = torch.full_like(current_norm, target_norm_scalar)
                except Exception as e:
                    logger.error(f"❌ Error creating target_norm_tensor at step {i}: {e}")
                    logger.error(f"   target_norm_scalar: {target_norm_scalar} (type: {type(target_norm_scalar)})")
                    logger.error(f"   current_norm shape: {current_norm.shape}")
                    # Skip this guidance step if there's an error
                    continue
                
                # Gentle correction towards target norm
                norm_ratio = target_norm_tensor / (current_norm + 1e-8)
                correction_strength = norm_guidance_strength * min(1.0, i / (num_inference_steps * 0.3))
                norm_correction = correction_strength * (norm_ratio - 1.0)
                x = x * (1.0 + norm_correction)
                
                if debug_generation and i % (norm_guidance_frequency * 4) == 0:
                    avg_current_norm = current_norm.mean().item()
                    logger.info(f"  Step {i}: Applied norm guidance, current norm: {avg_current_norm:.3f} -> target: {target_norm_scalar:.3f}")
        
        # IMPROVEMENT 3: Final scale correction
        current_norm = torch.norm(x, dim=-1, keepdim=True)  # [B, N, 1]
        
        # FIXED: Create final target_norm_tensor safely
        try:
            final_target_norm_tensor = torch.full_like(current_norm, target_norm_scalar)
            x = x * (final_target_norm_tensor / (current_norm + 1e-8))
        except Exception as e:
            logger.error(f"❌ Error in final scale correction: {e}")
            logger.error(f"   target_norm_scalar: {target_norm_scalar} (type: {type(target_norm_scalar)})")
            logger.error(f"   current_norm shape: {current_norm.shape}")
            # Skip final correction if there's an error
            logger.warning("⚠️ Skipping final scale correction due to error")
        
        # Optional normalization
        if normalize_output:
            x = F.normalize(x, p=2, dim=-1)
            if debug_generation:
                logger.info("Applied L2 normalization to output")
        
        if debug_generation:
            final_norm = torch.norm(x, dim=-1).mean().item()
            logger.info(f"🎯 Final norm: {final_norm:.3f} (target: {target_norm_scalar:.3f})")
            logger.info(f"✅ Scale-aware generation completed successfully")
        
        return x
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_clip_reproduction_model(
    config: Optional[BLIP3oCLIPDiTConfig] = None,
    training_mode: str = "patch_only",
    model_size: str = "base",
    use_3d_rope: bool = True,
    use_sandwich_norm: bool = True,
    # FIXED: Scale-aware generation parameters with proper type validation
    typical_clip_norm: Union[float, int] = 26.0,
    velocity_explosion_threshold: Union[float, int] = 100.0,
    norm_guidance_strength: Union[float, int] = 0.1,
    norm_guidance_frequency: int = 10,
    **kwargs
) -> BLIP3oCLIPDiTModel:
    """FIXED: Create CLIP reproduction model with scale-aware generation and proper type handling"""
    
    if config is None:
        size_configs = {
            "tiny": {"hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 6, "num_key_value_heads": 2},
            "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "num_key_value_heads": 4},
            "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "num_key_value_heads": 4},
            "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16, "num_key_value_heads": 8},
        }
        
        model_config = size_configs[model_size].copy()
        model_config.update({
            "num_tokens": 257 if training_mode == "cls_patch" else 256,
            "training_mode": training_mode,
            "eva_embedding_size": 4096,
            "clip_embedding_size": 1024,
            "intermediate_size": model_config["hidden_size"] * 4,
            "use_3d_rope": use_3d_rope,
            "use_sandwich_norm": use_sandwich_norm,
            # FIXED: Ensure scale-aware parameters are always Python floats/ints
            "typical_clip_norm": float(typical_clip_norm),
            "velocity_explosion_threshold": float(velocity_explosion_threshold),
            "norm_guidance_strength": float(norm_guidance_strength),
            "norm_guidance_frequency": int(norm_guidance_frequency),
            **kwargs
        })
        
        config = BLIP3oCLIPDiTConfig(**model_config)
    
    return BLIP3oCLIPDiTModel(config)