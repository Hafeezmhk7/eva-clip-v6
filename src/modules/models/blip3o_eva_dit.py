#!/usr/bin/env python3
"""
Fixed BLIP3-o DiT Model for EVA-CLIP Reproduction
Key fixes:
1. Proper gradient flow and initialization
2. Correct input/output handling
3. Better numerical stability
4. Fixed timestep embedding and attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import logging
from transformers import PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)


class BLIP3oEVADiTConfig(PretrainedConfig):
    """Configuration for BLIP3-o EVA DiT model"""
    model_type = "blip3o_eva_dit"
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        intermediate_size: int = 3072,
        clip_embedding_size: int = 1024,
        eva_embedding_size: int = 4096,
        num_tokens: int = 256,
        max_position_embeddings: int = 256,
        rms_norm_eps: float = 1e-6,
        dropout_prob: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_gradient_checkpointing: bool = False,
        training_mode: str = "patch_only",
        rope_theta: float = 10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.clip_embedding_size = clip_embedding_size
        self.eva_embedding_size = eva_embedding_size
        self.num_tokens = num_tokens
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.dropout_prob = dropout_prob
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        self.rope_theta = rope_theta


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


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_position_embeddings: int = 256, base: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding"""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TimestepEmbedder(nn.Module):
    """Fixed timestep embedding"""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        
        # Use SiLU activation for better gradients
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Better initialization
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


class Attention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, config: BLIP3oEVADiTConfig):
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
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights"""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight)
        # Smaller initialization for output projection
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1.0 / math.sqrt(self.config.num_hidden_layers))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        
        if key_value_states is not None:
            # Cross-attention
            kv_seq_len = key_value_states.shape[1]
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            # Self-attention
            kv_seq_len = q_len
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat k/v heads if needed
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        
        return self.o_proj(attn_output)


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config: BLIP3oEVADiTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Better initialization
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1.0 / math.sqrt(config.num_hidden_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class AdaLN(nn.Module):
    """Adaptive Layer Normalization"""
    def __init__(self, hidden_size: int, conditioning_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_size, 2 * hidden_size, bias=True)
        )
        
        # Initialize to identity transformation
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        # Ensure conditioning has correct shape
        if conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1)
        
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        
        # Broadcast if needed
        if shift.shape[1] == 1 and x.shape[1] > 1:
            shift = shift.expand(-1, x.shape[1], -1)
            scale = scale.expand(-1, x.shape[1], -1)
        
        normalized = self.norm(x)
        return normalized * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT transformer block"""
    def __init__(self, config: BLIP3oEVADiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Normalization layers
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm3 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Adaptive layer norms with timestep conditioning
        self.ada_ln1 = AdaLN(config.hidden_size, config.hidden_size)
        self.ada_ln2 = AdaLN(config.hidden_size, config.hidden_size)
        self.ada_ln3 = AdaLN(config.hidden_size, config.hidden_size)
        
        # Attention layers
        self.self_attn = Attention(config)
        self.cross_attn = Attention(config)
        
        # MLP
        self.mlp = MLP(config)
        
        # CLIP projection
        self.clip_proj = nn.Linear(config.clip_embedding_size, config.hidden_size, bias=True)
        nn.init.xavier_uniform_(self.clip_proj.weight)
        nn.init.zeros_(self.clip_proj.bias)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.ada_ln1(hidden_states, timestep_emb)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # Cross-attention with CLIP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ada_ln2(hidden_states, timestep_emb)
        clip_features = self.clip_proj(encoder_hidden_states)
        hidden_states = self.cross_attn(hidden_states, key_value_states=clip_features)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ada_ln3(hidden_states, timestep_emb)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class BLIP3oEVADiTModel(PreTrainedModel):
    """Fixed BLIP3-o DiT Model for EVA reproduction"""
    
    config_class = BLIP3oEVADiTConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: BLIP3oEVADiTConfig):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False
        
        # Input projection (EVA -> hidden)
        self.input_proj = nn.Linear(config.eva_embedding_size, config.hidden_size, bias=True)
        
        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output layers
        self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_adaln = AdaLN(config.hidden_size, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.eva_embedding_size, bias=True)
        
        # Initialize model
        self._init_weights()
        
        logger.info(f"BLIP3-o EVA DiT model initialized with {self.get_num_parameters():,} parameters")

    def _init_weights(self):
        """Initialize model weights"""
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # Positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Output projection - CRITICAL: Small initialization for flow matching
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, N, 4096] - Noisy EVA embeddings
        timestep: torch.Tensor,       # [B] - Flow matching timesteps
        encoder_hidden_states: torch.Tensor,  # [B, N, 1024] - CLIP conditioning
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project EVA input to hidden dimension
        x = self.input_proj(hidden_states)
        
        # Add positional embeddings
        if seq_len <= self.config.max_position_embeddings:
            x = x + self.pos_embed[:, :seq_len, :]
        
        # Get timestep embeddings
        timestep_emb = self.timestep_embedder(timestep)  # [B, hidden_size]
        
        # Pass through transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, encoder_hidden_states, timestep_emb, use_reentrant=False
                )
            else:
                x = block(x, encoder_hidden_states, timestep_emb)
        
        # Output projection
        x = self.output_norm(x)
        x = self.output_adaln(x, timestep_emb)
        velocity_pred = self.output_proj(x)
        
        if return_dict:
            return {"velocity_prediction": velocity_pred, "hidden_states": x}
        return velocity_pred
    
    @torch.no_grad()
    def generate(
        self,
        clip_features: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        normalize_output: bool = True,
    ) -> torch.Tensor:
        """Generate EVA embeddings using rectified flow"""
        device = clip_features.device
        batch_size, num_tokens, _ = clip_features.shape
        
        # Start from noise (t=0)
        x = torch.randn(
            batch_size, num_tokens, self.config.eva_embedding_size,
            device=device, generator=generator, dtype=clip_features.dtype
        )
        x = F.normalize(x, p=2, dim=-1)
        
        # Forward process (t=0 to t=1)
        dt = 1.0 / num_inference_steps
        
        for i in range(num_inference_steps):
            # Go from t=0 to t=1
            t = i * dt  # t: 0.0 → 0.98
            t_batch = torch.full((batch_size,), t, device=device, dtype=clip_features.dtype)
            
            # Get velocity prediction
            velocity = self.forward(
                hidden_states=x,
                timestep=t_batch,
                encoder_hidden_states=clip_features,
                return_dict=False
            )
            
            # Forward Euler step: follow the velocity field
            x = x + dt * velocity  # CHANGED: + instead of -
        
        if normalize_output:
            x = F.normalize(x, p=2, dim=-1)
        
        return x
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_eva_reproduction_model(
    config: Optional[BLIP3oEVADiTConfig] = None,
    training_mode: str = "patch_only",
    model_size: str = "base",
    **kwargs
) -> BLIP3oEVADiTModel:
    """Create EVA reproduction model"""
    
    if config is None:
        # Model size configurations
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
            "clip_embedding_size": 1024,
            "eva_embedding_size": 4096,
            "intermediate_size": model_config["hidden_size"] * 4,
            **kwargs
        })
        
        config = BLIP3oEVADiTConfig(**model_config)
    
    return BLIP3oEVADiTModel(config)