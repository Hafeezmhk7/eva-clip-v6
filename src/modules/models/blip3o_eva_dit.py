#!/usr/bin/env python3
"""
Modified BLIP3-o DiT Model for EVA-CLIP Reproduction Testing
src/modules/models/blip3o_eva_dit.py

This version tests reproducing EVA-CLIP embeddings from noisy EVA-CLIP embeddings,
using CLIP embeddings as conditioning.

KEY CHANGES:
- Input: Noisy EVA embeddings [B, N, 4096]
- Conditioning: CLIP embeddings [B, N, 1024]
- Output: Clean EVA embeddings [B, N, 4096]
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np
import logging
from transformers import PreTrainedModel, PretrainedConfig
from torch.utils.checkpoint import checkpoint
from functools import partial

logger = logging.getLogger(__name__)

class BLIP3oEVADiTConfig(PretrainedConfig):
    """Configuration class for EVA reproduction DiT model"""
    model_type = "blip3o_eva_dit"
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        clip_embedding_size: int = 1024,  # CLIP conditioning dimension
        eva_embedding_size: int = 4096,   # EVA input/output dimension
        num_tokens: int = 256,
        max_position_embeddings: int = 256,
        dropout_prob: float = 0.1,
        training_mode: str = "patch_only",
        use_gradient_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.clip_embedding_size = clip_embedding_size
        self.eva_embedding_size = eva_embedding_size
        self.num_tokens = num_tokens
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        self.training_mode = training_mode
        self.use_gradient_checkpointing = use_gradient_checkpointing


class RotaryPositionalEmbedding3D(nn.Module):
    """3D Rotary Position Embedding for spatial patch layout"""
    def __init__(self, dim: int, max_position_embeddings: int = 256, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim // 4, 2).float() / (dim // 4)))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        if seq_len not in [256, 257]:
            return x
        
        if seq_len == 257:
            cls_pos = torch.zeros(1, device=x.device)
            patch_x = torch.arange(16, device=x.device).repeat(16)
            patch_y = torch.arange(16, device=x.device).repeat_interleave(16)
            pos_x = torch.cat([cls_pos, patch_x])
            pos_y = torch.cat([cls_pos, patch_y])
        else:
            patch_x = torch.arange(16, device=x.device).repeat(16)
            patch_y = torch.arange(16, device=x.device).repeat_interleave(16)
            pos_x = patch_x
            pos_y = patch_y
        
        rope_dim = min(hidden_size, len(self.inv_freq) * 4)
        if rope_dim < 4:
            return x
        
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        
        x_rot = self._apply_2d_rotary_pos_emb(x_rope, pos_x, pos_y)
        return torch.cat([x_rot, x_pass], dim=-1)
    
    def _apply_2d_rotary_pos_emb(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, rope_dim = x.shape
        quarter_dim = rope_dim // 4
        
        x1, x2, x3, x4 = torch.split(x, quarter_dim, dim=-1)
        
        freqs_x = torch.outer(pos_x, self.inv_freq[:quarter_dim])
        freqs_y = torch.outer(pos_y, self.inv_freq[:quarter_dim])
        
        cos_x, sin_x = freqs_x.cos(), freqs_x.sin()
        cos_y, sin_y = freqs_y.cos(), freqs_y.sin()
        
        x1_rot = x1 * cos_x - x2 * sin_x
        x2_rot = x1 * sin_x + x2 * cos_x
        x3_rot = x3 * cos_y - x4 * sin_y
        x4_rot = x3 * sin_y + x4 * cos_y
        
        return torch.cat([x1_rot, x2_rot, x3_rot, x4_rot], dim=-1)


class TimestepEmbedder(nn.Module):
    """Timestep embedding with proper scaling"""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
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


class MultiHeadAttention(nn.Module):
    """Multi-head attention with gradient checkpointing support"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class BLIP3oEVADiTBlock(nn.Module):
    """DiT block for EVA reproduction with CLIP conditioning"""
    def __init__(self, config: BLIP3oEVADiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.self_attn = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.dropout_prob)
        self.cross_attn = MultiHeadAttention(config.hidden_size, config.num_attention_heads, config.dropout_prob)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout_prob),
        )
        
        # Project CLIP embeddings to hidden dimension for cross-attention
        self.clip_proj = nn.Linear(config.clip_embedding_size, config.hidden_size)
        self.rope = RotaryPositionalEmbedding3D(config.hidden_size)

    def _self_attention_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden = self.norm1(hidden_states)
        norm_hidden = self.rope(norm_hidden)
        attn_output = self.self_attn(norm_hidden, norm_hidden, norm_hidden)
        return hidden_states + attn_output
    
    def _cross_attention_block(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden = self.norm2(hidden_states)
        clip_features = self.clip_proj(encoder_hidden_states)  # Project CLIP to hidden_size
        cross_attn_output = self.cross_attn(norm_hidden, clip_features, clip_features)
        return hidden_states + cross_attn_output
    
    def _ffn_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden = self.norm3(hidden_states)
        ffn_output = self.ffn(norm_hidden)
        return hidden_states + ffn_output

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._self_attention_block(hidden_states)
        hidden_states = self._cross_attention_block(hidden_states, encoder_hidden_states)
        hidden_states = self._ffn_block(hidden_states)
        return hidden_states


class BLIP3oEVADiTModel(PreTrainedModel):
    """
    Modified DiT Model for EVA-CLIP Reproduction Testing
    
    Input: Noisy EVA embeddings [B, N, 4096]
    Conditioning: CLIP embeddings [B, N, 1024]
    Output: Clean EVA embeddings [B, N, 4096]
    """
    
    config_class = BLIP3oEVADiTConfig
    supports_gradient_checkpointing = True
    _supports_gradient_checkpointing = True
    
    def __init__(self, config: BLIP3oEVADiTConfig):
        super().__init__(config)
        self.config = config
        
        self.gradient_checkpointing = False
        
        # Input projection from EVA dimension to hidden dimension
        self.input_proj = nn.Linear(config.eva_embedding_size, config.hidden_size, bias=True)
        
        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BLIP3oEVADiTBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.output_proj = nn.Linear(config.hidden_size, config.eva_embedding_size, bias=True)
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize output projection to small values for stability
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        logger.info(f"✅ EVA Reproduction DiT model initialized:")
        logger.info(f"   Input: EVA embeddings [B, N, {config.eva_embedding_size}] (noisy)")
        logger.info(f"   Conditioning: CLIP embeddings [B, N, {config.clip_embedding_size}]")
        logger.info(f"   Output: EVA embeddings [B, N, {config.eva_embedding_size}] (clean)")
        logger.info(f"   Training mode: {config.training_mode}")
        logger.info(f"   Token count: {config.num_tokens}")
        logger.info(f"   Parameters: {self.get_num_parameters():,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BLIP3oEVADiTBlock):
            pass

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        
        self.gradient_checkpointing = True
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        logger.info("✅ Gradient checkpointing enabled")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.apply(partial(self._set_gradient_checkpointing, value=False))
        logger.info("Gradient checkpointing disabled")

    def _gradient_checkpointing_func(self, module, *args, **kwargs):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        if self.gradient_checkpointing and self.training:
            return checkpoint(create_custom_forward(module), *args, use_reentrant=False)
        else:
            return module(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, N, 4096] - Noisy EVA embeddings
        timestep: torch.Tensor,  # [B] - Flow matching timesteps
        encoder_hidden_states: torch.Tensor,  # [B, N, 1024] - CLIP conditioning
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for EVA reproduction
        
        Args:
            hidden_states: Noisy EVA embeddings [B, N, 4096]
            timestep: Flow matching timesteps [B]
            encoder_hidden_states: CLIP conditioning [B, N, 1024]
        
        Returns:
            Velocity prediction [B, N, 4096]
        """
        batch_size, seq_len, input_dim = hidden_states.shape
        
        # Validate inputs
        assert input_dim == self.config.eva_embedding_size, f"Expected {self.config.eva_embedding_size}-dim EVA input, got {input_dim}"
        assert seq_len in [256, 257], f"Expected 256 or 257 tokens, got {seq_len}"
        assert encoder_hidden_states.shape[2] == self.config.clip_embedding_size, f"Expected {self.config.clip_embedding_size}-dim CLIP features"
        
        # Project EVA input to hidden dimension
        x = self.input_proj(hidden_states)  # [B, N, hidden_size]
        
        # Add positional embeddings
        if seq_len <= self.config.max_position_embeddings:
            x = x + self.pos_embed[:, :seq_len, :]
        else:
            raise ValueError(f"Sequence length {seq_len} exceeds max position embeddings")
        
        # Add timestep embeddings
        timestep_emb = self.timestep_embedder(timestep)  # [B, hidden_size]
        timestep_emb = timestep_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, N, hidden_size]
        x = x + timestep_emb
        
        # Pass through transformer blocks with CLIP conditioning
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(block, x, encoder_hidden_states)
            else:
                x = block(x, encoder_hidden_states)
        
        # Output projection back to EVA dimension
        x = self.output_norm(x)
        velocity_pred = self.output_proj(x)  # [B, N, 4096]
        
        if return_dict:
            return {
                "velocity_prediction": velocity_pred,
                "hidden_states": x,
                "timestep_embeddings": timestep_emb,
            }
        return velocity_pred
    
    @torch.no_grad()
    def generate(
        self,
        clip_features: torch.Tensor,  # [B, N, 1024] - CLIP conditioning
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        normalize_output: bool = True,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate EVA embeddings from CLIP conditioning using rectified flow
        """
        device = clip_features.device
        batch_size, num_tokens, _ = clip_features.shape
        
        # Start from pure noise (same dimension as EVA: 4096)
        x = torch.randn(
            batch_size, num_tokens, self.config.eva_embedding_size,
            device=device,
            generator=generator,
            dtype=clip_features.dtype
        )
        
        # RECTIFIED FLOW: Linear timestep schedule from 1 (noise) to 0 (data)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        dt = 1.0 / num_inference_steps
        
        intermediates = [] if return_intermediate else None
        
        for i in range(num_inference_steps):
            # Current timestep
            t = torch.full((batch_size,), timesteps[i], device=device, dtype=clip_features.dtype)
            
            # Predict velocity
            velocity = self.forward(
                hidden_states=x,
                timestep=t,
                encoder_hidden_states=clip_features,
                return_dict=False
            )
            
            # Apply guidance if needed
            if guidance_scale != 1.0:
                velocity = velocity * guidance_scale
            
            # RECTIFIED FLOW: Euler integration
            # dx/dt = velocity, so x_next = x + dt * velocity
            x = x + dt * velocity
            
            if return_intermediate:
                intermediates.append(x.clone())
        
        # Final normalization for similarity comparison
        if normalize_output:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        if return_intermediate:
            return x, intermediates
        return x
    
    @torch.no_grad()
    def evaluate_similarity(
        self,
        clip_features: torch.Tensor,  # [B, N, 1024]
        target_eva_embeddings: torch.Tensor,  # [B, N, 4096]
        num_inference_steps: int = 50,
        normalize_embeddings: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate EVA embedding similarity between generated and target embeddings
        """
        # Generate EVA embeddings from CLIP conditioning
        generated = self.generate(
            clip_features=clip_features,
            num_inference_steps=num_inference_steps,
            normalize_output=normalize_embeddings
        )
        
        # Normalize targets if needed
        if normalize_embeddings:
            targets_norm = torch.nn.functional.normalize(target_eva_embeddings, p=2, dim=-1)
        else:
            targets_norm = target_eva_embeddings
        
        # Compute cosine similarity per patch
        per_patch_sim = torch.nn.functional.cosine_similarity(generated, targets_norm, dim=-1)
        
        # Compute per-image similarity (average over patches)
        per_image_sim = per_patch_sim.mean(dim=1)
        
        # Overall similarity
        overall_sim = per_image_sim.mean().item()
        
        return {
            'overall_eva_similarity': overall_sim,
            'per_image_mean': per_image_sim.mean().item(),
            'per_image_std': per_image_sim.std().item(),
            'per_patch_mean': per_patch_sim.mean().item(),
            'per_patch_std': per_patch_sim.std().item(),
            'high_quality_images': (per_image_sim > 0.7).float().mean().item(),
            'very_high_quality_images': (per_image_sim > 0.8).float().mean().item(),
            'num_inference_steps': num_inference_steps,
            'normalized': normalize_embeddings,
        }
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_eva_reproduction_model(
    config: Optional[BLIP3oEVADiTConfig] = None,
    training_mode: str = "patch_only",
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    use_gradient_checkpointing: bool = False,
    **kwargs
) -> BLIP3oEVADiTModel:
    """
    Create EVA reproduction DiT model
    """
    if config is None:
        num_tokens = 257 if training_mode == "cls_patch" else 256
        config = BLIP3oEVADiTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_tokens=num_tokens,
            max_position_embeddings=max(num_tokens, 257),
            training_mode=training_mode,
            use_gradient_checkpointing=use_gradient_checkpointing,
            clip_embedding_size=1024,  # CLIP conditioning
            eva_embedding_size=4096,   # EVA input/output
            **kwargs
        )
    
    model = BLIP3oEVADiTModel(config)
    
    if use_gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled during model creation")
        except Exception as e:
            logger.warning(f"⚠️ Could not enable gradient checkpointing during creation: {e}")
    
    return model