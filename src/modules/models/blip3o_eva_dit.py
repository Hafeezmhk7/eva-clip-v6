#!/usr/bin/env python3
"""
Fixed BLIP3-o DiT Model for EVA-CLIP Reproduction Testing
src/modules/models/blip3o_eva_dit.py

Key fixes:
- Better weight initialization for flow matching
- Improved architecture with proper normalization
- Added debugging features
- Fixed gradient flow issues
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
        dropout_prob: float = 0.0,  # Disable dropout for better training
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_gradient_checkpointing: bool = False,
        training_mode: str = "patch_only",
        zero_init_output: bool = True,  # Zero initialize output layer
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
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.initializer_range = initializer_range
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        self.zero_init_output = zero_init_output


class ImprovedTimestepEmbedder(nn.Module):
    """Improved timestep embedding with better scaling"""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        
        # Initialize with smaller weights for stability
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

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


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with timestep and optional conditioning"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.shift = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor, timestep_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        normalized = self.norm(x)
        if timestep_emb is not None:
            # Simple gating based on timestep
            scale = 1.0 + torch.tanh(timestep_emb[..., :x.shape[-1]]) * 0.1
            shift = torch.tanh(timestep_emb[..., x.shape[-1]:x.shape[-1]*2]) * 0.1
            return normalized * scale + shift
        return normalized * self.scale + self.shift


class ImprovedMultiHeadAttention(nn.Module):
    """Improved Multi-head attention with better initialization"""
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
        
        # Better initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)


class BLIP3oEVADiTBlock(nn.Module):
    """Improved DiT block with better normalization and initialization"""
    def __init__(self, config: BLIP3oEVADiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Layer normalization
        self.norm1 = AdaptiveLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = AdaptiveLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm3 = AdaptiveLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Self-attention
        self.self_attn = ImprovedMultiHeadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            config.attention_dropout
        )
        
        # Cross-attention with CLIP conditioning
        self.cross_attn = ImprovedMultiHeadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            config.attention_dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout),
        )
        
        # Project CLIP embeddings to hidden dimension
        self.clip_proj = nn.Linear(config.clip_embedding_size, config.hidden_size)
        
        # Initialize feed-forward with smaller weights
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        
        # Initialize projection
        nn.init.xavier_uniform_(self.clip_proj.weight)
        nn.init.zeros_(self.clip_proj.bias)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        timestep_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states, timestep_emb)
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + hidden_states
        
        # Cross-attention with CLIP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states, timestep_emb)
        clip_features = self.clip_proj(encoder_hidden_states)
        hidden_states = self.cross_attn(hidden_states, clip_features, clip_features)
        hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm3(hidden_states, timestep_emb)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class BLIP3oEVADiTModel(PreTrainedModel):
    """
    Fixed DiT Model for EVA-CLIP Reproduction Testing
    
    Key improvements:
    - Better initialization for flow matching
    - Improved normalization strategy
    - Added debugging features
    - Fixed gradient flow
    """
    
    config_class = BLIP3oEVADiTConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: BLIP3oEVADiTConfig):
        super().__init__(config)
        self.config = config
        
        self.gradient_checkpointing = False
        
        # Input projection from EVA dimension to hidden dimension
        self.input_proj = nn.Linear(config.eva_embedding_size, config.hidden_size, bias=True)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        
        # Timestep embedding
        self.timestep_embedder = ImprovedTimestepEmbedder(config.hidden_size)
        
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BLIP3oEVADiTBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output layers
        self.output_norm = AdaptiveLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_proj = nn.Linear(config.hidden_size, config.eva_embedding_size, bias=True)
        
        # Initialize output projection
        if config.zero_init_output:
            # Zero initialization for output helps with flow matching
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        else:
            nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
            nn.init.zeros_(self.output_proj.bias)
        
        # Track initialization
        self.register_buffer('initialized', torch.tensor(True))
        
        logger.info(f"✅ Fixed EVA Reproduction DiT model initialized:")
        logger.info(f"   Parameters: {self.get_num_parameters():,}")
        logger.info(f"   Zero init output: {config.zero_init_output}")
        logger.info(f"   Dropout disabled: {config.dropout_prob == 0.0}")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        logger.info("✅ Gradient checkpointing enabled")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _gradient_checkpointing_func(self, module, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
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
        Forward pass with improved stability
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project EVA input to hidden dimension
        x = self.input_proj(hidden_states)
        
        # Add positional embeddings
        if seq_len <= self.config.max_position_embeddings:
            x = x + self.pos_embed[:, :seq_len, :]
        
        # Get timestep embeddings
        timestep_emb = self.timestep_embedder(timestep)
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(
                    block, x, encoder_hidden_states, timestep_emb
                )
            else:
                x = block(x, encoder_hidden_states, timestep_emb)
        
        # Output projection
        x = self.output_norm(x, timestep_emb)
        velocity_pred = self.output_proj(x)
        
        if return_dict:
            return {
                "velocity_prediction": velocity_pred,
                "hidden_states": x,
            }
        return velocity_pred
    
    @torch.no_grad()
    def generate(
        self,
        clip_features: torch.Tensor,  # [B, N, 1024] - CLIP conditioning
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        guidance_scale: float = 1.0,
        return_intermediate: bool = False,
        normalize_output: bool = True,
        solver: str = "euler",  # "euler" or "heun"
    ) -> torch.Tensor:
        """
        Generate EVA embeddings using improved ODE solver
        """
        device = clip_features.device
        batch_size, num_tokens, _ = clip_features.shape
        
        # Start from normalized noise for better stability
        x = torch.randn(
            batch_size, num_tokens, self.config.eva_embedding_size,
            device=device, generator=generator, dtype=clip_features.dtype
        )
        x = F.normalize(x, p=2, dim=-1)
        
        # Setup timesteps (from 1 to 0)
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        dt = 1.0 / num_inference_steps
        
        intermediates = [] if return_intermediate else None
        
        for i in range(num_inference_steps):
            t = timesteps[i]
            t_batch = torch.full((batch_size,), t, device=device, dtype=clip_features.dtype)
            
            # Predict velocity
            with torch.cuda.amp.autocast(enabled=False):  # Disable for stability
                velocity = self.forward(
                    hidden_states=x,
                    timestep=t_batch,
                    encoder_hidden_states=clip_features,
                    return_dict=False
                )
            
            # Apply guidance if needed
            if guidance_scale != 1.0:
                velocity = velocity * guidance_scale
            
            if solver == "heun":
                # Heun's method (2nd order)
                # First step
                x_next = x - dt * velocity
                
                # Second step (if not last)
                if i < num_inference_steps - 1:
                    t_next = timesteps[i + 1]
                    t_next_batch = torch.full((batch_size,), t_next, device=device, dtype=clip_features.dtype)
                    velocity_next = self.forward(
                        hidden_states=x_next,
                        timestep=t_next_batch,
                        encoder_hidden_states=clip_features,
                        return_dict=False
                    )
                    if guidance_scale != 1.0:
                        velocity_next = velocity_next * guidance_scale
                    x = x - dt * 0.5 * (velocity + velocity_next)
                else:
                    x = x_next
            else:
                # Euler method
                x = x - dt * velocity
            
            if return_intermediate:
                intermediates.append(x.clone())
        
        # Final normalization
        if normalize_output:
            x = F.normalize(x, p=2, dim=-1)
        
        if return_intermediate:
            return x, intermediates
        return x
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def debug_forward(self, *args, **kwargs):
        """Debug forward pass to check intermediate values"""
        # Store original mode
        original_mode = self.training
        self.eval()
        
        # Get intermediate activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'norm': output.norm().item(),
                        'shape': output.shape,
                    }
            return hook
        
        # Register hooks
        hooks = []
        hooks.append(self.input_proj.register_forward_hook(hook_fn('input_proj')))
        hooks.append(self.output_proj.register_forward_hook(hook_fn('output_proj')))
        for i, block in enumerate(self.blocks):
            hooks.append(block.register_forward_hook(hook_fn(f'block_{i}')))
        
        # Forward pass
        output = self.forward(*args, **kwargs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Restore mode
        self.train(original_mode)
        
        return output, activations


def create_eva_reproduction_model(
    config: Optional[BLIP3oEVADiTConfig] = None,
    training_mode: str = "patch_only",
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    use_gradient_checkpointing: bool = False,
    zero_init_output: bool = True,
    dropout_prob: float = 0.0,
    **kwargs
) -> BLIP3oEVADiTModel:
    """
    Create fixed EVA reproduction DiT model
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
            zero_init_output=zero_init_output,
            dropout_prob=dropout_prob,
            clip_embedding_size=1024,
            eva_embedding_size=4096,
            **kwargs
        )
    
    model = BLIP3oEVADiTModel(config)
    
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model