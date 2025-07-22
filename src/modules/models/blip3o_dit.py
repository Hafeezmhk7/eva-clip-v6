"""
BLIP3-o DiT Model - Properly Aligned with Paper
src/modules/models/blip3o_dit.py

This implementation follows the BLIP3-o paper architecture:
1. Patch-level training with [B, 256, 1024] CLIP targets
2. Proper DiT architecture with timestep conditioning
3. EVA-CLIP conditioning via cross-attention
4. Attention pooling for global representation during inference
5. Flow matching for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import numpy as np
from transformers import PreTrainedModel, CLIPModel

from ..config.blip3o_config import BLIP3oDiTConfig


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drop labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    
    def __init__(self, hidden_size, num_heads, eva_dim=4096, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Cross-attention with EVA features
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, kdim=eva_dim, vdim=eva_dim,
            dropout=0.0, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            approx_gelu(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # AdaLN-Zero modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 8 * hidden_size)
        )

    def forward(self, x, c, eva_features=None):
        # c: timestep + label conditioning
        # eva_features: EVA-CLIP conditioning [B, 256, 4096]
        
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(9, dim=1)
        
        # Self-attention with adaLN-Zero
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Cross-attention with EVA features (if provided)
        if eva_features is not None:
            x_norm_cross = self.norm_cross(x)
            x_norm_cross = x_norm_cross * (1 + scale_cross.unsqueeze(1)) + shift_cross.unsqueeze(1)
            
            cross_attn_out, _ = self.cross_attn(x_norm_cross, eva_features, eva_features)
            x = x + gate_cross.unsqueeze(1) * cross_attn_out
        
        # Feed-forward with adaLN-Zero
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class AttentionPooling(nn.Module):
    """Attention pooling for global representation."""
    
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        
    def forward(self, x):
        # x: [B, N, D]
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # [B, N+1, D]
        x = x + self.positional_embedding
        x, _ = F.multi_head_attention_forward(
            query=x[:, :1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(1)


class BLIP3oDiTModel(PreTrainedModel):
    """
    BLIP3-o Diffusion Transformer - Aligned with Paper
    
    Architecture follows the BLIP3-o paper:
    1. Takes patch-level noisy CLIP embeddings [B, 256, 1024] as input
    2. Uses EVA-CLIP features [B, 256, 4096] as conditioning
    3. Outputs patch-level velocity predictions [B, 256, 1024]
    4. Can pool to global representation during inference
    """
    
    config_class = BLIP3oDiTConfig
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__(config)
        self.config = config
        
        # Model dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.depth = config.num_hidden_layers
        self.eva_dim = 4096  # EVA-CLIP dimension
        self.clip_dim = 1024  # CLIP patch dimension
        
        # Input projection from CLIP patches to hidden size
        self.x_embedder = nn.Linear(self.clip_dim, self.hidden_size)
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        
        # Label embedding (for class conditioning if needed)
        self.y_embedder = LabelEmbedder(
            num_classes=1000,  # Can be adjusted
            hidden_size=self.hidden_size,
            dropout_prob=0.1
        )
        
        # Position embedding for 256 patches (16x16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, self.hidden_size))
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads, self.eva_dim)
            for _ in range(self.depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(self.hidden_size, 1, self.clip_dim)
        
        # Attention pooling for global representation
        self.global_pool = AttentionPooling(self.clip_dim, num_heads=8)
        
        # CLIP visual projection for global features
        self.clip_projection = None
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def load_clip_projection(self, clip_model_name="openai/clip-vit-large-patch14"):
        """Load CLIP visual projection for global features."""
        try:
            clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.clip_projection = clip_model.visual_projection
            # Freeze the projection
            for param in self.clip_projection.parameters():
                param.requires_grad = False
            print(f"✅ Loaded CLIP projection: {self.clip_projection.weight.shape}")
        except Exception as e:
            print(f"⚠️ Failed to load CLIP projection: {e}")
            # Create fallback
            self.clip_projection = nn.Linear(1024, 768, bias=False)
            nn.init.xavier_uniform_(self.clip_projection.weight)
            self.clip_projection.requires_grad_(False)

    def forward(
        self,
        hidden_states,  # [B, 256, 1024] - Noisy CLIP patches
        timestep,       # [B] - Timesteps
        encoder_hidden_states,  # [B, 256, 4096] - EVA conditioning
        class_labels=None,  # [B] - Class labels (optional)
        return_dict=True,
        **kwargs
    ):
        """
        Forward pass following BLIP3-o architecture.
        
        Args:
            hidden_states: Noisy CLIP patch embeddings [B, 256, 1024]
            timestep: Timesteps [B]
            encoder_hidden_states: EVA-CLIP conditioning [B, 256, 4096]
            class_labels: Class labels [B] (optional)
            
        Returns:
            Velocity predictions [B, 256, 1024]
        """
        # Input validation
        batch_size, seq_len, _ = hidden_states.shape
        assert seq_len == 256, f"Expected 256 patches, got {seq_len}"
        assert hidden_states.shape[2] == self.clip_dim, f"Expected CLIP dim {self.clip_dim}, got {hidden_states.shape[2]}"
        
        # Project inputs to hidden size
        x = self.x_embedder(hidden_states)  # [B, 256, hidden_size]
        x = x + self.pos_embed  # Add positional embedding
        
        # Timestep embedding
        t = self.t_embedder(timestep)  # [B, hidden_size]
        
        # Class embedding (use dummy if not provided)
        if class_labels is None:
            class_labels = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
        y = self.y_embedder(class_labels, self.training)  # [B, hidden_size]
        
        # Conditioning vector
        c = t + y  # [B, hidden_size]
        
        # Pass through DiT blocks
        for block in self.blocks:
            x = block(x, c, encoder_hidden_states)
        
        # Final layer
        x = self.final_layer(x, c)  # [B, 256, 1024]
        
        if return_dict:
            return {"velocity_prediction": x}
        return x

    def generate_global_features(self, patch_features):
        """
        Generate global features from patch features using attention pooling.
        
        Args:
            patch_features: Patch features [B, 256, 1024]
            
        Returns:
            Global features [B, 768] or [B, 1024]
        """
        # Attention pooling
        global_features = self.global_pool(patch_features)  # [B, 1024]
        
        # Apply CLIP projection if available
        if self.clip_projection is not None:
            global_features = self.clip_projection(global_features)  # [B, 768]
        
        return global_features

    @torch.no_grad()
    def generate(
        self,
        eva_features,  # [B, 256, 4096]
        num_inference_steps=50,
        class_labels=None,
        generator=None,
        return_global=False,
    ):
        """
        Generate CLIP embeddings using flow matching sampling.
        
        Args:
            eva_features: EVA-CLIP conditioning [B, 256, 4096]
            num_inference_steps: Number of sampling steps
            class_labels: Class labels [B]
            generator: Random number generator
            return_global: Whether to return global features
            
        Returns:
            Generated CLIP embeddings [B, 256, 1024] or [B, 768] if return_global
        """
        device = eva_features.device
        batch_size = eva_features.shape[0]
        
        # Start from noise
        x = torch.randn(
            batch_size, 256, self.clip_dim,
            device=device,
            generator=generator,
            dtype=eva_features.dtype
        )
        
        # Flow matching sampling
        dt = 1.0 / num_inference_steps
        
        for step in range(num_inference_steps):
            t = step * dt
            timestep = torch.full((batch_size,), t, device=device)
            
            # Predict velocity
            velocity = self.forward(
                hidden_states=x,
                timestep=timestep,
                encoder_hidden_states=eva_features,
                class_labels=class_labels,
                return_dict=False
            )
            
            # Euler step
            x = x + dt * velocity
        
        # Return global features if requested
        if return_global:
            return self.generate_global_features(x)
        
        return x

    def get_num_parameters(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_blip3o_dit_model(
    config=None,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    load_clip_projection=True,
    **kwargs
):
    """Create BLIP3-o DiT model aligned with the paper."""
    if config is None:
        # Create default config
        from ..config.blip3o_config import BLIP3oDiTConfig
        config = BLIP3oDiTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            **kwargs
        )
    
    model = BLIP3oDiTModel(config)
    
    if load_clip_projection:
        model.load_clip_projection()
    
    print(f"✅ BLIP3-o DiT model created successfully")
    print(f"   Parameters: {model.get_num_parameters():,}")
    print(f"   Architecture: Patch-level training with global pooling")
    
    return model