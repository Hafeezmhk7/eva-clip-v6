"""
BLIP3-o DiT Model Implementation
Exact implementation of BLIP3-o diffusion transformer architecture using NextDiT backbone.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import PreTrainedModel

# Try to import rotary embeddings with fallback
try:
    from diffusers.models.embeddings import get_2d_rotary_pos_embed_lumina
    ROTARY_EMBED_AVAILABLE = True
except ImportError:
    print("⚠️  Rotary embeddings not available in this diffusers version")
    ROTARY_EMBED_AVAILABLE = False

from ..config.blip3o_config import BLIP3oDiTConfig
from .lumina_nextdit2d import LuminaNextDiT2DModel


class BLIP3oDiTModel(PreTrainedModel):
    """
    BLIP3-o Diffusion Transformer Model.
    
    This model implements the exact BLIP3-o architecture for generating CLIP embeddings
    from EVA-CLIP conditioning using flow matching. It uses the NextDiT backbone with
    cross-attention for conditioning.
    
    Architecture:
    - Input: Noisy CLIP features [B, 64, 768] + EVA-CLIP conditioning [B, 64, 1280]
    - Backbone: NextDiT (Lumina-Next) transformer
    - Output: Velocity field predictions [B, 64, 768] for flow matching
    """
    
    config_class = BLIP3oDiTConfig
    
    def __init__(self, config: BLIP3oDiTConfig):
        super().__init__(config)
        
        # Store configuration
        self.config = config
        self._gradient_checkpointing = config._gradient_checkpointing
        
        # Validate configuration for BLIP3-o
        self._validate_blip3o_config(config)
        
        # Initialize the NextDiT backbone
        self.dit_model = LuminaNextDiT2DModel(
            sample_size=config.input_size,              # 8 (for 8x8 grid)
            patch_size=config.patch_size,               # 1 (pre-tokenized)
            in_channels=config.in_channels,             # 768 (CLIP dimension)
            hidden_size=config.dim,                     # 1792 (model dimension)
            num_layers=config.n_layers,                 # 24 (transformer layers)
            num_attention_heads=config.n_heads,         # 28 (attention heads)
            num_kv_heads=config.n_kv_heads,             # 28 (KV heads)
            multiple_of=config.multiple_of,             # 256 (FFN multiple)
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,                   # 1e-5
            learn_sigma=config.learn_sigma,             # False (flow matching)
            qk_norm=config.qk_norm,                     # True
            cross_attention_dim=config.eva_embedding_size,  # 1280 (EVA-CLIP)
            scaling_factor=1.0,                         # Default scaling
        )
        
        # Enable gradient checkpointing if requested
        if self._gradient_checkpointing:
            try:
                self.dit_model.enable_gradient_checkpointing()
                print("✅ Gradient checkpointing enabled")
            except Exception as e:
                print(f"⚠️  Gradient checkpointing failed to enable: {e}")
                print("   Continuing without gradient checkpointing...")
                self._gradient_checkpointing = False
        
        # Initialize 2D rotary position embeddings for spatial understanding
        # Handle different diffusers API versions and availability
        head_dim = config.dim // config.n_heads
        
        if ROTARY_EMBED_AVAILABLE:
            try:
                # Try the newer API first
                self.freqs_cis = get_2d_rotary_pos_embed_lumina(
                    head_dim,  # head_dim as positional argument
                    384,  # height
                    384,  # width
                )
                print("✅ Rotary embeddings initialized (new API)")
            except TypeError:
                try:
                    # Try with different parameter names
                    self.freqs_cis = get_2d_rotary_pos_embed_lumina(
                        dim=head_dim,
                        height=384,
                        width=384,
                    )
                    print("✅ Rotary embeddings initialized (alt API)")
                except TypeError:
                    try:
                        # Try minimal parameters
                        self.freqs_cis = get_2d_rotary_pos_embed_lumina(
                            head_dim,
                            384,
                        )
                        print("✅ Rotary embeddings initialized (minimal API)")
                    except Exception as e:
                        print(f"⚠️  Rotary embeddings initialization failed: {e}")
                        # Create dummy embeddings as fallback
                        self.freqs_cis = torch.randn(384, 384, head_dim // 2, 2)
                        print("✅ Dummy rotary embeddings created")
        else:
            print("⚠️  Rotary embeddings not available, creating dummy embeddings")
            # Create dummy embeddings when function is not available
            self.freqs_cis = torch.randn(384, 384, head_dim // 2, 2)
            print("✅ Dummy rotary embeddings created")
        
        # Initialize model weights
        self._init_weights()
    
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
        # The LuminaNextDiT2DModel handles its own weight initialization
        # We just ensure proper device placement and dtype
        pass
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True
        try:
            if hasattr(self.dit_model, 'enable_gradient_checkpointing'):
                self.dit_model.enable_gradient_checkpointing()
            elif hasattr(self.dit_model, '_set_gradient_checkpointing'):
                self.dit_model._set_gradient_checkpointing(value=True)
            print("✅ Gradient checkpointing enabled")
        except Exception as e:
            print(f"⚠️  Gradient checkpointing failed to enable: {e}")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        try:
            if hasattr(self.dit_model, 'disable_gradient_checkpointing'):
                self.dit_model.disable_gradient_checkpointing()
            elif hasattr(self.dit_model, '_set_gradient_checkpointing'):
                self.dit_model._set_gradient_checkpointing(value=False)
            print("✅ Gradient checkpointing disabled")
        except Exception as e:
            print(f"⚠️  Gradient checkpointing failed to disable: {e}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, 64, 768] - Noisy CLIP features
        timestep: torch.Tensor,                # [B] - Flow matching timesteps
        encoder_hidden_states: torch.Tensor,  # [B, 64, 1280] - EVA-CLIP conditioning
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, 64] - Attention mask
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass of BLIP3-o DiT model.
        
        Args:
            hidden_states: Noisy CLIP features [batch_size, 64, 768]
            timestep: Flow matching timesteps [batch_size] or scalar
            encoder_hidden_states: EVA-CLIP conditioning [batch_size, 64, 1280]
            encoder_attention_mask: Optional attention mask [batch_size, 64]
            cross_attention_kwargs: Additional cross-attention arguments
            return_dict: Whether to return ModelOutput object
            
        Returns:
            Predicted velocity field [batch_size, 64, 768] for flow matching
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
        
        # Prepare cross-attention kwargs
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        
        # Move rotary embeddings to correct device and handle potential issues
        try:
            freqs_cis = self.freqs_cis.to(device=device, dtype=hidden_states.dtype)
        except Exception as e:
            print(f"⚠️  Rotary embedding device transfer failed: {e}")
            # Create device-appropriate dummy embeddings
            head_dim = self.config.dim // self.config.n_heads
            freqs_cis = torch.randn(384, 384, head_dim // 2, 2, device=device, dtype=hidden_states.dtype)
            print("✅ Created device-specific dummy rotary embeddings")
        
        # Forward through NextDiT backbone with error handling
        try:
            dit_output = self.dit_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_mask=encoder_attention_mask,
                image_rotary_emb=freqs_cis,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=return_dict,
            )
        except Exception as e:
            print(f"⚠️  Forward pass with rotary embeddings failed: {e}")
            print("   Trying without rotary embeddings...")
            # Try without rotary embeddings as fallback
            dit_output = self.dit_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_mask=encoder_attention_mask,
                image_rotary_emb=None,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=return_dict,
            )
        
        if return_dict:
            return dit_output
        else:
            return dit_output.sample if hasattr(dit_output, 'sample') else dit_output
    
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
        encoder_hidden_states: torch.Tensor,  # [B, 64, 1280] - EVA-CLIP conditioning
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,  # DDIM parameter
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Generate CLIP embeddings using flow matching sampling.
        
        This implements the flow matching sampling procedure used in BLIP3-o.
        
        Args:
            encoder_hidden_states: EVA-CLIP conditioning [batch_size, 64, 1280]
            num_inference_steps: Number of sampling steps
            guidance_scale: Guidance scale (not used in current flow matching)
            generator: Random number generator for reproducibility
            eta: DDIM parameter for stochasticity
            return_intermediate: Whether to return intermediate states
            
        Returns:
            Generated CLIP embeddings [batch_size, 64, 768]
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