"""
UPDATED BLIP3-o Configuration with Dual Supervision Support - COMPLETE VERSION
Place this file at: src/modules/config/blip3o_config.py

Includes:
1. Original BLIP3-o DiT configuration
2. NEW: MLP configuration for dual supervision
3. Flow matching configuration
4. NEW: Training configuration
5. Factory functions and defaults
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import math


@dataclass
class BLIP3oDiTConfig:
    """
    Configuration class for BLIP3-o Diffusion Transformer with Dual Supervision support.
    
    This configuration supports both standard and dual supervision training modes.
    """
    
    # ========================
    # Core Model Architecture
    # ========================
    
    # Input configuration
    input_size: int = 16              # Grid size (16x16 = 256 tokens)
    patch_size: int = 1               # Patch size (pre-tokenized, so 1)
    in_channels: int = 1024           # CLIP embedding dimension
    
    # Model dimensions
    dim: int = 768                    # Hidden dimension
    n_layers: int = 16                # Number of transformer layers
    n_heads: int = 12                 # Number of attention heads
    n_kv_heads: Optional[int] = None  # Number of key-value heads (defaults to n_heads)
    
    # Attention configuration
    qk_norm: bool = True              # Query-key normalization
    norm_eps: float = 1e-5            # Layer norm epsilon
    
    # Cross-attention configuration
    eva_embedding_size: int = 4096    # EVA-CLIP conditioning dimension
    
    # ========================
    # NEW: Dual Supervision MLP Configuration
    # ========================
    
    # Global adaptation MLP parameters
    mlp_hidden_dim: int = 2048        # Hidden dimension for adaptation MLP
    mlp_num_layers: int = 3           # Number of layers in adaptation MLP  
    mlp_dropout: float = 0.1          # Dropout rate for adaptation MLP
    mlp_activation: str = "gelu"      # Activation function for MLP
    
    # ========================
    # Training Configuration
    # ========================
    
    # Diffusion configuration
    learn_sigma: bool = False         # Whether to learn noise variance (False for flow matching)
    
    # Memory optimization
    _gradient_checkpointing: bool = True    # Enable gradient checkpointing
    
    # RoPE configuration  
    rope_base: float = 10000.0        # RoPE base frequency
    rope_scaling: Optional[Dict[str, Any]] = None  # RoPE scaling configuration
    
    # Initialization
    initializer_range: float = 0.02   # Standard deviation for weight initialization
    
    # ========================
    # Advanced Configuration
    # ========================
    
    # Attention optimizations
    use_flash_attention: bool = False     # Use Flash Attention (if available)
    attention_dropout: float = 0.0       # Attention dropout rate
    
    # Feed-forward configuration
    intermediate_size: Optional[int] = None  # FFN intermediate size (defaults to 4 * dim)
    hidden_dropout: float = 0.0          # Hidden layer dropout rate
    
    # Position encoding
    max_position_embeddings: int = 1024  # Maximum sequence length
    
    # Model type identification
    model_type: str = "blip3o_dit"
    
    # ========================
    # Compatibility and Validation
    # ========================
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        
        # Set default n_kv_heads if not specified
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        
        # Set default intermediate_size if not specified
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.dim
        
        # Validate dimensions
        self._validate_dimensions()
        
        # Adjust for RoPE compatibility if needed
        self._adjust_rope_compatibility()
    
    def _validate_dimensions(self):
        """Validate model dimensions and configuration."""
        
        # Basic dimension checks
        assert self.dim > 0, "Model dimension must be positive"
        assert self.n_layers > 0, "Number of layers must be positive"
        assert self.n_heads > 0, "Number of heads must be positive"
        assert self.n_kv_heads > 0, "Number of key-value heads must be positive"
        
        # Check head dimension compatibility
        assert self.dim % self.n_heads == 0, f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        
        # Input validation
        assert self.input_size > 0, "Input size must be positive"
        assert self.patch_size > 0, "Patch size must be positive"
        assert self.in_channels > 0, "Input channels must be positive"
        assert self.eva_embedding_size > 0, "EVA embedding size must be positive"
        
        # MLP validation
        assert self.mlp_hidden_dim > 0, "MLP hidden dimension must be positive"
        assert self.mlp_num_layers > 0, "MLP number of layers must be positive"
        assert 0.0 <= self.mlp_dropout <= 1.0, "MLP dropout must be between 0 and 1"
        assert self.mlp_activation in ["gelu", "relu", "silu"], f"Unknown activation: {self.mlp_activation}"
        
        # Flow matching validation
        assert self.learn_sigma is False, "BLIP3-o uses flow matching, learn_sigma must be False"
        
        # BLIP3-o specific validation
        assert self.in_channels == 1024, "CLIP embedding dimension must be 1024"
        assert self.eva_embedding_size == 4096, "EVA-CLIP dimension must be 4096"
        assert self.patch_size == 1, "Features are pre-tokenized, patch_size must be 1"
        assert self.input_size == 16, "Input size must be 16 for 256 tokens (16×16)"
    
    def _adjust_rope_compatibility(self):
        """Adjust dimensions for 3D RoPE compatibility."""
        head_dim = self.dim // self.n_heads
        
        # RoPE requires head_dim to be divisible by 4 for 3D
        if head_dim % 4 != 0:
            # Find compatible configuration
            for candidate_heads in range(1, self.dim + 1):
                if self.dim % candidate_heads == 0:
                    candidate_head_dim = self.dim // candidate_heads
                    if candidate_head_dim % 4 == 0:
                        print(f"⚠️ Adjusting n_heads from {self.n_heads} to {candidate_heads} for RoPE compatibility")
                        self.n_heads = candidate_heads
                        self.n_kv_heads = candidate_heads
                        head_dim = candidate_head_dim
                        break
            
            # If still not compatible, use safe defaults
            if head_dim % 4 != 0:
                print(f"⚠️ Using safe defaults for RoPE compatibility")
                self.dim = 512
                self.n_heads = 8
                self.n_kv_heads = 8
        
        final_head_dim = self.dim // self.n_heads
        assert final_head_dim % 4 == 0, f"Head dimension {final_head_dim} must be divisible by 4 for 3D RoPE"
    
    def get_num_tokens(self) -> int:
        """Get total number of tokens (patches)."""
        return self.input_size * self.input_size
    
    def get_head_dim(self) -> int:
        """Get attention head dimension."""
        return self.dim // self.n_heads
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            # Core architecture
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "qk_norm": self.qk_norm,
            "norm_eps": self.norm_eps,
            "eva_embedding_size": self.eva_embedding_size,
            
            # Dual supervision MLP
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "mlp_num_layers": self.mlp_num_layers,
            "mlp_dropout": self.mlp_dropout,
            "mlp_activation": self.mlp_activation,
            
            # Training
            "learn_sigma": self.learn_sigma,
            "_gradient_checkpointing": self._gradient_checkpointing,
            "rope_base": self.rope_base,
            "rope_scaling": self.rope_scaling,
            "initializer_range": self.initializer_range,
            
            # Advanced
            "use_flash_attention": self.use_flash_attention,
            "attention_dropout": self.attention_dropout,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout": self.hidden_dropout,
            "max_position_embeddings": self.max_position_embeddings,
            "model_type": self.model_type,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BLIP3oDiTConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class FlowMatchingConfig:
    """
    Configuration for flow matching loss computation.
    """
    
    # Flow matching parameters
    sigma_min: float = 1e-4           # Minimum noise level
    sigma_max: float = 1.0            # Maximum noise level
    prediction_type: str = "v_prediction"  # Prediction type ("v_prediction" or "epsilon")
    schedule_type: str = "linear"     # Noise schedule ("linear", "cosine", "sigmoid")
    
    # Embedding dimensions
    clip_dim: int = 1024             # CLIP embedding dimension
    eva_dim: int = 4096              # EVA-CLIP embedding dimension
    
    # Loss weights
    regularization_weight: float = 0.01  # Regularization loss weight
    
    # Enhanced loss parameters (for dual supervision)
    alignment_loss_weight: float = 0.1    # Alignment loss weight
    temporal_loss_weight: float = 0.05    # Temporal consistency weight
    
    # Progressive training
    use_progressive_training: bool = True  # Enable progressive timestep training
    min_timestep: float = 0.001       # Minimum timestep for progressive training
    max_timestep: float = 0.999       # Maximum timestep for progressive training
    
    def __post_init__(self):
        """Validate flow matching configuration."""
        assert self.prediction_type in ["v_prediction", "epsilon"], f"Invalid prediction type: {self.prediction_type}"
        assert self.schedule_type in ["linear", "cosine", "sigmoid"], f"Invalid schedule type: {self.schedule_type}"
        assert 0 <= self.sigma_min < self.sigma_max <= 10.0, f"Invalid sigma range: [{self.sigma_min}, {self.sigma_max}]"
        assert self.clip_dim > 0, "CLIP dimension must be positive"
        assert self.eva_dim > 0, "EVA dimension must be positive"
        assert 0 <= self.min_timestep < self.max_timestep <= 1.0, "Invalid timestep range"


@dataclass
class TrainingConfig:
    """
    Configuration for training parameters.
    """
    
    # Basic training parameters
    num_train_epochs: int = 8
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    warmup_steps: int = 100
    
    # Training optimization
    gradient_accumulation_steps: int = 6
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    
    # Logging and evaluation
    logging_steps: int = 50
    eval_steps: int = 250
    save_steps: int = 500
    
    # Memory optimization
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    
    # Model selection
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_global_cosine_mean"
    greater_is_better: bool = True
    
    # Dual supervision specific
    patch_loss_weight: float = 1.0
    global_loss_weight: float = 2.0
    flow_matching_loss_weight: float = 1.0
    use_cosine_similarity: bool = False
    
    def __post_init__(self):
        """Validate training configuration."""
        assert self.num_train_epochs > 0, "Number of epochs must be positive"
        assert self.per_device_train_batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 <= self.weight_decay <= 1.0, "Weight decay must be between 0 and 1"
        assert self.gradient_accumulation_steps > 0, "Gradient accumulation steps must be positive"


# ========================
# Factory Functions
# ========================

def get_default_blip3o_config() -> BLIP3oDiTConfig:
    """Get default BLIP3-o configuration."""
    return BLIP3oDiTConfig()


def get_small_blip3o_config() -> BLIP3oDiTConfig:
    """Get small BLIP3-o configuration for testing."""
    return BLIP3oDiTConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        mlp_hidden_dim=1024,
        mlp_num_layers=2,
    )


def get_large_blip3o_config() -> BLIP3oDiTConfig:
    """Get large BLIP3-o configuration for production."""
    return BLIP3oDiTConfig(
        dim=1024,
        n_layers=24,
        n_heads=16,
        mlp_hidden_dim=4096,
        mlp_num_layers=4,
    )


def get_dual_supervision_config(
    base_config: Optional[BLIP3oDiTConfig] = None,
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get configuration optimized for dual supervision training.
    
    Args:
        base_config: Base configuration to modify
        **kwargs: Additional parameters to override
        
    Returns:
        Configuration optimized for dual supervision
    """
    if base_config is None:
        base_config = get_default_blip3o_config()
    
    # Dual supervision optimizations
    dual_supervision_params = {
        "mlp_hidden_dim": 2048,      # Larger MLP for better adaptation
        "mlp_num_layers": 3,         # Deeper MLP for complex mappings
        "mlp_dropout": 0.1,          # Moderate dropout for regularization
        "_gradient_checkpointing": True,  # Memory optimization
        **kwargs  # Allow custom overrides
    }
    
    # Create new config with dual supervision parameters
    config_dict = base_config.to_dict()
    config_dict.update(dual_supervision_params)
    
    return BLIP3oDiTConfig.from_dict(config_dict)


def get_default_flow_matching_config() -> FlowMatchingConfig:
    """Get default flow matching configuration."""
    return FlowMatchingConfig()


def get_enhanced_flow_matching_config() -> FlowMatchingConfig:
    """Get enhanced flow matching configuration with alignment optimization."""
    return FlowMatchingConfig(
        alignment_loss_weight=0.1,
        temporal_loss_weight=0.05,
        use_progressive_training=True,
        schedule_type="cosine",  # Smoother transitions
    )


def get_dual_supervision_flow_matching_config() -> FlowMatchingConfig:
    """Get flow matching configuration optimized for dual supervision."""
    return FlowMatchingConfig(
        # Standard flow matching
        sigma_min=1e-4,
        sigma_max=1.0,
        prediction_type="v_prediction",
        schedule_type="linear",
        
        # Dual supervision optimizations
        alignment_loss_weight=0.15,     # Higher alignment emphasis
        temporal_loss_weight=0.05,
        use_progressive_training=True,
        
        # Regularization
        regularization_weight=0.01,
    )


def get_default_training_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_dual_supervision_training_config() -> TrainingConfig:
    """Get training configuration optimized for dual supervision."""
    return TrainingConfig(
        learning_rate=5e-5,  # Lower for dual supervision stability
        gradient_accumulation_steps=6,
        global_loss_weight=2.0,  # Higher weight for global alignment
        patch_loss_weight=1.0,
        flow_matching_loss_weight=1.0,
        metric_for_best_model="eval_global_cosine_mean",
    )


# ========================
# Configuration Validation
# ========================

def validate_config_compatibility(
    model_config: BLIP3oDiTConfig,
    flow_config: FlowMatchingConfig
) -> bool:
    """
    Validate compatibility between model and flow matching configurations.
    
    Args:
        model_config: Model configuration
        flow_config: Flow matching configuration
        
    Returns:
        True if compatible, raises assertion error if not
    """
    # Check dimension compatibility
    assert model_config.in_channels == flow_config.clip_dim, \
        f"Model input channels ({model_config.in_channels}) must match flow CLIP dim ({flow_config.clip_dim})"
    
    assert model_config.eva_embedding_size == flow_config.eva_dim, \
        f"Model EVA size ({model_config.eva_embedding_size}) must match flow EVA dim ({flow_config.eva_dim})"
    
    # Check training compatibility
    assert model_config.learn_sigma is False, \
        "Model must use flow matching (learn_sigma=False)"
    
    print("✅ Configuration compatibility validated")
    return True


# ========================
# Preset Configurations
# ========================

# Standard configurations
BLIP3O_SMALL = get_small_blip3o_config()
BLIP3O_DEFAULT = get_default_blip3o_config()
BLIP3O_LARGE = get_large_blip3o_config()

# Dual supervision configurations
BLIP3O_DUAL_SUPERVISION = get_dual_supervision_config()
FLOW_MATCHING_DUAL_SUPERVISION = get_dual_supervision_flow_matching_config()

# Enhanced configurations
FLOW_MATCHING_ENHANCED = get_enhanced_flow_matching_config()


def get_preset_config(preset_name: str) -> BLIP3oDiTConfig:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: Name of the preset configuration
        
    Returns:
        BLIP3oDiTConfig instance
    """
    presets = {
        "small": BLIP3O_SMALL,
        "default": BLIP3O_DEFAULT, 
        "large": BLIP3O_LARGE,
        "dual_supervision": BLIP3O_DUAL_SUPERVISION,
    }
    
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return presets[preset_name]


# ========================
# Export Summary
# ========================

__all__ = [
    # Main configuration classes
    "BLIP3oDiTConfig",
    "FlowMatchingConfig",
    "TrainingConfig",
    
    # Factory functions
    "get_default_blip3o_config",
    "get_small_blip3o_config", 
    "get_large_blip3o_config",
    "get_dual_supervision_config",
    "get_default_flow_matching_config",
    "get_enhanced_flow_matching_config",
    "get_dual_supervision_flow_matching_config",
    "get_default_training_config",
    "get_dual_supervision_training_config",
    
    # Utilities
    "validate_config_compatibility",
    "get_preset_config",
    
    # Presets
    "BLIP3O_SMALL",
    "BLIP3O_DEFAULT",
    "BLIP3O_LARGE",
    "BLIP3O_DUAL_SUPERVISION",
    "FLOW_MATCHING_DUAL_SUPERVISION",
    "FLOW_MATCHING_ENHANCED",
]


if __name__ == "__main__":
    # Test configuration creation and validation
    print("🧪 Testing BLIP3-o configurations...")
    
    # Test default configuration
    default_config = get_default_blip3o_config()
    print(f"✅ Default config: {default_config.dim}D, {default_config.n_layers}L, {default_config.n_heads}H")
    
    # Test dual supervision configuration
    dual_config = get_dual_supervision_config()
    print(f"✅ Dual supervision config: MLP {dual_config.mlp_hidden_dim}D, {dual_config.mlp_num_layers}L")
    
    # Test flow matching configuration
    flow_config = get_dual_supervision_flow_matching_config()
    print(f"✅ Flow matching config: {flow_config.prediction_type}, {flow_config.schedule_type}")
    
    # Test training configuration
    training_config = get_default_training_config()
    print(f"✅ Training config: {training_config.learning_rate} LR, {training_config.num_train_epochs} epochs")
    
    # Test compatibility
    validate_config_compatibility(dual_config, flow_config)
    
    print("🎉 All configuration tests passed!")