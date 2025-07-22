"""
BLIP3-o Configuration - Aligned with Paper
src/modules/config/blip3o_config.py

Configuration classes for BLIP3-o DiT model following the paper architecture.
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional


class BLIP3oDiTConfig(PretrainedConfig):
    """
    Configuration class for BLIP3-o DiT model.
    
    This configuration follows the BLIP3-o paper architecture with:
    - DiT backbone for patch-level training
    - EVA-CLIP conditioning
    - Flow matching training objective
    - Attention pooling for global representation
    """
    
    model_type = "blip3o_dit"
    
    def __init__(
        self,
        # Model architecture
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        
        # Input/output dimensions
        eva_embedding_size: int = 4096,  # EVA-CLIP dimension
        clip_embedding_size: int = 1024,  # CLIP patch dimension
        input_size: int = 16,  # 16x16 = 256 patches
        
        # Training configuration
        max_position_embeddings: int = 256,  # 16x16 patches
        num_classes: int = 1000,  # For class conditioning
        dropout_prob: float = 0.1,
        
        # Flow matching parameters
        prediction_type: str = "velocity",
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        
        # Global supervision
        use_global_supervision: bool = True,
        global_weight: float = 0.1,
        use_attention_pooling: bool = True,
        
        # Training optimizations
        use_gradient_checkpointing: bool = False,
        use_fp16: bool = True,
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Core architecture
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        
        # Input/output dimensions
        self.eva_embedding_size = eva_embedding_size
        self.clip_embedding_size = clip_embedding_size
        self.input_size = input_size
        
        # Training configuration
        self.max_position_embeddings = max_position_embeddings
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        # Flow matching
        self.prediction_type = prediction_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Global supervision
        self.use_global_supervision = use_global_supervision
        self.global_weight = global_weight
        self.use_attention_pooling = use_attention_pooling
        
        # Training optimizations
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_fp16 = use_fp16
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check head dimension compatibility
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Check input size
        if self.input_size * self.input_size != self.max_position_embeddings:
            raise ValueError(
                f"input_size^2 ({self.input_size}^2) must equal "
                f"max_position_embeddings ({self.max_position_embeddings})"
            )
        
        # Check prediction type
        if self.prediction_type not in ["velocity", "epsilon"]:
            raise ValueError(f"prediction_type must be 'velocity' or 'epsilon', got {self.prediction_type}")
    
    def get_head_dim(self):
        """Get attention head dimension."""
        return self.hidden_size // self.num_attention_heads
    
    def get_num_patches(self):
        """Get number of patches."""
        return self.input_size * self.input_size
    
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        return output


def get_blip3o_config(
    model_size: str = "base",
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get predefined BLIP3-o configuration.
    
    Args:
        model_size: Model size - "tiny", "small", "base", "large"
        **kwargs: Additional configuration overrides
        
    Returns:
        BLIP3oDiTConfig instance
    """
    # Predefined configurations aligned with common DiT sizes
    configs = {
        "tiny": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
        },
        "small": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
        },
        "xl": {
            "hidden_size": 1152,
            "num_hidden_layers": 18,
            "num_attention_heads": 18,
            "intermediate_size": 4608,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(configs.keys())}")
    
    # Get base config
    config_dict = configs[model_size].copy()
    
    # Apply overrides
    config_dict.update(kwargs)
    
    return BLIP3oDiTConfig(**config_dict)


def get_default_blip3o_config(**kwargs) -> BLIP3oDiTConfig:
    """Get default BLIP3-o configuration optimized for training."""
    return get_blip3o_config(
        model_size="base",
        use_global_supervision=True,
        use_attention_pooling=True,
        prediction_type="velocity",
        **kwargs
    )


def get_global_blip3o_config(
    model_size: str = "base",
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get BLIP3-o configuration optimized for global training.
    
    Args:
        model_size: Model size
        **kwargs: Additional overrides
        
    Returns:
        Configuration optimized for global supervision
    """
    return get_blip3o_config(
        model_size=model_size,
        use_global_supervision=True,
        global_weight=0.2,  # Higher weight for global supervision
        use_attention_pooling=True,
        prediction_type="velocity",
        **kwargs
    )


def get_dual_supervision_config(
    model_size: str = "base",
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get BLIP3-o configuration for dual supervision training.
    
    Args:
        model_size: Model size
        **kwargs: Additional overrides
        
    Returns:
        Configuration optimized for dual supervision
    """
    return get_blip3o_config(
        model_size=model_size,
        use_global_supervision=True,
        global_weight=0.3,  # Strong global supervision
        use_attention_pooling=True,
        prediction_type="velocity",
        **kwargs
    )


def create_config_from_pretrained(
    model_path: str,
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Create configuration from pretrained model.
    
    Args:
        model_path: Path to pretrained model
        **kwargs: Configuration overrides
        
    Returns:
        BLIP3oDiTConfig instance
    """
    import json
    from pathlib import Path
    
    config_path = Path(model_path) / "config.json"
    
    if not config_path.exists():
        # Fallback to default config
        print(f"⚠️ Config file not found at {config_path}, using default")
        return get_default_blip3o_config(**kwargs)
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Apply overrides
    config_dict.update(kwargs)
    
    return BLIP3oDiTConfig(**config_dict)


def validate_training_config(config: BLIP3oDiTConfig) -> Dict[str, Any]:
    """
    Validate configuration for training and return recommendations.
    
    Args:
        config: BLIP3-o configuration
        
    Returns:
        Dictionary with validation results and recommendations
    """
    results = {
        "valid": True,
        "warnings": [],
        "recommendations": [],
        "optimizations": []
    }
    
    # Check head dimension for efficiency
    head_dim = config.get_head_dim()
    if head_dim % 8 != 0:
        results["warnings"].append(f"Head dimension {head_dim} not optimal for GPU efficiency")
        results["recommendations"].append("Consider adjusting hidden_size for head_dim divisible by 8")
    
    # Check model size for memory
    param_estimate = (
        config.hidden_size * config.hidden_size * config.num_hidden_layers * 12 +  # Attention layers
        config.hidden_size * config.intermediate_size * config.num_hidden_layers * 2  # FFN layers
    ) / 1e6  # Convert to millions
    
    if param_estimate > 1000:  # > 1B parameters
        results["warnings"].append(f"Large model (~{param_estimate:.0f}M parameters)")
        results["recommendations"].append("Consider using gradient checkpointing and fp16")
        results["optimizations"].append("Enable use_gradient_checkpointing=True")
    
    # Check global supervision settings
    if config.use_global_supervision and config.global_weight == 0:
        results["warnings"].append("Global supervision enabled but weight is 0")
        results["recommendations"].append("Set global_weight > 0 for effective global supervision")
    
    return results


# Export commonly used configurations
TINY_CONFIG = get_blip3o_config("tiny")
SMALL_CONFIG = get_blip3o_config("small") 
BASE_CONFIG = get_blip3o_config("base")
LARGE_CONFIG = get_blip3o_config("large")

# Specialized configurations
GLOBAL_CONFIG = get_global_blip3o_config("base")
DUAL_SUPERVISION_CONFIG = get_dual_supervision_config("base")