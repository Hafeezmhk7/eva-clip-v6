"""
BLIP3-o Configuration - Updated for Patch-Level Training
src/modules/config/blip3o_config.py

Configuration classes for BLIP3-o patch-level DiT model following the paper architecture.
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional


class BLIP3oDiTConfig(PretrainedConfig):
    """
    Configuration class for BLIP3-o patch-level DiT model.
    
    This configuration follows the BLIP3-o paper architecture with:
    - Patch-level training on 256 CLIP tokens (1024-dim)
    - EVA-CLIP conditioning (256 tokens, 4096-dim)
    - Flow matching training objective
    - Image-to-text recall evaluation
    """
    
    model_type = "blip3o_patch_dit"
    
    def __init__(
        self,
        # Model architecture
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        
        # Input/output dimensions (BLIP3-o specific)
        eva_embedding_size: int = 4096,  # EVA-CLIP dimension
        clip_embedding_size: int = 1024,  # CLIP patch dimension
        num_patches: int = 256,  # 16x16 = 256 patches
        
        # Training configuration
        max_position_embeddings: int = 256,  # 16x16 patches
        dropout_prob: float = 0.1,
        
        # Flow matching parameters
        prediction_type: str = "velocity",
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        
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
        self.num_patches = num_patches
        
        # Training configuration
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        
        # Flow matching
        self.prediction_type = prediction_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
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
        
        # Check number of patches
        if self.num_patches != self.max_position_embeddings:
            raise ValueError(
                f"num_patches ({self.num_patches}) must equal "
                f"max_position_embeddings ({self.max_position_embeddings})"
            )
        
        # Check prediction type
        if self.prediction_type not in ["velocity", "epsilon"]:
            raise ValueError(f"prediction_type must be 'velocity' or 'epsilon', got {self.prediction_type}")
        
        # Check patch count (should be 256 for 16x16)
        if self.num_patches != 256:
            raise ValueError(f"Expected 256 patches (16x16), got {self.num_patches}")
    
    def get_head_dim(self):
        """Get attention head dimension."""
        return self.hidden_size // self.num_attention_heads
    
    def get_num_patches(self):
        """Get number of patches."""
        return self.num_patches
    
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        return output


def get_blip3o_patch_config(
    model_size: str = "base",
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get predefined BLIP3-o patch-level configuration.
    
    Args:
        model_size: Model size - "tiny", "small", "base", "large", "xl"
        **kwargs: Additional configuration overrides
        
    Returns:
        BLIP3oDiTConfig instance
    """
    # Predefined configurations optimized for patch-level training
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
    """Get default BLIP3-o configuration optimized for patch-level training."""
    return get_blip3o_patch_config(
        model_size="base",
        prediction_type="velocity",
        **kwargs
    )


def get_memory_optimized_config(
    num_gpus: int = 1,
    gpu_memory_gb: float = 40.0,
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get memory-optimized BLIP3-o configuration
    
    Args:
        num_gpus: Number of GPUs
        gpu_memory_gb: Memory per GPU in GB
        **kwargs: Additional overrides
        
    Returns:
        Memory-optimized configuration
    """
    # Choose model size based on available memory
    if gpu_memory_gb >= 40 and num_gpus >= 4:
        model_size = "large"
    elif gpu_memory_gb >= 24 and num_gpus >= 2:
        model_size = "base"
    elif gpu_memory_gb >= 16:
        model_size = "small"
    else:
        model_size = "tiny"
    
    return get_blip3o_patch_config(
        model_size=model_size,
        use_gradient_checkpointing=True,
        use_fp16=True,
        **kwargs
    )


def get_recall_optimized_config(
    model_size: str = "base",
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get BLIP3-o configuration optimized for image-to-text recall
    
    Args:
        model_size: Model size
        **kwargs: Additional overrides
        
    Returns:
        Configuration optimized for recall performance
    """
    return get_blip3o_patch_config(
        model_size=model_size,
        prediction_type="velocity",  # Better for recall
        dropout_prob=0.05,  # Lower dropout for better recall
        **kwargs
    )


def create_config_from_pretrained(
    model_path: str,
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Create configuration from pretrained model
    
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
    Validate configuration for patch-level training and return recommendations
    
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
        config.hidden_size * config.intermediate_size * config.num_hidden_layers * 2 +  # FFN layers
        config.eva_embedding_size * config.hidden_size +  # EVA projection
        config.hidden_size * config.clip_embedding_size  # Output projection
    ) / 1e6  # Convert to millions
    
    if param_estimate > 1000:  # > 1B parameters
        results["warnings"].append(f"Large model (~{param_estimate:.0f}M parameters)")
        results["recommendations"].append("Consider using gradient checkpointing and fp16")
        results["optimizations"].append("Enable use_gradient_checkpointing=True")
    
    # Check patch configuration
    if config.num_patches != 256:
        results["warnings"].append(f"Non-standard patch count: {config.num_patches} (expected 256)")
        results["recommendations"].append("Use 256 patches (16x16) for optimal BLIP3-o performance")
    
    # Check dimensions for BLIP3-o compatibility
    if config.eva_embedding_size != 4096:
        results["warnings"].append(f"Non-standard EVA dimension: {config.eva_embedding_size} (expected 4096)")
    
    if config.clip_embedding_size != 1024:
        results["warnings"].append(f"Non-standard CLIP dimension: {config.clip_embedding_size} (expected 1024)")
    
    return results


# Export commonly used configurations
TINY_CONFIG = get_blip3o_patch_config("tiny")
SMALL_CONFIG = get_blip3o_patch_config("small") 
BASE_CONFIG = get_blip3o_patch_config("base")
LARGE_CONFIG = get_blip3o_patch_config("large")

# Specialized configurations
RECALL_OPTIMIZED_CONFIG = get_recall_optimized_config("base")
MEMORY_OPTIMIZED_CONFIG = get_memory_optimized_config()