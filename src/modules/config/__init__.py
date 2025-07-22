"""
Configuration module for BLIP3-o DiT - Enhanced with Multi-GPU Support

Contains configuration classes for:
- Model architecture (BLIP3oDiTConfig)
- Flow matching loss (FlowMatchingConfig)  
- Training parameters (TrainingConfig)
- Memory-optimized configurations
- Multi-GPU specific configurations
"""

import logging

logger = logging.getLogger(__name__)

# Core configuration classes
try:
    from .blip3o_config import (
        BLIP3oDiTConfig,
        FlowMatchingConfig,
        TrainingConfig,
        get_default_blip3o_config,
        get_global_blip3o_config,
        get_multi_gpu_config,
        get_default_flow_matching_config,
        get_enhanced_flow_matching_config,
        get_default_training_config,
        validate_config_compatibility,
        print_config_summary,
    )
    logger.debug("✅ Core configuration classes loaded")
    CORE_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load core configuration classes: {e}")
    CORE_CONFIG_AVAILABLE = False
    # Set all to None
    BLIP3oDiTConfig = None
    FlowMatchingConfig = None
    TrainingConfig = None
    get_default_blip3o_config = None
    get_global_blip3o_config = None
    get_multi_gpu_config = None
    get_default_flow_matching_config = None
    get_enhanced_flow_matching_config = None
    get_default_training_config = None
    validate_config_compatibility = None
    print_config_summary = None

# Memory-optimized configurations  
MEMORY_OPTIMIZED_CONFIG_AVAILABLE = False
try:
    from .memory_optimized_config import (
        get_memory_optimized_model_configs,
        get_memory_optimized_training_args,
        estimate_memory_usage,
        recommend_configuration,
        print_memory_recommendations,
    )
    logger.debug("✅ Memory-optimized configurations loaded")
    MEMORY_OPTIMIZED_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Memory-optimized configurations not available: {e}")
    get_memory_optimized_model_configs = None
    get_memory_optimized_training_args = None
    estimate_memory_usage = None
    recommend_configuration = None
    print_memory_recommendations = None

# Build exports list
__all__ = [
    # Availability flags
    "CORE_CONFIG_AVAILABLE",
    "MEMORY_OPTIMIZED_CONFIG_AVAILABLE",
]

# Export core configuration if available
if CORE_CONFIG_AVAILABLE:
    __all__.extend([
        "BLIP3oDiTConfig",
        "FlowMatchingConfig", 
        "TrainingConfig",
        "get_default_blip3o_config",
        "get_global_blip3o_config",
        "get_multi_gpu_config",
        "get_default_flow_matching_config",
        "get_enhanced_flow_matching_config",
        "get_default_training_config",
        "validate_config_compatibility",
        "print_config_summary",
    ])

# Export memory-optimized configuration if available
if MEMORY_OPTIMIZED_CONFIG_AVAILABLE:
    __all__.extend([
        "get_memory_optimized_model_configs",
        "get_memory_optimized_training_args",
        "estimate_memory_usage",
        "recommend_configuration",
        "print_memory_recommendations",
    ])

def get_recommended_config(
    model_type: str = "auto",
    num_gpus: int = 1,
    gpu_memory_gb: float = 40.0,
    training_mode: str = "standard",
    **kwargs
):
    """
    Get recommended configuration based on hardware and requirements
    
    Args:
        model_type: "auto", "dual_supervision", "global", or "standard"
        num_gpus: Number of available GPUs
        gpu_memory_gb: Memory per GPU in GB
        training_mode: "standard", "memory_optimized", "fast"
        **kwargs: Additional config parameters
        
    Returns:
        Model configuration
    """
    if not CORE_CONFIG_AVAILABLE:
        raise RuntimeError("Core configuration not available")
    
    # Use memory-optimized recommendations if available
    if MEMORY_OPTIMIZED_CONFIG_AVAILABLE and training_mode == "memory_optimized":
        try:
            size_name, config, memory_info = recommend_configuration(
                available_gpu_memory_gb=gpu_memory_gb,
                num_gpus=num_gpus,
            )
            logger.info(f"Memory-optimized recommendation: {size_name}")
            return config
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}, falling back to standard")
    
    # Standard configuration selection
    if model_type == "auto":
        if num_gpus >= 4 and gpu_memory_gb >= 40:
            return get_global_blip3o_config("large", **kwargs)
        elif num_gpus >= 2 and gpu_memory_gb >= 24:
            return get_global_blip3o_config("medium", **kwargs)
        else:
            return get_global_blip3o_config("small", **kwargs)
    
    elif model_type == "global":
        if num_gpus >= 4:
            model_size = "large"
        elif num_gpus >= 2:
            model_size = "medium"
        else:
            model_size = "small"
        return get_global_blip3o_config(model_size, **kwargs)
    
    elif model_type == "multi_gpu":
        return get_multi_gpu_config(num_gpus, gpu_memory_gb, **kwargs)
    
    else:
        return get_default_blip3o_config(**kwargs)

def create_training_config(
    training_type: str = "standard",
    num_gpus: int = 1,
    **kwargs
):
    """
    Create training configuration
    
    Args:
        training_type: "standard", "enhanced", "memory_optimized"  
        num_gpus: Number of GPUs
        **kwargs: Additional training parameters
        
    Returns:
        Training configuration
    """
    if not CORE_CONFIG_AVAILABLE:
        raise RuntimeError("Core configuration not available")
    
    if training_type == "enhanced":
        return get_enhanced_flow_matching_config(**kwargs)
    elif training_type == "memory_optimized" and MEMORY_OPTIMIZED_CONFIG_AVAILABLE:
        # Use memory-optimized training args
        output_dir = kwargs.get('output_dir', './output')
        return get_memory_optimized_training_args(
            output_dir=output_dir,
            num_gpus=num_gpus,
            **kwargs
        )
    else:
        return get_default_training_config(**kwargs)

def validate_multi_gpu_config(
    model_config, 
    flow_config=None,
    training_config=None,
    num_gpus: int = 1
):
    """
    Validate configuration for multi-GPU training
    
    Args:
        model_config: Model configuration
        flow_config: Flow matching configuration (optional)
        training_config: Training configuration (optional) 
        num_gpus: Number of GPUs
        
    Returns:
        bool: True if configuration is valid
    """
    if not CORE_CONFIG_AVAILABLE:
        raise RuntimeError("Core configuration not available")
    
    try:
        # Basic model validation
        if hasattr(model_config, '_validate_config'):
            model_config._validate_config()
        
        # Multi-GPU specific checks
        head_dim = model_config.dim // model_config.n_heads
        if head_dim % 4 != 0:
            logger.error(f"Head dimension {head_dim} not compatible with 3D RoPE")
            return False
        
        # Flow matching compatibility
        if flow_config is not None:
            validate_config_compatibility(model_config, flow_config)
        
        # Memory estimation if available
        if MEMORY_OPTIMIZED_CONFIG_AVAILABLE:
            try:
                memory_info = estimate_memory_usage(model_config, batch_size=8)
                memory_per_gpu = memory_info['total_training_memory_gb']
                logger.info(f"Estimated memory per GPU: {memory_per_gpu:.1f} GB")
                
                if memory_per_gpu > 35:  # Conservative limit
                    logger.warning(f"High memory usage estimated: {memory_per_gpu:.1f} GB")
            except Exception as e:
                logger.warning(f"Memory estimation failed: {e}")
        
        logger.info("✅ Multi-GPU configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def print_config_status():
    """Print status of available configuration utilities"""
    print("⚙️ BLIP3-o Configuration Status")
    print("=" * 35)
    
    if CORE_CONFIG_AVAILABLE:
        print("  ✅ Core Configuration Classes")
        print("    - BLIP3oDiTConfig")
        print("    - FlowMatchingConfig") 
        print("    - TrainingConfig")
    else:
        print("  ❌ Core Configuration Classes")
    
    if MEMORY_OPTIMIZED_CONFIG_AVAILABLE:
        print("  ✅ Memory-Optimized Configurations")
        print("    - Memory usage estimation")
        print("    - Automatic configuration recommendation")
    else:
        print("  ❌ Memory-Optimized Configurations")
    
    print()
    print("Available factory functions:")
    if CORE_CONFIG_AVAILABLE:
        print("  ✅ get_recommended_config()")
        print("  ✅ create_training_config()")
        print("  ✅ validate_multi_gpu_config()")
    
    print("=" * 35)

# Add utility functions to exports
__all__.extend([
    "get_recommended_config",
    "create_training_config", 
    "validate_multi_gpu_config",
    "print_config_status",
])

# Log configuration module status
if CORE_CONFIG_AVAILABLE:
    logger.info("BLIP3-o configurations loaded successfully")
    if MEMORY_OPTIMIZED_CONFIG_AVAILABLE:
        logger.info("  ✅ Memory optimization features available")
else:
    logger.error("BLIP3-o configurations failed to load!")