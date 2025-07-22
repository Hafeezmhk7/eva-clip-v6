"""
Configuration module for BLIP3-o DiT - Patch-Level Training (Paper-Aligned)

Contains configuration classes for:
- Model architecture (BLIP3oDiTConfig)
- Flow matching loss (FlowMatchingConfig)  
- Training parameters (TrainingConfig)
- Memory-optimized configurations
"""

import logging

logger = logging.getLogger(__name__)

# Core configuration classes
try:
    from .blip3o_config import (
        BLIP3oDiTConfig,
        FlowMatchingConfig,
        TrainingConfig,
        get_blip3o_patch_config,
        get_default_blip3o_config,
        get_default_flow_matching_config,
        get_enhanced_flow_matching_config,
        get_default_training_config,
        validate_config_compatibility,
        print_config_summary,
        # Predefined configs
        TINY_CONFIG,
        SMALL_CONFIG,
        BASE_CONFIG,
        LARGE_CONFIG,
        RECALL_OPTIMIZED_CONFIG,
        MEMORY_OPTIMIZED_CONFIG,
        DEFAULT_FLOW_MATCHING_CONFIG,
        DEFAULT_TRAINING_CONFIG,
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
    get_blip3o_patch_config = None
    get_default_blip3o_config = None
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
        "get_blip3o_patch_config",
        "get_default_blip3o_config",
        "get_default_flow_matching_config",
        "get_enhanced_flow_matching_config",
        "get_default_training_config",
        "validate_config_compatibility",
        "print_config_summary",
        # Predefined configs
        "TINY_CONFIG",
        "SMALL_CONFIG", 
        "BASE_CONFIG",
        "LARGE_CONFIG",
        "RECALL_OPTIMIZED_CONFIG",
        "MEMORY_OPTIMIZED_CONFIG",
        "DEFAULT_FLOW_MATCHING_CONFIG",
        "DEFAULT_TRAINING_CONFIG",
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
    model_size: str = "base", 
    **kwargs
):
    """
    Get recommended configuration based on requirements
    
    Args:
        model_type: Ignored for patch-level (always patch)
        model_size: "tiny", "small", "base", "large"
        **kwargs: Additional config parameters
        
    Returns:
        Model configuration
    """
    if not CORE_CONFIG_AVAILABLE:
        raise RuntimeError("Core configuration not available")
    
    return get_blip3o_patch_config(model_size, **kwargs)

def create_training_config(
    training_type: str = "standard",
    **kwargs
):
    """
    Create training configuration
    
    Args:
        training_type: "standard" or "enhanced"
        **kwargs: Additional training parameters
        
    Returns:
        Training configuration
    """
    if not CORE_CONFIG_AVAILABLE:
        raise RuntimeError("Core configuration not available")
    
    if training_type == "enhanced":
        return get_enhanced_flow_matching_config(**kwargs)
    else:
        return get_default_training_config(**kwargs)

def validate_patch_config(
    model_config, 
    flow_config=None,
    training_config=None,
):
    """
    Validate configuration for patch-level training
    
    Args:
        model_config: Model configuration
        flow_config: Flow matching configuration (optional)
        training_config: Training configuration (optional) 
        
    Returns:
        bool: True if configuration is valid
    """
    if not CORE_CONFIG_AVAILABLE:
        raise RuntimeError("Core configuration not available")
    
    try:
        # Basic model validation
        if hasattr(model_config, '_validate_config'):
            model_config._validate_config()
        
        # Patch-level specific checks
        if model_config.num_patches != 256:
            logger.warning(f"Non-standard patch count: {model_config.num_patches} (expected 256)")
        
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
        
        logger.info("✅ Patch-level configuration validated successfully")
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
        print("    - BLIP3oDiTConfig (patch-level)")
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
        print("  ✅ validate_patch_config()")
    
    print("=" * 35)

# Add utility functions to exports
__all__.extend([
    "get_recommended_config",
    "create_training_config", 
    "validate_patch_config",
    "print_config_status",
])

# Log configuration module status
if CORE_CONFIG_AVAILABLE:
    logger.info("BLIP3-o configurations loaded successfully")
    if MEMORY_OPTIMIZED_CONFIG_AVAILABLE:
        logger.info("  ✅ Memory optimization features available")
else:
    logger.error("BLIP3-o configurations failed to load!")