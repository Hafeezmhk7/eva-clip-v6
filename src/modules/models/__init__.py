"""
Model modules for BLIP3-o DiT - Global Training Only

Contains:
- GlobalBLIP3oDiTModel: Global training model (primary)
- Model creation and loading utilities for global training
"""

import logging

logger = logging.getLogger(__name__)

# Import global model (your main model)
GLOBAL_MODEL_AVAILABLE = False
GlobalBLIP3oDiTModel = None
create_global_blip3o_dit_model = None

try:
    from .global_blip3o_dit import (
        GlobalBLIP3oDiTModel,
        create_global_blip3o_dit_model,
    )
    GLOBAL_MODEL_AVAILABLE = True
    logger.info("✅ Global BLIP3-o model loaded successfully")
    
except ImportError as e:
    GLOBAL_MODEL_AVAILABLE = False
    logger.error(f"❌ Failed to load global model: {e}")
    raise ImportError(f"Global BLIP3-o model is required but failed to load: {e}")

# Use global model as the main model (no fallbacks)
BLIP3oDiTModel = GlobalBLIP3oDiTModel
create_blip3o_dit_model = create_global_blip3o_dit_model
DEFAULT_MODEL_TYPE = "global"

logger.info("✅ Using Global BLIP3-o model as primary model")

# Build exports list
__all__ = [
    # Primary model interface
    "BLIP3oDiTModel",
    "create_blip3o_dit_model", 
    "DEFAULT_MODEL_TYPE",
    
    # Global model specific
    "GlobalBLIP3oDiTModel",
    "create_global_blip3o_dit_model",
    "GLOBAL_MODEL_AVAILABLE",
]

def get_model_class(model_type: str = "auto"):
    """
    Get the model class (always returns GlobalBLIP3oDiTModel)
    
    Args:
        model_type: Ignored, always returns global model
        
    Returns:
        GlobalBLIP3oDiTModel class
    """
    if not GLOBAL_MODEL_AVAILABLE:
        raise RuntimeError("Global BLIP3-o model not available")
    return GlobalBLIP3oDiTModel

def get_model_factory(model_type: str = "auto"):
    """
    Get the model factory function (always returns global factory)
    
    Args:
        model_type: Ignored, always returns global factory
        
    Returns:
        create_global_blip3o_dit_model function
    """
    if not GLOBAL_MODEL_AVAILABLE:
        raise RuntimeError("Global BLIP3-o model not available")
    return create_global_blip3o_dit_model

def create_model(config=None, **kwargs):
    """
    Create a BLIP3-o model instance (always global model)
    
    Args:
        config: Model configuration
        **kwargs: Additional arguments
        
    Returns:
        GlobalBLIP3oDiTModel instance
    """
    if not GLOBAL_MODEL_AVAILABLE:
        raise RuntimeError("Global BLIP3-o model not available")
        
    if config is not None:
        return create_global_blip3o_dit_model(config=config, **kwargs)
    else:
        return create_global_blip3o_dit_model(**kwargs)

def print_model_status():
    """Print status of available models"""
    print("🏗️ BLIP3-o Models Status")
    print("=" * 30)
    print(f"Model type: {DEFAULT_MODEL_TYPE}")
    print()
    print("Available model:")
    
    if GLOBAL_MODEL_AVAILABLE:
        print("  ✅ Global BLIP3-o DiT (Primary Model)")
        print("    - Direct global feature training")
        print("    - Optimized for recall performance")
        print("    - Multi-GPU compatible")
    else:
        print("  ❌ Global BLIP3-o DiT (REQUIRED)")
    
    print("=" * 30)

# Add utility functions to exports
__all__.extend([
    "get_model_class",
    "get_model_factory",
    "create_model",
    "print_model_status",
])

# Ensure the model is available
if not GLOBAL_MODEL_AVAILABLE:
    logger.error("❌ Global BLIP3-o model is required but not available!")
    raise ImportError("Global BLIP3-o model is required for this project")

logger.info("BLIP3-o global model loaded successfully")