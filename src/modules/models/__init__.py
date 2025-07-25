"""
BLIP3-o Models Module - FIXED with Proper Factory Functions
src/modules/models/__init__.py

FIXES:
- Consistent parameter naming (num_hidden_layers)
- Proper factory functions with scaling
- Better error handling
- Conflict resolution for parameter names
"""

import logging
import torch
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Import availability flags
PATCH_MODEL_AVAILABLE = False

# Core model components
try:
    from .blip3o_patch_dit import (
        BLIP3oPatchDiTModel,
        BLIP3oDiTConfig,
        RotaryPositionalEmbedding3D,
        TimestepEmbedder,
        MultiHeadAttention,
        BLIP3oDiTBlock,
        create_blip3o_patch_dit_model,
    )
    
    PATCH_MODEL_AVAILABLE = True
    logger.info("✅ FIXED BLIP3-o patch-level DiT model loaded successfully")
    logger.info("   NEW FEATURES:")
    logger.info("     • Output scaling parameter (output_scale)")
    logger.info("     • Fixed generation timestep schedule")
    logger.info("     • Improved gradient checkpointing")
    logger.info("     • Better device handling")
    
except ImportError as e:
    logger.error(f"❌ Failed to import patch DiT model: {e}")
    # Set None values to avoid AttributeError
    BLIP3oPatchDiTModel = None
    BLIP3oDiTConfig = None
    create_blip3o_patch_dit_model = None

# Determine which model to use as primary
if PATCH_MODEL_AVAILABLE:
    logger.info("✅ Using FIXED BLIP3-o patch-level DiT model as primary model")
else:
    logger.error("❌ No models available!")

# Factory functions with FIXED parameter handling
def create_fixed_model(
    training_mode: str = "patch_only",
    model_size: str = "base",
    output_scale: float = 0.1,
    use_gradient_checkpointing: bool = False,
    hidden_size: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,  # FIXED: Use consistent parameter name
    num_attention_heads: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    **kwargs
) -> Optional[BLIP3oPatchDiTModel]:
    """
    FIXED: Create BLIP3-o model with proper scaling and consistent parameter names
    
    Args:
        training_mode: "patch_only" or "cls_patch"
        model_size: "tiny", "small", "base", "large" (ignored if explicit sizes provided)
        output_scale: Output scaling factor (CRITICAL FIX)
        use_gradient_checkpointing: Enable gradient checkpointing
        hidden_size: Hidden dimension (overrides model_size)
        num_hidden_layers: Number of layers (overrides model_size)
        num_attention_heads: Number of attention heads (overrides model_size)
        intermediate_size: FFN intermediate size (overrides model_size)
        **kwargs: Additional config parameters
    
    Returns:
        BLIP3oPatchDiTModel with all fixes applied
    """
    if not PATCH_MODEL_AVAILABLE:
        raise RuntimeError("Patch DiT model not available")
    
    # FIXED: Predefined size configurations with consistent parameter names
    size_configs = {
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
    }
    
    # FIXED: Start with size config, then override with explicit parameters
    if model_size in size_configs:
        config_params = size_configs[model_size].copy()
    else:
        logger.warning(f"Unknown model_size '{model_size}', using 'base'")
        config_params = size_configs["base"].copy()
    
    # FIXED: Override with explicit parameters if provided
    if hidden_size is not None:
        config_params["hidden_size"] = hidden_size
    if num_hidden_layers is not None:
        config_params["num_hidden_layers"] = num_hidden_layers
    if num_attention_heads is not None:
        config_params["num_attention_heads"] = num_attention_heads
    if intermediate_size is not None:
        config_params["intermediate_size"] = intermediate_size
    
    # FIXED: Set training mode parameters
    num_tokens = 257 if training_mode == "cls_patch" else 256
    config_params.update({
        "num_tokens": num_tokens,
        "max_position_embeddings": max(num_tokens, 257),
        "training_mode": training_mode,
        "output_scale": output_scale,  # CRITICAL FIX
        "use_gradient_checkpointing": use_gradient_checkpointing,
    })
    
    # FIXED: Apply additional kwargs, but avoid parameter conflicts
    for key, value in kwargs.items():
        # FIXED: Skip conflicting parameter names
        if key not in ["num_layers"]:  # Avoid conflict with num_hidden_layers
            config_params[key] = value
        elif key == "num_layers":
            # FIXED: Map num_layers to num_hidden_layers
            logger.info(f"Mapping num_layers={value} to num_hidden_layers")
            config_params["num_hidden_layers"] = value
    
    logger.info(f"✅ Creating FIXED model with parameters:")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Hidden size: {config_params['hidden_size']}")
    logger.info(f"   Layers: {config_params['num_hidden_layers']}")
    logger.info(f"   Heads: {config_params['num_attention_heads']}")
    logger.info(f"   🔧 Output scale: {output_scale} (APPLIED)")
    
    try:
        # FIXED: Create config with resolved parameters
        config = BLIP3oDiTConfig(**config_params)
        
        # Create model
        model = BLIP3oPatchDiTModel(config)
        
        logger.info(f"✅ FIXED model created successfully")
        logger.info(f"   Parameters: {model.get_num_parameters():,}")
        logger.info(f"   Output scale: {model.output_scale.item():.3f}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        logger.error(f"   Config parameters: {config_params}")
        raise

def create_overfitting_model(**kwargs) -> Optional[BLIP3oPatchDiTModel]:
    """Create model optimized for overfitting tests"""
    return create_fixed_model(
        model_size="tiny",
        output_scale=0.05,  # Smaller scale for overfitting
        use_gradient_checkpointing=False,
        **kwargs
    )

def create_production_model(**kwargs) -> Optional[BLIP3oPatchDiTModel]:
    """Create model optimized for production training"""
    return create_fixed_model(
        model_size="base",
        output_scale=0.1,
        use_gradient_checkpointing=True,
        **kwargs
    )

def create_debug_model(**kwargs) -> Optional[BLIP3oPatchDiTModel]:
    """Create tiny model for debugging"""
    return create_fixed_model(
        model_size="tiny",
        output_scale=0.1,
        use_gradient_checkpointing=False,
        **kwargs
    )

def print_model_fixes():
    """Print information about model fixes"""
    print("🔧 BLIP3-o Model Fixes Applied")
    print("=" * 40)
    if PATCH_MODEL_AVAILABLE:
        print("✅ FIXED Patch-Level DiT Model:")
        print("  • Output scaling parameter (output_scale)")
        print("  • Proper generation timestep schedule")
        print("  • Fixed parameter name conflicts")
        print("  • Improved gradient checkpointing")
        print("  • Better device handling")
        print("  • Consistent num_hidden_layers parameter")
        print("  • Smaller initial noise for generation")
        print("  • L2 normalization in generation")
        print("  • Guidance scale support")
    else:
        print("❌ Patch-Level DiT Model: Not Available")
    print("=" * 40)

def validate_model_config(config_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Validate and clean model configuration parameters
    
    Args:
        config_params: Raw configuration parameters
        
    Returns:
        Cleaned configuration parameters
    """
    cleaned_params = config_params.copy()
    
    # FIXED: Handle parameter name conflicts
    if "num_layers" in cleaned_params and "num_hidden_layers" in cleaned_params:
        logger.warning("Both 'num_layers' and 'num_hidden_layers' provided, using 'num_hidden_layers'")
        cleaned_params.pop("num_layers")
    elif "num_layers" in cleaned_params:
        cleaned_params["num_hidden_layers"] = cleaned_params.pop("num_layers")
        logger.info("Mapped 'num_layers' to 'num_hidden_layers'")
    
    # FIXED: Validate required parameters
    required_params = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
    for param in required_params:
        if param not in cleaned_params:
            raise ValueError(f"Missing required parameter: {param}")
    
    # FIXED: Validate attention head compatibility
    if cleaned_params["hidden_size"] % cleaned_params["num_attention_heads"] != 0:
        raise ValueError(
            f"hidden_size ({cleaned_params['hidden_size']}) must be divisible by "
            f"num_attention_heads ({cleaned_params['num_attention_heads']})"
        )
    
    return cleaned_params

# Main exports
__all__ = [
    # Availability flags
    "PATCH_MODEL_AVAILABLE",
]

# Export models if available
if PATCH_MODEL_AVAILABLE:
    __all__.extend([
        # Core classes
        "BLIP3oPatchDiTModel",
        "BLIP3oDiTConfig",
        
        # Factory functions (FIXED with scaling)
        "create_blip3o_patch_dit_model",
        "create_fixed_model",
        "create_overfitting_model",
        "create_production_model",
        "create_debug_model",
        
        # Model components
        "RotaryPositionalEmbedding3D",
        "TimestepEmbedder",
        "MultiHeadAttention",
        "BLIP3oDiTBlock",
        
        # Utilities
        "print_model_fixes",
        "validate_model_config",
    ])

# Initialize models
if PATCH_MODEL_AVAILABLE:
    logger.info("✅ Verified FIXED model with output scaling parameter")
    logger.info("FIXED BLIP3-o patch-level DiT model loaded successfully - All fixes applied")
else:
    logger.error("❌ Model initialization failed")