"""
Model modules for BLIP3-o DiT - UPDATED with Scaling Fixes
src/modules/models/__init__.py
Contains:
- BLIP3oPatchDiTModel: FIXED patch-level DiT model with output scaling
- Model creation and loading utilities with scaling parameters
- Paper-aligned architecture with EVA-CLIP conditioning
"""

import logging

logger = logging.getLogger(__name__)

# Import FIXED patch-level model (primary model following BLIP3-o paper)
PATCH_MODEL_AVAILABLE = False
BLIP3oPatchDiTModel = None
create_blip3o_patch_dit_model = None

try:
    from .blip3o_patch_dit import (
        # Core model classes
        BLIP3oPatchDiTModel,
        BLIP3oDiTConfig,
        
        # Factory functions (UPDATED with scaling parameters)
        create_blip3o_patch_dit_model,
        
        # Model components
        RotaryPositionalEmbedding3D,
        TimestepEmbedder,
        MultiHeadAttention,
        BLIP3oDiTBlock,
    )
    PATCH_MODEL_AVAILABLE = True
    logger.info("✅ FIXED BLIP3-o patch-level DiT model loaded successfully")
    logger.info("   NEW FEATURES:")
    logger.info("     • Output scaling parameter (output_scale)")
    logger.info("     • Fixed generation timestep schedule")
    logger.info("     • Improved gradient checkpointing")
    logger.info("     • Better device handling")
    
except ImportError as e:
    PATCH_MODEL_AVAILABLE = False
    logger.error(f"❌ Failed to load FIXED patch-level DiT model: {e}")
    raise ImportError(f"FIXED BLIP3-o patch-level DiT model is required but failed to load: {e}")

# Use patch-level model as the main model (paper-aligned)
BLIP3oDiTModel = BLIP3oPatchDiTModel
create_blip3o_dit_model = create_blip3o_patch_dit_model
DEFAULT_MODEL_TYPE = "patch_level_fixed"

logger.info("✅ Using FIXED BLIP3-o patch-level DiT model as primary model")

# Verify that we have the fixed version with scaling parameters
try:
    # Test that we can create a model with the new scaling parameters
    test_config = BLIP3oDiTConfig(
        hidden_size=384,  # Small for testing
        num_hidden_layers=2,
        num_attention_heads=6,
        output_scale=0.1,  # NEW: Test scaling parameter
    )
    
    test_model = create_blip3o_patch_dit_model(
        config=test_config,
        output_scale=0.1,  # NEW: Test scaling parameter
    )
    
    # Verify the model has the scaling parameter
    if hasattr(test_model, 'output_scale'):
        logger.info("✅ Verified FIXED model with output scaling parameter")
    else:
        logger.warning("⚠️ Model may not have the latest fixes")
    
    del test_model, test_config  # Clean up
    
except Exception as e:
    logger.error(f"❌ Failed to verify FIXED model version: {e}")
    logger.error("   The model file may not be the complete fixed version")

# Build exports list with all functions
__all__ = [
    # Availability flags
    "PATCH_MODEL_AVAILABLE",
    "DEFAULT_MODEL_TYPE",
    
    # Primary model interface (paper-aligned with fixes)
    "BLIP3oDiTModel",
    "BLIP3oDiTConfig", 
    "create_blip3o_dit_model",
    
    # FIXED patch-level model specific
    "BLIP3oPatchDiTModel",
    "create_blip3o_patch_dit_model",
    
    # Model components
    "RotaryPositionalEmbedding3D",
    "TimestepEmbedder", 
    "MultiHeadAttention",
    "BLIP3oDiTBlock",
]

def get_model_class(model_type: str = "auto"):
    """
    Get the FIXED model class (always returns BLIP3oPatchDiTModel)
    
    Args:
        model_type: Ignored, always returns FIXED patch-level model
        
    Returns:
        BLIP3oPatchDiTModel class with all fixes
    """
    if not PATCH_MODEL_AVAILABLE:
        raise RuntimeError("FIXED BLIP3-o patch-level DiT model not available")
    return BLIP3oPatchDiTModel

def get_model_factory(model_type: str = "auto"):
    """
    Get the FIXED model factory function
    
    Args:
        model_type: Ignored, always returns FIXED patch-level factory
        
    Returns:
        create_blip3o_patch_dit_model function with scaling support
    """
    if not PATCH_MODEL_AVAILABLE:
        raise RuntimeError("FIXED BLIP3-o patch-level DiT model not available")
    return create_blip3o_patch_dit_model

def create_model(config=None, **kwargs):
    """
    Create a FIXED BLIP3-o model instance with scaling parameters
    
    Args:
        config: Model configuration
        **kwargs: Additional arguments including scaling parameters
        
    Returns:
        BLIP3oPatchDiTModel instance with all fixes applied
    """
    if not PATCH_MODEL_AVAILABLE:
        raise RuntimeError("FIXED BLIP3-o patch-level DiT model not available")
        
    if config is not None:
        return create_blip3o_patch_dit_model(config=config, **kwargs)
    else:
        return create_blip3o_patch_dit_model(**kwargs)

def create_fixed_model(
    training_mode: str = "patch_only",
    output_scale: float = 0.1,  # NEW: Default scaling parameter
    **kwargs
):
    """
    Create a FIXED BLIP3-o model with recommended scaling parameters
    
    Args:
        training_mode: "patch_only" or "cls_patch"
        output_scale: Output scaling factor (CRITICAL for fixing scale mismatch)
        **kwargs: Additional model parameters
        
    Returns:
        BLIP3oPatchDiTModel with all fixes applied
    """
    if not PATCH_MODEL_AVAILABLE:
        raise RuntimeError("FIXED BLIP3-o patch-level DiT model not available")
    
    # Set recommended defaults
    defaults = {
        'training_mode': training_mode,
        'output_scale': output_scale,
        'use_gradient_checkpointing': False,
    }
    
    # Override with user parameters
    defaults.update(kwargs)
    
    return create_blip3o_patch_dit_model(**defaults)

def create_overfitting_model(**kwargs):
    """
    Create model optimized for overfitting tests
    """
    defaults = {
        'training_mode': 'patch_only',
        'output_scale': 0.1,
        'model_size': 'base',
        'use_gradient_checkpointing': False,  # Disable for overfitting test
    }
    
    defaults.update(kwargs)
    return create_fixed_model(**defaults)

def create_production_model(**kwargs):
    """
    Create model optimized for production training
    """
    defaults = {
        'training_mode': 'patch_only',
        'output_scale': 0.1,
        'model_size': 'base',
        'use_gradient_checkpointing': True,  # Enable for memory efficiency
    }
    
    defaults.update(kwargs)
    return create_fixed_model(**defaults)

def load_pretrained_model(model_path: str, **kwargs):
    """
    Load a pretrained FIXED BLIP3-o patch-level DiT model
    
    Args:
        model_path: Path to pretrained model
        **kwargs: Additional arguments
        
    Returns:
        Loaded BLIP3oPatchDiTModel instance with fixes
    """
    from pathlib import Path
    import torch
    import json
    
    model_path = Path(model_path)
    
    # Load config
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Ensure output_scale is in config (for backward compatibility)
        if 'output_scale' not in config_dict:
            config_dict['output_scale'] = 0.1
            logger.info("Added default output_scale=0.1 to loaded config")
        
        config = BLIP3oDiTConfig(**config_dict)
    else:
        # Use default config with fixes
        from ..config.blip3o_config import get_default_blip3o_config
        config = get_default_blip3o_config()
        config.output_scale = 0.1  # Ensure scaling is applied
        logger.warning(f"No config found at {config_file}, using default with fixes")
    
    # Create model with fixes
    model = create_blip3o_patch_dit_model(config=config, **kwargs)
    
    # Load weights
    weight_files = [
        model_path / "pytorch_model.bin",
        model_path / "model.safetensors",
        model_path / "pytorch_model.safetensors"
    ]
    
    weight_file = None
    for wf in weight_files:
        if wf.exists():
            weight_file = wf
            break
    
    if weight_file is None:
        logger.warning(f"No weight file found in {model_path}, returning untrained model")
        return model
    
    logger.info(f"Loading weights from: {weight_file}")
    
    # Load weights
    if weight_file.suffix == ".bin":
        state_dict = torch.load(weight_file, map_location='cpu')
    else:
        from safetensors.torch import load_file
        state_dict = load_file(str(weight_file))
    
    # Load state dict (allow missing keys for new parameters)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.info(f"Missing keys when loading model: {len(missing_keys)} keys")
        logger.info("This is expected when loading models without the latest fixes")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading model: {len(unexpected_keys)} keys")
    
    logger.info(f"✅ FIXED model loaded successfully from {model_path}")
    
    return model

def print_model_status():
    """Print status of available FIXED models"""
    print("🏗️ FIXED BLIP3-o DiT Models Status")
    print("=" * 40)
    print(f"Model type: {DEFAULT_MODEL_TYPE}")
    print()
    print("Available model (Paper-Aligned with FIXES):")
    
    if PATCH_MODEL_AVAILABLE:
        print("  ✅ BLIP3-o Patch DiT (FIXED Primary Model)")
        print("    - 256-token patch-level training")
        print("    - EVA-CLIP conditioning (4096-dim)")
        print("    - CLIP output (1024-dim)")
        print("    - Flow matching training objective")
        print("    - Image-to-text recall optimization")
        print("    - 3D Rotary Position Embedding")
        print("    - Multi-head attention with spatial encoding")
        print("    - ✅ OUTPUT SCALING (output_scale parameter)")
        print("    - ✅ FIXED generation timestep schedule")
        print("    - ✅ Improved gradient checkpointing")
        print("    - Multi-GPU compatible")
        print("    - Paper-aligned architecture")
    else:
        print("  ❌ BLIP3-o Patch DiT (REQUIRED)")
    
    print()
    print("Model components (FIXED):")
    print("  ✅ RotaryPositionalEmbedding3D (spatial-temporal)")
    print("  ✅ TimestepEmbedder (flow matching)")
    print("  ✅ MultiHeadAttention (with 3D RoPE)")
    print("  ✅ BLIP3oDiTBlock (patch-conditioned)")
    
    print()
    print("Architecture details (FIXED):")
    print("  📐 Input: EVA-CLIP patches [B, 256, 4096]")
    print("  🎯 Output: CLIP patches [B, 256, 1024] (with scaling)")
    print("  🔄 Conditioning: Cross-attention with EVA features")
    print("  📊 Evaluation: Image-to-text recall metrics")
    print("  🔧 Scaling: Output scaling to fix norm mismatch")
    
    print("=" * 40)

def print_model_fixes():
    """Print information about the model fixes applied"""
    print("🔧 BLIP3-o Model Fixes Applied:")
    print("=" * 40)
    print("✅ Scale Mismatch Solution:")
    print("   • output_scale parameter added to model")
    print("   • Learnable output scaling layer")
    print("   • Proper initialization of output projection")
    print()
    print("✅ Generation Improvements:")
    print("   • Fixed timestep schedule (0 to 1.0)")
    print("   • Proper Euler integration")
    print("   • Correct normalization in generation")
    print("   • Guidance scale support")
    print()
    print("✅ Training Enhancements:")
    print("   • Better gradient checkpointing")
    print("   • Improved device handling")
    print("   • Proper weight initialization")
    print("   • Enhanced forward pass")
    print("=" * 40)

# Add new functions to exports
__all__.extend([
    "create_fixed_model",
    "create_overfitting_model",
    "create_production_model",
    "print_model_fixes",
])

# Ensure the FIXED patch-level model is available
if not PATCH_MODEL_AVAILABLE:
    logger.error("❌ FIXED BLIP3-o patch-level DiT model is required but not available!")
    raise ImportError("FIXED BLIP3-o patch-level DiT model is required for this project")

logger.info("FIXED BLIP3-o patch-level DiT model loaded successfully - All fixes applied")