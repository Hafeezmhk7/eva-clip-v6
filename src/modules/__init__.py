"""
BLIP3-o Modules - UPDATED with All Fixes Applied
src/modules/__init__.py

Main module initialization with:
- FIXED flow matching loss with scaling parameters
- FIXED DiT model with output scaling  
- Updated trainers and datasets
- Comprehensive fix verification
- Better error handling for missing components
"""

import logging
import sys
from pathlib import Path

# Setup module-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("🚀 Initializing BLIP3-o modules with ALL FIXES")
logger.info("=" * 50)

# Import availability flags
MODEL_AVAILABLE = False
LOSS_AVAILABLE = False
TRAINER_AVAILABLE = False
DATASET_AVAILABLE = False

# 1. Import FIXED models
try:
    from .models import (
        # Core classes
        BLIP3oPatchDiTModel,
        BLIP3oDiTConfig,
        
        # Factory functions (with scaling)
        create_blip3o_patch_dit_model,
        create_fixed_model,
        create_overfitting_model,
        create_production_model,
        
        # Model components
        RotaryPositionalEmbedding3D,
        TimestepEmbedder,
        MultiHeadAttention,
        BLIP3oDiTBlock,
        
        # Utilities
        print_model_fixes,
        PATCH_MODEL_AVAILABLE,
    )
    
    MODEL_AVAILABLE = PATCH_MODEL_AVAILABLE
    logger.info("✅ FIXED models imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import FIXED models: {e}")
    logger.error("   Ensure blip3o_patch_dit.py has all the fixes applied")
    # Set None values to avoid AttributeError later
    BLIP3oPatchDiTModel = None
    BLIP3oDiTConfig = None
    create_blip3o_patch_dit_model = None
    create_fixed_model = None

# 2. Import FIXED losses  
try:
    from .losses import (
        # Core classes
        BLIP3oFlowMatchingLoss,
        
        # Factory functions (with scaling)
        create_blip3o_flow_matching_loss,
        create_debug_loss,
        create_production_loss,
        get_fixed_loss_function,
        get_overfitting_loss_function,
        
        # Utilities
        analyze_loss_scaling,
        print_loss_fixes,
        FLOW_MATCHING_LOSS_AVAILABLE,
    )
    
    LOSS_AVAILABLE = FLOW_MATCHING_LOSS_AVAILABLE
    logger.info("✅ FIXED losses imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import FIXED losses: {e}")
    logger.error("   Ensure blip3o_flow_matching_loss.py is the complete fixed version")
    # Set None values to avoid AttributeError later
    BLIP3oFlowMatchingLoss = None
    create_blip3o_flow_matching_loss = None
    get_fixed_loss_function = None

# 3. Import trainers
try:
    from .trainers import (
        BLIP3oTrainingOnlyTrainer,
        create_training_only_args,
        TRAINING_ONLY_TRAINER_AVAILABLE,
    )
    
    TRAINER_AVAILABLE = TRAINING_ONLY_TRAINER_AVAILABLE
    logger.info("✅ Trainers imported successfully")
    
    # Try to import unified trainer if available
    try:
        from .trainers import (
            BLIP3oUnifiedTrainer,
            create_unified_training_args,
            UNIFIED_TRAINER_AVAILABLE,
        )
        logger.info("✅ Unified trainer imported successfully")
    except ImportError as e:
        logger.warning(f"⚠️ Unified trainer not available: {e}")
        BLIP3oUnifiedTrainer = None
        create_unified_training_args = None
    
except ImportError as e:
    logger.error(f"❌ Failed to import trainers: {e}")
    # Set None values to avoid AttributeError later
    BLIP3oTrainingOnlyTrainer = None
    create_training_only_args = None

# 4. Import datasets with better error handling
try:
    from .datasets import (
        create_flexible_dataloaders,
        DATASET_AVAILABLE as DATASET_MODULE_AVAILABLE,
    )
    
    DATASET_AVAILABLE = DATASET_MODULE_AVAILABLE
    logger.info("✅ FIXED datasets imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Failed to import datasets: {e}")
    # Try to import at least the flag
    try:
        from .datasets import DATASET_AVAILABLE as DATASET_MODULE_AVAILABLE
        DATASET_AVAILABLE = DATASET_MODULE_AVAILABLE
        logger.warning("⚠️ Partial dataset import - flag only")
    except ImportError:
        DATASET_AVAILABLE = False
        logger.error("❌ Complete dataset import failure")
    
    # Set None values to avoid AttributeError later
    create_flexible_dataloaders = None

# 5. Import config if available
try:
    from .config import (
        get_default_blip3o_config,
        BLIP3oDiTConfig as ConfigClass,
    )
    
    logger.info("✅ Config imported successfully")
    
except ImportError as e:
    logger.warning(f"⚠️ Config import failed: {e}")
    get_default_blip3o_config = None

# Check overall module status
ALL_MODULES_AVAILABLE = all([
    MODEL_AVAILABLE,
    LOSS_AVAILABLE, 
    TRAINER_AVAILABLE,
    DATASET_AVAILABLE,
])

if ALL_MODULES_AVAILABLE:
    logger.info("🎉 ALL BLIP3-o modules with FIXES loaded successfully!")
else:
    logger.warning("⚠️ Some modules failed to load - check individual import errors")

logger.info("=" * 50)

# Main exports
__all__ = [
    # Availability flags
    "MODEL_AVAILABLE",
    "LOSS_AVAILABLE", 
    "TRAINER_AVAILABLE",
    "DATASET_AVAILABLE",
    "ALL_MODULES_AVAILABLE",
]

# Export models if available
if MODEL_AVAILABLE and BLIP3oPatchDiTModel is not None:
    __all__.extend([
        # Core model classes
        "BLIP3oPatchDiTModel",
        "BLIP3oDiTConfig",
        
        # Factory functions (FIXED with scaling)
        "create_blip3o_patch_dit_model",
        "create_fixed_model",
        "create_overfitting_model", 
        "create_production_model",
        
        # Model components
        "RotaryPositionalEmbedding3D",
        "TimestepEmbedder",
        "MultiHeadAttention", 
        "BLIP3oDiTBlock",
        
        # Utilities
        "print_model_fixes",
    ])

# Export losses if available
if LOSS_AVAILABLE and BLIP3oFlowMatchingLoss is not None:
    __all__.extend([
        # Core loss classes
        "BLIP3oFlowMatchingLoss",
        
        # Factory functions (FIXED with scaling)
        "create_blip3o_flow_matching_loss",
        "create_debug_loss",
        "create_production_loss",
        "get_fixed_loss_function",
        "get_overfitting_loss_function",
        
        # Utilities
        "analyze_loss_scaling",
        "print_loss_fixes",
    ])

# Export trainers if available
if TRAINER_AVAILABLE and BLIP3oTrainingOnlyTrainer is not None:
    __all__.extend([
        "BLIP3oTrainingOnlyTrainer",
        "create_training_only_args",
    ])

# Export unified trainer if available
if 'BLIP3oUnifiedTrainer' in locals() and BLIP3oUnifiedTrainer is not None:
    __all__.extend([
        "BLIP3oUnifiedTrainer",
        "create_unified_training_args",
    ])

# Export datasets if available
if DATASET_AVAILABLE and create_flexible_dataloaders is not None:
    __all__.extend([
        "create_flexible_dataloaders",
    ])

# High-level convenience functions
def create_complete_fixed_setup(
    training_mode: str = "patch_only",
    velocity_scale: float = 0.1,
    output_scale: float = 0.1,
    **kwargs
):
    """
    Create a complete BLIP3-o setup with all fixes applied
    
    Args:
        training_mode: "patch_only" or "cls_patch"
        velocity_scale: Velocity scaling for loss function
        output_scale: Output scaling for model
        **kwargs: Additional parameters
        
    Returns:
        tuple: (model, loss_function) with all fixes applied
    """
    if not (MODEL_AVAILABLE and LOSS_AVAILABLE):
        raise RuntimeError("Model and loss modules must be available")
    
    if create_fixed_model is None or get_fixed_loss_function is None:
        raise RuntimeError("Model and loss factory functions not available")
    
    # Create FIXED model with scaling
    model = create_fixed_model(
        training_mode=training_mode,
        output_scale=output_scale,
        **kwargs
    )
    
    # Create FIXED loss function with scaling
    loss_fn = get_fixed_loss_function(
        velocity_scale=velocity_scale,
        **kwargs
    )
    
    logger.info(f"✅ Created complete FIXED setup:")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Velocity scale: {velocity_scale}")
    logger.info(f"   Output scale: {output_scale}")
    
    return model, loss_fn

def create_overfitting_setup(**kwargs):
    """
    Create setup optimized for overfitting tests
    
    Returns:
        tuple: (model, loss_function) for overfitting test
    """
    if not (MODEL_AVAILABLE and LOSS_AVAILABLE):
        raise RuntimeError("Model and loss modules must be available")
    
    if create_overfitting_model is None or get_overfitting_loss_function is None:
        raise RuntimeError("Overfitting factory functions not available")
    
    model = create_overfitting_model(**kwargs)
    loss_fn = get_overfitting_loss_function(**kwargs)
    
    logger.info("✅ Created overfitting test setup with all fixes")
    
    return model, loss_fn

def create_production_setup(**kwargs):
    """
    Create setup optimized for production training
    
    Returns:
        tuple: (model, loss_function) for production training
    """
    if not (MODEL_AVAILABLE and LOSS_AVAILABLE):
        raise RuntimeError("Model and loss modules must be available")
    
    if create_production_model is None or create_production_loss is None:
        raise RuntimeError("Production factory functions not available")
    
    model = create_production_model(**kwargs)
    loss_fn = create_production_loss(**kwargs)
    
    logger.info("✅ Created production setup with all fixes")
    
    return model, loss_fn

def print_all_fixes():
    """Print comprehensive information about all fixes applied"""
    print("🔧 BLIP3-o Complete Fixes Applied")
    print("=" * 60)
    print()
    
    if MODEL_AVAILABLE and 'print_model_fixes' in globals():
        try:
            print_model_fixes()
        except Exception as e:
            print(f"⚠️ Could not print model fixes: {e}")
        print()
    
    if LOSS_AVAILABLE and 'print_loss_fixes' in globals():
        try:
            print_loss_fixes()
        except Exception as e:
            print(f"⚠️ Could not print loss fixes: {e}")
        print()
    
    print("📊 Module Status:")
    print(f"   Models: {'✅ Available' if MODEL_AVAILABLE else '❌ Not Available'}")
    print(f"   Losses: {'✅ Available' if LOSS_AVAILABLE else '❌ Not Available'}")
    print(f"   Trainers: {'✅ Available' if TRAINER_AVAILABLE else '❌ Not Available'}")
    print(f"   Datasets: {'✅ Available' if DATASET_AVAILABLE else '❌ Not Available'}")
    print(f"   Overall: {'✅ All Ready' if ALL_MODULES_AVAILABLE else '⚠️ Some Issues'}")
    print("=" * 60)

def validate_fixes():
    """Validate that all fixes are properly applied"""
    validation_results = {}
    
    # Validate model fixes
    if MODEL_AVAILABLE and create_fixed_model is not None:
        try:
            test_model = create_fixed_model(
                model_size='tiny',
                output_scale=0.1
            )
            
            # Check if model has scaling parameter
            has_scaling = hasattr(test_model, 'output_scale')
            validation_results['model_scaling'] = has_scaling
            
            # Check if model can be created with new parameters
            validation_results['model_creation'] = True
            
            del test_model
            
        except Exception as e:
            validation_results['model_creation'] = False
            validation_results['model_error'] = str(e)
    else:
        validation_results['model_creation'] = False
        validation_results['model_error'] = "Model factory not available"
    
    # Validate loss fixes
    if LOSS_AVAILABLE and get_fixed_loss_function is not None:
        try:
            test_loss = get_fixed_loss_function(
                velocity_scale=0.1,
                adaptive_scaling=True
            )
            
            # Check if loss has new methods
            has_scaling_methods = hasattr(test_loss, 'get_scaling_info')
            validation_results['loss_scaling'] = has_scaling_methods
            
            # Check if loss can be created with new parameters
            validation_results['loss_creation'] = True
            
            del test_loss
            
        except Exception as e:
            validation_results['loss_creation'] = False
            validation_results['loss_error'] = str(e)
    else:
        validation_results['loss_creation'] = False
        validation_results['loss_error'] = "Loss factory not available"
    
    # Print validation results
    print("🔍 Fix Validation Results:")
    print("=" * 30)
    for key, value in validation_results.items():
        if key.endswith('_error'):
            print(f"   ❌ {key}: {value}")
        else:
            status = "✅" if value else "❌"
            print(f"   {status} {key}: {value}")
    
    all_valid = all(
        v for k, v in validation_results.items() 
        if not k.endswith('_error')
    )
    
    if all_valid:
        print("✅ All fixes validated successfully!")
    else:
        print("⚠️ Some fixes may not be properly applied")
    
    return validation_results

# Add convenience functions to exports
__all__.extend([
    "create_complete_fixed_setup",
    "create_overfitting_setup",
    "create_production_setup", 
    "print_all_fixes",
    "validate_fixes",
])

# Run validation on import only if all modules are available
if ALL_MODULES_AVAILABLE:
    logger.info("🔍 Running fix validation...")
    try:
        validation_results = validate_fixes()
        
        if all(v for k, v in validation_results.items() if not k.endswith('_error')):
            logger.info("✅ All fixes validated - ready for use!")
        else:
            logger.warning("⚠️ Some validation issues detected - check output above")
    except Exception as e:
        logger.warning(f"⚠️ Validation failed: {e}")
else:
    logger.warning("⚠️ Cannot validate fixes - some modules unavailable")

logger.info("🏁 BLIP3-o module initialization complete")