"""
Training utilities for BLIP3-o DiT - FIXED with Unified Trainer
src/modules/trainers/__init__.py

Contains:
- BLIP3oUnifiedTrainer: Single trainer that handles everything
- Training argument creation utilities for all modes
- Overfitting and production training support
"""

import logging

logger = logging.getLogger(__name__)

# Import UNIFIED trainer (replaces both flexible and training-only trainers)
UNIFIED_TRAINER_AVAILABLE = False

try:
    from .blip3o_unified_trainer import (
        # Main trainer class
        BLIP3oUnifiedTrainer,
        
        # Training argument factories
        create_unified_training_args,
        create_training_only_args,
        create_training_with_eval_args,
        create_overfitting_training_args,
        create_production_training_args,
    )
    
    UNIFIED_TRAINER_AVAILABLE = True
    logger.info("✅ UNIFIED BLIP3-o trainer loaded successfully")
    logger.info("   Features:")
    logger.info("     • Training-only mode (no evaluation during training)")
    logger.info("     • Training+evaluation mode (periodic evaluation)")
    logger.info("     • All scaling fixes applied")
    logger.info("     • Overfitting test support")
    logger.info("     • Production training support")
    logger.info("     • Comprehensive metrics and monitoring")
    
except ImportError as e:
    UNIFIED_TRAINER_AVAILABLE = False
    logger.error(f"❌ Failed to load UNIFIED BLIP3-o trainer: {e}")
    logger.error("   Make sure blip3o_unified_trainer.py exists and has no import errors")

# Set availability flags for compatibility
FLEXIBLE_TRAINER_AVAILABLE = UNIFIED_TRAINER_AVAILABLE  # Unified replaces flexible
TRAINING_ONLY_TRAINER_AVAILABLE = UNIFIED_TRAINER_AVAILABLE  # Unified replaces training-only

if UNIFIED_TRAINER_AVAILABLE:
    # Create aliases for backward compatibility
    BLIP3oFlexibleTrainer = BLIP3oUnifiedTrainer  # Alias for old code
    BLIP3oTrainingOnlyTrainer = BLIP3oUnifiedTrainer  # Alias for old code
    create_blip3o_flexible_training_args = create_training_with_eval_args  # Alias
    DEFAULT_TRAINER = "unified"
    
    logger.info("✅ Unified trainer available - replaces both flexible and training-only trainers")
else:
    DEFAULT_TRAINER = None
    logger.error("❌ No BLIP3-o trainers available")

__all__ = [
    # Availability flags
    "UNIFIED_TRAINER_AVAILABLE",
    "FLEXIBLE_TRAINER_AVAILABLE",  # For compatibility
    "TRAINING_ONLY_TRAINER_AVAILABLE",  # For compatibility
    "DEFAULT_TRAINER",
]

# Export unified trainer if available
if UNIFIED_TRAINER_AVAILABLE:
    __all__.extend([
        # Main trainer class
        "BLIP3oUnifiedTrainer",
        
        # Backward compatibility aliases
        "BLIP3oFlexibleTrainer",
        "BLIP3oTrainingOnlyTrainer",
        
        # Training argument factories
        "create_unified_training_args",
        "create_training_only_args",
        "create_training_with_eval_args",
        "create_overfitting_training_args",
        "create_production_training_args",
        
        # Backward compatibility aliases
        "create_blip3o_flexible_training_args",
    ])

def get_trainer_class(trainer_type: str = "auto"):
    """
    Get trainer class - now always returns unified trainer
    
    Args:
        trainer_type: "auto", "unified", "flexible", or "training_only"
                     (all return the same unified trainer)
    
    Returns:
        BLIP3oUnifiedTrainer class
    """
    if not UNIFIED_TRAINER_AVAILABLE:
        raise ValueError("Unified trainer not available")
    
    # All trainer types now return the unified trainer
    return BLIP3oUnifiedTrainer

def get_training_args_factory(trainer_type: str = "auto"):
    """
    Get training args factory based on type
    
    Args:
        trainer_type: "auto", "unified", "flexible", "training_only", 
                     "overfitting", or "production"
    
    Returns:
        Appropriate training args factory function
    """
    if not UNIFIED_TRAINER_AVAILABLE:
        raise ValueError("Unified trainer not available")
    
    if trainer_type in ["auto", "unified"]:
        return create_unified_training_args
    elif trainer_type in ["flexible", "training_with_eval"]:
        return create_training_with_eval_args
    elif trainer_type == "training_only":
        return create_training_only_args
    elif trainer_type == "overfitting":
        return create_overfitting_training_args
    elif trainer_type == "production":
        return create_production_training_args
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def create_trainer(
    model, 
    flow_matching_loss, 
    trainer_type="auto", 
    enable_evaluation=False,
    **kwargs
):
    """
    Create unified trainer instance
    
    Args:
        model: BLIP3-o model
        flow_matching_loss: Flow matching loss function
        trainer_type: Type of trainer configuration
        enable_evaluation: Whether to enable evaluation during training
        **kwargs: Additional trainer arguments
    
    Returns:
        BLIP3oUnifiedTrainer instance
    """
    if not UNIFIED_TRAINER_AVAILABLE:
        raise RuntimeError("Unified trainer not available")
    
    # Set enable_evaluation based on trainer_type if not explicitly set
    if trainer_type in ["flexible", "training_with_eval", "overfitting"]:
        enable_evaluation = True
    elif trainer_type in ["training_only", "production"]:
        enable_evaluation = False
    
    return BLIP3oUnifiedTrainer(
        model=model,
        flow_matching_loss=flow_matching_loss,
        enable_evaluation=enable_evaluation,
        **kwargs
    )

def create_overfitting_trainer(model, flow_matching_loss, **kwargs):
    """
    Create trainer optimized for overfitting tests
    
    Returns:
        BLIP3oUnifiedTrainer configured for overfitting
    """
    defaults = {
        'enable_evaluation': True,
        'enable_same_data_eval': True,
        'detailed_logging': True,
        'training_mode': 'patch_only',
    }
    defaults.update(kwargs)
    
    return create_trainer(
        model=model,
        flow_matching_loss=flow_matching_loss,
        trainer_type="overfitting",
        **defaults
    )

def create_production_trainer(model, flow_matching_loss, **kwargs):
    """
    Create trainer optimized for production training
    
    Returns:
        BLIP3oUnifiedTrainer configured for production
    """
    defaults = {
        'enable_evaluation': False,  # Usually disabled for production
        'detailed_logging': True,
        'training_mode': 'patch_only',
    }
    defaults.update(kwargs)
    
    return create_trainer(
        model=model,
        flow_matching_loss=flow_matching_loss,
        trainer_type="production",
        **defaults
    )

def print_trainer_status():
    """Print status of available trainers"""
    print("🏋️ BLIP3-o Trainers Status")
    print("=" * 40)
    print(f"Default trainer: {DEFAULT_TRAINER}")
    
    print("\nAvailable trainer:")
    if UNIFIED_TRAINER_AVAILABLE:
        print("  ✅ Unified BLIP3-o Trainer")
        print("    - Training-only mode (no evaluation)")
        print("    - Training+evaluation mode (periodic evaluation)")
        print("    - Overfitting test support")
        print("    - Production training support")
        print("    - All scaling fixes applied")
        print("    - Comprehensive metrics and monitoring")
        print("    - Replaces both flexible and training-only trainers")
    else:
        print("  ❌ Unified BLIP3-o Trainer")
    
    print("\nBackward compatibility:")
    print("  ✅ BLIP3oFlexibleTrainer (alias to unified)")
    print("  ✅ BLIP3oTrainingOnlyTrainer (alias to unified)")
    print("  ✅ All old function names supported")
    
    print("\nTraining features:")
    print("  📊 Objective: Patch-level flow matching")
    print("  📐 Input: EVA-CLIP patches [B, N, 4096]")
    print("  🎯 Output: CLIP patches [B, N, 1024]")
    print("  🔄 Loss: Pure flow matching (BLIP3-o paper)")
    print("  🔧 Scaling: All fixes applied (velocity & output scaling)")
    print("=" * 40)

def validate_trainer_setup():
    """Validate that trainer setup is working correctly"""
    validation_results = {}
    
    if UNIFIED_TRAINER_AVAILABLE:
        try:
            # Test creating trainer args
            test_args = create_training_only_args(
                output_dir="./test",
                num_train_epochs=1
            )
            validation_results['training_args_creation'] = True
            
            # Test that aliases work
            trainer_class = get_trainer_class("auto")
            validation_results['trainer_class_access'] = trainer_class == BLIP3oUnifiedTrainer
            
            # Test different trainer types
            for trainer_type in ["training_only", "overfitting", "production"]:
                try:
                    factory = get_training_args_factory(trainer_type)
                    validation_results[f'{trainer_type}_factory'] = True
                except Exception:
                    validation_results[f'{trainer_type}_factory'] = False
            
        except Exception as e:
            validation_results['trainer_validation_error'] = str(e)
    else:
        validation_results['trainer_available'] = False
    
    # Print validation results
    print("🔍 Trainer Validation Results:")
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
        print("✅ All trainer setup validated successfully!")
    else:
        print("⚠️ Some trainer setup issues detected")
    
    return validation_results

# Add new functions to exports
__all__.extend([
    "create_trainer",
    "create_overfitting_trainer",
    "create_production_trainer",
    "get_trainer_class",
    "get_training_args_factory",
    "print_trainer_status",
    "validate_trainer_setup",
])

# Enhanced status logging
if UNIFIED_TRAINER_AVAILABLE:
    logger.info("BLIP3-o unified trainer loaded successfully")
    logger.info(f"Default trainer: {DEFAULT_TRAINER}")
    logger.info("Backward compatibility: All old trainer names work as aliases")
else:
    logger.warning("No BLIP3-o trainers available")

logger.info("BLIP3-o trainer initialization complete")