"""
Training utilities for BLIP3-o DiT - Both Original and Enhanced Versions

Contains:
- BLIP3oPatchTrainer: Original trainer for patch-level flow matching
- BLIP3oPatchTrainerEnhanced: Enhanced trainer with convergence optimization
- Training argument creation utilities
- Paper-aligned training pipeline
"""

import logging

logger = logging.getLogger(__name__)

# Import original patch-level trainer (currently running)
PATCH_TRAINER_AVAILABLE = False
try:
    from .blip3o_patch_trainer import (
        BLIP3oPatchTrainer,
        create_blip3o_patch_training_args,
    )
    logger.debug("✅ BLIP3-o original patch-level trainer loaded")
    PATCH_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load original BLIP3-o patch-level trainer: {e}")
    PATCH_TRAINER_AVAILABLE = False

# Import enhanced patch-level trainer (new version)
ENHANCED_TRAINER_AVAILABLE = False
try:
    from .blip3o_patch_trainer_enhanced import (
        BLIP3oPatchTrainerEnhanced,
        create_blip3o_enhanced_training_args,
    )
    logger.debug("✅ BLIP3-o enhanced patch-level trainer loaded")
    ENHANCED_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Enhanced BLIP3-o trainer not available: {e}")
    ENHANCED_TRAINER_AVAILABLE = False

# Set defaults based on availability
if ENHANCED_TRAINER_AVAILABLE:
    # Prefer enhanced trainer as default for new trainings
    BLIP3oTrainer = BLIP3oPatchTrainerEnhanced
    create_training_args = create_blip3o_enhanced_training_args
    DEFAULT_TRAINER = "enhanced_patch_level"
    logger.info("✅ Using enhanced patch-level trainer as default")
elif PATCH_TRAINER_AVAILABLE:
    # Fallback to original trainer
    BLIP3oTrainer = BLIP3oPatchTrainer
    create_training_args = create_blip3o_patch_training_args
    DEFAULT_TRAINER = "patch_level"
    logger.info("✅ Using original patch-level trainer as default")
else:
    # No trainers available
    BLIP3oTrainer = None
    create_training_args = None
    DEFAULT_TRAINER = None
    logger.error("❌ No BLIP3-o trainers available")

__all__ = [
    # Availability flags
    "PATCH_TRAINER_AVAILABLE",
    "ENHANCED_TRAINER_AVAILABLE", 
    "DEFAULT_TRAINER",
]

# Export original trainer components if available
if PATCH_TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oPatchTrainer",
        "create_blip3o_patch_training_args",
    ])

# Export enhanced trainer components if available
if ENHANCED_TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oPatchTrainerEnhanced",
        "create_blip3o_enhanced_training_args",
    ])

# Export default interface if available
if BLIP3oTrainer is not None:
    __all__.extend([
        "BLIP3oTrainer",
        "create_training_args",
    ])

def get_trainer_class(trainer_type: str = "auto"):
    """
    Get the appropriate trainer class
    
    Args:
        trainer_type: "auto", "original", "enhanced", "patch_level"
        
    Returns:
        Trainer class
    """
    if trainer_type == "auto":
        if BLIP3oTrainer is None:
            raise ValueError("No trainer class available")
        return BLIP3oTrainer
        
    elif trainer_type in ("original", "patch_level"):
        if not PATCH_TRAINER_AVAILABLE:
            raise ValueError("Original patch-level trainer not available")
        return BLIP3oPatchTrainer
        
    elif trainer_type == "enhanced":
        if not ENHANCED_TRAINER_AVAILABLE:
            raise ValueError("Enhanced trainer not available")
        return BLIP3oPatchTrainerEnhanced
        
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def get_training_args_factory(trainer_type: str = "auto"):
    """
    Get the appropriate training args factory
    
    Args:
        trainer_type: "auto", "original", "enhanced", "patch_level"
        
    Returns:
        Training args factory function
    """
    if trainer_type == "auto":
        if create_training_args is None:
            raise ValueError("No training args factory available")
        return create_training_args
        
    elif trainer_type in ("original", "patch_level"):
        if not PATCH_TRAINER_AVAILABLE:
            raise ValueError("Original patch-level trainer not available")
        return create_blip3o_patch_training_args
        
    elif trainer_type == "enhanced":
        if not ENHANCED_TRAINER_AVAILABLE:
            raise ValueError("Enhanced trainer not available")
        return create_blip3o_enhanced_training_args
        
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def create_trainer(model, flow_matching_loss, trainer_type="auto", **kwargs):
    """
    Create a BLIP3-o trainer instance
    
    Args:
        model: BLIP3-o model
        flow_matching_loss: Flow matching loss function
        trainer_type: "auto", "original", "enhanced", "patch_level"
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer instance
    """
    trainer_class = get_trainer_class(trainer_type)
    
    return trainer_class(
        model=model,
        flow_matching_loss=flow_matching_loss,
        **kwargs
    )

def print_trainer_status():
    """
    Print status of available trainers
    """
    print("🏋️ BLIP3-o Trainers Status")
    print("=" * 40)
    print(f"Default trainer: {DEFAULT_TRAINER}")
    print()
    print("Available trainers:")
    
    if PATCH_TRAINER_AVAILABLE:
        print("  ✅ Original BLIP3-o Patch Trainer")
        print("    - 256-token patch-level training")
        print("    - Flow matching optimization")
        print("    - Basic recall evaluation")
        print("    - Standard hyperparameters")
        print("    - Multi-GPU distributed training")
        print("    - Paper-aligned training pipeline")
    else:
        print("  ❌ Original BLIP3-o Patch Trainer")
        
    if ENHANCED_TRAINER_AVAILABLE:
        print("  ✅ Enhanced BLIP3-o Patch Trainer (RECOMMENDED)")
        print("    - 256-token patch-level training")
        print("    - Enhanced flow matching optimization")
        print("    - Advanced convergence monitoring")
        print("    - Cosine LR scheduling with decay")
        print("    - Optimized hyperparameters")
        print("    - Enhanced logging and progress tracking")
        print("    - Pure training mode (no evaluation)")
        print("    - Superior convergence optimization")
    else:
        print("  ❌ Enhanced BLIP3-o Patch Trainer")
    
    if not any([PATCH_TRAINER_AVAILABLE, ENHANCED_TRAINER_AVAILABLE]):
        print("  ⚠️  No trainers available!")
        print("  💡 Make sure trainer files are properly implemented")
    
    print()
    print("Training features comparison:")
    print("  Original: Basic training with standard features")
    print("  Enhanced: Advanced convergence optimization")
    print("  📊 Objective: Patch-level flow matching")
    print("  📐 Input: EVA-CLIP patches [B, 256, 4096]")
    print("  🎯 Output: CLIP patches [B, 256, 1024]")
    print("  📊 Evaluation: Image-to-text recall")
    print("  🔄 Loss: Flow matching + contrastive")
    print("  💾 Memory: Optimized for multi-GPU")
    
    print("=" * 40)

def get_recommended_trainer():
    """
    Get the recommended trainer based on availability
    
    Returns:
        Recommended trainer class and type
    """
    if ENHANCED_TRAINER_AVAILABLE:
        return BLIP3oPatchTrainerEnhanced, "enhanced"
    elif PATCH_TRAINER_AVAILABLE:
        return BLIP3oPatchTrainer, "original"
    else:
        raise ValueError("No trainers available")

def create_enhanced_trainer_if_available(model, flow_matching_loss, **kwargs):
    """
    Create enhanced trainer if available, otherwise fallback to original
    
    Args:
        model: BLIP3-o model
        flow_matching_loss: Flow matching loss function
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer instance and trainer type used
    """
    if ENHANCED_TRAINER_AVAILABLE:
        trainer = BLIP3oPatchTrainerEnhanced(
            model=model,
            flow_matching_loss=flow_matching_loss,
            **kwargs
        )
        return trainer, "enhanced"
    elif PATCH_TRAINER_AVAILABLE:
        trainer = BLIP3oPatchTrainer(
            model=model,
            flow_matching_loss=flow_matching_loss,
            **kwargs
        )
        return trainer, "original"
    else:
        raise ValueError("No trainers available")

# Add utility functions to exports
__all__.extend([
    "get_trainer_class",
    "get_training_args_factory",
    "create_trainer",
    "print_trainer_status",
    "get_recommended_trainer",
    "create_enhanced_trainer_if_available",
])

# Enhanced status logging
trainer_status = []
if PATCH_TRAINER_AVAILABLE:
    trainer_status.append("original")
if ENHANCED_TRAINER_AVAILABLE:
    trainer_status.append("enhanced")

if trainer_status:
    logger.info(f"BLIP3-o trainers loaded successfully: {', '.join(trainer_status)}")
    if DEFAULT_TRAINER:
        logger.info(f"Default trainer: {DEFAULT_TRAINER}")
else:
    logger.warning("No BLIP3-o trainers available")

# Version compatibility message
if ENHANCED_TRAINER_AVAILABLE and PATCH_TRAINER_AVAILABLE:
    logger.info("Both trainer versions available - enhanced trainer recommended for new training")
elif PATCH_TRAINER_AVAILABLE:
    logger.info("Original trainer available - enhanced trainer can be added for better convergence")
elif ENHANCED_TRAINER_AVAILABLE:
    logger.info("Enhanced trainer available - original trainer as fallback")