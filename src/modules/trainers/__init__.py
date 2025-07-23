"""
Training utilities for BLIP3-o DiT - Enhanced Flexible Training
src/modules/trainers/__init__.py
Contains:
- BLIP3oFlexibleTrainer: Enhanced trainer with CLS+Patch support
- Training argument creation utilities
- Paper-aligned training pipeline
"""

import logging

logger = logging.getLogger(__name__)

# Import flexible trainer
FLEXIBLE_TRAINER_AVAILABLE = False
try:
    from .blip3o_flexible_trainer import (
        BLIP3oFlexibleTrainer,
        create_blip3o_flexible_training_args,
    )
    logger.debug("✅ BLIP3-o flexible trainer loaded")
    FLEXIBLE_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load flexible BLIP3-o trainer: {e}")
    FLEXIBLE_TRAINER_AVAILABLE = False

# Set defaults based on availability
if FLEXIBLE_TRAINER_AVAILABLE:
    BLIP3oTrainer = BLIP3oFlexibleTrainer
    create_training_args = create_blip3o_flexible_training_args
    DEFAULT_TRAINER = "flexible"
    logger.info("✅ Using flexible trainer as default")
else:
    BLIP3oTrainer = None
    create_training_args = None
    DEFAULT_TRAINER = None
    logger.error("❌ No BLIP3-o trainers available")

__all__ = [
    # Availability flags
    "FLEXIBLE_TRAINER_AVAILABLE",
    "DEFAULT_TRAINER",
]

# Export flexible trainer components if available
if FLEXIBLE_TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oFlexibleTrainer",
        "create_blip3o_flexible_training_args",
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
        trainer_type: "auto" or "flexible"
        
    Returns:
        Trainer class
    """
    if trainer_type == "auto":
        if BLIP3oTrainer is None:
            raise ValueError("No trainer class available")
        return BLIP3oTrainer
        
    elif trainer_type == "flexible":
        if not FLEXIBLE_TRAINER_AVAILABLE:
            raise ValueError("Flexible trainer not available")
        return BLIP3oFlexibleTrainer
        
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def get_training_args_factory(trainer_type: str = "auto"):
    """
    Get the appropriate training args factory
    
    Args:
        trainer_type: "auto" or "flexible"
        
    Returns:
        Training args factory function
    """
    if trainer_type == "auto":
        if create_training_args is None:
            raise ValueError("No training args factory available")
        return create_training_args
        
    elif trainer_type == "flexible":
        if not FLEXIBLE_TRAINER_AVAILABLE:
            raise ValueError("Flexible trainer not available")
        return create_blip3o_flexible_training_args
        
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def create_trainer(model, flow_matching_loss, trainer_type="auto", **kwargs):
    """
    Create a BLIP3-o trainer instance
    
    Args:
        model: BLIP3-o model
        flow_matching_loss: Flow matching loss function
        trainer_type: "auto" or "flexible"
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
    
    if FLEXIBLE_TRAINER_AVAILABLE:
        print("  ✅ Flexible BLIP3-o Trainer")
        print("    - Both 256 (patch-only) and 257 (CLS+patch) token modes")
        print("    - Flexible shard selection for training")
        print("    - Same-data evaluation (overfitting tests)")
        print("    - Pure flow matching loss (BLIP3-o paper aligned)")
        print("    - Detailed training metrics and progress tracking")
        print("    - Multi-GPU distributed training")
        print("    - Enhanced logging and evaluation")
    else:
        print("  ❌ Flexible BLIP3-o Trainer")
    
    if not FLEXIBLE_TRAINER_AVAILABLE:
        print("  ⚠️  No trainers available!")
        print("  💡 Make sure trainer files are properly implemented")
    
    print()
    print("Training features:")
    print("  📊 Objective: Patch-level flow matching")
    print("  📐 Input: EVA-CLIP patches [B, N, 4096] (N=256 or 257)")
    print("  🎯 Output: CLIP patches [B, N, 1024] (N=256 or 257)")
    print("  📊 Evaluation: Image-to-text recall")
    print("  🔄 Loss: Pure flow matching (BLIP3-o paper)")
    print("  💾 Memory: Optimized for multi-GPU")
    print("  🧪 Testing: Same-data evaluation for overfitting")
    
    print("=" * 40)

def get_recommended_trainer():
    """
    Get the recommended trainer based on availability
    
    Returns:
        Recommended trainer class and type
    """
    if FLEXIBLE_TRAINER_AVAILABLE:
        return BLIP3oFlexibleTrainer, "flexible"
    else:
        raise ValueError("No trainers available")

# Add utility functions to exports
__all__.extend([
    "get_trainer_class",
    "get_training_args_factory",
    "create_trainer",
    "print_trainer_status",
    "get_recommended_trainer",
])

# Enhanced status logging
trainer_status = []
if FLEXIBLE_TRAINER_AVAILABLE:
    trainer_status.append("flexible")

if trainer_status:
    logger.info(f"BLIP3-o trainers loaded successfully: {', '.join(trainer_status)}")
    if DEFAULT_TRAINER:
        logger.info(f"Default trainer: {DEFAULT_TRAINER}")
else:
    logger.warning("No BLIP3-o trainers available")

# Ensure flexible trainer is available
if not FLEXIBLE_TRAINER_AVAILABLE:
    logger.error("❌ BLIP3-o flexible trainer is required but not available!")
    raise ImportError("BLIP3-o flexible trainer is required for this project")

logger.info("BLIP3-o flexible trainer loaded successfully - Enhanced features available")