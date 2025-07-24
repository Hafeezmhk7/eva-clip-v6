"""
Training utilities for BLIP3-o DiT - Enhanced Flexible Training
src/modules/trainers/__init__.py
Contains:
- BLIP3oFlexibleTrainer: Trainer with evaluation support
- BLIP3oTrainingOnlyTrainer: Trainer without evaluation
- Training argument creation utilities
"""

import logging
from functools import partial

logger = logging.getLogger(__name__)

# Import trainers
FLEXIBLE_TRAINER_AVAILABLE = False
TRAINING_ONLY_TRAINER_AVAILABLE = False

try:
    from .blip3o_flexible_trainer import (
        BLIP3oFlexibleTrainer,
        create_blip3o_flexible_training_args,
    )
    logger.debug("✅ BLIP3-o flexible trainer loaded")
    FLEXIBLE_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load flexible BLIP3-o trainer: {e}")

try:
    from .blip3o_training_only_trainer import (
        BLIP3oTrainingOnlyTrainer,
        create_training_only_args,
    )
    logger.debug("✅ BLIP3-o training-only trainer loaded")
    TRAINING_ONLY_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load training-only trainer: {e}")

# Set defaults based on availability
if FLEXIBLE_TRAINER_AVAILABLE and TRAINING_ONLY_TRAINER_AVAILABLE:
    DEFAULT_TRAINER = "both"
    logger.info("✅ Both trainers available")
elif FLEXIBLE_TRAINER_AVAILABLE:
    DEFAULT_TRAINER = "flexible"
    logger.info("✅ Using flexible trainer as default")
elif TRAINING_ONLY_TRAINER_AVAILABLE:
    DEFAULT_TRAINER = "training_only"
    logger.info("✅ Using training-only trainer as default")
else:
    DEFAULT_TRAINER = None
    logger.error("❌ No BLIP3-o trainers available")

__all__ = [
    # Availability flags
    "FLEXIBLE_TRAINER_AVAILABLE",
    "TRAINING_ONLY_TRAINER_AVAILABLE",
    "DEFAULT_TRAINER",
]

# Export flexible trainer if available
if FLEXIBLE_TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oFlexibleTrainer",
        "create_blip3o_flexible_training_args",
    ])

# Export training-only trainer if available
if TRAINING_ONLY_TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oTrainingOnlyTrainer",
        "create_training_only_args",
    ])

def get_trainer_class(trainer_type: str = "auto"):
    """Get trainer class based on type"""
    if trainer_type == "auto":
        if DEFAULT_TRAINER == "flexible":
            return BLIP3oFlexibleTrainer
        elif DEFAULT_TRAINER == "training_only":
            return BLIP3oTrainingOnlyTrainer
        else:
            raise ValueError("No default trainer available")
    
    elif trainer_type == "flexible":
        if not FLEXIBLE_TRAINER_AVAILABLE:
            raise ValueError("Flexible trainer not available")
        return BLIP3oFlexibleTrainer
    
    elif trainer_type == "training_only":
        if not TRAINING_ONLY_TRAINER_AVAILABLE:
            raise ValueError("Training-only trainer not available")
        return BLIP3oTrainingOnlyTrainer
    
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def get_training_args_factory(trainer_type: str = "auto"):
    """Get training args factory based on type"""
    if trainer_type == "auto":
        if DEFAULT_TRAINER == "flexible":
            return create_blip3o_flexible_training_args
        elif DEFAULT_TRAINER == "training_only":
            return create_training_only_args
        else:
            raise ValueError("No default args factory available")
    
    elif trainer_type == "flexible":
        if not FLEXIBLE_TRAINER_AVAILABLE:
            raise ValueError("Flexible trainer not available")
        return create_blip3o_flexible_training_args
    
    elif trainer_type == "training_only":
        if not TRAINING_ONLY_TRAINER_AVAILABLE:
            raise ValueError("Training-only trainer not available")
        return create_training_only_args
    
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def create_trainer(model, flow_matching_loss, trainer_type="auto", **kwargs):
    """Create trainer instance"""
    trainer_class = get_trainer_class(trainer_type)
    return trainer_class(
        model=model,
        flow_matching_loss=flow_matching_loss,
        **kwargs
    )

def print_trainer_status():
    """Print status of available trainers"""
    print("🏋️ BLIP3-o Trainers Status")
    print("=" * 40)
    print(f"Default trainer: {DEFAULT_TRAINER}")
    
    print("\nAvailable trainers:")
    if FLEXIBLE_TRAINER_AVAILABLE:
        print("  ✅ Flexible BLIP3-o Trainer")
        print("    - Supports evaluation during training")
        print("    - Both 256/257 token modes")
    else:
        print("  ❌ Flexible BLIP3-o Trainer")
    
    if TRAINING_ONLY_TRAINER_AVAILABLE:
        print("  ✅ Training-Only BLIP3-o Trainer")
        print("    - Training only (no evaluation)")
        print("    - Reports loss, learning rate, training metrics")
    else:
        print("  ❌ Training-Only BLIP3-o Trainer")
    
    print("\nTraining features:")
    print("  📊 Objective: Patch-level flow matching")
    print("  📐 Input: EVA-CLIP patches [B, N, 4096]")
    print("  🎯 Output: CLIP patches [B, N, 1024]")
    print("  🔄 Loss: Pure flow matching (BLIP3-o paper)")
    print("=" * 40)

# Enhanced status logging
if FLEXIBLE_TRAINER_AVAILABLE or TRAINING_ONLY_TRAINER_AVAILABLE:
    logger.info("BLIP3-o trainers loaded successfully")
    if DEFAULT_TRAINER:
        logger.info(f"Default trainer: {DEFAULT_TRAINER}")
else:
    logger.warning("No BLIP3-o trainers available")

logger.info("BLIP3-o trainer initialization complete")