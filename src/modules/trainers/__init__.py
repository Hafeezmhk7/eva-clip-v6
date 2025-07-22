"""
Training utilities for BLIP3-o DiT - Global Training Only

Contains:
- EnhancedBLIP3oTrainer: Primary trainer with enhanced error handling
- Training argument creation utility
"""

import logging

logger = logging.getLogger(__name__)

# Import enhanced trainer (required)
try:
    from .global_blip3o_trainer import (
        EnhancedBLIP3oTrainer,
        create_enhanced_training_args,
    )
    logger.debug("✅ Enhanced BLIP3-o trainer loaded")
except ImportError as e:
    logger.error(f"❌ Failed to load Enhanced BLIP3-o trainer: {e}")
    raise ImportError("EnhancedBLIP3oTrainer is required but not available")

# Set defaults
BLIP3oTrainer = EnhancedBLIP3oTrainer
create_training_args = create_enhanced_training_args
DEFAULT_TRAINER = "enhanced"

logger.info("✅ Using Enhanced BLIP3-o trainer as default")

__all__ = [
    "BLIP3oTrainer",
    "create_training_args",
    "DEFAULT_TRAINER",
    "EnhancedBLIP3oTrainer",
    "create_enhanced_training_args",
    "get_trainer_class",
    "get_training_args_factory",
    "create_trainer",
    "print_trainer_status",
]

def get_trainer_class(trainer_type: str = "auto"):
    """
    Get the trainer class (only 'enhanced' supported)
    """
    if trainer_type in ("auto", "enhanced"):
        return EnhancedBLIP3oTrainer
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")

def get_training_args_factory(trainer_type: str = "auto"):
    """
    Get the training args factory (only 'enhanced' supported)
    """
    if trainer_type in ("auto", "enhanced"):
        return create_enhanced_training_args
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")

def create_trainer(model, flow_matching_loss, **kwargs):
    """
    Create a BLIP3-o trainer instance
    """
    return EnhancedBLIP3oTrainer(
        model=model,
        flow_matching_loss=flow_matching_loss,
        **kwargs
    )

def print_trainer_status():
    """
    Print status of available trainers
    """
    print("🏋️ BLIP3-o Trainers Status")
    print("=" * 30)
    print(f"Default trainer: {DEFAULT_TRAINER}\n")
    print("Available trainers:")
    print("  ✅ Enhanced BLIP3-o Trainer (Primary)")
    print("    - Better error handling")
    print("    - Memory optimization")
    print("    - Enhanced debugging")
    print("=" * 30)
