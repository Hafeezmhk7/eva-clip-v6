"""
Training utilities for BLIP3-o DiT - FIXED for Dual Supervision
"""

from .blip3o_trainer import (
    BLIP3oTrainer as StandardBLIP3oTrainer,
    create_blip3o_training_args as create_standard_training_args,
)

# Import dual supervision trainer with better error handling
DUAL_SUPERVISION_TRAINER_AVAILABLE = False
try:
    from .dual_supervision_blip3o_trainer import (
        DualSupervisionBLIP3oTrainer,
        create_blip3o_training_args as create_dual_supervision_training_args,
    )
    # Use dual supervision as default
    BLIP3oTrainer = DualSupervisionBLIP3oTrainer
    create_blip3o_training_args = create_dual_supervision_training_args
    DUAL_SUPERVISION_TRAINER_AVAILABLE = True
    print("✅ Dual supervision trainer loaded successfully")
    
except ImportError as e:
    # Use standard trainer as fallback
    BLIP3oTrainer = StandardBLIP3oTrainer
    create_blip3o_training_args = create_standard_training_args
    DUAL_SUPERVISION_TRAINER_AVAILABLE = False
    print(f"⚠️ Dual supervision trainer import failed: {e}")
    print("⚠️ Using standard trainer as fallback")
    
except Exception as e:
    # Handle other errors
    BLIP3oTrainer = StandardBLIP3oTrainer
    create_blip3o_training_args = create_standard_training_args
    DUAL_SUPERVISION_TRAINER_AVAILABLE = False
    print(f"⚠️ Unexpected error loading dual supervision trainer: {e}")
    print("⚠️ Using standard trainer as fallback")

__all__ = [
    "BLIP3oTrainer",
    "create_blip3o_training_args",
    "DUAL_SUPERVISION_TRAINER_AVAILABLE",
]

if DUAL_SUPERVISION_TRAINER_AVAILABLE:
    __all__.extend([
        "DualSupervisionBLIP3oTrainer",
        "StandardBLIP3oTrainer",
        "create_standard_training_args",
        "create_dual_supervision_training_args",
    ])