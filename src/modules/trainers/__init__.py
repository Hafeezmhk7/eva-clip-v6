"""
Training utilities for BLIP3-o DiT - Updated for Dual Supervision
"""

from .blip3o_trainer import (
    BLIP3oTrainer as StandardBLIP3oTrainer,
    create_blip3o_training_args as create_standard_training_args,
)

# Import dual supervision trainer
try:
    from .dual_supervision_blip3o_trainer import (
        DualSupervisionBLIP3oTrainer,
        create_blip3o_training_args as create_dual_supervision_training_args,
    )
    # Use dual supervision as default
    BLIP3oTrainer = DualSupervisionBLIP3oTrainer
    create_blip3o_training_args = create_dual_supervision_training_args
    DUAL_SUPERVISION_TRAINER_AVAILABLE = True
except ImportError:
    # Use standard trainer as fallback
    BLIP3oTrainer = StandardBLIP3oTrainer
    create_blip3o_training_args = create_standard_training_args
    DUAL_SUPERVISION_TRAINER_AVAILABLE = False
    print("⚠️ Using standard trainer")

__all__ = [
    "BLIP3oTrainer",
    "create_blip3o_training_args",
]

if DUAL_SUPERVISION_TRAINER_AVAILABLE:
    __all__.extend([
        "DualSupervisionBLIP3oTrainer",
    ])