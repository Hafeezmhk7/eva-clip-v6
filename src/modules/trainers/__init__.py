"""
BLIP3-o Trainers Module
src/modules/trainers/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flag
TRAINER_AVAILABLE = False

try:
    from .blip3o_trainer import (
        BLIP3oTrainer,
        create_training_args,
    )
    TRAINER_AVAILABLE = True
    logger.info("✅ BLIP3-o trainer loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import trainer: {e}")
    BLIP3oTrainer = None
    create_training_args = None

# Main exports
__all__ = [
    "TRAINER_AVAILABLE",
]

if TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oTrainer",
        "create_training_args",
    ])