"""
src/modules/__init__.py
Top-level module initialization
"""

# Package version
__version__ = "1.0.0"

# Import key components for easier access
from .models.blip3o_patch_dit import BLIP3oPatchDiTModel, create_blip3o_patch_dit_model
from .losses.blip3o_flow_matching_loss import BLIP3oFlowMatchingLoss, create_blip3o_flow_matching_loss
from .trainers import (
    BLIP3oFlexibleTrainer, 
    create_blip3o_flexible_training_args,
    BLIP3oTrainingOnlyTrainer,
    create_training_only_args
)

# Log initialization
import logging
logger = logging.getLogger(__name__)
logger.info(f"BLIP3-o Enhanced Training Module v{__version__} initialized")