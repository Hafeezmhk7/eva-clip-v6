"""
BLIP3-o Modules - Simplified
src/modules/__init__.py

Minimal imports for core functionality
"""

# Core availability flags
MODEL_AVAILABLE = False
LOSS_AVAILABLE = False
TRAINER_AVAILABLE = False
DATASET_AVAILABLE = False

# Try importing core components
try:
    from .models.blip3o_patch_dit import BLIP3oPatchDiTModel, BLIP3oDiTConfig, create_blip3o_patch_dit_model
    MODEL_AVAILABLE = True
except ImportError:
    pass

try:
    from .losses.blip3o_flow_matching_loss import BLIP3oFlowMatchingLoss, create_blip3o_flow_matching_loss
    LOSS_AVAILABLE = True
except ImportError:
    pass

try:
    from .trainers.blip3o_trainer import BLIP3oTrainer, create_training_args
    TRAINER_AVAILABLE = True
except ImportError:
    pass

try:
    from .datasets.blip3o_dataset import create_flexible_dataloaders
    DATASET_AVAILABLE = True
except ImportError:
    pass

# Export main components
__all__ = [
    "MODEL_AVAILABLE",
    "LOSS_AVAILABLE", 
    "TRAINER_AVAILABLE",
    "DATASET_AVAILABLE",
]

if MODEL_AVAILABLE:
    __all__.extend(["BLIP3oPatchDiTModel", "BLIP3oDiTConfig", "create_blip3o_patch_dit_model"])

if LOSS_AVAILABLE:
    __all__.extend(["BLIP3oFlowMatchingLoss", "create_blip3o_flow_matching_loss"])

if TRAINER_AVAILABLE:
    __all__.extend(["BLIP3oTrainer", "create_training_args"])

if DATASET_AVAILABLE:
    __all__.extend(["create_flexible_dataloaders"])