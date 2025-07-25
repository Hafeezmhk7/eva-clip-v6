"""
BLIP3-o Modules - Main Package
src/modules/__init__.py

Main entry point for all BLIP3-o modules
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flags
MODEL_AVAILABLE = False
LOSS_AVAILABLE = False
TRAINER_AVAILABLE = False
DATASET_AVAILABLE = False
CONFIG_AVAILABLE = False

# Try importing core components
try:
    from .models.blip3o_patch_dit import BLIP3oPatchDiTModel, BLIP3oDiTConfig, create_blip3o_patch_dit_model
    MODEL_AVAILABLE = True
    logger.info("✅ BLIP3-o model loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Model import failed: {e}")

try:
    from .losses.blip3o_flow_matching_loss import BLIP3oFlowMatchingLoss, create_blip3o_flow_matching_loss
    LOSS_AVAILABLE = True
    logger.info("✅ BLIP3-o loss loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Loss import failed: {e}")

try:
    from .trainers.blip3o_trainer import BLIP3oTrainer, create_training_args
    TRAINER_AVAILABLE = True
    logger.info("✅ BLIP3-o trainer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Trainer import failed: {e}")

try:
    from .datasets.blip3o_dataset import create_blip3o_dataloaders, BLIP3oEmbeddingDataset
    DATASET_AVAILABLE = True
    logger.info("✅ BLIP3-o dataset loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Dataset import failed: {e}")

try:
    from .config.blip3o_config import get_blip3o_config, create_config_from_args
    CONFIG_AVAILABLE = True
    logger.info("✅ BLIP3-o config loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Config import failed: {e}")

# Export main components
__all__ = [
    "MODEL_AVAILABLE",
    "LOSS_AVAILABLE", 
    "TRAINER_AVAILABLE",
    "DATASET_AVAILABLE",
    "CONFIG_AVAILABLE",
]

if MODEL_AVAILABLE:
    __all__.extend(["BLIP3oPatchDiTModel", "BLIP3oDiTConfig", "create_blip3o_patch_dit_model"])

if LOSS_AVAILABLE:
    __all__.extend(["BLIP3oFlowMatchingLoss", "create_blip3o_flow_matching_loss"])

if TRAINER_AVAILABLE:
    __all__.extend(["BLIP3oTrainer", "create_training_args"])

if DATASET_AVAILABLE:
    __all__.extend(["create_blip3o_dataloaders", "BLIP3oEmbeddingDataset"])

if CONFIG_AVAILABLE:
    __all__.extend(["get_blip3o_config", "create_config_from_args"])


def check_environment():
    """Check if all required components are available"""
    status = {
        'model': MODEL_AVAILABLE,
        'loss': LOSS_AVAILABLE,
        'trainer': TRAINER_AVAILABLE,
        'dataset': DATASET_AVAILABLE,
        'config': CONFIG_AVAILABLE,
    }
    
    all_available = all(status.values())
    
    if all_available:
        logger.info("🎉 All BLIP3-o components loaded successfully!")
    else:
        missing = [name for name, available in status.items() if not available]
        logger.warning(f"⚠️ Missing components: {missing}")
    
    return status, all_available


def get_version_info():
    """Get version and component information"""
    return {
        'blip3o_implementation': 'fixed_flow_matching_v1',
        'components': {
            'model': MODEL_AVAILABLE,
            'loss': LOSS_AVAILABLE,
            'trainer': TRAINER_AVAILABLE,
            'dataset': DATASET_AVAILABLE,
            'config': CONFIG_AVAILABLE,
        },
        'features': [
            'rectified_flow_matching',
            'patch_level_training',
            'evaluation_during_training',
            'proper_normalization',
            'blip3o_paper_aligned',
        ]
    }


# Initialize on import
_status, _all_available = check_environment()

if not _all_available:
    logger.warning("Some BLIP3-o components failed to load. Check individual imports.")
else:
    logger.info("BLIP3-o modules package ready for training and evaluation!")