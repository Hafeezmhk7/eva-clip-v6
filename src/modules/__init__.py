# src/modules/__init__.py
"""
BLIP3-o Modules - Updated for EVA Reproduction Testing
src/modules/__init__.py

Main entry point for all BLIP3-o modules including EVA reproduction components
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flags for original components
MODEL_AVAILABLE = False
LOSS_AVAILABLE = False
TRAINER_AVAILABLE = False
DATASET_AVAILABLE = False
CONFIG_AVAILABLE = False

# Import availability flags for EVA reproduction components
EVA_MODEL_AVAILABLE = False
EVA_LOSS_AVAILABLE = False
EVA_TRAINER_AVAILABLE = False
EVA_DATASET_AVAILABLE = False

# Try importing original core components
try:
    from .models.blip3o_patch_dit import BLIP3oPatchDiTModel, BLIP3oDiTConfig, create_blip3o_patch_dit_model
    MODEL_AVAILABLE = True
    logger.info("✅ Original BLIP3-o model loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Original model import failed: {e}")

try:
    from .losses.blip3o_flow_matching_loss import BLIP3oFlowMatchingLoss, create_blip3o_flow_matching_loss
    LOSS_AVAILABLE = True
    logger.info("✅ Original BLIP3-o loss loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Original loss import failed: {e}")

try:
    from .trainers.blip3o_trainer import BLIP3oTrainer, create_training_args
    TRAINER_AVAILABLE = True
    logger.info("✅ Original BLIP3-o trainer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Original trainer import failed: {e}")

try:
    from .datasets.blip3o_dataset import create_blip3o_dataloaders, BLIP3oEmbeddingDataset
    DATASET_AVAILABLE = True
    logger.info("✅ Original BLIP3-o dataset loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Original dataset import failed: {e}")

try:
    from .config.blip3o_config import get_blip3o_config, create_config_from_args
    CONFIG_AVAILABLE = True
    logger.info("✅ Original BLIP3-o config loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Original config import failed: {e}")

# Try importing EVA reproduction components
try:
    from .models.blip3o_eva_dit import BLIP3oEVADiTModel, BLIP3oEVADiTConfig, create_eva_reproduction_model
    EVA_MODEL_AVAILABLE = True
    logger.info("✅ EVA reproduction model loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ EVA model import failed: {e}")

try:
    from .losses.blip3o_eva_loss import BLIP3oEVAFlowMatchingLoss, create_eva_reproduction_loss
    EVA_LOSS_AVAILABLE = True
    logger.info("✅ EVA reproduction loss loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ EVA loss import failed: {e}")

try:
    from .trainers.blip3o_eva_trainer import BLIP3oEVATrainer, create_eva_training_args
    EVA_TRAINER_AVAILABLE = True
    logger.info("✅ EVA reproduction trainer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ EVA trainer import failed: {e}")

try:
    from .datasets.blip3o_eva_dataset import create_eva_reproduction_dataloaders, BLIP3oEVAReproductionDataset
    EVA_DATASET_AVAILABLE = True
    logger.info("✅ EVA reproduction dataset loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ EVA dataset import failed: {e}")

# Export main components
__all__ = [
    # Availability flags
    "MODEL_AVAILABLE",
    "LOSS_AVAILABLE", 
    "TRAINER_AVAILABLE",
    "DATASET_AVAILABLE",
    "CONFIG_AVAILABLE",
    "EVA_MODEL_AVAILABLE",
    "EVA_LOSS_AVAILABLE",
    "EVA_TRAINER_AVAILABLE",
    "EVA_DATASET_AVAILABLE",
]

# Original components
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

# EVA reproduction components
if EVA_MODEL_AVAILABLE:
    __all__.extend(["BLIP3oEVADiTModel", "BLIP3oEVADiTConfig", "create_eva_reproduction_model"])

if EVA_LOSS_AVAILABLE:
    __all__.extend(["BLIP3oEVAFlowMatchingLoss", "create_eva_reproduction_loss"])

if EVA_TRAINER_AVAILABLE:
    __all__.extend(["BLIP3oEVATrainer", "create_eva_training_args"])

if EVA_DATASET_AVAILABLE:
    __all__.extend(["create_eva_reproduction_dataloaders", "BLIP3oEVAReproductionDataset"])


def check_environment():
    """Check if all required components are available"""
    original_status = {
        'model': MODEL_AVAILABLE,
        'loss': LOSS_AVAILABLE,
        'trainer': TRAINER_AVAILABLE,
        'dataset': DATASET_AVAILABLE,
        'config': CONFIG_AVAILABLE,
    }
    
    eva_status = {
        'eva_model': EVA_MODEL_AVAILABLE,
        'eva_loss': EVA_LOSS_AVAILABLE,
        'eva_trainer': EVA_TRAINER_AVAILABLE,
        'eva_dataset': EVA_DATASET_AVAILABLE,
    }
    
    all_original_available = all(original_status.values())
    all_eva_available = all(eva_status.values())
    
    if all_original_available:
        logger.info("🎉 All original BLIP3-o components loaded successfully!")
    else:
        missing = [name for name, available in original_status.items() if not available]
        logger.warning(f"⚠️ Missing original components: {missing}")
    
    if all_eva_available:
        logger.info("🎉 All EVA reproduction components loaded successfully!")
    else:
        missing = [name for name, available in eva_status.items() if not available]
        logger.warning(f"⚠️ Missing EVA components: {missing}")
    
    return {
        'original': original_status,
        'eva_reproduction': eva_status,
        'all_original_available': all_original_available,
        'all_eva_available': all_eva_available,
    }


def get_version_info():
    """Get version and component information"""
    return {
        'blip3o_implementation': 'eva_reproduction_test_v1',
        'original_components': {
            'model': MODEL_AVAILABLE,
            'loss': LOSS_AVAILABLE,
            'trainer': TRAINER_AVAILABLE,
            'dataset': DATASET_AVAILABLE,
            'config': CONFIG_AVAILABLE,
        },
        'eva_reproduction_components': {
            'eva_model': EVA_MODEL_AVAILABLE,
            'eva_loss': EVA_LOSS_AVAILABLE,
            'eva_trainer': EVA_TRAINER_AVAILABLE,
            'eva_dataset': EVA_DATASET_AVAILABLE,
        },
        'features': [
            'rectified_flow_matching',
            'patch_level_training',
            'evaluation_during_training',
            'proper_normalization',
            'blip3o_paper_aligned',
            'eva_reproduction_testing',
            'dit_architecture_validation',
        ]
    }


# Initialize on import
_status = check_environment()

if not _status['all_original_available']:
    logger.warning("Some original BLIP3-o components failed to load. Check individual imports.")

if not _status['all_eva_available']:
    logger.warning("Some EVA reproduction components failed to load. Check individual imports.")

if _status['all_original_available'] and _status['all_eva_available']:
    logger.info("🎉 BLIP3-o modules package ready for both original training and EVA reproduction testing!")
elif _status['all_original_available']:
    logger.info("✅ BLIP3-o modules package ready for original training!")
elif _status['all_eva_available']:
    logger.info("✅ BLIP3-o modules package ready for EVA reproduction testing!")
else:
    logger.warning("⚠️ Some components missing. Check imports before proceeding.")


# =============================================================================
# src/modules/models/__init__.py
"""
BLIP3-o Models Module - Updated for EVA Reproduction
src/modules/models/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flags
MODEL_AVAILABLE = False
EVA_MODEL_AVAILABLE = False

# Original model
try:
    from .blip3o_patch_dit import (
        BLIP3oPatchDiTModel,
        BLIP3oDiTConfig,
        create_blip3o_patch_dit_model,
    )
    MODEL_AVAILABLE = True
    logger.info("✅ Original BLIP3-o patch-level DiT model loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import original model: {e}")
    BLIP3oPatchDiTModel = None
    BLIP3oDiTConfig = None
    create_blip3o_patch_dit_model = None

# EVA reproduction model
try:
    from .blip3o_eva_dit import (
        BLIP3oEVADiTModel,
        BLIP3oEVADiTConfig,
        create_eva_reproduction_model,
    )
    EVA_MODEL_AVAILABLE = True
    logger.info("✅ EVA reproduction DiT model loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import EVA model: {e}")
    BLIP3oEVADiTModel = None
    BLIP3oEVADiTConfig = None
    create_eva_reproduction_model = None

# Main exports
__all__ = [
    "MODEL_AVAILABLE",
    "EVA_MODEL_AVAILABLE",
]

if MODEL_AVAILABLE:
    __all__.extend([
        "BLIP3oPatchDiTModel",
        "BLIP3oDiTConfig", 
        "create_blip3o_patch_dit_model",
    ])

if EVA_MODEL_AVAILABLE:
    __all__.extend([
        "BLIP3oEVADiTModel",
        "BLIP3oEVADiTConfig",
        "create_eva_reproduction_model",
    ])


# =============================================================================
# src/modules/losses/__init__.py
"""
BLIP3-o Losses Module - Updated for EVA Reproduction
src/modules/losses/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flags
LOSS_AVAILABLE = False
EVA_LOSS_AVAILABLE = False

# Original loss
try:
    from .blip3o_flow_matching_loss import (
        BLIP3oFlowMatchingLoss,
        create_blip3o_flow_matching_loss,
    )
    LOSS_AVAILABLE = True
    logger.info("✅ Original BLIP3-o flow matching loss loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import original loss: {e}")
    BLIP3oFlowMatchingLoss = None
    create_blip3o_flow_matching_loss = None

# EVA reproduction loss
try:
    from .blip3o_eva_loss import (
        BLIP3oEVAFlowMatchingLoss,
        create_eva_reproduction_loss,
    )
    EVA_LOSS_AVAILABLE = True
    logger.info("✅ EVA reproduction flow matching loss loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import EVA loss: {e}")
    BLIP3oEVAFlowMatchingLoss = None
    create_eva_reproduction_loss = None

# Main exports
__all__ = [
    "LOSS_AVAILABLE",
    "EVA_LOSS_AVAILABLE",
]

if LOSS_AVAILABLE:
    __all__.extend([
        "BLIP3oFlowMatchingLoss",
        "create_blip3o_flow_matching_loss",
    ])

if EVA_LOSS_AVAILABLE:
    __all__.extend([
        "BLIP3oEVAFlowMatchingLoss",
        "create_eva_reproduction_loss",
    ])


# =============================================================================
# src/modules/trainers/__init__.py
"""
BLIP3-o Trainers Module - Updated for EVA Reproduction
src/modules/trainers/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flags
TRAINER_AVAILABLE = False
EVA_TRAINER_AVAILABLE = False

# Original trainer
try:
    from .blip3o_trainer import (
        BLIP3oTrainer,
        create_training_args,
    )
    TRAINER_AVAILABLE = True
    logger.info("✅ Original BLIP3-o trainer loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import original trainer: {e}")
    BLIP3oTrainer = None
    create_training_args = None

# EVA reproduction trainer
try:
    from .blip3o_eva_trainer import (
        BLIP3oEVATrainer,
        create_eva_training_args,
    )
    EVA_TRAINER_AVAILABLE = True
    logger.info("✅ EVA reproduction trainer loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import EVA trainer: {e}")
    BLIP3oEVATrainer = None
    create_eva_training_args = None

# Main exports
__all__ = [
    "TRAINER_AVAILABLE",
    "EVA_TRAINER_AVAILABLE",
]

if TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oTrainer",
        "create_training_args",
    ])

if EVA_TRAINER_AVAILABLE:
    __all__.extend([
        "BLIP3oEVATrainer",
        "create_eva_training_args",
    ])


# =============================================================================
# src/modules/datasets/__init__.py
"""
BLIP3-o Datasets Module - Updated for EVA Reproduction
src/modules/datasets/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flags
DATASET_AVAILABLE = False
EVA_DATASET_AVAILABLE = False

# Original dataset
try:
    from .blip3o_dataset import (
        create_blip3o_dataloaders,
        BLIP3oEmbeddingDataset,
        blip3o_collate_fn_fixed,
    )
    DATASET_AVAILABLE = True
    logger.info("✅ Original BLIP3-o dataset loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import original dataset: {e}")
    create_blip3o_dataloaders = None
    BLIP3oEmbeddingDataset = None
    blip3o_collate_fn_fixed = None

# EVA reproduction dataset
try:
    from .blip3o_eva_dataset import (
        create_eva_reproduction_dataloaders,
        BLIP3oEVAReproductionDataset,
        blip3o_eva_collate_fn,
    )
    EVA_DATASET_AVAILABLE = True
    logger.info("✅ EVA reproduction dataset loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import EVA dataset: {e}")
    create_eva_reproduction_dataloaders = None
    BLIP3oEVAReproductionDataset = None
    blip3o_eva_collate_fn = None

# Main exports
__all__ = [
    "DATASET_AVAILABLE",
    "EVA_DATASET_AVAILABLE",
]

if DATASET_AVAILABLE:
    __all__.extend([
        "create_blip3o_dataloaders",
        "BLIP3oEmbeddingDataset",
        "blip3o_collate_fn_fixed",
    ])

if EVA_DATASET_AVAILABLE:
    __all__.extend([
        "create_eva_reproduction_dataloaders",
        "BLIP3oEVAReproductionDataset", 
        "blip3o_eva_collate_fn",
    ])