# src/modules/__init__.py
"""
BLIP3-o Modules - CLIP Reproduction from EVA Embeddings
Single comprehensive module init that handles all components
Updated for new file names: blip3o_datasets.py, blip3o_config.py, blip3o_fm_loss.py, blip3o_dit.py, blip3o_trainer.py
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import availability flags
MODEL_AVAILABLE = False
LOSS_AVAILABLE = False
TRAINER_AVAILABLE = False
DATASET_AVAILABLE = False
CONFIG_AVAILABLE = False

# Store imported components
_imported_components = {}

# =============================================================================
# MODEL IMPORTS (blip3o_dit.py)
# =============================================================================
try:
    from blip3o_dit import (
        BLIP3oCLIPDiTModel,
        BLIP3oCLIPDiTConfig, 
        create_clip_reproduction_model
    )
    MODEL_AVAILABLE = True
    _imported_components.update({
        'BLIP3oCLIPDiTModel': BLIP3oCLIPDiTModel,
        'BLIP3oCLIPDiTConfig': BLIP3oCLIPDiTConfig,
        'create_clip_reproduction_model': create_clip_reproduction_model,
    })
    logger.info("✅ CLIP DiT model loaded from blip3o_dit.py")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import from blip3o_dit.py: {e}")
    # Fallback: try with different path
    try:
        import blip3o_dit as dit_module
        BLIP3oCLIPDiTModel = getattr(dit_module, 'BLIP3oCLIPDiTModel', None)
        BLIP3oCLIPDiTConfig = getattr(dit_module, 'BLIP3oCLIPDiTConfig', None)
        create_clip_reproduction_model = getattr(dit_module, 'create_clip_reproduction_model', None)
        
        if all([BLIP3oCLIPDiTModel, BLIP3oCLIPDiTConfig, create_clip_reproduction_model]):
            MODEL_AVAILABLE = True
            _imported_components.update({
                'BLIP3oCLIPDiTModel': BLIP3oCLIPDiTModel,
                'BLIP3oCLIPDiTConfig': BLIP3oCLIPDiTConfig,
                'create_clip_reproduction_model': create_clip_reproduction_model,
            })
            logger.info("✅ CLIP DiT model loaded (fallback)")
        else:
            logger.error("❌ Could not load DiT model components")
    except Exception as fallback_e:
        logger.error(f"❌ Fallback import also failed: {fallback_e}")

# =============================================================================
# LOSS IMPORTS (blip3o_fm_loss.py)
# =============================================================================
try:
    from blip3o_fm_loss import (
        BLIP3oCLIPFlowMatchingLoss,
        create_clip_reproduction_loss
    )
    LOSS_AVAILABLE = True
    _imported_components.update({
        'BLIP3oCLIPFlowMatchingLoss': BLIP3oCLIPFlowMatchingLoss,
        'create_clip_reproduction_loss': create_clip_reproduction_loss,
    })
    logger.info("✅ Flow matching loss loaded from blip3o_fm_loss.py")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import from blip3o_fm_loss.py: {e}")
    try:
        import blip3o_fm_loss as loss_module
        BLIP3oCLIPFlowMatchingLoss = getattr(loss_module, 'BLIP3oCLIPFlowMatchingLoss', None)
        create_clip_reproduction_loss = getattr(loss_module, 'create_clip_reproduction_loss', None)
        
        if all([BLIP3oCLIPFlowMatchingLoss, create_clip_reproduction_loss]):
            LOSS_AVAILABLE = True
            _imported_components.update({
                'BLIP3oCLIPFlowMatchingLoss': BLIP3oCLIPFlowMatchingLoss,
                'create_clip_reproduction_loss': create_clip_reproduction_loss,
            })
            logger.info("✅ Flow matching loss loaded (fallback)")
        else:
            logger.error("❌ Could not load loss components")
    except Exception as fallback_e:
        logger.error(f"❌ Loss fallback import failed: {fallback_e}")

# =============================================================================
# TRAINER IMPORTS (blip3o_trainer.py) 
# =============================================================================
try:
    from blip3o_trainer import (
        BLIP3oCLIPTrainer,
        create_clip_trainer
    )
    TRAINER_AVAILABLE = True
    _imported_components.update({
        'BLIP3oCLIPTrainer': BLIP3oCLIPTrainer,
        'create_clip_trainer': create_clip_trainer,
    })
    logger.info("✅ CLIP trainer loaded from blip3o_trainer.py")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import from blip3o_trainer.py: {e}")
    try:
        import blip3o_trainer as trainer_module
        BLIP3oCLIPTrainer = getattr(trainer_module, 'BLIP3oCLIPTrainer', None)
        create_clip_trainer = getattr(trainer_module, 'create_clip_trainer', None)
        
        if all([BLIP3oCLIPTrainer, create_clip_trainer]):
            TRAINER_AVAILABLE = True
            _imported_components.update({
                'BLIP3oCLIPTrainer': BLIP3oCLIPTrainer,
                'create_clip_trainer': create_clip_trainer,
            })
            logger.info("✅ CLIP trainer loaded (fallback)")
        else:
            logger.error("❌ Could not load trainer components")
    except Exception as fallback_e:
        logger.error(f"❌ Trainer fallback import failed: {fallback_e}")

# =============================================================================
# DATASET IMPORTS (blip3o_datasets.py)
# =============================================================================
try:
    from blip3o_datasets import (
        create_clip_reproduction_dataloaders,
        BLIP3oCLIPReproductionDataset,
        clip_reproduction_collate_fn
    )
    DATASET_AVAILABLE = True
    _imported_components.update({
        'create_clip_reproduction_dataloaders': create_clip_reproduction_dataloaders,
        'BLIP3oCLIPReproductionDataset': BLIP3oCLIPReproductionDataset,
        'clip_reproduction_collate_fn': clip_reproduction_collate_fn,
    })
    logger.info("✅ CLIP datasets loaded from blip3o_datasets.py")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import from blip3o_datasets.py: {e}")
    try:
        import blip3o_datasets as dataset_module
        create_clip_reproduction_dataloaders = getattr(dataset_module, 'create_clip_reproduction_dataloaders', None)
        BLIP3oCLIPReproductionDataset = getattr(dataset_module, 'BLIP3oCLIPReproductionDataset', None)
        clip_reproduction_collate_fn = getattr(dataset_module, 'clip_reproduction_collate_fn', None)
        
        if all([create_clip_reproduction_dataloaders, BLIP3oCLIPReproductionDataset, clip_reproduction_collate_fn]):
            DATASET_AVAILABLE = True
            _imported_components.update({
                'create_clip_reproduction_dataloaders': create_clip_reproduction_dataloaders,
                'BLIP3oCLIPReproductionDataset': BLIP3oCLIPReproductionDataset,
                'clip_reproduction_collate_fn': clip_reproduction_collate_fn,
            })
            logger.info("✅ CLIP datasets loaded (fallback)")
        else:
            logger.error("❌ Could not load dataset components")
    except Exception as fallback_e:
        logger.error(f"❌ Dataset fallback import failed: {fallback_e}")

# =============================================================================
# CONFIG IMPORTS (blip3o_config.py)
# =============================================================================
try:
    from blip3o_config import (
        get_blip3o_clip_config,
        create_config_from_args,
        BLIP3oCLIPDiTConfig,
        print_config_summary,
        validate_blip3o_clip_architecture,
        FlowMatchingConfig,
        TrainingConfig,
        EvaluationConfig
    )
    CONFIG_AVAILABLE = True
    _imported_components.update({
        'get_blip3o_clip_config': get_blip3o_clip_config,
        'create_config_from_args': create_config_from_args,
        'BLIP3oCLIPDiTConfig': BLIP3oCLIPDiTConfig,
        'print_config_summary': print_config_summary,
        'validate_blip3o_clip_architecture': validate_blip3o_clip_architecture,
        'FlowMatchingConfig': FlowMatchingConfig,
        'TrainingConfig': TrainingConfig,
        'EvaluationConfig': EvaluationConfig,
    })
    logger.info("✅ Configuration loaded from blip3o_config.py")
except ImportError as e:
    logger.warning(f"⚠️ Failed to import from blip3o_config.py: {e}")
    try:
        import blip3o_config as config_module
        config_components = [
            'get_blip3o_clip_config', 'create_config_from_args', 'BLIP3oCLIPDiTConfig',
            'print_config_summary', 'validate_blip3o_clip_architecture',
            'FlowMatchingConfig', 'TrainingConfig', 'EvaluationConfig'
        ]
        
        config_imports = {}
        for comp in config_components:
            val = getattr(config_module, comp, None)
            if val is not None:
                config_imports[comp] = val
        
        if len(config_imports) >= 3:  # At least basic components
            CONFIG_AVAILABLE = True
            _imported_components.update(config_imports)
            logger.info(f"✅ Configuration loaded (fallback): {len(config_imports)} components")
        else:
            logger.error("❌ Could not load sufficient config components")
    except Exception as fallback_e:
        logger.error(f"❌ Config fallback import failed: {fallback_e}")

# =============================================================================
# EXPORT ALL COMPONENTS
# =============================================================================

# Main availability flags
__all__ = [
    # Availability flags
    "MODEL_AVAILABLE",
    "LOSS_AVAILABLE", 
    "TRAINER_AVAILABLE",
    "DATASET_AVAILABLE",
    "CONFIG_AVAILABLE",
]

# Add available components to exports
if MODEL_AVAILABLE:
    __all__.extend(["BLIP3oCLIPDiTModel", "BLIP3oCLIPDiTConfig", "create_clip_reproduction_model"])

if LOSS_AVAILABLE:
    __all__.extend(["BLIP3oCLIPFlowMatchingLoss", "create_clip_reproduction_loss"])

if TRAINER_AVAILABLE:
    __all__.extend(["BLIP3oCLIPTrainer", "create_clip_trainer"])

if DATASET_AVAILABLE:
    __all__.extend(["create_clip_reproduction_dataloaders", "BLIP3oCLIPReproductionDataset", "clip_reproduction_collate_fn"])

if CONFIG_AVAILABLE:
    __all__.extend([
        "get_blip3o_clip_config", "create_config_from_args", "BLIP3oCLIPDiTConfig",
        "print_config_summary", "validate_blip3o_clip_architecture",
        "FlowMatchingConfig", "TrainingConfig", "EvaluationConfig"
    ])

# Make imported components available at module level
locals().update(_imported_components)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
        logger.info("🎉 All CLIP reproduction components loaded successfully!")
    else:
        missing = [name for name, available in status.items() if not available]
        logger.warning(f"⚠️ Missing components: {missing}")
    
    return {
        'component_status': status,
        'all_available': all_available,
        'missing_components': [name for name, available in status.items() if not available],
        'available_components': [name for name, available in status.items() if available],
    }

def get_version_info():
    """Get version and component information"""
    return {
        'implementation': 'clip_reproduction_from_eva',
        'version': '1.0.0',
        'file_mapping': {
            'model': 'blip3o_dit.py',
            'loss': 'blip3o_fm_loss.py',
            'trainer': 'blip3o_trainer.py',
            'dataset': 'blip3o_datasets.py',
            'config': 'blip3o_config.py',
            'training_script': 'train_dit.py',
        },
        'component_status': {
            'model': MODEL_AVAILABLE,
            'loss': LOSS_AVAILABLE,
            'trainer': TRAINER_AVAILABLE,
            'dataset': DATASET_AVAILABLE,
            'config': CONFIG_AVAILABLE,
        },
        'features': [
            'eva_to_clip_reproduction',
            'minimal_normalization',
            'rectified_flow_matching',
            'blip3o_dit_architecture',
            '3d_rope_attention',
            'grouped_query_attention',
            'sandbox_normalization',
            'overfitting_test_capability',
        ],
        'task_description': {
            'input': 'EVA embeddings [B, N, 4096]',
            'output': 'CLIP embeddings [B, N, 1024]',
            'method': 'Rectified Flow Matching with BLIP3-o DiT',
            'normalization': 'Minimal (only for evaluation similarity)',
        }
    }

def get_all_components():
    """Get all available components as a dictionary"""
    return _imported_components.copy()

def create_full_pipeline(
    model_size: str = "base",
    training_mode: str = "patch_only",
    debug_mode: bool = False,
    **kwargs
):
    """Create a complete pipeline with all components"""
    
    if not all([MODEL_AVAILABLE, LOSS_AVAILABLE, TRAINER_AVAILABLE, DATASET_AVAILABLE, CONFIG_AVAILABLE]):
        missing = check_environment()['missing_components']
        raise ImportError(f"Cannot create pipeline. Missing components: {missing}")
    
    # Create model
    model = create_clip_reproduction_model(
        model_size=model_size,
        training_mode=training_mode,
        **kwargs
    )
    
    # Create loss function
    loss_fn = create_clip_reproduction_loss(
        prediction_type="velocity",
        flow_type="rectified",
        debug_mode=debug_mode
    )
    
    # Create config
    model_config = get_blip3o_clip_config(model_size, training_mode)
    
    return {
        'model': model,
        'loss_fn': loss_fn,
        'model_config': model_config,
        'create_dataloaders': create_clip_reproduction_dataloaders,
        'create_trainer': create_clip_trainer,
    }

def print_environment_status():
    """Print detailed environment status"""
    print("🔍 BLIP3-o CLIP Reproduction Environment Status")
    print("=" * 60)
    
    status = check_environment()
    version_info = get_version_info()
    
    print("📄 File Mapping:")
    for component, filename in version_info['file_mapping'].items():
        available = status['component_status'].get(component, False)
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component}: {filename}")
    
    print(f"\n📊 Component Status:")
    for component, available in status['component_status'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component.capitalize()}: {'Available' if available else 'Not Available'}")
    
    if status['all_available']:
        print(f"\n🎉 All components available! Ready for training.")
        print(f"🚫 Using minimal normalization approach")
        print(f"🎯 Task: EVA [4096] → CLIP [1024] reproduction")
    else:
        print(f"\n⚠️ Missing components: {', '.join(status['missing_components'])}")
        print(f"Available components: {', '.join(status['available_components'])}")
    
    print("=" * 60)

# =============================================================================
# INITIALIZATION
# =============================================================================

# Run environment check on import
_env_status = check_environment()

# Log initialization status
if _env_status['all_available']:
    logger.info("🎉 BLIP3-o CLIP reproduction modules fully initialized!")
    logger.info("🚫 Using minimal normalization approach")
    logger.info("🎯 Ready for EVA → CLIP reproduction training")
else:
    logger.warning(f"⚠️ Partial initialization. Missing: {_env_status['missing_components']}")

# Export environment status for external access
ENVIRONMENT_STATUS = _env_status

# Cleanup
del _env_status, _imported_components