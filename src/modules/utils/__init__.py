"""
Utility functions for BLIP3-o DiT - Enhanced Multi-GPU Support

Contains:
- System setup and environment utilities
- Multi-GPU patches and enhancements
- Temp directory management
- GPU diagnostics and debugging
- Training pipeline setup helpers
"""

import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Import temp manager
TEMP_MANAGER_AVAILABLE = False
try:
    from .temp_manager import (
        SnelliusTempManager,
        get_temp_manager,
        setup_snellius_environment,
    )
    logger.debug("✅ Temp manager loaded")
    TEMP_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Temp manager not available: {e}")
    SnelliusTempManager = None
    get_temp_manager = None
    setup_snellius_environment = None

# Import enhanced multi-GPU patches
ENHANCED_PATCHES_AVAILABLE = False
try:
    from ..multi_gpu_patches_enhanced import (
        apply_all_enhanced_patches,
        detect_gpu_environment,
        apply_gpu_fixes,
        enhanced_ddp_init,
        create_gpu_debug_report,
    )
    logger.debug("✅ Enhanced multi-GPU patches loaded")
    ENHANCED_PATCHES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Enhanced multi-GPU patches not available: {e}")
    apply_all_enhanced_patches = None
    detect_gpu_environment = None
    apply_gpu_fixes = None
    enhanced_ddp_init = None
    create_gpu_debug_report = None

# Build exports list
__all__ = [
    # Availability flags
    "TEMP_MANAGER_AVAILABLE",
    "ENHANCED_PATCHES_AVAILABLE",
]

# Export temp manager if available
if TEMP_MANAGER_AVAILABLE:
    __all__.extend([
        "SnelliusTempManager",
        "get_temp_manager", 
        "setup_snellius_environment",
    ])

# Export enhanced patches if available
if ENHANCED_PATCHES_AVAILABLE:
    __all__.extend([
        "apply_all_enhanced_patches",
        "detect_gpu_environment",
        "apply_gpu_fixes",
        "enhanced_ddp_init", 
        "create_gpu_debug_report",
    ])

def setup_blip3o_environment(
    project_name: str = "blip3o_workspace",
    apply_patches: bool = True,
    setup_temp_dirs: bool = True,
    enable_debug: bool = False
) -> Dict[str, Any]:
    """
    Complete BLIP3-o environment setup
    
    Args:
        project_name: Name for temp workspace
        apply_patches: Whether to apply enhanced multi-GPU patches
        setup_temp_dirs: Whether to setup temp directories
        enable_debug: Whether to enable debug logging
        
    Returns:
        Dictionary with setup results
    """
    results = {
        'temp_manager': None,
        'patches_applied': False,
        'gpu_info': None,
        'warnings': [],
        'errors': []
    }
    
    # Setup logging level
    if enable_debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("🐛 Debug logging enabled")
    
    # Setup temp directories
    if setup_temp_dirs and TEMP_MANAGER_AVAILABLE:
        try:
            results['temp_manager'] = setup_snellius_environment(project_name)
            logger.info("✅ Temp directories configured")
        except Exception as e:
            results['warnings'].append(f"Temp setup failed: {e}")
            logger.warning(f"⚠️ Temp setup failed: {e}")
    
    # Apply enhanced patches
    if apply_patches and ENHANCED_PATCHES_AVAILABLE:
        try:
            patch_results = apply_all_enhanced_patches()
            results['patches_applied'] = True
            results['gpu_info'] = patch_results.get('gpu_info')
            logger.info("✅ Enhanced multi-GPU patches applied")
        except Exception as e:
            results['errors'].append(f"Patch application failed: {e}")
            logger.error(f"❌ Patch application failed: {e}")
    
    # Create debug report
    if ENHANCED_PATCHES_AVAILABLE:
        try:
            debug_report = create_gpu_debug_report()
            logger.info("✅ GPU debug report created")
        except Exception as e:
            results['warnings'].append(f"Debug report failed: {e}")
    
    return results

def validate_training_setup(
    embeddings_dir: str,
    output_dir: str,
    check_gpu: bool = True,
    check_memory: bool = True
) -> Dict[str, Any]:
    """
    Validate complete training setup
    
    Args:
        embeddings_dir: Path to embeddings directory
        output_dir: Path to output directory  
        check_gpu: Whether to check GPU setup
        check_memory: Whether to check memory requirements
        
    Returns:
        Validation results
    """
    from pathlib import Path
    import json
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check embeddings
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        results['errors'].append(f"Embeddings directory not found: {embeddings_dir}")
        results['valid'] = False
    else:
        manifest_path = embeddings_path / "embeddings_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                logger.info(f"✅ Found embeddings: {manifest.get('total_samples', 0):,} samples")
            except Exception as e:
                results['warnings'].append(f"Could not read manifest: {e}")
        else:
            results['warnings'].append("Embeddings manifest not found")
    
    # Check output directory
    output_path = Path(output_dir) 
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Output directory ready: {output_dir}")
    except Exception as e:
        results['errors'].append(f"Cannot create output directory: {e}")
        results['valid'] = False
    
    # Check GPU setup
    if check_gpu and ENHANCED_PATCHES_AVAILABLE:
        try:
            gpu_info = detect_gpu_environment()
            if not gpu_info['cuda_available']:
                results['warnings'].append("CUDA not available")
            elif gpu_info['gpu_count'] == 0:
                results['warnings'].append("No GPUs detected")
            else:
                logger.info(f"✅ GPU setup: {gpu_info['gpu_count']} GPUs available")
                
            for issue in gpu_info['issues']:
                results['warnings'].append(f"GPU issue: {issue}")
                
        except Exception as e:
            results['warnings'].append(f"GPU check failed: {e}")
    
    # Memory recommendations
    if check_memory:
        import torch
        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    if memory_gb < 20:
                        results['warnings'].append(f"GPU {i} has limited memory: {memory_gb:.1f}GB")
                    logger.info(f"GPU {i}: {memory_gb:.1f}GB memory")
            except Exception as e:
                results['warnings'].append(f"Memory check failed: {e}")
    
    # Recommendations
    if results['warnings']:
        results['recommendations'].extend([
            "Check GPU allocation with 'nvidia-smi'",
            "Verify SLURM job has requested GPUs",
            "Consider reducing batch size if memory issues",
            "Use --cpu_fallback flag if GPU issues persist"
        ])
    
    return results

def create_training_script(
    embeddings_dir: str,
    output_dir: str,
    script_path: str = "run_training.py",
    **training_args
) -> str:
    """
    Create a complete training script
    
    Args:
        embeddings_dir: Path to embeddings
        output_dir: Output directory
        script_path: Path for generated script
        **training_args: Training arguments
        
    Returns:
        Path to created script
    """
    
    script_content = f'''#!/usr/bin/env python3
"""
Generated BLIP3-o Training Script
Auto-generated by BLIP3-o utils
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """Main training function"""
    
    # Setup enhanced environment
    from src.modules.utils import setup_blip3o_environment
    setup_results = setup_blip3o_environment(
        apply_patches=True,
        setup_temp_dirs=True
    )
    
    if setup_results['errors']:
        print("❌ Environment setup failed:")
        for error in setup_results['errors']:
            print(f"  {error}")
        return 1
    
    # Validate setup
    from src.modules.utils import validate_training_setup
    validation = validate_training_setup(
        embeddings_dir="{embeddings_dir}",
        output_dir="{output_dir}"
    )
    
    if not validation['valid']:
        print("❌ Training setup validation failed:")
        for error in validation['errors']:
            print(f"  {error}")
        return 1
    
    if validation['warnings']:
        print("⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"  {warning}")
    
    # Import training components
    from src.modules.models import create_blip3o_dit_model
    from src.modules.losses import get_loss_function
    from src.modules.trainers import get_trainer_class, get_training_args_factory
    from src.modules.datasets import create_dataloaders
    from src.modules.config import get_recommended_config
    
    # Create configuration
    model_config = get_recommended_config(
        model_type="auto",
        training_mode="enhanced"
    )
    
    # Create model
    model = create_blip3o_dit_model(config=model_config)
    
    # Create loss
    loss_fn = get_loss_function(loss_type="auto")
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(
        chunked_embeddings_dir="{embeddings_dir}",
        batch_size={training_args.get('batch_size', 8)},
        eval_batch_size={training_args.get('eval_batch_size', 4)}
    )
    
    # Create training args
    training_args_factory = get_training_args_factory("auto")
    training_args = training_args_factory(
        output_dir="{output_dir}",
        num_train_epochs={training_args.get('num_epochs', 5)},
        per_device_train_batch_size={training_args.get('batch_size', 8)},
        learning_rate={training_args.get('learning_rate', 1e-4)}
    )
    
    # Create trainer
    trainer_class = get_trainer_class("auto")
    trainer = trainer_class(
        model=model,
        args=training_args,
        flow_matching_loss=loss_fn,
        train_dataset=None,  # Will use custom dataloaders
        eval_dataset=None
    )
    
    # Override dataloaders
    trainer.get_train_dataloader = lambda: train_dataloader
    trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
    
    # Start training
    print("🚀 Starting enhanced BLIP3-o training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    print("✅ Training completed successfully!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"✅ Training script created: {script_path}")
    return script_path

def print_utils_status():
    """Print status of utility functions"""
    print("🔧 BLIP3-o Utils Status")
    print("=" * 25)
    
    if TEMP_MANAGER_AVAILABLE:
        print("  ✅ Temp Manager")
        print("    - Snellius environment setup")
        print("    - Structured directory management")
    else:
        print("  ❌ Temp Manager")
    
    if ENHANCED_PATCHES_AVAILABLE:
        print("  ✅ Enhanced Multi-GPU Patches")
        print("    - GPU environment detection")
        print("    - Automatic fixes")
        print("    - DDP enhancements")
    else:
        print("  ❌ Enhanced Multi-GPU Patches")
    
    print()
    print("Available functions:")
    print("  ✅ setup_blip3o_environment()")
    print("  ✅ validate_training_setup()")
    print("  ✅ create_training_script()")
    
    print("=" * 25)

# Add utility functions to exports
__all__.extend([
    "setup_blip3o_environment",
    "validate_training_setup", 
    "create_training_script",
    "print_utils_status",
])

# Log utils module status
logger.info("BLIP3-o utils loaded successfully")
if ENHANCED_PATCHES_AVAILABLE:
    logger.info("  ✅ Enhanced multi-GPU features available")
if TEMP_MANAGER_AVAILABLE:
    logger.info("  ✅ Temp management features available")