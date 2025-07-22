"""
Training utilities for BLIP3-o DiT - Patch-Level Training (Paper-Aligned)

Contains:
- BLIP3oPatchTrainer: Primary trainer for patch-level flow matching
- Training argument creation utilities
- Paper-aligned training pipeline
"""

import logging

logger = logging.getLogger(__name__)

# Import patch-level trainer (required for paper alignment)
PATCH_TRAINER_AVAILABLE = False
try:
    from .blip3o_patch_trainer import (
        BLIP3oPatchTrainer,
        create_blip3o_patch_training_args,
    )
    logger.debug("✅ BLIP3-o patch-level trainer loaded")
    PATCH_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load BLIP3-o patch-level trainer: {e}")
    raise ImportError("BLIP3oPatchTrainer is required but not available")

# Set defaults (paper-aligned)
BLIP3oTrainer = BLIP3oPatchTrainer
create_training_args = create_blip3o_patch_training_args
DEFAULT_TRAINER = "patch_level"

logger.info("✅ Using BLIP3-o patch-level trainer as default")

__all__ = [
    # Primary trainer interface (paper-aligned)
    "BLIP3oTrainer",
    "create_training_args",
    "DEFAULT_TRAINER",
    
    # Patch-level trainer specific
    "BLIP3oPatchTrainer",
    "create_blip3o_patch_training_args",
    "PATCH_TRAINER_AVAILABLE",
    
    # Utility functions
    "get_trainer_class",
    "get_training_args_factory",
    "create_trainer",
    "print_trainer_status",
]

def get_trainer_class(trainer_type: str = "auto"):
    """
    Get the trainer class (always patch-level for paper alignment)
    """
    if trainer_type in ("auto", "patch_level", "patch"):
        if not PATCH_TRAINER_AVAILABLE:
            raise RuntimeError("Patch-level trainer not available")
        return BLIP3oPatchTrainer
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}. Use 'patch_level' for paper alignment.")

def get_training_args_factory(trainer_type: str = "auto"):
    """
    Get the training args factory (always patch-level for paper alignment)
    """
    if trainer_type in ("auto", "patch_level", "patch"):
        if not PATCH_TRAINER_AVAILABLE:
            raise RuntimeError("Patch-level trainer not available")
        return create_blip3o_patch_training_args
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}. Use 'patch_level' for paper alignment.")

def create_trainer(model, flow_matching_loss, **kwargs):
    """
    Create a BLIP3-o trainer instance (always patch-level for paper alignment)
    """
    if not PATCH_TRAINER_AVAILABLE:
        raise RuntimeError("Patch-level trainer not available")
    
    return BLIP3oPatchTrainer(
        model=model,
        flow_matching_loss=flow_matching_loss,
        **kwargs
    )

def print_trainer_status():
    """
    Print status of available trainers
    """
    print("🏋️ BLIP3-o Trainers Status")
    print("=" * 35)
    print(f"Default trainer: {DEFAULT_TRAINER}")
    print()
    print("Available trainer (Paper-Aligned):")
    
    if PATCH_TRAINER_AVAILABLE:
        print("  ✅ BLIP3-o Patch Trainer (Primary)")
        print("    - 256-token patch-level training")
        print("    - Flow matching optimization")
        print("    - Image-to-text recall evaluation")
        print("    - Enhanced error handling")
        print("    - Memory optimization")
        print("    - Multi-GPU distributed training")
        print("    - Paper-aligned training pipeline")
        print("    - Recall@K metrics tracking")
        print("    - Contrastive loss support")
    else:
        print("  ❌ BLIP3-o Patch Trainer (REQUIRED)")
    
    print()
    print("Training features:")
    print("  🎯 Objective: Patch-level flow matching")
    print("  📐 Input: EVA-CLIP patches [B, 256, 4096]")
    print("  🎯 Output: CLIP patches [B, 256, 1024]")
    print("  📊 Evaluation: Image-to-text recall")
    print("  🔄 Loss: Flow matching + contrastive")
    print("  💾 Memory: Optimized for multi-GPU")
    
    print("=" * 35)

# Ensure the trainer is available
if not PATCH_TRAINER_AVAILABLE:
    logger.error("❌ BLIP3-o patch-level trainer is required but not available!")
    raise ImportError("BLIP3-o patch-level trainer is required for this project")

logger.info("BLIP3-o patch-level trainer loaded successfully - Paper-aligned training")