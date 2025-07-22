"""
Training utilities for BLIP3-o DiT - Global Training Only

Contains:
- GlobalBLIP3oTrainer: Global feature trainer (primary)
- EnhancedBLIP3oTrainer: Enhanced trainer with better error handling
- Training argument creation utilities for global training
"""

import logging

logger = logging.getLogger(__name__)

# Import global trainer (your main trainer)
GLOBAL_TRAINER_AVAILABLE = False
try:
    from .global_blip3o_trainer import (
        GlobalBLIP3oTrainer,
        create_global_training_args,
    )
    logger.debug("✅ Global BLIP3-o trainer loaded")
    GLOBAL_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load global trainer: {e}")
    GLOBAL_TRAINER_AVAILABLE = False
    GlobalBLIP3oTrainer = None
    create_global_training_args = None

# Import enhanced trainer (optional enhancement)
ENHANCED_TRAINER_AVAILABLE = False
try:
    from .blip3o_trainer_enhanced import (
        EnhancedBLIP3oTrainer,
        create_enhanced_training_args,
    )
    logger.debug("✅ Enhanced BLIP3-o trainer loaded")
    ENHANCED_TRAINER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Enhanced trainer not available: {e}")
    EnhancedBLIP3oTrainer = None
    create_enhanced_training_args = None

# Determine the best trainer (prefer enhanced if available, otherwise global)
if ENHANCED_TRAINER_AVAILABLE and GLOBAL_TRAINER_AVAILABLE:
    # Use enhanced trainer as default
    BLIP3oTrainer = EnhancedBLIP3oTrainer
    create_training_args = create_enhanced_training_args
    DEFAULT_TRAINER = "enhanced"
    logger.info("✅ Using Enhanced BLIP3-o trainer as default")
elif GLOBAL_TRAINER_AVAILABLE:
    # Use global trainer
    BLIP3oTrainer = GlobalBLIP3oTrainer
    create_training_args = create_global_training_args
    DEFAULT_TRAINER = "global"
    logger.info("✅ Using Global BLIP3-o trainer as default")
else:
    # No trainer available - this is an error
    BLIP3oTrainer = None
    create_training_args = None
    DEFAULT_TRAINER = None
    logger.error("❌ No BLIP3-o trainer available!")
    raise ImportError("Global BLIP3-o trainer is required but not available")

# Build exports list
__all__ = [
    # Primary trainer interface
    "BLIP3oTrainer",
    "create_training_args",
    "DEFAULT_TRAINER",
]

# Export global trainer (always available)
if GLOBAL_TRAINER_AVAILABLE:
    __all__.extend([
        "GlobalBLIP3oTrainer",
        "create_global_training_args",
        "GLOBAL_TRAINER_AVAILABLE",
    ])

# Export enhanced trainer if available
if ENHANCED_TRAINER_AVAILABLE:
    __all__.extend([
        "EnhancedBLIP3oTrainer", 
        "create_enhanced_training_args",
        "ENHANCED_TRAINER_AVAILABLE",
    ])

def get_trainer_class(trainer_type: str = "auto"):
    """
    Get the appropriate trainer class
    
    Args:
        trainer_type: "auto", "enhanced", or "global"
        
    Returns:
        Trainer class
    """
    if trainer_type == "auto":
        if BLIP3oTrainer is None:
            raise RuntimeError("No trainer available")
        return BLIP3oTrainer
    elif trainer_type == "enhanced":
        if not ENHANCED_TRAINER_AVAILABLE:
            raise ValueError("Enhanced trainer not available, using global trainer instead")
        return EnhancedBLIP3oTrainer
    elif trainer_type == "global":
        if not GLOBAL_TRAINER_AVAILABLE:
            raise RuntimeError("Global trainer not available")
        return GlobalBLIP3oTrainer
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def get_training_args_factory(trainer_type: str = "auto"):
    """
    Get the appropriate training args factory function
    
    Args:
        trainer_type: "auto", "enhanced", or "global"
        
    Returns:
        Training args factory function
    """
    if trainer_type == "auto":
        if create_training_args is None:
            raise RuntimeError("No training args factory available")
        return create_training_args
    elif trainer_type == "enhanced":
        if not ENHANCED_TRAINER_AVAILABLE:
            logger.warning("Enhanced trainer not available, using global training args")
            return create_global_training_args
        return create_enhanced_training_args
    elif trainer_type == "global":
        if not GLOBAL_TRAINER_AVAILABLE:
            raise RuntimeError("Global trainer not available")
        return create_global_training_args
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def create_trainer(
    model,
    flow_matching_loss,
    trainer_type: str = "auto",
    **kwargs
):
    """
    Create a BLIP3-o trainer instance
    
    Args:
        model: Model instance
        flow_matching_loss: Loss function
        trainer_type: "auto", "enhanced", or "global"
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer instance
    """
    trainer_class = get_trainer_class(trainer_type)
    return trainer_class(
        model=model,
        flow_matching_loss=flow_matching_loss,
        **kwargs
    )

def print_trainer_status():
    """Print status of available trainers"""
    print("🏋️ BLIP3-o Trainers Status")
    print("=" * 30)
    print(f"Default trainer: {DEFAULT_TRAINER}")
    print()
    print("Available trainers:")
    
    if GLOBAL_TRAINER_AVAILABLE:
        print("  ✅ Global BLIP3-o Trainer (Primary)")
        print("    - Direct global feature training")
        print("    - Optimized for global embeddings")
        print("    - Multi-GPU compatible")
    else:
        print("  ❌ Global BLIP3-o Trainer (REQUIRED)")
        
    if ENHANCED_TRAINER_AVAILABLE:
        print("  ✅ Enhanced BLIP3-o Trainer (Recommended)")
        print("    - Better error handling")
        print("    - Memory optimization")
        print("    - Enhanced debugging")
    else:
        print("  ❌ Enhanced BLIP3-o Trainer")
    
    print("=" * 30)

# Add utility functions to exports
__all__.extend([
    "get_trainer_class",
    "get_training_args_factory", 
    "create_trainer",
    "print_trainer_status",
])

# Ensure at least global trainer is available
if not GLOBAL_TRAINER_AVAILABLE:
    logger.error("❌ Global BLIP3-o trainer is required but not available!")
    raise ImportError("Global BLIP3-o trainer is required for this project")

# Log trainer module status
logger.info(f"BLIP3-o global trainers loaded successfully (default: {DEFAULT_TRAINER})")