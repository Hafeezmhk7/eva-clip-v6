# ============================================================================
# FIXED: src/modules/losses/__init__.py
# ============================================================================

"""
Loss functions module for BLIP3-o DiT - FIXED for Dual Supervision
"""

from .flow_matching_loss import (
    FlowMatchingLoss,
    BLIP3oFlowMatchingLoss,  # ✅ This one stays here (from flow_matching_loss.py)
    create_blip3o_flow_matching_loss,
)

# Import dual supervision components with better error handling
DUAL_SUPERVISION_AVAILABLE = False
try:
    from .dual_supervision_flow_matching_loss import (
        DualSupervisionFlowMatchingLoss,  # ✅ CORRECT: This is the right class name
        create_dual_supervision_loss,
    )
    DUAL_SUPERVISION_AVAILABLE = True
    print("✅ Dual supervision loss loaded successfully")
    
except ImportError as e:
    DUAL_SUPERVISION_AVAILABLE = False
    print(f"⚠️ Dual supervision loss import failed: {e}")
    print("⚠️ Dual supervision loss not available")
    
except Exception as e:
    DUAL_SUPERVISION_AVAILABLE = False
    print(f"⚠️ Unexpected error loading dual supervision loss: {e}")
    print("⚠️ Dual supervision loss not available")

__all__ = [
    "FlowMatchingLoss",
    "BLIP3oFlowMatchingLoss",  # From flow_matching_loss.py
    "create_blip3o_flow_matching_loss",
    "DUAL_SUPERVISION_AVAILABLE",
]

if DUAL_SUPERVISION_AVAILABLE:
    __all__.extend([
        "DualSupervisionFlowMatchingLoss",  # From dual_supervision_flow_matching_loss.py
        "create_dual_supervision_loss",
    ])

# ============================================================================
# FIXED: src/modules/models/__init__.py  
# ============================================================================

"""
Model modules for BLIP3-o DiT - FIXED for Dual Supervision
"""

from .blip3o_dit import (
    BLIP3oDiTModel,
    create_blip3o_dit_model as create_standard_blip3o_dit_model,
)

# Import dual supervision model with better error handling
DUAL_SUPERVISION_MODEL_AVAILABLE = False
try:
    from .dual_supervision_blip3o_dit import (
        DualSupervisionBLIP3oDiTModel,
        create_blip3o_dit_model as create_dual_supervision_blip3o_dit_model,
        load_dual_supervision_blip3o_dit_model,
    )
    # Use dual supervision as default
    create_blip3o_dit_model = create_dual_supervision_blip3o_dit_model
    DUAL_SUPERVISION_MODEL_AVAILABLE = True
    print("✅ Dual supervision model loaded successfully")
    
except ImportError as e:
    # Use standard model as fallback
    create_blip3o_dit_model = create_standard_blip3o_dit_model
    DUAL_SUPERVISION_MODEL_AVAILABLE = False
    print(f"⚠️ Dual supervision model import failed: {e}")
    print("⚠️ Using standard model as fallback")

except Exception as e:
    # Handle other errors
    create_blip3o_dit_model = create_standard_blip3o_dit_model
    DUAL_SUPERVISION_MODEL_AVAILABLE = False
    print(f"⚠️ Unexpected error loading dual supervision model: {e}")
    print("⚠️ Using standard model as fallback")

__all__ = [
    "BLIP3oDiTModel",
    "create_blip3o_dit_model",
    "DUAL_SUPERVISION_MODEL_AVAILABLE",
]

if DUAL_SUPERVISION_MODEL_AVAILABLE:
    __all__.extend([
        "DualSupervisionBLIP3oDiTModel",
        "load_dual_supervision_blip3o_dit_model",
    ])

# ============================================================================
# FIXED: src/modules/trainers/__init__.py
# ============================================================================

"""
Training utilities for BLIP3-o DiT - FIXED for Dual Supervision
"""

from .blip3o_trainer import (
    BLIP3oTrainer as StandardBLIP3oTrainer,
    create_blip3o_training_args as create_standard_training_args,
)

# Import dual supervision trainer with better error handling
DUAL_SUPERVISION_TRAINER_AVAILABLE = False
try:
    from .dual_supervision_blip3o_trainer import (
        DualSupervisionBLIP3oTrainer,
        create_blip3o_training_args as create_dual_supervision_training_args,
    )
    # Use dual supervision as default
    BLIP3oTrainer = DualSupervisionBLIP3oTrainer
    create_blip3o_training_args = create_dual_supervision_training_args
    DUAL_SUPERVISION_TRAINER_AVAILABLE = True
    print("✅ Dual supervision trainer loaded successfully")
    
except ImportError as e:
    # Use standard trainer as fallback
    BLIP3oTrainer = StandardBLIP3oTrainer
    create_blip3o_training_args = create_standard_training_args
    DUAL_SUPERVISION_TRAINER_AVAILABLE = False
    print(f"⚠️ Dual supervision trainer import failed: {e}")
    print("⚠️ Using standard trainer as fallback")
    
except Exception as e:
    # Handle other errors
    BLIP3oTrainer = StandardBLIP3oTrainer
    create_blip3o_training_args = create_standard_training_args
    DUAL_SUPERVISION_TRAINER_AVAILABLE = False
    print(f"⚠️ Unexpected error loading dual supervision trainer: {e}")
    print("⚠️ Using standard trainer as fallback")

__all__ = [
    "BLIP3oTrainer",
    "create_blip3o_training_args", 
    "DUAL_SUPERVISION_TRAINER_AVAILABLE",
]

if DUAL_SUPERVISION_TRAINER_AVAILABLE:
    __all__.extend([
        "DualSupervisionBLIP3oTrainer",
        "StandardBLIP3oTrainer",
        "create_standard_training_args",
        "create_dual_supervision_training_args",
    ])