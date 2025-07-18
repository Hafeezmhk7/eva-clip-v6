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