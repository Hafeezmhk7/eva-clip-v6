"""
Model modules for BLIP3-o DiT - Updated for Dual Supervision
"""

from .blip3o_dit import (
    BLIP3oDiTModel,
    create_blip3o_dit_model as create_standard_blip3o_dit_model,
)

# Import dual supervision model
try:
    from .dual_supervision_blip3o_dit import (
        DualSupervisionBLIP3oDiTModel,
        create_blip3o_dit_model as create_dual_supervision_blip3o_dit_model,
    )
    # Use dual supervision as default
    create_blip3o_dit_model = create_dual_supervision_blip3o_dit_model
    DUAL_SUPERVISION_MODEL_AVAILABLE = True
except ImportError:
    # Use standard model as fallback
    create_blip3o_dit_model = create_standard_blip3o_dit_model
    DUAL_SUPERVISION_MODEL_AVAILABLE = False
    print("⚠️ Using standard model")

__all__ = [
    "BLIP3oDiTModel",
    "create_blip3o_dit_model",
]

if DUAL_SUPERVISION_MODEL_AVAILABLE:
    __all__.extend([
        "DualSupervisionBLIP3oDiTModel",
    ])