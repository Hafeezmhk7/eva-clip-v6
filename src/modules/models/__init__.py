"""
Models module for BLIP3-o DiT.

Contains:
- BLIP3oDiTModel: Main diffusion transformer model
- Utility functions for model creation and loading
"""

from .blip3o_dit import (
    BLIP3oDiTModel,
    create_blip3o_dit_model,
    load_blip3o_dit_model,
)

# Import lumina_nextdit2d if it exists in the same directory
try:
    from .lumina_nextdit2d import LuminaNextDiT2DModel
    __all__ = [
        "BLIP3oDiTModel",
        "create_blip3o_dit_model", 
        "load_blip3o_dit_model",
        "LuminaNextDiT2DModel",
    ]
except ImportError:
    __all__ = [
        "BLIP3oDiTModel",
        "create_blip3o_dit_model",
        "load_blip3o_dit_model",
    ]