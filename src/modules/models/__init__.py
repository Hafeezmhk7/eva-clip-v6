"""
BLIP3-o Models Module - Simplified
src/modules/models/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flag
MODEL_AVAILABLE = False

try:
    from .blip3o_patch_dit import (
        BLIP3oPatchDiTModel,
        BLIP3oDiTConfig,
        create_blip3o_patch_dit_model,
    )
    MODEL_AVAILABLE = True
    logger.info("✅ BLIP3-o patch-level DiT model loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import model: {e}")
    BLIP3oPatchDiTModel = None
    BLIP3oDiTConfig = None
    create_blip3o_patch_dit_model = None

# Main exports
__all__ = [
    "MODEL_AVAILABLE",
]

if MODEL_AVAILABLE:
    __all__.extend([
        "BLIP3oPatchDiTModel",
        "BLIP3oDiTConfig", 
        "create_blip3o_patch_dit_model",
    ])