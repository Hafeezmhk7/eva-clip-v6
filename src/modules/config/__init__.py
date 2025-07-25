"""
BLIP3-o Config Module - Simplified
src/modules/config/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flag
CONFIG_AVAILABLE = False

try:
    from .blip3o_config import BLIP3oDiTConfig, get_blip3o_patch_config
    CONFIG_AVAILABLE = True
    logger.info("✅ BLIP3-o config loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import config: {e}")
    BLIP3oDiTConfig = None
    get_blip3o_patch_config = None

# Main exports
__all__ = [
    "CONFIG_AVAILABLE",
]

if CONFIG_AVAILABLE:
    __all__.extend([
        "BLIP3oDiTConfig",
        "get_blip3o_patch_config",
    ])