"""
BLIP3-o Utils Module - Simplified
src/modules/utils/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flag
UTILS_AVAILABLE = False

try:
    from .temp_manager import SnelliusTempManager, setup_snellius_environment
    UTILS_AVAILABLE = True
    logger.info("✅ BLIP3-o utils loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import utils: {e}")
    SnelliusTempManager = None
    setup_snellius_environment = None

# Main exports
__all__ = [
    "UTILS_AVAILABLE",
]

if UTILS_AVAILABLE:
    __all__.extend([
        "SnelliusTempManager",
        "setup_snellius_environment",
    ])