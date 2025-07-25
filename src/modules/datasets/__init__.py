"""
BLIP3-o Datasets Module - Simplified
src/modules/datasets/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flag
DATASET_AVAILABLE = False

try:
    from .blip3o_dataset import create_flexible_dataloaders
    DATASET_AVAILABLE = True
    logger.info("✅ BLIP3-o dataset loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import dataset: {e}")
    create_flexible_dataloaders = None

# Main exports
__all__ = [
    "DATASET_AVAILABLE",
]

if DATASET_AVAILABLE:
    __all__.extend([
        "create_flexible_dataloaders",
    ])