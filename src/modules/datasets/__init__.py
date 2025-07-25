"""
BLIP3-o Datasets Module
src/modules/datasets/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flag
DATASET_AVAILABLE = False

try:
    from .blip3o_dataset import (
        BLIP3oEmbeddingDataset,
        create_blip3o_dataloaders,
        blip3o_collate_fn,
    )
    DATASET_AVAILABLE = True
    logger.info("✅ BLIP3-o dataset components loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import dataset: {e}")
    BLIP3oEmbeddingDataset = None
    create_blip3o_dataloaders = None
    blip3o_collate_fn = None

# Main exports
__all__ = [
    "DATASET_AVAILABLE",
]

if DATASET_AVAILABLE:
    __all__.extend([
        "BLIP3oEmbeddingDataset",
        "create_blip3o_dataloaders", 
        "blip3o_collate_fn",
    ])

# Backward compatibility aliases
if DATASET_AVAILABLE:
    create_flexible_dataloaders = create_blip3o_dataloaders
    __all__.append("create_flexible_dataloaders")