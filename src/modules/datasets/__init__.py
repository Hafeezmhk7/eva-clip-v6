"""
Dataset utilities for BLIP3-o DiT training - FIXED.

Contains:
- BLIP3oEmbeddingDataset: Dataset for loading EVA-CLIP and CLIP embeddings
- Dataloader creation utilities
- Collation functions
"""

from .blip3o_dataset import (
    BLIP3oEmbeddingDataset,
    chunked_collate_fn,
    create_chunked_dataloader,
    create_chunked_dataloaders,
    test_chunked_dataset,
)

# Create aliases for backward compatibility
create_blip3o_dataloader = create_chunked_dataloader
create_blip3o_dataloaders = create_chunked_dataloaders
blip3o_collate_fn = chunked_collate_fn
test_blip3o_dataset = test_chunked_dataset

__all__ = [
    "BLIP3oEmbeddingDataset",
    
    # Main functions
    "create_chunked_dataloader",
    "create_chunked_dataloaders",
    "chunked_collate_fn",
    "test_chunked_dataset",
    
    # Compatibility aliases
    "create_blip3o_dataloader",
    "create_blip3o_dataloaders",
    "blip3o_collate_fn",
    "test_blip3o_dataset",
]