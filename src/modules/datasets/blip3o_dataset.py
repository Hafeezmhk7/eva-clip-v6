"""
Dataset utilities for BLIP3-o DiT training - Enhanced with Multi-GPU Support

Contains:
- BLIP3oEmbeddingDataset: Dataset for loading EVA-CLIP and CLIP embeddings
- Enhanced dataloader creation utilities with DDP support
- Collation functions optimized for multi-GPU training
- Testing and validation utilities
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Core dataset components
try:
    from .blip3o_dataset import (
        BLIP3oEmbeddingDataset,
        chunked_collate_fn,
        create_chunked_dataloader,
        create_chunked_dataloaders,
        test_chunked_dataset,
    )
    logger.debug("✅ Core dataset components loaded")
    CORE_DATASET_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load core dataset components: {e}")
    CORE_DATASET_AVAILABLE = False
    BLIP3oEmbeddingDataset = None
    chunked_collate_fn = None
    create_chunked_dataloader = None
    create_chunked_dataloaders = None
    test_chunked_dataset = None

# Enhanced multi-GPU dataset utilities
ENHANCED_DDP_AVAILABLE = False
create_enhanced_ddp_dataloader = None
create_enhanced_ddp_dataloaders = None

try:
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    
    def create_enhanced_ddp_dataloader(
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = None,
        drop_last: bool = True,
        collate_fn=None,
        **kwargs
    ):
        """Create DataLoader with enhanced DDP support"""
        from torch.utils.data import DataLoader
        
        # Auto-detect pin_memory
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        
        sampler = None
        shuffle_for_dataloader = shuffle
        
        # Use DistributedSampler if in distributed mode
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                drop_last=drop_last
            )
            shuffle_for_dataloader = False
            logger.debug(f"Created DistributedSampler for rank {dist.get_rank()}")
        
        # Adjust num_workers for stability in distributed mode
        if dist.is_available() and dist.is_initialized():
            num_workers = min(num_workers, 2)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_for_dataloader,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            **kwargs
        )
        
        return dataloader
    
    def create_enhanced_ddp_dataloaders(
        chunked_embeddings_dir,
        batch_size: int = 32,
        eval_batch_size: int = None,
        eval_split_ratio: float = 0.1,
        normalize_embeddings: bool = True,
        delete_after_use: bool = False,
        num_workers: int = 4,
        pin_memory: bool = None,
        **kwargs
    ):
        """Create train and eval dataloaders with enhanced DDP support"""
        
        if eval_batch_size is None:
            eval_batch_size = batch_size * 2
        
        if not CORE_DATASET_AVAILABLE:
            raise RuntimeError("Core dataset components not available")
        
        # Create training dataset
        train_dataset = BLIP3oEmbeddingDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split="train",
            eval_split_ratio=eval_split_ratio,
            normalize_embeddings=normalize_embeddings,
            shuffle_shards=True,
            shuffle_within_shard=True,
            delete_after_use=delete_after_use,
        )
        
        # Create training dataloader
        train_dataloader = create_enhanced_ddp_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=chunked_collate_fn,
            **kwargs
        )
        
        # Create evaluation dataloader if needed
        eval_dataloader = None
        if eval_split_ratio > 0:
            eval_dataset = BLIP3oEmbeddingDataset(
                chunked_embeddings_dir=chunked_embeddings_dir,
                split="eval",
                eval_split_ratio=eval_split_ratio,
                normalize_embeddings=normalize_embeddings,
                shuffle_shards=False,
                shuffle_within_shard=False,
                delete_after_use=False,
            )
            
            eval_dataloader = create_enhanced_ddp_dataloader(
                eval_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=min(num_workers, 2),  # Conservative for eval
                pin_memory=pin_memory,
                collate_fn=chunked_collate_fn,
                **kwargs
            )
        
        return train_dataloader, eval_dataloader
    
    ENHANCED_DDP_AVAILABLE = True
    logger.debug("✅ Enhanced DDP dataloader utilities created")
    
except ImportError as e:
    ENHANCED_DDP_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced DDP utilities not available: {e}")

# Determine best dataloader creation functions
if ENHANCED_DDP_AVAILABLE and CORE_DATASET_AVAILABLE:
    # Use enhanced versions as default
    create_dataloader = create_enhanced_ddp_dataloader
    create_dataloaders = create_enhanced_ddp_dataloaders
    DEFAULT_DATALOADER_TYPE = "enhanced_ddp"
    logger.info("✅ Using enhanced DDP dataloaders as default")
elif CORE_DATASET_AVAILABLE:
    # Use standard versions
    create_dataloader = create_chunked_dataloader
    create_dataloaders = create_chunked_dataloaders
    DEFAULT_DATALOADER_TYPE = "standard"
    logger.info("✅ Using standard dataloaders as default")
else:
    # No dataloaders available
    create_dataloader = None
    create_dataloaders = None
    DEFAULT_DATALOADER_TYPE = None
    logger.error("❌ No dataloader functions available!")

# Build exports list
__all__ = [
    # Availability flags
    "CORE_DATASET_AVAILABLE",
    "ENHANCED_DDP_AVAILABLE",
    "DEFAULT_DATALOADER_TYPE",
]

# Export core components if available
if CORE_DATASET_AVAILABLE:
    __all__.extend([
        "BLIP3oEmbeddingDataset",
        "chunked_collate_fn",
        "create_chunked_dataloader", 
        "create_chunked_dataloaders",
        "test_chunked_dataset",
    ])

# Export enhanced components if available
if ENHANCED_DDP_AVAILABLE:
    __all__.extend([
        "create_enhanced_ddp_dataloader",
        "create_enhanced_ddp_dataloaders",
    ])

# Export default functions if available
if create_dataloader is not None:
    __all__.extend([
        "create_dataloader",
        "create_dataloaders",
    ])

# Backward compatibility aliases
if CORE_DATASET_AVAILABLE:
    create_blip3o_dataloader = create_chunked_dataloader
    create_blip3o_dataloaders = create_chunked_dataloaders
    blip3o_collate_fn = chunked_collate_fn
    test_blip3o_dataset = test_chunked_dataset
    
    __all__.extend([
        "create_blip3o_dataloader",
        "create_blip3o_dataloaders", 
        "blip3o_collate_fn",
        "test_blip3o_dataset",
    ])

def get_dataloader_factory(dataloader_type: str = "auto"):
    """
    Get the appropriate dataloader factory function
    
    Args:
        dataloader_type: "auto", "enhanced_ddp", or "standard"
        
    Returns:
        Dataloader factory function
    """
    if dataloader_type == "auto":
        return create_dataloaders
    elif dataloader_type == "enhanced_ddp":
        if not ENHANCED_DDP_AVAILABLE:
            raise ValueError("Enhanced DDP dataloaders not available")
        return create_enhanced_ddp_dataloaders
    elif dataloader_type == "standard":
        if not CORE_DATASET_AVAILABLE:
            raise ValueError("Standard dataloaders not available")
        return create_chunked_dataloaders
    else:
        raise ValueError(f"Unknown dataloader type: {dataloader_type}")

def print_dataset_status():
    """Print status of available dataset utilities"""
    print("📊 BLIP3-o Dataset Status")
    print("=" * 30)
    print(f"Default dataloader: {DEFAULT_DATALOADER_TYPE}")
    print()
    print("Available components:")
    
    if CORE_DATASET_AVAILABLE:
        print("  ✅ Core Dataset (BLIP3oEmbeddingDataset)")
        print("  ✅ Standard Dataloaders")
    else:
        print("  ❌ Core Dataset")
        print("  ❌ Standard Dataloaders")
        
    if ENHANCED_DDP_AVAILABLE:
        print("  ✅ Enhanced DDP Dataloaders (Recommended)")
    else:
        print("  ❌ Enhanced DDP Dataloaders")
    
    # Check distributed status
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            print(f"  ✅ Distributed training active (rank {rank}/{world_size})")
        else:
            print("  🔧 Distributed training available but not initialized")
    else:
        print("  ❌ Distributed training not available")
    
    print("=" * 30)

# Add utility functions to exports
__all__.extend([
    "get_dataloader_factory",
    "print_dataset_status",
])

# Log dataset module status
if DEFAULT_DATALOADER_TYPE:
    logger.info(f"BLIP3-o datasets loaded successfully (default: {DEFAULT_DATALOADER_TYPE})")
else:
    logger.error("BLIP3-o datasets failed to load!")