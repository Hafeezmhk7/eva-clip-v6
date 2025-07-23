"""
Dataset utilities for BLIP3-o DiT training - Enhanced with Fixed Gradient Flow

UPDATES:
- Integrated with fixed gradient flow setup
- Enhanced multi-GPU support with proper tensor handling
- Updated collate functions for gradient-aware training
- Maintains backward compatibility

Contains:
- BLIP3oEmbeddingDataset: Dataset for loading EVA-CLIP and CLIP embeddings
- Enhanced dataloader creation utilities with DDP support
- Updated collation functions optimized for gradient flow
- Testing and validation utilities
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Core dataset components
try:
    from .blip3o_dataset import (
        BLIP3oEmbeddingDataset,
        chunked_collate_fn,  # UPDATED with gradient flow setup
        create_chunked_dataloader,
        create_chunked_dataloaders,
        test_chunked_dataset,
    )
    logger.debug("✅ Core dataset components loaded (with gradient flow fixes)")
    CORE_DATASET_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load core dataset components: {e}")
    CORE_DATASET_AVAILABLE = False
    BLIP3oEmbeddingDataset = None
    chunked_collate_fn = None
    create_chunked_dataloader = None
    create_chunked_dataloaders = None
    test_chunked_dataset = None

# Enhanced multi-GPU dataset utilities with gradient flow support
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
        """
        Create DataLoader with enhanced DDP support and gradient flow setup
        
        UPDATED: Uses gradient-aware collate function by default
        """
        from torch.utils.data import DataLoader, IterableDataset
        
        # Auto-detect pin_memory
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        
        # Use updated collate function if none provided
        if collate_fn is None and CORE_DATASET_AVAILABLE:
            collate_fn = chunked_collate_fn  # UPDATED with gradient flow
            logger.debug("Using gradient-aware collate function")
        
        sampler = None
        shuffle_for_dataloader = False
        
        # Handle IterableDataset: cannot use shuffle in DataLoader
        if isinstance(dataset, IterableDataset):
            shuffle_for_dataloader = False
            logger.info("IterableDataset detected: Forcing shuffle=False for DataLoader")
        
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
        """
        Create train and eval dataloaders with enhanced DDP support and gradient flow
        
        UPDATED: Uses gradient-aware dataset and collate functions
        """
        
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
        
        # Create training dataloader with gradient-aware collate function
        train_dataloader = create_enhanced_ddp_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=chunked_collate_fn,  # UPDATED gradient-aware collate
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
                collate_fn=chunked_collate_fn,  # UPDATED gradient-aware collate
                **kwargs
            )
        
        return train_dataloader, eval_dataloader
    
    ENHANCED_DDP_AVAILABLE = True
    logger.debug("✅ Enhanced DDP dataloader utilities created (with gradient flow)")
    
except ImportError as e:
    ENHANCED_DDP_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced DDP utilities not available: {e}")

# Determine best dataloader creation functions
if ENHANCED_DDP_AVAILABLE and CORE_DATASET_AVAILABLE:
    # Use enhanced versions as default
    create_dataloader = create_enhanced_ddp_dataloader
    create_dataloaders = create_enhanced_ddp_dataloaders
    DEFAULT_DATALOADER_TYPE = "enhanced_ddp_gradient_flow"
    logger.info("✅ Using enhanced DDP dataloaders with gradient flow as default")
elif CORE_DATASET_AVAILABLE:
    # Use standard versions
    create_dataloader = create_chunked_dataloader
    create_dataloaders = create_chunked_dataloaders
    DEFAULT_DATALOADER_TYPE = "standard_gradient_flow"
    logger.info("✅ Using standard dataloaders with gradient flow as default")
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
        "chunked_collate_fn",  # UPDATED with gradient flow
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
    blip3o_collate_fn = chunked_collate_fn  # UPDATED with gradient flow
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
        Dataloader factory function with gradient flow support
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

def create_gradient_aware_dataloaders(
    chunked_embeddings_dir,
    batch_size: int = 32,
    eval_batch_size: int = None,
    eval_split_ratio: float = 0.1,
    normalize_embeddings: bool = True,
    delete_after_use: bool = False,
    num_workers: int = 4,
    pin_memory: bool = None,
    use_ddp: bool = None,
    **kwargs
):
    """
    Create dataloaders with proper gradient flow setup for BLIP3-o training
    
    This is the recommended function for creating dataloaders that work with
    the fixed gradient flow implementation.
    
    Args:
        chunked_embeddings_dir: Path to chunked embeddings
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size (defaults to batch_size * 2)
        eval_split_ratio: Ratio for evaluation split
        normalize_embeddings: Whether to normalize embeddings
        delete_after_use: Whether to delete shards after processing
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory (auto-detected if None)
        use_ddp: Whether to use DDP (auto-detected if None)
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (train_dataloader, eval_dataloader) with gradient flow support
    """
    # Auto-detect DDP usage
    if use_ddp is None:
        use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    # Choose appropriate factory
    if use_ddp and ENHANCED_DDP_AVAILABLE:
        logger.info("Creating DDP dataloaders with gradient flow support")
        return create_enhanced_ddp_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_split_ratio=eval_split_ratio,
            normalize_embeddings=normalize_embeddings,
            delete_after_use=delete_after_use,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
    elif CORE_DATASET_AVAILABLE:
        logger.info("Creating standard dataloaders with gradient flow support")
        return create_chunked_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_split_ratio=eval_split_ratio,
            normalize_embeddings=normalize_embeddings,
            delete_after_use=delete_after_use,
            **kwargs
        )
    else:
        raise RuntimeError("No gradient-aware dataloaders available")

def print_dataset_status():
    """Print status of available dataset utilities"""
    print("📊 BLIP3-o Dataset Status (Updated)")
    print("=" * 35)
    print(f"Default dataloader: {DEFAULT_DATALOADER_TYPE}")
    print()
    print("Available components:")
    
    if CORE_DATASET_AVAILABLE:
        print("  ✅ Core Dataset (BLIP3oEmbeddingDataset)")
        print("  ✅ Standard Dataloaders (with gradient flow)")
        print("  ✅ Gradient-aware collate function")
    else:
        print("  ❌ Core Dataset")
        print("  ❌ Standard Dataloaders")
        
    if ENHANCED_DDP_AVAILABLE:
        print("  ✅ Enhanced DDP Dataloaders (Recommended)")
        print("  ✅ Multi-GPU gradient flow support")
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
    
    print()
    print("Gradient flow features:")
    if CORE_DATASET_AVAILABLE:
        print("  ✅ Pre-computed noisy inputs with gradients")
        print("  ✅ Proper tensor detachment for targets/conditioning")
        print("  ✅ Flow matching timestep generation")
        print("  ✅ Rectified flow interpolation")
        print("  ✅ Compatible with fixed trainer")
    else:
        print("  ❌ Gradient flow features not available")
    
    print("=" * 35)

def test_gradient_flow_setup(chunked_embeddings_dir, batch_size: int = 4):
    """
    Test the gradient flow setup with actual data
    
    Args:
        chunked_embeddings_dir: Path to chunked embeddings
        batch_size: Batch size for testing
    """
    print("🧪 Testing Gradient Flow Setup")
    print("=" * 35)
    
    try:
        # Create dataloader with gradient flow
        train_dataloader, eval_dataloader = create_gradient_aware_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=batch_size,
            eval_split_ratio=0.1,
            delete_after_use=False,  # Don't delete during testing
        )
        
        print(f"✅ Created dataloaders successfully")
        print(f"   Train dataloader: {len(train_dataloader):,} batches")
        if eval_dataloader:
            print(f"   Eval dataloader: {len(eval_dataloader):,} batches")
        
        # Test gradient flow on actual batch
        print(f"🧪 Testing gradient flow on real batch...")
        batch = next(iter(train_dataloader))
        
        # Check required keys
        required_keys = ['eva_embeddings', 'clip_embeddings', 'hidden_states', 'timesteps']
        for key in required_keys:
            if key not in batch:
                print(f"❌ Missing required key: {key}")
                return False
            else:
                print(f"✅ Found key: {key}")
        
        # Check tensor shapes
        eva_shape = batch['eva_embeddings'].shape
        clip_shape = batch['clip_embeddings'].shape
        hidden_shape = batch['hidden_states'].shape
        timestep_shape = batch['timesteps'].shape
        
        print(f"✅ Tensor shapes:")
        print(f"   EVA embeddings: {eva_shape}")
        print(f"   CLIP embeddings: {clip_shape}")
        print(f"   Hidden states: {hidden_shape}")
        print(f"   Timesteps: {timestep_shape}")
        
        # Check gradient requirements
        eva_grad = batch['eva_embeddings'].requires_grad
        clip_grad = batch['clip_embeddings'].requires_grad
        hidden_grad = batch['hidden_states'].requires_grad
        
        print(f"✅ Gradient requirements:")
        print(f"   EVA embeddings: {eva_grad} (should be False)")
        print(f"   CLIP embeddings: {clip_grad} (should be False)")
        print(f"   Hidden states: {hidden_grad} (should be True)")
        
        # Validate gradient flow
        gradient_flow_ok = (
            not eva_grad and 
            not clip_grad and 
            hidden_grad and
            eva_shape == (batch_size, 256, 4096) and
            clip_shape == (batch_size, 256, 1024) and
            hidden_shape == (batch_size, 256, 1024) and
            timestep_shape == (batch_size,)
        )
        
        if gradient_flow_ok:
            print("🎉 Gradient flow setup is PERFECT!")
            print("✅ Ready for BLIP3-o training with fixed gradient flow")
        else:
            print("❌ Gradient flow setup has issues")
            if eva_grad:
                print("   ❌ EVA embeddings shouldn't require gradients (conditioning)")
            if clip_grad:
                print("   ❌ CLIP embeddings shouldn't require gradients (targets)")
            if not hidden_grad:
                print("   ❌ Hidden states MUST require gradients (model input)")
        
        return gradient_flow_ok
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Add utility functions to exports
__all__.extend([
    "get_dataloader_factory",
    "create_gradient_aware_dataloaders",
    "print_dataset_status",
    "test_gradient_flow_setup",
])

# Log dataset module status
if DEFAULT_DATALOADER_TYPE:
    logger.info(f"BLIP3-o datasets loaded successfully with gradient flow (default: {DEFAULT_DATALOADER_TYPE})")
else:
    logger.error("BLIP3-o datasets failed to load!")

# Final gradient flow check
if CORE_DATASET_AVAILABLE and chunked_collate_fn:
    logger.info("✅ Gradient flow setup is active and ready for training")
else:
    logger.warning("⚠️ Gradient flow setup may not be available")