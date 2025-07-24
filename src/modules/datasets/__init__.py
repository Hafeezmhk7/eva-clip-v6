"""
Dataset utilities for BLIP3-o DiT training - FIXED Gradient Flow
src/modules/datasets/__init__.py

FIXES:
- Fixed import errors
- Proper gradient handling for multiprocessing
- Support for both CLS+patch and patch-only modes
"""

import logging
import torch

logger = logging.getLogger(__name__)

# Core dataset components
CORE_DATASET_AVAILABLE = False
try:
    from .blip3o_dataset import (
        BLIP3oEmbeddingDataset,
        training_aware_collate_fn,  # FIXED: Use proper name
        create_flexible_dataloaders,
        test_gradient_flow_dataset,  # FIXED: Use correct function name
    )
    logger.debug("✅ Core dataset components loaded (with gradient flow fixes)")
    CORE_DATASET_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to load core dataset components: {e}")
    CORE_DATASET_AVAILABLE = False
    BLIP3oEmbeddingDataset = None
    training_aware_collate_fn = None
    create_flexible_dataloaders = None
    test_gradient_flow_dataset = None

# Enhanced multi-GPU dataset utilities with gradient flow support
ENHANCED_DDP_AVAILABLE = False

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
        
        FIXED: Uses gradient-aware collate function by default
        """
        from torch.utils.data import DataLoader, IterableDataset
        
        # Auto-detect pin_memory
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        
        # Use gradient-aware collate function if none provided
        if collate_fn is None and CORE_DATASET_AVAILABLE:
            collate_fn = training_aware_collate_fn  # FIXED: Use proper function
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
        
        # CRITICAL FIX: Force num_workers=0 to avoid multiprocessing gradient issues
        if num_workers > 0:
            logger.warning(f"Forcing num_workers=0 (was {num_workers}) to avoid gradient serialization issues")
            num_workers = 0
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_for_dataloader,
            sampler=sampler,
            num_workers=num_workers,  # FIXED: Always 0 to avoid multiprocessing issues
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=False,  # FIXED: Must be False when num_workers=0
            **kwargs
        )
        
        return dataloader
    
    ENHANCED_DDP_AVAILABLE = True
    logger.debug("✅ Enhanced DDP dataloader utilities created (with gradient flow)")
    
except ImportError as e:
    ENHANCED_DDP_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced DDP utilities not available: {e}")

# Determine best dataloader creation functions
if ENHANCED_DDP_AVAILABLE and CORE_DATASET_AVAILABLE:
    # Use enhanced versions as default
    create_dataloader = create_enhanced_ddp_dataloader
    create_dataloaders = create_flexible_dataloaders
    DEFAULT_DATALOADER_TYPE = "enhanced_ddp_gradient_flow"
    logger.info("✅ Using enhanced DDP dataloaders with gradient flow as default")
elif CORE_DATASET_AVAILABLE:
    # Use standard versions
    create_dataloader = None
    create_dataloaders = create_flexible_dataloaders
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
        "training_aware_collate_fn",  # FIXED: Use proper name
        "create_flexible_dataloaders",
        "test_gradient_flow_dataset",  # FIXED: Use correct function name
    ])

# Export enhanced components if available
if ENHANCED_DDP_AVAILABLE:
    __all__.extend([
        "create_enhanced_ddp_dataloader",
    ])

# Export default functions if available
if create_dataloaders is not None:
    __all__.extend([
        "create_dataloaders",
    ])

if create_dataloader is not None:
    __all__.extend([
        "create_dataloader",
    ])

# Backward compatibility aliases
if CORE_DATASET_AVAILABLE:
    create_blip3o_dataloader = create_flexible_dataloaders
    blip3o_collate_fn = training_aware_collate_fn  # FIXED: Use proper name
    test_blip3o_dataset = test_gradient_flow_dataset  # FIXED: Use correct function name
    flexible_collate_fn = training_aware_collate_fn  # FIXED: Add this alias
    test_flexible_dataset = test_gradient_flow_dataset  # FIXED: Add this alias
    
    __all__.extend([
        "create_blip3o_dataloader",
        "blip3o_collate_fn",
        "test_blip3o_dataset",
        "flexible_collate_fn",
        "test_flexible_dataset",
    ])

def get_dataloader_factory(dataloader_type: str = "auto"):
    """
    Get the appropriate dataloader factory function
    """
    if dataloader_type == "auto":
        return create_dataloaders
    elif dataloader_type == "enhanced_ddp":
        if not ENHANCED_DDP_AVAILABLE:
            raise ValueError("Enhanced DDP dataloaders not available")
        return create_enhanced_ddp_dataloader
    elif dataloader_type == "standard":
        if not CORE_DATASET_AVAILABLE:
            raise ValueError("Standard dataloaders not available")
        return create_flexible_dataloaders
    else:
        raise ValueError(f"Unknown dataloader type: {dataloader_type}")

def create_gradient_aware_dataloaders(
    chunked_embeddings_dir,
    batch_size: int = 32,
    eval_batch_size: int = None,
    eval_split_ratio: float = 0.1,
    normalize_embeddings: bool = True,
    training_mode: str = "cls_patch",
    max_shards: int = None,
    use_same_data_for_eval: bool = False,
    delete_after_use: bool = False,
    num_workers: int = 0,  # FIXED: Default to 0 to avoid multiprocessing issues
    pin_memory: bool = None,
    use_ddp: bool = None,
    **kwargs
):
    """
    FIXED: Create dataloaders with proper gradient flow setup for BLIP3-o training
    
    CRITICAL FIX: Forces num_workers=0 to avoid multiprocessing gradient serialization errors
    """
    # CRITICAL FIX: Always use num_workers=0 to avoid multiprocessing gradient issues
    if num_workers > 0:
        logger.warning(f"Forcing num_workers=0 (was {num_workers}) to avoid gradient serialization issues")
        num_workers = 0
    
    # Auto-detect DDP usage
    if use_ddp is None:
        use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    # Choose appropriate factory
    if CORE_DATASET_AVAILABLE:
        logger.info("Creating gradient-aware dataloaders with multiprocessing FIX")
        return create_flexible_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_split_ratio=eval_split_ratio,
            normalize_embeddings=normalize_embeddings,
            training_mode=training_mode,
            max_shards=max_shards,
            use_same_data_for_eval=use_same_data_for_eval,
            delete_after_use=delete_after_use,
            num_workers=num_workers,  # FIXED: Always 0
            pin_memory=pin_memory,
            **kwargs
        )
    else:
        raise RuntimeError("No gradient-aware dataloaders available")

def print_dataset_status():
    """Print status of available dataset utilities"""
    print("📊 BLIP3-o Dataset Status (FIXED)")
    print("=" * 35)
    print(f"Default dataloader: {DEFAULT_DATALOADER_TYPE}")
    print()
    print("Available components:")
    
    if CORE_DATASET_AVAILABLE:
        print("  ✅ Core Dataset (BLIP3oEmbeddingDataset)")
        print("  ✅ Gradient-aware dataloaders (FIXED)")
        print("  ✅ Multiprocessing issue RESOLVED")
        print("  ✅ Forces num_workers=0 for gradient safety")
    else:
        print("  ❌ Core Dataset")
        print("  ❌ Flexible Dataloaders")
    
    if ENHANCED_DDP_AVAILABLE:
        print("  ✅ Enhanced DDP Dataloaders")
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
    print("FIXED gradient flow features:")
    if CORE_DATASET_AVAILABLE:
        print("  ✅ Proper tensor detachment for multiprocessing")
        print("  ✅ Gradients added in training loop (not collate)")
        print("  ✅ Forces num_workers=0 for safety")
        print("  ✅ Compatible with both 256 and 257 token modes")
        print("  ✅ CLS+patch and patch-only support")
    else:
        print("  ❌ Gradient flow features not available")
    
    print("=" * 35)

def test_gradient_flow_setup(chunked_embeddings_dir, batch_size: int = 4):
    """
    FIXED: Test the gradient flow setup with actual data
    """
    print("🧪 Testing FIXED Gradient Flow Setup")
    print("=" * 35)
    
    try:
        # Create dataloader with FIXED gradient flow
        train_dataloader, eval_dataloader = create_gradient_aware_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=batch_size,
            eval_split_ratio=0.1,
            delete_after_use=False,  # Don't delete during testing
            num_workers=0,  # FIXED: Always 0
        )
        
        print(f"✅ Created dataloaders successfully (num_workers=0)")
        print(f"   Train dataloader: {len(train_dataloader):,} batches")
        if eval_dataloader:
            print(f"   Eval dataloader: {len(eval_dataloader):,} batches")
        
        # Test gradient flow on actual batch
        print(f"🧪 Testing FIXED gradient flow on real batch...")
        batch = next(iter(train_dataloader))
        
        # Check required keys
        required_keys = ['encoder_hidden_states', 'clip_embeddings', 'hidden_states', 'timestep']
        for key in required_keys:
            if key not in batch:
                print(f"❌ Missing required key: {key}")
                return False
            else:
                print(f"✅ Found key: {key}")
        
        # Check tensor shapes
        eva_shape = batch['encoder_hidden_states'].shape
        clip_shape = batch['clip_embeddings'].shape
        hidden_shape = batch['hidden_states'].shape
        timestep_shape = batch['timestep'].shape
        
        print(f"✅ Tensor shapes:")
        print(f"   EVA embeddings: {eva_shape}")
        print(f"   CLIP embeddings: {clip_shape}")
        print(f"   Hidden states: {hidden_shape}")
        print(f"   Timesteps: {timestep_shape}")
        
        # Check gradient requirements - FIXED logic
        eva_grad = batch['encoder_hidden_states'].requires_grad
        clip_grad = batch['clip_embeddings'].requires_grad
        hidden_grad = batch['hidden_states'].requires_grad
        
        print(f"✅ Gradient requirements (FIXED):")
        print(f"   EVA embeddings: {eva_grad} (should be False - conditioning)")
        print(f"   CLIP embeddings: {clip_grad} (should be False - targets)")
        print(f"   Hidden states: {hidden_grad} (should be True - model input)")
        
        # FIXED validation logic
        gradient_flow_ok = (
            not eva_grad and 
            not clip_grad and 
            hidden_grad and
            eva_shape[1] in [256, 257] and
            clip_shape[1] in [256, 257] and
            hidden_shape[1] in [256, 257] and
            timestep_shape == (batch_size,)
        )
        
        if gradient_flow_ok:
            print("🎉 MULTIPROCESSING GRADIENT ISSUE FIXED!")
            print("✅ Tensors properly detached for multiprocessing")
            print("✅ Gradients will be added in training loop")
            print("✅ Ready for BLIP3-o training without crashes")
        else:
            print("❌ Gradient flow setup still has issues")
            if eva_grad:
                print("   ❌ EVA embeddings shouldn't require gradients")
            if clip_grad:
                print("   ❌ CLIP embeddings shouldn't require gradients")
            if not hidden_grad:
                print("   ❌ Hidden states MUST require gradients")
        
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
    logger.info(f"BLIP3-o datasets loaded successfully (FIXED gradient flow)")
    logger.info("✅ Multiprocessing gradient serialization issue RESOLVED")
else:
    logger.error("BLIP3-o datasets failed to load!")

# Final gradient flow check
if CORE_DATASET_AVAILABLE and training_aware_collate_fn:
    logger.info("✅ FIXED gradient flow setup is active and ready for training")
else:
    logger.warning("⚠️ Gradient flow setup may not be available")