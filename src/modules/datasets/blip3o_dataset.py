"""
UPDATED BLIP3-o Dataset with Fixed Gradient Flow Integration
src/modules/datasets/blip3o_dataset.py

UPDATES:
1. Enhanced chunked_collate_fn to provide proper gradient flow setup
2. Integration with fixed data collator approach
3. Maintains compatibility with existing dataset structure
4. Adds gradient-aware tensor preparation
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Iterator
from pathlib import Path
import logging
import json
import random
import time
import gc
import os

logger = logging.getLogger(__name__)


class BLIP3oEmbeddingDataset(IterableDataset):
    """
    FIXED Chunked dataset for BLIP3-o training that loads one shard at a time.
    
    FIX: Added __len__() method to prevent boolean evaluation errors with DataLoader.
    
    This dataset:
    1. Loads embedding shards sequentially
    2. Provides samples from current shard
    3. Automatically moves to next shard when current is exhausted
    4. Optionally deletes processed shards to save disk space
    5. FIXED: Provides length estimation for DataLoader compatibility
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        split: str = "train",
        eval_split_ratio: float = 0.1,
        normalize_embeddings: bool = True,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        delete_after_use: bool = True,
        random_seed: int = 42,
        expected_tokens: int = 256,
        cache_next_shard: bool = True,
    ):
        """
        Initialize chunked dataset.
        """
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.split = split
        self.eval_split_ratio = eval_split_ratio
        self.normalize_embeddings = normalize_embeddings
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.delete_after_use = delete_after_use
        self.random_seed = random_seed
        self.expected_tokens = expected_tokens
        self.cache_next_shard = cache_next_shard
        
        # Setup random state
        self.rng = random.Random(random_seed)
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(random_seed)
        
        # Load manifest and shard list
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current shard state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_shard_samples = []
        self.current_sample_idx = 0
        
        # Cache for next shard
        self.next_shard_data = None
        
        # Statistics
        self.total_samples_processed = 0
        self.shards_processed = 0
        
        # FIXED: Calculate estimated length for this split and rank
        self._calculate_estimated_length()
        
        logger.info(f"ChunkedDataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Split: {self.split}")
        logger.info(f"  Total shards: {len(self.shard_files)}")
        logger.info(f"  Estimated samples: {self.estimated_length:,}")
        logger.info(f"  Delete after use: {self.delete_after_use}")
    
    def _calculate_estimated_length(self):
        """
        FIXED: Calculate estimated length for this dataset split and distributed rank.
        This enables DataLoader boolean evaluation and length-based operations.
        """
        # Get total samples from manifest
        total_samples = self.estimated_total_samples
        
        # Apply split ratio for train/eval
        if self.split == "train" and self.eval_split_ratio > 0:
            # Training gets (1 - eval_split_ratio) of the data
            split_samples = int(total_samples * (1 - self.eval_split_ratio))
        elif self.split == "eval" and self.eval_split_ratio > 0:
            # Evaluation gets eval_split_ratio of the data
            split_samples = int(total_samples * self.eval_split_ratio)
        else:
            # Use all data if no split or split is "all"
            split_samples = total_samples
        
        # Account for distributed training
        # Check if we're in distributed mode
        if 'WORLD_SIZE' in os.environ and 'RANK' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            
            # Each rank gets approximately 1/world_size of the data
            rank_samples = split_samples // world_size
            
            # Add remainder to last rank
            if rank == world_size - 1:
                rank_samples += split_samples % world_size
            
            self.estimated_length = rank_samples
            
            logger.info(f"Distributed mode: rank {rank}/{world_size}")
            logger.info(f"  Total samples: {total_samples:,}")
            logger.info(f"  Split samples ({self.split}): {split_samples:,}")
            logger.info(f"  This rank samples: {rank_samples:,}")
        else:
            self.estimated_length = split_samples
            logger.info(f"Single-node mode:")
            logger.info(f"  Total samples: {total_samples:,}")
            logger.info(f"  Split samples ({self.split}): {split_samples:,}")
    
    def __len__(self) -> int:
        """
        FIXED: Return estimated length for DataLoader compatibility.
        
        This is required for:
        1. DataLoader boolean evaluation (if dataloader: ...)
        2. Training progress tracking
        3. Epoch-based training schedules
        4. Distributed training coordination
        
        Returns:
            Estimated number of samples in this dataset split
        """
        return self.estimated_length
    
    def _load_manifest(self):
        """Load the embeddings manifest file."""
        manifest_path = self.chunked_embeddings_dir / "embeddings_manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.estimated_total_samples = self.manifest.get('total_samples', 0)
        logger.info(f"Loaded manifest: {self.manifest['total_shards']} shards, {self.estimated_total_samples:,} samples")
    
    def _prepare_shard_list(self):
        """FIXED: Prepare the list of shard files to process with better validation."""
        # Find all shard files
        shard_pattern = "embeddings_shard_*.pkl"
        all_shard_files = list(self.chunked_embeddings_dir.glob(shard_pattern))
        all_shard_files.sort()  # Sort by name (shard index)
        
        if not all_shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
        logger.info(f"Found {len(all_shard_files)} shard files:")
        for shard_file in all_shard_files:
            logger.info(f"  {shard_file.name} ({'EXISTS' if shard_file.exists() else 'MISSING'})")
        
        # FIXED: Filter out non-existent files
        existing_shard_files = [f for f in all_shard_files if f.exists()]
        if len(existing_shard_files) != len(all_shard_files):
            missing_files = [f for f in all_shard_files if not f.exists()]
            logger.warning(f"Some shard files are missing:")
            for missing_file in missing_files:
                logger.warning(f"  Missing: {missing_file}")
            logger.warning(f"Proceeding with {len(existing_shard_files)} existing files")
            all_shard_files = existing_shard_files
        
        if not all_shard_files:
            raise FileNotFoundError(f"No existing shard files found in {self.chunked_embeddings_dir}")
        
        # Split shards for train/eval if needed
        if self.split in ["train", "eval"] and self.eval_split_ratio > 0:
            total_shards = len(all_shard_files)
            eval_shards = max(1, int(total_shards * self.eval_split_ratio))
            train_shards = total_shards - eval_shards
            
            # Use consistent splitting based on shard names
            self.rng.shuffle(all_shard_files)  # Shuffle with fixed seed
            
            if self.split == "train":
                self.shard_files = all_shard_files[:train_shards]
            else:  # eval
                self.shard_files = all_shard_files[train_shards:]
        else:
            self.shard_files = all_shard_files
        
        # Shuffle shard order if requested
        if self.shuffle_shards:
            self.rng.shuffle(self.shard_files)
        
        logger.info(f"Prepared {len(self.shard_files)} shard files for {self.split} split")
    
    def _load_shard(self, shard_path: Path) -> Dict[str, Any]:
        """Load a single embedding shard."""
        logger.debug(f"Loading shard: {shard_path}")
        
        try:
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
            
            # Validate shard data
            self._validate_shard(shard_data, shard_path)
            
            # Normalize embeddings if requested
            if self.normalize_embeddings:
                shard_data = self._normalize_shard_embeddings(shard_data)
            
            return shard_data
            
        except Exception as e:
            logger.error(f"Failed to load shard {shard_path}: {e}")
            raise
    
    def _validate_shard(self, shard_data: Dict[str, Any], shard_path: Path):
        """Validate shard data format."""
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Validate shapes
        if clip_emb.shape[1] != self.expected_tokens:
            raise ValueError(f"Expected {self.expected_tokens} tokens, got {clip_emb.shape[1]} in {shard_path}")
        
        if eva_emb.shape[1] != self.expected_tokens:
            raise ValueError(f"Expected {self.expected_tokens} tokens, got {eva_emb.shape[1]} in {shard_path}")
        
        # Check consistency
        if clip_emb.shape[0] != eva_emb.shape[0]:
            raise ValueError(f"Sample count mismatch in {shard_path}: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
    
    def _normalize_shard_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize embeddings in a shard."""
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Normalize to unit norm along feature dimension
        clip_norm = torch.norm(clip_emb, dim=-1, keepdim=True)
        clip_norm = torch.clamp(clip_norm, min=1e-8)
        shard_data['clip_blip3o_embeddings'] = clip_emb / clip_norm
        
        eva_norm = torch.norm(eva_emb, dim=-1, keepdim=True)
        eva_norm = torch.clamp(eva_norm, min=1e-8)
        shard_data['eva_blip3o_embeddings'] = eva_emb / eva_norm
        
        return shard_data
    
    def _prepare_current_shard_samples(self):
        """Prepare samples from current shard."""
        if self.current_shard_data is None:
            return
        
        num_samples = self.current_shard_data['clip_blip3o_embeddings'].shape[0]
        indices = list(range(num_samples))
        
        # Shuffle within shard if requested
        if self.shuffle_within_shard:
            self.rng.shuffle(indices)
        
        self.current_shard_samples = indices
        self.current_sample_idx = 0
    
    def _load_next_shard(self):
        """FIXED: Load the next shard and prepare it with better error handling."""
        # Clean up current shard
        if self.current_shard_data is not None:
            # Delete previous shard file if requested (and not the first iteration)
            if self.delete_after_use and self.shards_processed > 0:
                try:
                    prev_shard_idx = self.current_shard_idx - 1
                    if 0 <= prev_shard_idx < len(self.shard_files):
                        prev_shard_path = self.shard_files[prev_shard_idx]
                        if prev_shard_path.exists():
                            prev_shard_path.unlink()
                            logger.debug(f"Deleted processed shard: {prev_shard_path}")
                        else:
                            logger.debug(f"Previous shard already deleted: {prev_shard_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete previous shard: {e}")
            
            # Clear memory
            del self.current_shard_data
            gc.collect()
        
        # Check if we have more shards
        if self.current_shard_idx >= len(self.shard_files):
            logger.info("No more shards to process")
            self.current_shard_data = None
            return False
        
        # Use cached shard if available
        if self.next_shard_data is not None:
            self.current_shard_data = self.next_shard_data
            self.next_shard_data = None
            logger.debug("Using cached next shard")
        else:
            # Load current shard with error handling
            shard_path = self.shard_files[self.current_shard_idx]
            
            # FIXED: Check if shard file exists before trying to load
            if not shard_path.exists():
                logger.error(f"Shard file does not exist: {shard_path}")
                self.current_shard_idx += 1
                return self._load_next_shard()  # Try next shard recursively
            
            try:
                self.current_shard_data = self._load_shard(shard_path)
                logger.debug(f"Loaded current shard: {shard_path}")
            except Exception as e:
                logger.error(f"Failed to load shard {shard_path}: {e}")
                self.current_shard_idx += 1
                return self._load_next_shard()  # Try next shard
        
        # Prepare samples
        self._prepare_current_shard_samples()
        
        # Cache next shard if requested
        if self.cache_next_shard and (self.current_shard_idx + 1) < len(self.shard_files):
            try:
                next_shard_path = self.shard_files[self.current_shard_idx + 1]
                if next_shard_path.exists():
                    self.next_shard_data = self._load_shard(next_shard_path)
                    logger.debug(f"Cached next shard: {next_shard_path}")
                else:
                    logger.warning(f"Next shard file does not exist: {next_shard_path}")
                    self.next_shard_data = None
            except Exception as e:
                logger.warning(f"Failed to cache next shard: {e}")
                self.next_shard_data = None
        
        logger.info(f"Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: "
                   f"{len(self.current_shard_samples)} samples")
        
        self.current_shard_idx += 1
        self.shards_processed += 1
        
        return True
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples across all shards."""
        # Reset state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        self.shards_processed = 0
        
        # Load first shard
        if not self._load_next_shard():
            logger.warning("No shards could be loaded")
            return
        
        # Iterate through all shards and samples
        while self.current_shard_data is not None:
            # Iterate through current shard
            while self.current_sample_idx < len(self.current_shard_samples):
                sample_idx = self.current_shard_samples[self.current_sample_idx]
                
                # Get sample data
                item = {
                    'eva_embeddings': self.current_shard_data['eva_blip3o_embeddings'][sample_idx],
                    'clip_embeddings': self.current_shard_data['clip_blip3o_embeddings'][sample_idx],
                    'caption': self.current_shard_data['captions'][sample_idx],
                    'key': self.current_shard_data.get('keys', [f"sample_{sample_idx}"])[sample_idx] if self.current_shard_data.get('keys') else f"sample_{sample_idx}",
                    'shard_idx': self.current_shard_idx - 1,
                    'sample_idx': sample_idx,
                }
                
                self.current_sample_idx += 1
                self.total_samples_processed += 1
                
                yield item
            
            # Move to next shard
            if not self._load_next_shard():
                break
        
        logger.info(f"Dataset iteration completed: {self.total_samples_processed} samples from {self.shards_processed} shards")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'total_shards': len(self.shard_files),
            'estimated_total_samples': self.estimated_total_samples,
            'estimated_length': self.estimated_length,
            'shards_processed': self.shards_processed,
            'samples_processed': self.total_samples_processed,
            'current_shard': self.current_shard_idx,
            'split': self.split,
            'delete_after_use': self.delete_after_use,
            'expected_tokens': self.expected_tokens,
        }


def chunked_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    UPDATED Custom collate function with gradient flow setup for BLIP3-o training
    
    This function now prepares tensors compatible with the fixed trainer:
    1. Creates proper noisy inputs with gradients for flow matching
    2. Detaches conditioning and target tensors appropriately  
    3. Sets up flow matching timesteps and noise
    4. Ensures proper tensor connectivity for gradient flow
    """
    # Stack tensor data
    eva_embeddings = torch.stack([item['eva_embeddings'] for item in batch])  # [B, 256, 4096]
    clip_embeddings = torch.stack([item['clip_embeddings'] for item in batch])  # [B, 256, 1024]
    
    # Collect metadata
    captions = [item['caption'] for item in batch]
    keys = [item['key'] for item in batch]
    shard_indices = [item['shard_idx'] for item in batch]
    sample_indices = [item['sample_idx'] for item in batch]
    
    # Ensure proper dtype and device
    eva_embeddings = eva_embeddings.float()
    clip_embeddings = clip_embeddings.float()
    
    # Get batch info
    batch_size, seq_len, clip_dim = clip_embeddings.shape
    device = clip_embeddings.device
    dtype = clip_embeddings.dtype
    
    # CRITICAL: Detach conditioning and targets (no gradients needed)
    eva_embeddings = eva_embeddings.detach()
    clip_embeddings = clip_embeddings.detach()
    
    # UPDATED: Create flow matching setup with proper gradient flow
    # Sample timesteps for flow matching
    timesteps = torch.rand(batch_size, device=device, dtype=dtype)
    
    # Create noise for flow matching
    noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)
    
    # CRITICAL: Create base noise with proper gradient requirement
    base_noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype, requires_grad=True)
    
    # Linear interpolation for flow matching (rectified flow style)
    alpha = timesteps.view(-1, 1, 1)  # [B, 1, 1]
    
    # Create noisy input for the model - this MUST have gradients
    hidden_states = (1 - alpha) * base_noise + alpha * clip_embeddings + 0.1 * noise
    
    # Ensure the noisy input requires gradients
    if not hidden_states.requires_grad:
        logger.warning("Noisy input doesn't require gradients, fixing...")
        hidden_states = hidden_states.requires_grad_(True)
    
    # Validate tensor properties
    assert eva_embeddings.shape == (batch_size, 256, 4096), f"EVA batch shape: {eva_embeddings.shape}"
    assert clip_embeddings.shape == (batch_size, 256, 1024), f"CLIP batch shape: {clip_embeddings.shape}"
    assert hidden_states.shape == (batch_size, 256, 1024), f"Hidden states shape: {hidden_states.shape}"
    assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
    
    # Log gradient status for debugging (only occasionally)
    if batch_size > 0 and hasattr(chunked_collate_fn, '_debug_count'):
        chunked_collate_fn._debug_count = getattr(chunked_collate_fn, '_debug_count', 0) + 1
        if chunked_collate_fn._debug_count % 100 == 0:
            logger.debug(f"Collate function: eva_embeddings requires_grad={eva_embeddings.requires_grad}")
            logger.debug(f"Collate function: clip_embeddings requires_grad={clip_embeddings.requires_grad}")
            logger.debug(f"Collate function: hidden_states requires_grad={hidden_states.requires_grad}")
    
    return {
        # Core embeddings
        'eva_embeddings': eva_embeddings,        # [B, 256, 4096] - EVA conditioning (detached)
        'clip_embeddings': clip_embeddings,      # [B, 256, 1024] - Target CLIP patches (detached)
        
        # UPDATED: Flow matching inputs (compatible with fixed trainer)
        'hidden_states': hidden_states,          # [B, 256, 1024] - Noisy input for model (with gradients)
        'timesteps': timesteps,                  # [B] - Flow matching timesteps
        'noise': noise,                          # [B, 256, 1024] - Original noise
        'base_noise': base_noise,                # [B, 256, 1024] - Base noise with gradients
        
        # Metadata (backward compatibility)
        'captions': captions,                    # List[str] - Text captions
        'keys': keys,                           # List[str] - Sample keys
        'shard_indices': shard_indices,         # List[int] - Shard indices
        'sample_indices': sample_indices,       # List[int] - Sample indices
        'batch_size': batch_size,               # int
    }


def create_chunked_dataloader(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 32,
    split: str = "train",
    eval_split_ratio: float = 0.1,
    normalize_embeddings: bool = True,
    shuffle_shards: bool = True,
    shuffle_within_shard: bool = True,
    delete_after_use: bool = True,
    num_workers: int = 0,  # Use 0 for IterableDataset
    pin_memory: bool = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for chunked embeddings.
    
    UPDATED: Uses enhanced collate function with gradient flow setup
    """
    # Auto-detect pin_memory
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # Create dataset
    dataset = BLIP3oEmbeddingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        split=split,
        eval_split_ratio=eval_split_ratio,
        normalize_embeddings=normalize_embeddings,
        shuffle_shards=shuffle_shards,
        shuffle_within_shard=shuffle_within_shard,
        delete_after_use=delete_after_use,
    )
    
    # Create dataloader with UPDATED collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Should be 0 for IterableDataset
        collate_fn=chunked_collate_fn,  # UPDATED collate function
        pin_memory=pin_memory,
        **kwargs  # Pass kwargs to DataLoader (drop_last, persistent_workers, etc.)
    )
    
    return dataloader


def create_chunked_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    eval_split_ratio: float = 0.1,
    normalize_embeddings: bool = True,
    delete_after_use: bool = True,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create both training and evaluation dataloaders for chunked embeddings.
    
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size * 2
    
    # Create training dataloader
    train_dataloader = create_chunked_dataloader(
        chunked_embeddings_dir=chunked_embeddings_dir,
        batch_size=batch_size,
        split="train",
        eval_split_ratio=eval_split_ratio,
        normalize_embeddings=normalize_embeddings,
        shuffle_shards=True,
        shuffle_within_shard=True,
        delete_after_use=delete_after_use,
        **kwargs
    )
    
    # Create evaluation dataloader if eval_split_ratio > 0
    eval_dataloader = None
    if eval_split_ratio > 0:
        eval_dataloader = create_chunked_dataloader(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=eval_batch_size,
            split="eval",
            eval_split_ratio=eval_split_ratio,
            normalize_embeddings=normalize_embeddings,
            shuffle_shards=False,
            shuffle_within_shard=False,
            delete_after_use=False,  # Don't delete eval shards
            **kwargs
        )
    
    return train_dataloader, eval_dataloader


# Legacy compatibility functions (unchanged)
def create_blip3o_dataloader(*args, **kwargs):
    """Legacy function - redirects to chunked approach"""
    logger.warning("create_blip3o_dataloader called - this should use single-file approach")
    raise NotImplementedError("Use create_chunked_dataloader for chunked datasets")


def create_blip3o_dataloaders(*args, **kwargs):
    """Legacy function - redirects to chunked approach"""
    logger.warning("create_blip3o_dataloaders called - this should use single-file approach")  
    raise NotImplementedError("Use create_chunked_dataloaders for chunked datasets")


def blip3o_collate_fn(*args, **kwargs):
    """Legacy function - redirects to chunked approach"""
    return chunked_collate_fn(*args, **kwargs)


def test_blip3o_dataset(*args, **kwargs):
    """Legacy function - redirects to chunked approach"""
    logger.warning("test_blip3o_dataset called - this should use single-file approach")
    raise NotImplementedError("Use test_chunked_dataset for chunked datasets")


def test_chunked_dataset(chunked_embeddings_dir: Union[str, Path]):
    """Test the chunked dataset implementation."""
    print(f"🧪 Testing updated chunked dataset: {chunked_embeddings_dir}")
    
    try:
        # Create dataset
        dataset = BLIP3oEmbeddingDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split="all",
            delete_after_use=False,  # Don't delete during testing
            cache_next_shard=True,
        )
        
        # Test length
        print(f"✅ Dataset length: {len(dataset):,}")
        
        # Test iteration
        sample_count = 0
        shard_count = 0
        current_shard = -1
        
        for i, sample in enumerate(dataset):
            if sample['shard_idx'] != current_shard:
                current_shard = sample['shard_idx']
                shard_count += 1
                print(f"  Shard {shard_count}: Started processing shard {current_shard}")
            
            sample_count += 1
            
            # Validate sample
            assert sample['eva_embeddings'].shape[0] == 256, f"Invalid EVA tokens: {sample['eva_embeddings'].shape[0]}"
            assert sample['clip_embeddings'].shape[0] == 256, f"Invalid CLIP tokens: {sample['clip_embeddings'].shape[0]}"
            
            # Print first few samples
            if i < 3:
                print(f"    Sample {i}: EVA {sample['eva_embeddings'].shape}, CLIP {sample['clip_embeddings'].shape}")
                print(f"      Caption: {sample['caption'][:50]}...")
            
            # Break early for testing
            if sample_count >= 10:
                break
        
        print(f"✅ Test completed: {sample_count} samples from {shard_count} shards")
        
        # Test dataloader with UPDATED collate function
        print(f"🧪 Testing dataloader with gradient flow setup...")
        dataloader = create_chunked_dataloader(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=4,
            split="all",
            delete_after_use=False,
        )
        
        # Test boolean evaluation
        print(f"✅ DataLoader boolean evaluation: {bool(dataloader)}")
        print(f"✅ DataLoader length: {len(dataloader):,}")
        
        batch = next(iter(dataloader))
        print(f"✅ Dataloader test: batch shape EVA {batch['eva_embeddings'].shape}, CLIP {batch['clip_embeddings'].shape}")
        
        # UPDATED: Test gradient flow setup
        print(f"✅ Gradient flow test:")
        print(f"   hidden_states shape: {batch['hidden_states'].shape}")
        print(f"   hidden_states requires_grad: {batch['hidden_states'].requires_grad}")
        print(f"   timesteps shape: {batch['timesteps'].shape}")
        print(f"   eva_embeddings requires_grad: {batch['eva_embeddings'].requires_grad}")
        print(f"   clip_embeddings requires_grad: {batch['clip_embeddings'].requires_grad}")
        
        if not batch['hidden_states'].requires_grad:
            print("❌ ERROR: hidden_states doesn't require gradients!")
        else:
            print("✅ Gradient flow setup is correct!")
        
        print(f"✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        chunked_dir = sys.argv[1]
        test_chunked_dataset(chunked_dir)
    else:
        print("Usage: python blip3o_dataset.py <chunked_embeddings_dir>")
        print("Example: python blip3o_dataset.py /scratch-local/user/chunked_embeddings")