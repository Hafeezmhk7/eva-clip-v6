"""
FIXED Chunked Dataset implementation for BLIP3-o training with sequential shard loading.
Place this file as: src/modules/datasets/blip3o_dataset.py
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

logger = logging.getLogger(__name__)


class BLIP3oEmbeddingDataset(IterableDataset):
    """
    Chunked dataset for BLIP3-o training that loads one shard at a time.
    
    This dataset:
    1. Loads embedding shards sequentially
    2. Provides samples from current shard
    3. Automatically moves to next shard when current is exhausted
    4. Optionally deletes processed shards to save disk space
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
        
        logger.info(f"ChunkedDataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Split: {self.split}")
        logger.info(f"  Total shards: {len(self.shard_files)}")
        logger.info(f"  Estimated samples: {self.estimated_total_samples:,}")
        logger.info(f"  Delete after use: {self.delete_after_use}")
    
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
        """Prepare the list of shard files to process."""
        # Find all shard files
        shard_pattern = "embeddings_shard_*.pkl"
        all_shard_files = list(self.chunked_embeddings_dir.glob(shard_pattern))
        all_shard_files.sort()  # Sort by name (shard index)
        
        if not all_shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
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
        """Load the next shard and prepare it."""
        # Clean up current shard
        if self.current_shard_data is not None:
            # Delete previous shard file if requested
            if self.delete_after_use and self.current_shard_idx > 0:
                prev_shard_path = self.shard_files[self.current_shard_idx - 1]
                try:
                    prev_shard_path.unlink()
                    logger.debug(f"Deleted processed shard: {prev_shard_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete shard {prev_shard_path}: {e}")
            
            # Clear memory
            del self.current_shard_data
            gc.collect()
        
        # Check if we have more shards
        if self.current_shard_idx >= len(self.shard_files):
            self.current_shard_data = None
            return False
        
        # Use cached shard if available
        if self.next_shard_data is not None:
            self.current_shard_data = self.next_shard_data
            self.next_shard_data = None
        else:
            # Load current shard
            shard_path = self.shard_files[self.current_shard_idx]
            self.current_shard_data = self._load_shard(shard_path)
        
        # Prepare samples
        self._prepare_current_shard_samples()
        
        # Cache next shard if requested
        if self.cache_next_shard and (self.current_shard_idx + 1) < len(self.shard_files):
            try:
                next_shard_path = self.shard_files[self.current_shard_idx + 1]
                self.next_shard_data = self._load_shard(next_shard_path)
                logger.debug(f"Cached next shard: {next_shard_path}")
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
            'shards_processed': self.shards_processed,
            'samples_processed': self.total_samples_processed,
            'current_shard': self.current_shard_idx,
            'split': self.split,
            'delete_after_use': self.delete_after_use,
            'expected_tokens': self.expected_tokens,
        }


def chunked_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for chunked dataset."""
    # Stack tensor data
    eva_embeddings = torch.stack([item['eva_embeddings'] for item in batch])
    clip_embeddings = torch.stack([item['clip_embeddings'] for item in batch])
    
    # Collect metadata
    captions = [item['caption'] for item in batch]
    keys = [item['key'] for item in batch]
    shard_indices = [item['shard_idx'] for item in batch]
    sample_indices = [item['sample_idx'] for item in batch]
    
    return {
        'eva_embeddings': eva_embeddings,
        'clip_embeddings': clip_embeddings,
        'captions': captions,
        'keys': keys,
        'shard_indices': shard_indices,
        'sample_indices': sample_indices,
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
        **kwargs
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Should be 0 for IterableDataset
        collate_fn=chunked_collate_fn,
        pin_memory=pin_memory,
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


# Legacy compatibility for single-file approach
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
    print(f"🧪 Testing chunked dataset: {chunked_embeddings_dir}")
    
    try:
        # Create dataset
        dataset = BLIP3oEmbeddingDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split="all",
            delete_after_use=False,  # Don't delete during testing
            cache_next_shard=True,
        )
        
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
        
        # Test dataloader
        print(f"🧪 Testing dataloader...")
        dataloader = create_chunked_dataloader(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=4,
            split="all",
            delete_after_use=False,
        )
        
        batch = next(iter(dataloader))
        print(f"✅ Dataloader test: batch shape EVA {batch['eva_embeddings'].shape}, CLIP {batch['clip_embeddings'].shape}")
        
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