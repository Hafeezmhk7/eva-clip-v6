"""
Dataset implementation for BLIP3-o embedding training.
Handles loading and preprocessing of EVA-CLIP and CLIP embeddings for flow matching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BLIP3oEmbeddingDataset(Dataset):
    """
    Dataset class for BLIP3-o embedding training.
    
    Loads pre-extracted EVA-CLIP and CLIP embeddings in 64-token format
    for flow matching training. The dataset handles:
    - EVA-CLIP embeddings (1280-dim, 64 tokens) as conditioning
    - CLIP embeddings (768-dim, 64 tokens) as targets
    - Proper normalization and preprocessing
    - Memory-efficient loading
    """
    
    def __init__(
        self,
        embeddings_path: Union[str, Path],
        subset_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        eva_normalize: bool = True,
        clip_normalize: bool = True,
        cache_in_memory: bool = True,
        split: str = "train",
        eval_split_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize the BLIP3-o embedding dataset.
        
        Args:
            embeddings_path: Path to the pickle file containing embeddings
            subset_size: Optional subset size for debugging/testing
            normalize_embeddings: Whether to normalize embeddings (applies to both)
            eva_normalize: Whether to normalize EVA-CLIP embeddings specifically
            clip_normalize: Whether to normalize CLIP embeddings specifically
            cache_in_memory: Whether to keep all data in memory
            split: Dataset split ("train" or "eval")
            eval_split_ratio: Ratio of data to use for evaluation
            random_seed: Random seed for reproducible splitting
        """
        self.embeddings_path = Path(embeddings_path)
        self.subset_size = subset_size
        self.normalize_embeddings = normalize_embeddings
        self.eva_normalize = eva_normalize and normalize_embeddings
        self.clip_normalize = clip_normalize and normalize_embeddings
        self.cache_in_memory = cache_in_memory
        self.split = split
        self.eval_split_ratio = eval_split_ratio
        self.random_seed = random_seed
        
        # Load and process the embeddings
        self._load_embeddings()
        self._split_dataset()
        
        # Apply subset if requested
        if subset_size is not None:
            self._apply_subset(subset_size)
        
        # Log dataset information
        self._log_dataset_info()
    
    def _load_embeddings(self):
        """Load embeddings from pickle file and validate format."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
        
        logger.info(f"Loading embeddings from {self.embeddings_path}")
        
        try:
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings: {e}")
        
        # Extract embeddings - ensure we use the 64-token BLIP3-o format
        if 'eva_blip3o_embeddings' not in data or 'clip_blip3o_embeddings' not in data:
            raise ValueError(
                "Expected 'eva_blip3o_embeddings' and 'clip_blip3o_embeddings' in data. "
                "Make sure you're using the BLIP3-o compatible embeddings."
            )
        
        self.eva_embeddings = data['eva_blip3o_embeddings']    # [N, 64, 1280]
        self.clip_embeddings = data['clip_blip3o_embeddings']  # [N, 64, 768]
        self.captions = data.get('captions', [])
        self.keys = data.get('keys', [])
        self.config = data.get('config', {})
        
        # Convert to torch tensors
        self.eva_embeddings = self._to_torch_tensor(self.eva_embeddings)
        self.clip_embeddings = self._to_torch_tensor(self.clip_embeddings)
        
        # Validate shapes
        self._validate_embeddings()
        
        # Apply normalization
        if self.eva_normalize:
            logger.info("Normalizing EVA-CLIP embeddings")
            self.eva_embeddings = self._normalize_embeddings(self.eva_embeddings)
        
        if self.clip_normalize:
            logger.info("Normalizing CLIP embeddings")
            self.clip_embeddings = self._normalize_embeddings(self.clip_embeddings)
        
        logger.info(f"Loaded {self.eva_embeddings.shape[0]} embedding pairs")
    
    def _to_torch_tensor(self, data) -> torch.Tensor:
        """Convert numpy array or other format to torch tensor."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            return data.float()
        else:
            return torch.tensor(data, dtype=torch.float32)
    
    def _validate_embeddings(self):
        """Validate embedding shapes and consistency."""
        eva_shape = self.eva_embeddings.shape
        clip_shape = self.clip_embeddings.shape
        
        # Check batch dimension consistency
        if eva_shape[0] != clip_shape[0]:
            raise ValueError(
                f"Mismatch in number of samples: EVA {eva_shape[0]} vs CLIP {clip_shape[0]}"
            )
        
        # Check token dimension (should be 64)
        if eva_shape[1] != 64:
            raise ValueError(f"EVA embeddings should have 64 tokens, got {eva_shape[1]}")
        if clip_shape[1] != 64:
            raise ValueError(f"CLIP embeddings should have 64 tokens, got {clip_shape[1]}")
        
        # Check feature dimensions
        if eva_shape[2] != 1280:
            raise ValueError(f"EVA embeddings should be 1280-dim, got {eva_shape[2]}")
        if clip_shape[2] != 768:
            raise ValueError(f"CLIP embeddings should be 768-dim, got {clip_shape[2]}")
        
        logger.info(f"Validated embeddings: EVA {eva_shape}, CLIP {clip_shape}")
    
    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Normalize embeddings to unit norm along the feature dimension.
        
        Args:
            embeddings: Input embeddings [..., feature_dim]
            
        Returns:
            Normalized embeddings
        """
        # Compute L2 norm along the last dimension
        norm = torch.norm(embeddings, dim=-1, keepdim=True)
        # Avoid division by zero
        norm = torch.clamp(norm, min=1e-8)
        return embeddings / norm
    
    def _split_dataset(self):
        """Split dataset into train/eval based on split parameter."""
        if self.split not in ["train", "eval", "all"]:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'eval', or 'all'")
        
        if self.split == "all":
            # Use all data
            return
        
        # Set random seed for reproducible splits
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        
        total_size = len(self.eva_embeddings)
        eval_size = int(total_size * self.eval_split_ratio)
        train_size = total_size - eval_size
        
        # Create random permutation
        indices = torch.randperm(total_size, generator=generator)
        
        if self.split == "train":
            selected_indices = indices[:train_size]
        else:  # eval
            selected_indices = indices[train_size:]
        
        # Apply split
        self.eva_embeddings = self.eva_embeddings[selected_indices]
        self.clip_embeddings = self.clip_embeddings[selected_indices]
        
        # Handle captions and keys if they exist
        if self.captions:
            self.captions = [self.captions[i] for i in selected_indices.tolist()]
        if self.keys:
            self.keys = [self.keys[i] for i in selected_indices.tolist()]
        
        logger.info(f"Using {self.split} split: {len(selected_indices)} samples")
    
    def _apply_subset(self, subset_size: int):
        """Apply subset for debugging/testing."""
        current_size = len(self.eva_embeddings)
        if subset_size >= current_size:
            logger.warning(f"Subset size {subset_size} >= dataset size {current_size}")
            return
        
        # Use random subset
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        indices = torch.randperm(current_size, generator=generator)[:subset_size]
        
        self.eva_embeddings = self.eva_embeddings[indices]
        self.clip_embeddings = self.clip_embeddings[indices]
        
        if self.captions:
            self.captions = [self.captions[i] for i in indices.tolist()]
        if self.keys:
            self.keys = [self.keys[i] for i in indices.tolist()]
        
        logger.info(f"Applied subset of size {subset_size}")
    
    def _log_dataset_info(self):
        """Log dataset information."""
        logger.info(f"Dataset created with {len(self)} samples")
        logger.info(f"EVA-CLIP embeddings shape: {self.eva_embeddings.shape}")
        logger.info(f"CLIP embeddings shape: {self.clip_embeddings.shape}")
        logger.info(f"Split: {self.split}")
        logger.info(f"Normalization - EVA: {self.eva_normalize}, CLIP: {self.clip_normalize}")
    
    def __len__(self) -> int:
        return len(self.eva_embeddings)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - eva_embeddings: [64, 1280] EVA-CLIP conditioning
            - clip_embeddings: [64, 768] CLIP targets
            - caption: Text caption (if available)
            - key: Unique identifier (if available)
            - index: Dataset index
        """
        item = {
            'eva_embeddings': self.eva_embeddings[idx],      # [64, 1280]
            'clip_embeddings': self.clip_embeddings[idx],    # [64, 768]
            'index': idx,
        }
        
        # Add optional metadata
        if self.captions and idx < len(self.captions):
            item['caption'] = self.captions[idx]
        else:
            item['caption'] = f"sample_{idx}"
        
        if self.keys and idx < len(self.keys):
            item['key'] = self.keys[idx]
        else:
            item['key'] = f"key_{idx}"
        
        return item
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        eva_flat = self.eva_embeddings.view(-1, self.eva_embeddings.shape[-1])
        clip_flat = self.clip_embeddings.view(-1, self.clip_embeddings.shape[-1])
        
        return {
            'num_samples': len(self),
            'num_tokens': self.eva_embeddings.shape[1],
            'eva_dim': self.eva_embeddings.shape[-1],
            'clip_dim': self.clip_embeddings.shape[-1],
            'eva_stats': {
                'mean': eva_flat.mean().item(),
                'std': eva_flat.std().item(),
                'min': eva_flat.min().item(),
                'max': eva_flat.max().item(),
                'norm_mean': torch.norm(self.eva_embeddings, dim=-1).mean().item(),
            },
            'clip_stats': {
                'mean': clip_flat.mean().item(),
                'std': clip_flat.std().item(),
                'min': clip_flat.min().item(),
                'max': clip_flat.max().item(),
                'norm_mean': torch.norm(self.clip_embeddings, dim=-1).mean().item(),
            },
            'split': self.split,
            'normalized': {
                'eva': self.eva_normalize,
                'clip': self.clip_normalize,
            }
        }


def blip3o_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for BLIP3-o batching.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched dictionary with proper tensor stacking
    """
    # Stack tensor data
    eva_embeddings = torch.stack([item['eva_embeddings'] for item in batch])
    clip_embeddings = torch.stack([item['clip_embeddings'] for item in batch])
    indices = torch.tensor([item['index'] for item in batch])
    
    # Collect string data
    captions = [item['caption'] for item in batch]
    keys = [item['key'] for item in batch]
    
    return {
        'eva_embeddings': eva_embeddings,      # [B, 64, 1280]
        'clip_embeddings': clip_embeddings,    # [B, 64, 768]
        'captions': captions,                  # List[str]
        'keys': keys,                          # List[str]
        'indices': indices,                    # [B]
    }


def create_blip3o_dataloader(
    embeddings_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    split: str = "train",
    subset_size: Optional[int] = None,
    normalize_embeddings: bool = True,
    eval_split_ratio: float = 0.1,
    pin_memory: bool = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for BLIP3-o training.
    
    Args:
        embeddings_path: Path to embeddings pickle file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        split: Dataset split ("train", "eval", or "all")
        subset_size: Optional subset size for debugging
        normalize_embeddings: Whether to normalize embeddings
        eval_split_ratio: Ratio for train/eval split
        pin_memory: Whether to pin memory (auto-detected if None)
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    # Auto-detect pin_memory based on CUDA availability
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # Create dataset
    dataset = BLIP3oEmbeddingDataset(
        embeddings_path=embeddings_path,
        subset_size=subset_size,
        normalize_embeddings=normalize_embeddings,
        split=split,
        eval_split_ratio=eval_split_ratio,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=blip3o_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,  # Keep all samples
        **kwargs
    )
    
    return dataloader


def create_blip3o_dataloaders(
    embeddings_path: Union[str, Path],
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    subset_size: Optional[int] = None,
    normalize_embeddings: bool = True,
    eval_split_ratio: float = 0.1,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create both training and evaluation dataloaders.
    
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
        eval_dataloader is None if eval_split_ratio is 0
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size * 2  # Larger batch size for evaluation
    
    # Create training dataloader
    train_dataloader = create_blip3o_dataloader(
        embeddings_path=embeddings_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        split="train",
        subset_size=subset_size,
        normalize_embeddings=normalize_embeddings,
        eval_split_ratio=eval_split_ratio,
        **kwargs
    )
    
    # Create evaluation dataloader if eval_split_ratio > 0
    eval_dataloader = None
    if eval_split_ratio > 0:
        eval_dataloader = create_blip3o_dataloader(
            embeddings_path=embeddings_path,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            split="eval",
            subset_size=subset_size,
            normalize_embeddings=normalize_embeddings,
            eval_split_ratio=eval_split_ratio,
            **kwargs
        )
    
    return train_dataloader, eval_dataloader


def test_blip3o_dataset(embeddings_path: Union[str, Path]):
    """Test dataset loading and basic functionality."""
    print("Testing BLIP3-o dataset...")
    
    try:
        # Create dataset
        dataset = BLIP3oEmbeddingDataset(
            embeddings_path=embeddings_path,
            subset_size=100,  # Small subset for testing
            split="all"
        )
        
        # Print statistics
        stats = dataset.get_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        # Test single item
        item = dataset[0]
        print(f"\nSample item:")
        print(f"  EVA embeddings shape: {item['eva_embeddings'].shape}")
        print(f"  CLIP embeddings shape: {item['clip_embeddings'].shape}")
        print(f"  Caption: {item['caption']}")
        print(f"  Key: {item['key']}")
        
        # Test dataloader
        dataloader = create_blip3o_dataloader(
            embeddings_path=embeddings_path,
            batch_size=8,
            subset_size=50,
            num_workers=0,  # Avoid multiprocessing in testing
            split="all"
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch shapes:")
        print(f"  EVA embeddings: {batch['eva_embeddings'].shape}")
        print(f"  CLIP embeddings: {batch['clip_embeddings'].shape}")
        print(f"  Number of captions: {len(batch['captions'])}")
        print(f"  Number of keys: {len(batch['keys'])}")
        
        # Test train/eval split
        train_loader, eval_loader = create_blip3o_dataloaders(
            embeddings_path=embeddings_path,
            batch_size=8,
            subset_size=50,
            num_workers=0,
            eval_split_ratio=0.2
        )
        
        train_batch = next(iter(train_loader))
        eval_batch = next(iter(eval_loader))
        
        print(f"\nTrain/Eval split test:")
        print(f"  Train batch size: {train_batch['eva_embeddings'].shape[0]}")
        print(f"  Eval batch size: {eval_batch['eva_embeddings'].shape[0]}")
        
        print("\n✅ Dataset test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        embeddings_path = sys.argv[1]
        test_blip3o_dataset(embeddings_path)
    else:
        print("Usage: python blip3o_dataset.py <embeddings_path>")
        print("Example: python blip3o_dataset.py path/to/blip3o_grid_embeddings.pkl")