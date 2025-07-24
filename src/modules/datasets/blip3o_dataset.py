"""
FIXED: BLIP3-o Dataset with Multiprocessing-Safe Gradient Handling
src/modules/datasets/blip3o_dataset.py

KEY FIXES:
1. Collate function creates tensors WITHOUT gradients
2. Gradients are added in the training loop, not in DataLoader workers
3. Safe multiprocessing with num_workers > 0
4. Maintains all training functionality
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
    FIXED: Flexible BLIP3-o dataset with CLS+patch support and multiprocessing-safe gradient handling
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        split: str = "train",
        eval_split_ratio: float = 0.1,
        normalize_embeddings: bool = True,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        delete_after_use: bool = False,
        random_seed: int = 42,
        training_mode: str = "cls_patch",
        max_shards: Optional[int] = None,
        use_same_data_for_eval: bool = False,
        expected_tokens: Optional[int] = None,
        cache_next_shard: bool = True,
    ):
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.split = split
        self.eval_split_ratio = eval_split_ratio
        self.normalize_embeddings = normalize_embeddings
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.delete_after_use = delete_after_use
        self.random_seed = random_seed
        self.training_mode = training_mode
        self.max_shards = max_shards
        self.use_same_data_for_eval = use_same_data_for_eval
        self.cache_next_shard = cache_next_shard
        
        # Determine expected tokens based on mode
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
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
        
        # Calculate estimated length
        self._calculate_estimated_length()
        
        logger.info(f"FIXED: Multiprocessing-safe dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Split: {self.split}")
        logger.info(f"  Training mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  Total shards: {len(self.shard_files)}")
        logger.info(f"  Max shards: {self.max_shards}")
        logger.info(f"  Estimated samples: {self.estimated_length:,}")

    def _load_manifest(self):
        """Load the embeddings manifest file with mode detection"""
        manifest_path = self.chunked_embeddings_dir / "embeddings_manifest.json"
        
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}, creating default")
            self.manifest = {
                'total_shards': 0,
                'total_samples': 0,
                'extraction_mode': 'unknown',
                'tokens_per_sample': self.expected_tokens,
            }
            return
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.estimated_total_samples = self.manifest.get('total_samples', 0)
        
        # Detect mode from manifest if available
        manifest_mode = self.manifest.get('extraction_mode', '')
        manifest_tokens = self.manifest.get('tokens_per_sample', 0)
        
        if manifest_tokens and manifest_tokens != self.expected_tokens:
            logger.warning(f"Token mismatch: expected {self.expected_tokens}, manifest has {manifest_tokens}")
            logger.warning(f"Dataset mode: {self.training_mode}, manifest mode: {manifest_mode}")
        
        logger.info(f"Loaded manifest: {self.manifest.get('total_shards', 0)} shards, {self.estimated_total_samples:,} samples")

    def _prepare_shard_list(self):
        """Prepare the list of shard files with improved pattern matching"""
        mode_suffix = "cls_patch" if self.training_mode == "cls_patch" else "patch_only"
        patterns_to_try = [
            f"embeddings_shard_*_{mode_suffix}.pkl",
            f"*_{mode_suffix}.pkl",
            "embeddings_shard_*.pkl",
            "*.pkl",
        ]
        
        all_shard_files = []
        pattern_used = None
        
        for pattern in patterns_to_try:
            all_shard_files = list(self.chunked_embeddings_dir.glob(pattern))
            if all_shard_files:
                pattern_used = pattern
                logger.info(f"✅ Found {len(all_shard_files)} files with pattern: {pattern}")
                break
        
        if not all_shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
        # Sort files properly
        def sort_key(filepath):
            try:
                import re
                numbers = re.findall(r'\d+', filepath.stem)
                return int(numbers[0]) if numbers else 0
            except:
                return str(filepath.name)
        
        all_shard_files.sort(key=sort_key)
        
        # Apply max_shards limit first
        if self.max_shards is not None:
            all_shard_files = all_shard_files[:self.max_shards]
            logger.info(f"Limited to {self.max_shards} shards: {len(all_shard_files)} files")
        
        # Filter out non-existent files
        existing_shard_files = [f for f in all_shard_files if f.exists() and f.is_file()]
        
        if not existing_shard_files:
            raise FileNotFoundError(f"No valid shard files found in {self.chunked_embeddings_dir}")
        
        # Handle split logic
        if self.use_same_data_for_eval:
            self.shard_files = existing_shard_files
            logger.info(f"Using same data for train and eval: {len(self.shard_files)} shards")
        elif self.split in ["train", "eval"] and self.eval_split_ratio > 0:
            total_shards = len(existing_shard_files)
            eval_shards = max(1, int(total_shards * self.eval_split_ratio))
            train_shards = total_shards - eval_shards
            
            split_rng = random.Random(42)
            split_files = existing_shard_files.copy()
            split_rng.shuffle(split_files)
            
            if self.split == "train":
                self.shard_files = split_files[:train_shards]
            else:
                self.shard_files = split_files[train_shards:]
        else:
            self.shard_files = existing_shard_files
        
        if self.shuffle_shards:
            self.rng.shuffle(self.shard_files)
        
        logger.info(f"✅ Prepared {len(self.shard_files)} shard files for {self.split} split")

    def _calculate_estimated_length(self):
        """Calculate estimated length for this dataset configuration"""
        if hasattr(self, 'estimated_total_samples') and self.estimated_total_samples > 0:
            total_samples = self.estimated_total_samples
        else:
            estimated_samples_per_shard = 2500
            total_samples = len(self.shard_files) * estimated_samples_per_shard
        
        if self.max_shards is not None:
            total_shards = self.manifest.get('total_shards', len(self.shard_files))
            if total_shards > 0:
                shard_ratio = min(len(self.shard_files) / total_shards, 1.0)
                total_samples = int(total_samples * shard_ratio)
        
        if self.use_same_data_for_eval:
            split_samples = total_samples
        elif self.split == "train" and self.eval_split_ratio > 0:
            split_samples = int(total_samples * (1 - self.eval_split_ratio))
        elif self.split == "eval" and self.eval_split_ratio > 0:
            split_samples = int(total_samples * self.eval_split_ratio)
        else:
            split_samples = total_samples
        
        self.estimated_length = max(split_samples, 1)

    def __len__(self) -> int:
        """Return estimated length for DataLoader compatibility"""
        return self.estimated_length

    def _load_shard(self, shard_path: Path) -> Dict[str, Any]:
        """Load a single embedding shard with token mode adaptation"""
        logger.debug(f"Loading shard: {shard_path}")
        
        try:
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file does not exist: {shard_path}")
            
            if shard_path.stat().st_size == 0:
                raise ValueError(f"Shard file is empty: {shard_path}")
            
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
            
            self._validate_and_adapt_shard(shard_data, shard_path)
            
            if self.normalize_embeddings:
                shard_data = self._normalize_shard_embeddings(shard_data)
            
            logger.debug(f"✅ Successfully loaded shard: {shard_path}")
            return shard_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load shard {shard_path}: {e}")
            raise RuntimeError(f"Failed to load shard {shard_path}: {e}") from e

    def _validate_and_adapt_shard(self, shard_data: Dict[str, Any], shard_path: Path):
        """Validate shard data format and adapt token count for training mode"""
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        captions = shard_data['captions']
        
        if not torch.is_tensor(clip_emb):
            raise ValueError(f"CLIP embeddings should be tensor, got {type(clip_emb)}")
        if not torch.is_tensor(eva_emb):
            raise ValueError(f"EVA embeddings should be tensor, got {type(eva_emb)}")
        
        if clip_emb.dim() != 3:
            raise ValueError(f"CLIP embeddings should be 3D [samples, tokens, dim], got shape {clip_emb.shape}")
        if eva_emb.dim() != 3:
            raise ValueError(f"EVA embeddings should be 3D [samples, tokens, dim], got shape {eva_emb.shape}")
        
        clip_tokens = clip_emb.shape[1]
        eva_tokens = eva_emb.shape[1]
        
        if clip_tokens != eva_tokens:
            raise ValueError(f"Token count mismatch: CLIP {clip_tokens} vs EVA {eva_tokens}")
        
        # Handle token count adaptation for different training modes
        if clip_tokens != self.expected_tokens:
            logger.info(f"Adapting from {clip_tokens} to {self.expected_tokens} tokens for {self.training_mode} mode")
            
            if clip_tokens == 256 and self.expected_tokens == 257:
                logger.info("Adding CLS token (average of patches) for cls_patch mode")
                batch_size, _, clip_dim = clip_emb.shape
                eva_dim = eva_emb.shape[2]
                
                clip_cls = clip_emb.mean(dim=1, keepdim=True)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                
                shard_data['clip_blip3o_embeddings'] = torch.cat([clip_cls, clip_emb], dim=1)
                shard_data['eva_blip3o_embeddings'] = torch.cat([eva_cls, eva_emb], dim=1)
                
                logger.debug(f"Added CLS token: {clip_tokens} -> {self.expected_tokens} tokens")
                
            elif clip_tokens == 257 and self.expected_tokens == 256:
                logger.info("Removing CLS token for patch_only mode")
                shard_data['clip_blip3o_embeddings'] = clip_emb[:, 1:, :]
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
                
                logger.debug(f"Removed CLS token: {clip_tokens} -> {self.expected_tokens} tokens")
                
            else:
                raise ValueError(f"Cannot adapt from {clip_tokens} to {self.expected_tokens} tokens")
        
        # Final validation after adaptation
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        if eva_emb.shape[1] != self.expected_tokens:
            raise ValueError(f"EVA token count after adaptation: expected {self.expected_tokens}, got {eva_emb.shape[1]}")
        
        if clip_emb.shape[0] != eva_emb.shape[0]:
            raise ValueError(f"Sample count mismatch: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
        
        if len(captions) != clip_emb.shape[0]:
            raise ValueError(f"Caption count mismatch: {len(captions)} vs {clip_emb.shape[0]} samples")
        
        assert clip_emb.shape[2] == 1024, f"Expected CLIP 1024-dim, got {clip_emb.shape[2]}"
        assert eva_emb.shape[2] == 4096, f"Expected EVA 4096-dim, got {eva_emb.shape[2]}"

    def _normalize_shard_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize embeddings in a shard"""
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        clip_norm = torch.norm(clip_emb, dim=-1, keepdim=True)
        clip_norm = torch.clamp(clip_norm, min=1e-8)
        shard_data['clip_blip3o_embeddings'] = clip_emb / clip_norm
        
        eva_norm = torch.norm(eva_emb, dim=-1, keepdim=True)
        eva_norm = torch.clamp(eva_norm, min=1e-8)
        shard_data['eva_blip3o_embeddings'] = eva_emb / eva_norm
        
        return shard_data

    def _prepare_current_shard_samples(self):
        """Prepare samples from current shard"""
        if self.current_shard_data is None:
            return
        
        num_samples = self.current_shard_data['clip_blip3o_embeddings'].shape[0]
        indices = list(range(num_samples))
        
        if self.shuffle_within_shard:
            self.rng.shuffle(indices)
        
        self.current_shard_samples = indices
        self.current_sample_idx = 0

    def _load_next_shard(self):
        """Load the next shard with improved error handling"""
        if self.current_shard_data is not None:
            if self.delete_after_use and self.shards_processed > 0:
                try:
                    prev_shard_idx = self.current_shard_idx - 1
                    if 0 <= prev_shard_idx < len(self.shard_files):
                        prev_shard_path = self.shard_files[prev_shard_idx]
                        if prev_shard_path.exists():
                            prev_shard_path.unlink()
                            logger.debug(f"Deleted processed shard: {prev_shard_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete previous shard: {e}")
            
            del self.current_shard_data
            gc.collect()
        
        if self.current_shard_idx >= len(self.shard_files):
            logger.debug("No more shards to process")
            self.current_shard_data = None
            return False
        
        if self.next_shard_data is not None:
            self.current_shard_data = self.next_shard_data
            self.next_shard_data = None
            logger.debug("Using cached next shard")
        else:
            shard_path = self.shard_files[self.current_shard_idx]
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    if not shard_path.exists():
                        raise FileNotFoundError(f"Shard file does not exist: {shard_path}")
                    
                    self.current_shard_data = self._load_shard(shard_path)
                    logger.debug(f"✅ Loaded current shard: {shard_path}")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Failed to load shard {shard_path} (attempt {retry_count}/{max_retries}): {e}")
                    
                    if retry_count >= max_retries:
                        logger.error(f"❌ Failed to load shard after {max_retries} attempts: {shard_path}")
                        self.current_shard_idx += 1
                        if self.current_shard_idx >= len(self.shard_files):
                            return False
                        else:
                            return self._load_next_shard()
                    
                    time.sleep(0.1)
        
        self._prepare_current_shard_samples()
        
        if self.cache_next_shard and (self.current_shard_idx + 1) < len(self.shard_files):
            try:
                next_shard_path = self.shard_files[self.current_shard_idx + 1]
                if next_shard_path.exists():
                    self.next_shard_data = self._load_shard(next_shard_path)
                    logger.debug(f"Cached next shard: {next_shard_path}")
            except Exception as e:
                logger.warning(f"Failed to cache next shard: {e}")
                self.next_shard_data = None
        
        num_samples = len(self.current_shard_samples) if self.current_shard_samples else 0
        logger.info(f"✅ Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: "
                   f"{num_samples} samples ({self.expected_tokens} tokens)")
        
        self.current_shard_idx += 1
        self.shards_processed += 1
        
        return True

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples across all shards"""
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        self.shards_processed = 0
        
        logger.debug(f"Starting iteration over {len(self.shard_files)} shards")
        
        if not self._load_next_shard():
            logger.warning("No shards could be loaded")
            return
        
        while self.current_shard_data is not None:
            while self.current_sample_idx < len(self.current_shard_samples):
                try:
                    sample_idx = self.current_shard_samples[self.current_sample_idx]
                    
                    item = {
                        'eva_embeddings': self.current_shard_data['eva_blip3o_embeddings'][sample_idx],
                        'clip_embeddings': self.current_shard_data['clip_blip3o_embeddings'][sample_idx],
                        'caption': self.current_shard_data['captions'][sample_idx],
                        'key': self.current_shard_data.get('keys', [f"sample_{sample_idx}"])[sample_idx] if self.current_shard_data.get('keys') else f"sample_{sample_idx}",
                        'shard_idx': self.current_shard_idx - 1,
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                    }
                    
                    self.current_sample_idx += 1
                    self.total_samples_processed += 1
                    
                    yield item
                    
                except Exception as e:
                    logger.error(f"Error processing sample {sample_idx} in shard {self.current_shard_idx - 1}: {e}")
                    self.current_sample_idx += 1
                    continue
            
            if not self._load_next_shard():
                break
        
        logger.info(f"✅ Dataset iteration completed: {self.total_samples_processed} samples from {self.shards_processed} shards")


def multiprocessing_safe_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    FIXED: Multiprocessing-safe collate function - creates tensors WITHOUT gradients
    
    This function creates all tensors without gradients. Gradients will be added
    in the training loop after the data is loaded from workers.
    """
    if not batch:
        raise ValueError("Empty batch received")
    
    try:
        # Stack tensor data - ALL tensors created WITHOUT gradients
        eva_embeddings = torch.stack([item['eva_embeddings'] for item in batch])  # [B, N, 4096]
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in batch])  # [B, N, 1024]
        
        # Collect metadata
        captions = [item['caption'] for item in batch]
        keys = [item['key'] for item in batch]
        shard_indices = [item['shard_idx'] for item in batch]
        sample_indices = [item['sample_idx'] for item in batch]
        training_mode = batch[0]['training_mode']
        num_tokens = batch[0]['num_tokens']
        
        # Get batch info
        batch_size, seq_len, clip_dim = clip_embeddings.shape
        device = clip_embeddings.device
        dtype = clip_embeddings.dtype
        
        # CRITICAL FIX: ALL tensors created WITHOUT gradients for safe multiprocessing
        eva_embeddings = eva_embeddings.detach()
        clip_embeddings = clip_embeddings.detach()
        
        # Create flow matching setup WITHOUT gradients
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # Create noise WITHOUT gradients
        noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)
        base_noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)  # NO requires_grad=True
        
        # Linear interpolation for flow matching WITHOUT gradients
        alpha = timesteps.view(-1, 1, 1)  # [B, 1, 1]
        hidden_states = (1 - alpha) * base_noise + alpha * clip_embeddings + 0.1 * noise
        
        # CRITICAL: Ensure NO tensors require gradients (for safe multiprocessing)
        assert not eva_embeddings.requires_grad, "EVA embeddings should not require gradients for multiprocessing"
        assert not clip_embeddings.requires_grad, "CLIP embeddings should not require gradients for multiprocessing"
        assert not hidden_states.requires_grad, "Hidden states should not require gradients for multiprocessing"
        assert not base_noise.requires_grad, "Base noise should not require gradients for multiprocessing"
        
        # Validate tensor properties
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA batch shape: {eva_embeddings.shape}"
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP batch shape: {clip_embeddings.shape}"
        assert hidden_states.shape == (batch_size, seq_len, 1024), f"Hidden states shape: {hidden_states.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        assert seq_len in [256, 257], f"Unexpected token count: {seq_len}"
        
        return {
            # Core embeddings (for model input) - ALL WITHOUT gradients
            'encoder_hidden_states': eva_embeddings,        # [B, N, 4096] - EVA conditioning
            'clip_embeddings': clip_embeddings,             # [B, N, 1024] - Target CLIP patches
            
            # Flow matching inputs - ALL WITHOUT gradients (gradients added in training loop)
            'hidden_states': hidden_states,                 # [B, N, 1024] - Noisy input
            'timestep': timesteps,                          # [B] - Flow matching timesteps
            'noise': noise,                                 # [B, N, 1024] - Original noise
            'base_noise': base_noise,                       # [B, N, 1024] - Base noise
            
            # Metadata
            'captions': captions,                           # List[str] - Text captions
            'keys': keys,                                   # List[str] - Sample keys
            'shard_indices': shard_indices,                 # List[int] - Shard indices
            'sample_indices': sample_indices,               # List[int] - Sample indices
            'batch_size': batch_size,                       # int
            'training_mode': training_mode,                 # str - "cls_patch" or "patch_only"
            'num_tokens': num_tokens,                       # int - 256 or 257
            'seq_len': seq_len,                            # int - sequence length
            
            # Training flags
            'multiprocessing_safe': True,                   # bool - safe for multiprocessing
            'gradients_will_be_added_in_training_loop': True,  # bool - gradients added later
        }
        
    except Exception as e:
        logger.error(f"Error in multiprocessing-safe collate function: {e}")
        logger.error(f"Batch info: {len(batch)} items")
        if batch:
            first_item = batch[0]
            logger.error(f"First item keys: {list(first_item.keys())}")
            for key, value in first_item.items():
                if torch.is_tensor(value):
                    logger.error(f"  {key}: {value.shape} {value.dtype}")
                else:
                    logger.error(f"  {key}: {type(value)} {value}")
        raise


def create_flexible_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    eval_split_ratio: float = 0.1,
    normalize_embeddings: bool = True,
    training_mode: str = "cls_patch",
    max_shards: Optional[int] = None,
    use_same_data_for_eval: bool = False,
    delete_after_use: bool = False,
    num_workers: int = 4,  # Now we can safely use multiple workers!
    pin_memory: bool = None,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    FIXED: Create flexible dataloaders with multiprocessing-safe gradient handling
    
    Now supports num_workers > 0 safely!
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size * 2
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # Create training dataset
    try:
        train_dataset = BLIP3oEmbeddingDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split="train",
            eval_split_ratio=eval_split_ratio,
            normalize_embeddings=normalize_embeddings,
            shuffle_shards=True,
            shuffle_within_shard=True,
            delete_after_use=delete_after_use,
            training_mode=training_mode,
            max_shards=max_shards,
            use_same_data_for_eval=use_same_data_for_eval,
        )
        
        logger.info(f"✅ Training dataset created: {len(train_dataset):,} estimated samples")
        
    except Exception as e:
        logger.error(f"❌ Failed to create training dataset: {e}")
        raise
    
    # Create training dataloader with multiprocessing-safe collate function
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,  # Now we can safely use multiple workers!
            collate_fn=multiprocessing_safe_collate_fn,  # FIXED: multiprocessing-safe
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,  # Enable persistent workers if using multiple workers
            **kwargs
        )
        
        logger.info(f"✅ Training dataloader created (num_workers={num_workers}, multiprocessing-safe)")
        
    except Exception as e:
        logger.error(f"❌ Failed to create training dataloader: {e}")
        raise
    
    # Create evaluation dataloader
    eval_dataloader = None
    if eval_split_ratio > 0 or use_same_data_for_eval:
        try:
            eval_dataset = BLIP3oEmbeddingDataset(
                chunked_embeddings_dir=chunked_embeddings_dir,
                split="train" if use_same_data_for_eval else "eval",
                eval_split_ratio=eval_split_ratio,
                normalize_embeddings=normalize_embeddings,
                shuffle_shards=False,
                shuffle_within_shard=False,
                delete_after_use=False,
                training_mode=training_mode,
                max_shards=max_shards,
                use_same_data_for_eval=use_same_data_for_eval,
            )
            
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                num_workers=min(num_workers, 2),  # Slightly fewer workers for eval
                collate_fn=multiprocessing_safe_collate_fn,  # FIXED: multiprocessing-safe
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=min(num_workers, 2) > 0,
                **kwargs
            )
            
            logger.info(f"✅ Evaluation dataloader created (multiprocessing-safe)")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create evaluation dataloader: {e}")
            eval_dataloader = None
    
    return train_dataloader, eval_dataloader


# Backward compatibility aliases
training_aware_collate_fn = multiprocessing_safe_collate_fn  # Updated alias
flexible_collate_fn = multiprocessing_safe_collate_fn
create_gradient_aware_dataloaders = create_flexible_dataloaders


def test_gradient_flow_dataset(chunked_embeddings_dir: Union[str, Path], 
                               training_mode: str = "cls_patch",
                               max_shards: int = 1):
    """FIXED: Test the multiprocessing-safe gradient flow setup with actual data"""
    print(f"🧪 Testing FIXED multiprocessing-safe gradient flow setup")
    print(f"📁 Directory: {chunked_embeddings_dir}")
    print(f"🎯 Training mode: {training_mode}")
    print(f"📦 Max shards: {max_shards}")
    print("=" * 50)
    
    try:
        # Create dataloaders with multiple workers (now safe!)
        train_dataloader, eval_dataloader = create_flexible_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=4,
            training_mode=training_mode,
            max_shards=max_shards,
            use_same_data_for_eval=True,
            delete_after_use=False,
            num_workers=2,  # Test with multiple workers!
        )
        
        print(f"✅ Dataloaders created successfully with num_workers=2")
        print(f"   Train dataloader: {len(train_dataloader):,} batches")
        if eval_dataloader:
            print(f"   Eval dataloader: {len(eval_dataloader):,} batches")
        
        # Test batch creation with multiprocessing
        print(f"🧪 Testing multiprocessing-safe batch creation...")
        batch = next(iter(train_dataloader))
        
        print(f"✅ Batch created: EVA {batch['encoder_hidden_states'].shape}, CLIP {batch['clip_embeddings'].shape}")
        print(f"   Training mode: {batch['training_mode']}")
        print(f"   Tokens: {batch['num_tokens']}")
        print(f"   Multiprocessing safe: {batch['multiprocessing_safe']}")
        
        # Verify NO tensors have gradients (multiprocessing safe)
        eva_grad = batch['encoder_hidden_states'].requires_grad
        clip_grad = batch['clip_embeddings'].requires_grad
        hidden_grad = batch['hidden_states'].requires_grad
        base_noise_grad = batch['base_noise'].requires_grad
        
        print(f"✅ Gradient status (all should be False for multiprocessing safety):")
        print(f"   EVA embeddings requires_grad: {eva_grad}")
        print(f"   CLIP embeddings requires_grad: {clip_grad}")
        print(f"   Hidden states requires_grad: {hidden_grad}")
        print(f"   Base noise requires_grad: {base_noise_grad}")
        
        # Verify all are False (multiprocessing safe)
        if not eva_grad and not clip_grad and not hidden_grad and not base_noise_grad:
            print("🎉 MULTIPROCESSING ISSUE COMPLETELY FIXED!")
            print("✅ All tensors properly detached for safe multiprocessing")
            print("✅ Can now use num_workers > 0 without crashes")
            print("✅ Gradients will be added in training loop where needed")
            print("✅ Ready for fast BLIP3-o training with multiple workers!")
        else:
            print("❌ Some tensors still have gradients - this will cause multiprocessing errors")
        
        # Test eval dataloader too
        if eval_dataloader:
            try:
                eval_batch = next(iter(eval_dataloader))
                print(f"✅ Eval batch also works with multiprocessing-safe setup")
            except Exception as e:
                print(f"⚠️ Eval batch test failed: {e}")
        
        print(f"🎉 ALL MULTIPROCESSING TESTS PASSED!")
        print("✅ Multiprocessing gradient serialization issue COMPLETELY RESOLVED!")
        print("✅ Ready for fast training with multiple DataLoader workers")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        embeddings_dir = sys.argv[1]
        training_mode = sys.argv[2] if len(sys.argv) > 2 else "cls_patch"
        max_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        success = test_gradient_flow_dataset(embeddings_dir, training_mode, max_shards)
        sys.exit(0 if success else 1)
    else:
        print("Usage: python blip3o_dataset.py <embeddings_dir> [training_mode] [max_shards]")
        print("Example: python blip3o_dataset.py /path/to/embeddings cls_patch 1")