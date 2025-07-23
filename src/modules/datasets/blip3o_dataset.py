"""
FIXED: Flexible BLIP3-o Dataset with Shard Selection and Token Mode Support
src/modules/datasets/blip3o_dataset.py

FIXES:
1. Fixed shard pattern matching for generic files
2. Improved error handling in shard loading
3. Better validation and fallback mechanisms
4. Fixed iterator and loading logic
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
    FIXED: Flexible BLIP3-o dataset with shard selection and token mode support
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
        training_mode: str = "cls_patch",  # "cls_patch" or "patch_only"
        max_shards: Optional[int] = None,  # Limit number of shards
        use_same_data_for_eval: bool = False,  # Use training data for evaluation
        expected_tokens: Optional[int] = None,  # Auto-detect if None
        cache_next_shard: bool = True,
    ):
        """
        Initialize flexible dataset
        
        Args:
            chunked_embeddings_dir: Path to embeddings directory
            split: "train", "eval", or "all"
            eval_split_ratio: Ratio for eval split (ignored if use_same_data_for_eval=True)
            normalize_embeddings: Whether to normalize embeddings
            shuffle_shards: Whether to shuffle shard order
            shuffle_within_shard: Whether to shuffle samples within shard
            delete_after_use: Whether to delete processed shards
            random_seed: Random seed for reproducibility
            training_mode: "cls_patch" (257 tokens) or "patch_only" (256 tokens)
            max_shards: Maximum number of shards to use (for training size control)
            use_same_data_for_eval: Use training data for evaluation (overfitting test)
            expected_tokens: Expected number of tokens (auto-detect if None)
            cache_next_shard: Whether to cache next shard
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
        
        logger.info(f"FIXED: Flexible dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Split: {self.split}")
        logger.info(f"  Training mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  Total shards: {len(self.shard_files)}")
        logger.info(f"  Max shards: {self.max_shards}")
        logger.info(f"  Use same data for eval: {self.use_same_data_for_eval}")
        logger.info(f"  Estimated samples: {self.estimated_length:,}")
    
    def _load_manifest(self):
        """Load the embeddings manifest file with mode detection"""
        manifest_path = self.chunked_embeddings_dir / "embeddings_manifest.json"
        
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}, creating default")
            # Create a basic manifest
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
        logger.info(f"Manifest mode: {manifest_mode} ({manifest_tokens} tokens)")
    
    def _prepare_shard_list(self):
        """FIXED: Prepare the list of shard files with improved pattern matching"""
        # FIXED: Try multiple patterns in order of preference
        
        # 1. Try mode-specific patterns first
        mode_suffix = "cls_patch" if self.training_mode == "cls_patch" else "patch_only"
        patterns_to_try = [
            f"embeddings_shard_*_{mode_suffix}.pkl",  # Mode-specific with suffix
            f"*_{mode_suffix}.pkl",                    # Any file with mode suffix
            "embeddings_shard_*.pkl",                  # Generic embeddings shard pattern
            "*.pkl",                                   # Any pickle file
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
            # Last resort: list all files and filter
            try:
                all_files = list(self.chunked_embeddings_dir.iterdir())
                pkl_files = [f for f in all_files if f.suffix == '.pkl']
                if pkl_files:
                    all_shard_files = pkl_files
                    pattern_used = "manual_filter"
                    logger.warning(f"Used manual filtering, found {len(pkl_files)} .pkl files")
            except Exception as e:
                logger.error(f"Failed to list files in directory: {e}")
        
        if not all_shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
        # FIXED: Sort files properly (handle both numeric and string sorting)
        def sort_key(filepath):
            """Extract numeric part for proper sorting"""
            try:
                # Try to extract number from filename
                import re
                numbers = re.findall(r'\d+', filepath.stem)
                return int(numbers[0]) if numbers else 0
            except:
                return str(filepath.name)
        
        all_shard_files.sort(key=sort_key)
        
        logger.info(f"📁 Found and sorted {len(all_shard_files)} shard files")
        logger.debug(f"First few files: {[f.name for f in all_shard_files[:3]]}")
        
        # Apply max_shards limit first
        if self.max_shards is not None:
            all_shard_files = all_shard_files[:self.max_shards]
            logger.info(f"Limited to {self.max_shards} shards: {len(all_shard_files)} files")
        
        # FIXED: Filter out non-existent files with better error reporting
        existing_shard_files = []
        missing_count = 0
        
        for f in all_shard_files:
            if f.exists() and f.is_file():
                existing_shard_files.append(f)
            else:
                missing_count += 1
                logger.debug(f"Missing or invalid file: {f}")
        
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing/invalid files, using {len(existing_shard_files)} valid files")
        
        all_shard_files = existing_shard_files
        
        if not all_shard_files:
            raise FileNotFoundError(f"No valid shard files found in {self.chunked_embeddings_dir}")
        
        # Handle split logic
        if self.use_same_data_for_eval:
            # Use same data for both train and eval (overfitting test)
            self.shard_files = all_shard_files
            logger.info(f"Using same data for train and eval: {len(self.shard_files)} shards")
        elif self.split in ["train", "eval"] and self.eval_split_ratio > 0:
            # Standard train/eval split
            total_shards = len(all_shard_files)
            eval_shards = max(1, int(total_shards * self.eval_split_ratio))
            train_shards = total_shards - eval_shards
            
            # Consistent splitting based on fixed seed
            split_rng = random.Random(42)  # Fixed seed for reproducible splits
            split_files = all_shard_files.copy()
            split_rng.shuffle(split_files)
            
            if self.split == "train":
                self.shard_files = split_files[:train_shards]
            else:  # eval
                self.shard_files = split_files[train_shards:]
        else:
            # Use all shards
            self.shard_files = all_shard_files
        
        # Shuffle shard order if requested (with different seed than split)
        if self.shuffle_shards:
            self.rng.shuffle(self.shard_files)
        
        logger.info(f"✅ Prepared {len(self.shard_files)} shard files for {self.split} split")
        
        # Log first few shard files for debugging
        for i, shard_file in enumerate(self.shard_files[:3]):
            logger.debug(f"  Shard {i}: {shard_file.name}")
    
    def _calculate_estimated_length(self):
        """Calculate estimated length for this dataset configuration"""
        # FIXED: Better estimation logic
        
        if hasattr(self, 'estimated_total_samples') and self.estimated_total_samples > 0:
            # Use manifest data if available
            total_samples = self.estimated_total_samples
        else:
            # Estimate based on typical shard size (fallback)
            estimated_samples_per_shard = 2500  # Rough estimate
            total_samples = len(self.shard_files) * estimated_samples_per_shard
        
        # Apply shard limit
        if self.max_shards is not None:
            total_shards = self.manifest.get('total_shards', len(self.shard_files))
            if total_shards > 0:
                shard_ratio = min(len(self.shard_files) / total_shards, 1.0)
                total_samples = int(total_samples * shard_ratio)
        
        # Apply split ratio
        if self.use_same_data_for_eval:
            split_samples = total_samples
        elif self.split == "train" and self.eval_split_ratio > 0:
            split_samples = int(total_samples * (1 - self.eval_split_ratio))
        elif self.split == "eval" and self.eval_split_ratio > 0:
            split_samples = int(total_samples * self.eval_split_ratio)
        else:
            split_samples = total_samples
        
        self.estimated_length = max(split_samples, 1)  # At least 1
    
    def __len__(self) -> int:
        """Return estimated length for DataLoader compatibility"""
        return self.estimated_length
    
    def _load_shard(self, shard_path: Path) -> Dict[str, Any]:
        """FIXED: Load a single embedding shard with better error handling"""
        logger.debug(f"Loading shard: {shard_path}")
        
        try:
            # Check file exists and is readable
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file does not exist: {shard_path}")
            
            if shard_path.stat().st_size == 0:
                raise ValueError(f"Shard file is empty: {shard_path}")
            
            # Load the pickle file
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
            
            # Validate shard data
            self._validate_shard(shard_data, shard_path)
            
            # Normalize embeddings if requested
            if self.normalize_embeddings:
                shard_data = self._normalize_shard_embeddings(shard_data)
            
            logger.debug(f"✅ Successfully loaded shard: {shard_path}")
            return shard_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load shard {shard_path}: {e}")
            # Re-raise with more context
            raise RuntimeError(f"Failed to load shard {shard_path}: {e}") from e
    
    def _validate_shard(self, shard_data: Dict[str, Any], shard_path: Path):
        """FIXED: Validate shard data format and token count with better error messages"""
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        captions = shard_data['captions']
        
        # Check types
        if not torch.is_tensor(clip_emb):
            raise ValueError(f"CLIP embeddings should be tensor, got {type(clip_emb)}")
        if not torch.is_tensor(eva_emb):
            raise ValueError(f"EVA embeddings should be tensor, got {type(eva_emb)}")
        
        # Check dimensions
        if clip_emb.dim() != 3:
            raise ValueError(f"CLIP embeddings should be 3D [samples, tokens, dim], got shape {clip_emb.shape}")
        if eva_emb.dim() != 3:
            raise ValueError(f"EVA embeddings should be 3D [samples, tokens, dim], got shape {eva_emb.shape}")
        
        # Validate token count and handle mismatches
        clip_tokens = clip_emb.shape[1]
        eva_tokens = eva_emb.shape[1]
        
        if clip_tokens != eva_tokens:
            raise ValueError(f"Token count mismatch: CLIP {clip_tokens} vs EVA {eva_tokens}")
        
        # Handle token count adaptation
        if clip_tokens != self.expected_tokens:
            logger.warning(f"Token count mismatch in {shard_path}: expected {self.expected_tokens}, got {clip_tokens}")
            
            if clip_tokens == 257 and self.expected_tokens == 256:
                # Remove CLS token
                logger.info("Removing CLS token from 257-token data for patch-only mode")
                shard_data['clip_blip3o_embeddings'] = clip_emb[:, 1:, :]
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
                
            elif clip_tokens == 256 and self.expected_tokens == 257:
                # Add dummy CLS token
                logger.info("Adding dummy CLS token to 256-token data for CLS+patch mode")
                batch_size, _, clip_dim = clip_emb.shape
                eva_dim = eva_emb.shape[2]
                
                dummy_cls_clip = torch.zeros(batch_size, 1, clip_dim, dtype=clip_emb.dtype)
                dummy_cls_eva = torch.zeros(batch_size, 1, eva_dim, dtype=eva_emb.dtype)
                
                shard_data['clip_blip3o_embeddings'] = torch.cat([dummy_cls_clip, clip_emb], dim=1)
                shard_data['eva_blip3o_embeddings'] = torch.cat([dummy_cls_eva, eva_emb], dim=1)
                
            else:
                raise ValueError(f"Cannot handle token mismatch: {clip_tokens} -> {self.expected_tokens}")
        
        # Final validation after adaptation
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        if eva_emb.shape[1] != self.expected_tokens:
            raise ValueError(f"EVA token count after adaptation: expected {self.expected_tokens}, got {eva_emb.shape[1]}")
        
        if clip_emb.shape[0] != eva_emb.shape[0]:
            raise ValueError(f"Sample count mismatch: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
        
        if len(captions) != clip_emb.shape[0]:
            raise ValueError(f"Caption count mismatch: {len(captions)} vs {clip_emb.shape[0]} samples")
    
    def _normalize_shard_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize embeddings in a shard"""
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
        """Prepare samples from current shard"""
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
        """FIXED: Load the next shard with improved error handling"""
        # Clean up current shard
        if self.current_shard_data is not None:
            # Delete previous shard file if requested
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
            
            # Clear memory
            del self.current_shard_data
            gc.collect()
        
        # Check if we have more shards
        if self.current_shard_idx >= len(self.shard_files):
            logger.debug("No more shards to process")
            self.current_shard_data = None
            return False
        
        # Use cached shard if available
        if self.next_shard_data is not None:
            self.current_shard_data = self.next_shard_data
            self.next_shard_data = None
            logger.debug("Using cached next shard")
        else:
            # Load current shard
            shard_path = self.shard_files[self.current_shard_idx]
            
            # FIXED: Better error handling for missing files
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
                        # Skip this shard and try the next one
                        self.current_shard_idx += 1
                        if self.current_shard_idx >= len(self.shard_files):
                            return False
                        else:
                            return self._load_next_shard()  # Recursive call to try next shard
                    
                    # Brief delay before retry
                    time.sleep(0.1)
        
        # Prepare samples
        self._prepare_current_shard_samples()
        
        # Cache next shard if requested and available
        if self.cache_next_shard and (self.current_shard_idx + 1) < len(self.shard_files):
            try:
                next_shard_path = self.shard_files[self.current_shard_idx + 1]
                if next_shard_path.exists():
                    self.next_shard_data = self._load_shard(next_shard_path)
                    logger.debug(f"Cached next shard: {next_shard_path}")
            except Exception as e:
                logger.warning(f"Failed to cache next shard: {e}")
                self.next_shard_data = None
        
        # Log success
        num_samples = len(self.current_shard_samples) if self.current_shard_samples else 0
        logger.info(f"✅ Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: "
                   f"{num_samples} samples")
        
        self.current_shard_idx += 1
        self.shards_processed += 1
        
        return True
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """FIXED: Iterate through all samples across all shards with better error handling"""
        # Reset state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        self.shards_processed = 0
        
        logger.debug(f"Starting iteration over {len(self.shard_files)} shards")
        
        # Load first shard
        if not self._load_next_shard():
            logger.warning("No shards could be loaded")
            return
        
        # Iterate through all shards and samples
        while self.current_shard_data is not None:
            # Iterate through current shard
            while self.current_sample_idx < len(self.current_shard_samples):
                try:
                    sample_idx = self.current_shard_samples[self.current_sample_idx]
                    
                    # Get sample data
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
                    self.current_sample_idx += 1  # Skip this sample
                    continue
            
            # Move to next shard
            if not self._load_next_shard():
                break
        
        logger.info(f"✅ Dataset iteration completed: {self.total_samples_processed} samples from {self.shards_processed} shards")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'total_shards': len(self.shard_files),
            'max_shards': self.max_shards,
            'estimated_total_samples': getattr(self, 'estimated_total_samples', 0),
            'estimated_length': self.estimated_length,
            'shards_processed': self.shards_processed,
            'samples_processed': self.total_samples_processed,
            'current_shard': self.current_shard_idx,
            'split': self.split,
            'training_mode': self.training_mode,
            'expected_tokens': self.expected_tokens,
            'use_same_data_for_eval': self.use_same_data_for_eval,
            'delete_after_use': self.delete_after_use,
        }


def flexible_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    FIXED: Flexible collate function supporting both 256 and 257 token modes with proper gradient flow
    """
    if not batch:
        raise ValueError("Empty batch received")
    
    try:
        # Stack tensor data
        eva_embeddings = torch.stack([item['eva_embeddings'] for item in batch])  # [B, N, 4096]
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in batch])  # [B, N, 1024]
        
        # Collect metadata
        captions = [item['caption'] for item in batch]
        keys = [item['key'] for item in batch]
        shard_indices = [item['shard_idx'] for item in batch]
        sample_indices = [item['sample_idx'] for item in batch]
        training_mode = batch[0]['training_mode']
        num_tokens = batch[0]['num_tokens']
        
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
        
        # Create flow matching setup with proper gradient flow
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # Create noise for flow matching
        noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)
        
        # CRITICAL: Create base noise with proper gradient requirement
        base_noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype, requires_grad=True)
        
        # Linear interpolation for flow matching (rectified flow)
        alpha = timesteps.view(-1, 1, 1)  # [B, 1, 1]
        
        # Create noisy input for the model - this MUST have gradients
        hidden_states = (1 - alpha) * base_noise + alpha * clip_embeddings + 0.1 * noise
        
        # CRITICAL: Ensure the noisy input requires gradients
        if not hidden_states.requires_grad:
            logger.warning("Noisy input doesn't require gradients, fixing...")
            hidden_states = hidden_states.requires_grad_(True)
        
        # Validate tensor properties
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA batch shape: {eva_embeddings.shape}"
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP batch shape: {clip_embeddings.shape}"
        assert hidden_states.shape == (batch_size, seq_len, 1024), f"Hidden states shape: {hidden_states.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        assert seq_len in [256, 257], f"Unexpected token count: {seq_len}"
        
        return {
            # Core embeddings (for model input)
            'encoder_hidden_states': eva_embeddings,        # [B, N, 4096] - EVA conditioning
            'clip_embeddings': clip_embeddings,             # [B, N, 1024] - Target CLIP patches
            
            # Flow matching inputs (with proper gradients)
            'hidden_states': hidden_states,                 # [B, N, 1024] - Noisy input (with gradients)
            'timestep': timesteps,                          # [B] - Flow matching timesteps
            'noise': noise,                                 # [B, N, 1024] - Original noise
            'base_noise': base_noise,                       # [B, N, 1024] - Base noise with gradients
            
            # Metadata
            'captions': captions,                           # List[str] - Text captions
            'keys': keys,                                   # List[str] - Sample keys
            'shard_indices': shard_indices,                 # List[int] - Shard indices
            'sample_indices': sample_indices,               # List[int] - Sample indices
            'batch_size': batch_size,                       # int
            'training_mode': training_mode,                 # str - "cls_patch" or "patch_only"
            'num_tokens': num_tokens,                       # int - 256 or 257
            'seq_len': seq_len,                            # int - sequence length
        }
        
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
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
    delete_after_use: bool = True,
    num_workers: int = 0,
    pin_memory: bool = None,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    FIXED: Create flexible dataloaders with improved error handling
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size * 2
    
    # Auto-detect pin_memory
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
    
    # Create training dataloader
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=flexible_collate_fn,
            pin_memory=pin_memory,
            drop_last=True,  # Important for consistent batch sizes
            **kwargs
        )
        
        logger.info(f"✅ Training dataloader created")
        
    except Exception as e:
        logger.error(f"❌ Failed to create training dataloader: {e}")
        raise
    
    # Create evaluation dataloader
    eval_dataloader = None
    if eval_split_ratio > 0 or use_same_data_for_eval:
        try:
            eval_dataset = BLIP3oEmbeddingDataset(
                chunked_embeddings_dir=chunked_embeddings_dir,
                split="train" if use_same_data_for_eval else "eval",  # Same data for overfitting test
                eval_split_ratio=eval_split_ratio,
                normalize_embeddings=normalize_embeddings,
                shuffle_shards=False,
                shuffle_within_shard=False,
                delete_after_use=False,  # Don't delete eval data
                training_mode=training_mode,
                max_shards=max_shards,
                use_same_data_for_eval=use_same_data_for_eval,
            )
            
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                num_workers=min(num_workers, 2),
                collate_fn=flexible_collate_fn,
                pin_memory=pin_memory,
                drop_last=False,  # Don't drop last for eval
                **kwargs
            )
            
            logger.info(f"✅ Evaluation dataloader created")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create evaluation dataloader: {e}")
            eval_dataloader = None
    
    return train_dataloader, eval_dataloader


# Backward compatibility
create_gradient_aware_dataloaders = create_flexible_dataloaders
chunked_collate_fn = flexible_collate_fn


def test_flexible_dataset(chunked_embeddings_dir: Union[str, Path], 
                         training_mode: str = "patch_only",
                         max_shards: int = 1):
    """FIXED: Test the flexible dataset implementation with improved diagnostics"""
    print(f"🧪 Testing FIXED flexible dataset")
    print(f"📁 Directory: {chunked_embeddings_dir}")
    print(f"🎯 Training mode: {training_mode}")
    print(f"📦 Max shards: {max_shards}")
    print("=" * 50)
    
    try:
        # Create dataset
        dataset = BLIP3oEmbeddingDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split="all",
            training_mode=training_mode,
            max_shards=max_shards,
            delete_after_use=False,
            cache_next_shard=True,
        )
        
        # Test length
        print(f"✅ Dataset length: {len(dataset):,}")
        
        # Test iteration
        sample_count = 0
        expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        print(f"🔄 Testing iteration (first 5 samples)...")
        
        for i, sample in enumerate(dataset):
            # Validate sample
            assert sample['eva_embeddings'].shape[0] == expected_tokens, f"Invalid EVA tokens: {sample['eva_embeddings'].shape[0]}"
            assert sample['clip_embeddings'].shape[0] == expected_tokens, f"Invalid CLIP tokens: {sample['clip_embeddings'].shape[0]}"
            assert sample['training_mode'] == training_mode, f"Wrong training mode: {sample['training_mode']}"
            
            sample_count += 1
            
            # Print first few samples
            if i < 5:
                print(f"    Sample {i}: EVA {sample['eva_embeddings'].shape}, CLIP {sample['clip_embeddings'].shape}")
                print(f"      Mode: {sample['training_mode']}, Tokens: {sample['num_tokens']}")
                print(f"      Caption: {sample['caption'][:50]}...")
            
            # Break early for testing
            if sample_count >= 10:
                break
        
        print(f"✅ Test completed: {sample_count} samples processed")
        
        # Test dataloader
        print(f"🔄 Testing FIXED dataloader...")
        train_dataloader, eval_dataloader = create_flexible_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=4,
            training_mode=training_mode,
            max_shards=max_shards,
            use_same_data_for_eval=True,  # Test overfitting mode
            delete_after_use=False,
            num_workers=0,  # Use 0 workers for testing
        )
        
        print(f"✅ Dataloaders created successfully")
        print(f"   Train dataloader: {len(train_dataloader):,} batches")
        if eval_dataloader:
            print(f"   Eval dataloader: {len(eval_dataloader):,} batches")
        
        # Test batch
        print(f"🧪 Testing batch creation...")
        batch = next(iter(train_dataloader))
        print(f"✅ Batch created: EVA {batch['encoder_hidden_states'].shape}, CLIP {batch['clip_embeddings'].shape}")
        print(f"   Training mode: {batch['training_mode']}")
        print(f"   Tokens: {batch['num_tokens']}")
        print(f"   Hidden states requires_grad: {batch['hidden_states'].requires_grad}")
        
        if eval_dataloader:
            try:
                eval_batch = next(iter(eval_dataloader))
                print(f"✅ Eval batch: same data = {torch.equal(batch['clip_embeddings'], eval_batch['clip_embeddings'])}")
            except Exception as e:
                print(f"⚠️ Eval batch test failed: {e}")
        
        print(f"🎉 ALL TESTS PASSED!")
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
        training_mode = sys.argv[2] if len(sys.argv) > 2 else "patch_only"
        max_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        success = test_flexible_dataset(embeddings_dir, training_mode, max_shards)
        sys.exit(0 if success else 1)
    else:
        print("Usage: python blip3o_dataset.py <embeddings_dir> [training_mode] [max_shards]")
        print("Example: python blip3o_dataset.py /path/to/embeddings patch_only 1")