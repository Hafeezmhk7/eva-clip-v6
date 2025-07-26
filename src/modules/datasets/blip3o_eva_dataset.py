#!/usr/bin/env python3
"""
Fixed BLIP3-o Dataset for EVA-CLIP Reproduction Testing
src/modules/datasets/blip3o_eva_dataset.py

MAJOR FIXES:
1. Better normalization handling
2. Fixed shape validation and adaptation
3. Improved error handling and recovery
4. Better memory management
5. Fixed collate function with proper tensor handling
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
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BLIP3oEVAReproductionDataset(IterableDataset):
    """
    Fixed dataset for EVA-CLIP reproduction testing with robust error handling
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
        training_mode: str = "patch_only",
        max_shards: Optional[int] = None,
        use_same_data_for_eval: bool = False,
        expected_tokens: Optional[int] = None,
        # New stability parameters
        max_retries: int = 3,
        skip_corrupted: bool = True,
        validate_shapes: bool = True,
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
        self.max_retries = max_retries
        self.skip_corrupted = skip_corrupted
        self.validate_shapes = validate_shapes
        
        # Determine expected tokens based on mode
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
        # Setup random state
        self.rng = random.Random(random_seed)
        
        # Load manifest and shard list
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current shard state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_shard_samples = []
        self.current_sample_idx = 0
        
        # Statistics
        self.total_samples_processed = 0
        self.shards_processed = 0
        self.corrupted_samples_skipped = 0
        
        # Calculate estimated length
        self._calculate_estimated_length()
        
        logger.info(f"✅ Fixed EVA-CLIP Reproduction Dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Split: {self.split}")
        logger.info(f"  Training mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  TARGET: EVA-CLIP embeddings [B, N, 4096] (to reproduce)")
        logger.info(f"  CONDITIONING: CLIP embeddings [B, N, 1024]")
        logger.info(f"  L2 Normalization: {self.normalize_embeddings}")
        logger.info(f"  Total shards: {len(self.shard_files)}")
        logger.info(f"  Estimated samples: {self.estimated_length:,}")
        logger.info(f"  Skip corrupted: {self.skip_corrupted}")

    def _load_manifest(self):
        """Load the embeddings manifest file with error handling"""
        manifest_path = self.chunked_embeddings_dir / "embeddings_manifest.json"
        
        try:
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    self.manifest = json.load(f)
                logger.info(f"Loaded manifest: {self.manifest.get('total_shards', 0)} shards, {self.manifest.get('total_samples', 0):,} samples")
            else:
                logger.warning(f"Manifest not found: {manifest_path}, creating default")
                self.manifest = {
                    'total_shards': 0,
                    'total_samples': 0,
                    'extraction_mode': 'unknown',
                    'tokens_per_sample': self.expected_tokens,
                }
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}, using defaults")
            self.manifest = {
                'total_shards': 0,
                'total_samples': 0,
                'extraction_mode': 'unknown',
                'tokens_per_sample': self.expected_tokens,
            }
        
        self.estimated_total_samples = self.manifest.get('total_samples', 0)

    def _prepare_shard_list(self):
        """Prepare the list of shard files with better error handling"""
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
        """Load a single embedding shard with robust error handling and validation"""
        logger.debug(f"Loading shard: {shard_path}")
        
        for attempt in range(self.max_retries):
            try:
                if not shard_path.exists():
                    raise FileNotFoundError(f"Shard file does not exist: {shard_path}")
                
                if shard_path.stat().st_size == 0:
                    raise ValueError(f"Shard file is empty: {shard_path}")
                
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                # Validate and adapt shard
                self._validate_and_adapt_shard(shard_data, shard_path)
                
                # Apply normalization if enabled
                if self.normalize_embeddings:
                    shard_data = self._normalize_shard_embeddings(shard_data)
                
                logger.debug(f"✅ Successfully loaded shard: {shard_path}")
                return shard_data
                
            except Exception as e:
                logger.warning(f"❌ Attempt {attempt + 1}/{self.max_retries} failed for shard {shard_path}: {e}")
                if attempt == self.max_retries - 1:
                    if self.skip_corrupted:
                        logger.warning(f"⚠️ Skipping corrupted shard: {shard_path}")
                        return None
                    else:
                        raise RuntimeError(f"Failed to load shard {shard_path} after {self.max_retries} attempts: {e}")
                time.sleep(0.1)

    def _validate_and_adapt_shard(self, shard_data: Dict[str, Any], shard_path: Path):
        """Validate shard data format and adapt token count for training mode"""
        if shard_data is None:
            raise ValueError("Shard data is None")
            
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        captions = shard_data['captions']
        
        # Convert to tensors if needed
        if not torch.is_tensor(clip_emb):
            clip_emb = torch.tensor(clip_emb, dtype=torch.float32)
            shard_data['clip_blip3o_embeddings'] = clip_emb
        if not torch.is_tensor(eva_emb):
            eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
            shard_data['eva_blip3o_embeddings'] = eva_emb
        
        # Validate dimensions
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
            logger.debug(f"Adapting from {clip_tokens} to {self.expected_tokens} tokens for {self.training_mode} mode")
            
            if clip_tokens == 256 and self.expected_tokens == 257:
                logger.debug("Adding CLS token (average of patches) for cls_patch mode")
                batch_size, _, clip_dim = clip_emb.shape
                eva_dim = eva_emb.shape[2]
                
                clip_cls = clip_emb.mean(dim=1, keepdim=True)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                
                shard_data['clip_blip3o_embeddings'] = torch.cat([clip_cls, clip_emb], dim=1)
                shard_data['eva_blip3o_embeddings'] = torch.cat([eva_cls, eva_emb], dim=1)
                
            elif clip_tokens == 257 and self.expected_tokens == 256:
                logger.debug("Removing CLS token for patch_only mode")
                shard_data['clip_blip3o_embeddings'] = clip_emb[:, 1:, :]
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
                
            else:
                raise ValueError(f"Cannot adapt from {clip_tokens} to {self.expected_tokens} tokens")
        
        # Final validation after adaptation
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        if self.validate_shapes:
            if eva_emb.shape[1] != self.expected_tokens:
                raise ValueError(f"EVA token count after adaptation: expected {self.expected_tokens}, got {eva_emb.shape[1]}")
            
            if clip_emb.shape[0] != eva_emb.shape[0]:
                raise ValueError(f"Sample count mismatch: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
            
            if len(captions) != clip_emb.shape[0]:
                raise ValueError(f"Caption count mismatch: {len(captions)} vs {clip_emb.shape[0]} samples")
            
            if clip_emb.shape[2] != 1024:
                raise ValueError(f"Expected CLIP 1024-dim, got {clip_emb.shape[2]}")
            if eva_emb.shape[2] != 4096:
                raise ValueError(f"Expected EVA 4096-dim, got {eva_emb.shape[2]}")

    def _normalize_shard_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply L2 normalization to embeddings with better error handling"""
        try:
            clip_emb = shard_data['clip_blip3o_embeddings']
            eva_emb = shard_data['eva_blip3o_embeddings']
            
            # Check for invalid values
            if torch.isnan(clip_emb).any() or torch.isinf(clip_emb).any():
                logger.warning("Found NaN/Inf in CLIP embeddings, replacing with zeros")
                clip_emb = torch.nan_to_num(clip_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
                logger.warning("Found NaN/Inf in EVA embeddings, replacing with zeros")
                eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Check initial norms
            initial_clip_norm = torch.norm(clip_emb, dim=-1).mean().item()
            initial_eva_norm = torch.norm(eva_emb, dim=-1).mean().item()
            
            # Apply L2 normalization with epsilon for stability
            eps = 1e-8
            clip_emb_normalized = F.normalize(clip_emb + eps, p=2, dim=-1)
            eva_emb_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
            
            # Check final norms (should be ~1.0)
            final_clip_norm = torch.norm(clip_emb_normalized, dim=-1).mean().item()
            final_eva_norm = torch.norm(eva_emb_normalized, dim=-1).mean().item()
            
            # Log normalization status
            logger.debug(f"L2 Normalization applied:")
            logger.debug(f"  CLIP: {initial_clip_norm:.3f} -> {final_clip_norm:.3f}")
            logger.debug(f"  EVA:  {initial_eva_norm:.3f} -> {final_eva_norm:.3f}")
            
            # Verify normalization was successful
            if abs(final_clip_norm - 1.0) > 0.1:
                logger.warning(f"CLIP L2 normalization may have failed: norm = {final_clip_norm:.3f}")
            if abs(final_eva_norm - 1.0) > 0.1:
                logger.warning(f"EVA L2 normalization may have failed: norm = {final_eva_norm:.3f}")
            
            # Update shard data with normalized embeddings
            shard_data['clip_blip3o_embeddings'] = clip_emb_normalized
            shard_data['eva_blip3o_embeddings'] = eva_emb_normalized
            
            # Add normalization metadata
            shard_data['normalization_applied'] = True
            shard_data['initial_clip_norm'] = initial_clip_norm
            shard_data['initial_eva_norm'] = initial_eva_norm
            shard_data['final_clip_norm'] = final_clip_norm
            shard_data['final_eva_norm'] = final_eva_norm
            
            return shard_data
            
        except Exception as e:
            logger.error(f"Failed to normalize embeddings: {e}")
            raise

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
        """Load the next shard with error handling"""
        # Clean up previous shard
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
        
        # Check if more shards available
        if self.current_shard_idx >= len(self.shard_files):
            logger.debug("No more shards to process")
            self.current_shard_data = None
            return False
        
        # Try to load next shard
        while self.current_shard_idx < len(self.shard_files):
            shard_path = self.shard_files[self.current_shard_idx]
            
            try:
                self.current_shard_data = self._load_shard(shard_path)
                
                if self.current_shard_data is not None:
                    self._prepare_current_shard_samples()
                    
                    num_samples = len(self.current_shard_samples) if self.current_shard_samples else 0
                    
                    # Log normalization status for this shard
                    norm_status = "✅ Normalized" if self.current_shard_data.get('normalization_applied', False) else "⚠️ Not normalized"
                    final_eva_norm = self.current_shard_data.get('final_eva_norm', 'unknown')
                    
                    logger.info(f"✅ Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: "
                               f"{num_samples} samples ({self.expected_tokens} tokens) - {norm_status}")
                    if isinstance(final_eva_norm, float):
                        logger.info(f"   EVA norm: {final_eva_norm:.3f} (should be ~1.0)")
                    
                    self.current_shard_idx += 1
                    self.shards_processed += 1
                    return True
                else:
                    # Shard was corrupted and skipped
                    self.current_shard_idx += 1
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Failed to load shard {shard_path}: {e}")
                if self.skip_corrupted:
                    logger.warning(f"⚠️ Skipping corrupted shard and continuing")
                    self.current_shard_idx += 1
                    continue
                else:
                    raise
        
        # No more shards
        self.current_shard_data = None
        return False

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples across all shards with robust error handling"""
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        self.shards_processed = 0
        self.corrupted_samples_skipped = 0
        
        logger.debug(f"Starting iteration over {len(self.shard_files)} shards for EVA reproduction test")
        
        if not self._load_next_shard():
            logger.warning("No shards could be loaded")
            return
        
        while self.current_shard_data is not None:
            while self.current_sample_idx < len(self.current_shard_samples):
                try:
                    sample_idx = self.current_shard_samples[self.current_sample_idx]
                    
                    # Extract sample data with validation
                    clip_emb = self.current_shard_data['clip_blip3o_embeddings'][sample_idx]
                    eva_emb = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Validate sample shapes
                    if self.validate_shapes:
                        if clip_emb.shape != (self.expected_tokens, 1024):
                            raise ValueError(f"Invalid CLIP shape: {clip_emb.shape}")
                        if eva_emb.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {eva_emb.shape}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(clip_emb).any() or torch.isnan(eva_emb).any():
                        if self.skip_corrupted:
                            self.corrupted_samples_skipped += 1
                            self.current_sample_idx += 1
                            continue
                        else:
                            raise ValueError("NaN detected in embeddings")
                    
                    # Create sample item
                    item = {
                        'clip_embeddings': clip_emb,  # [N, 1024] - CONDITIONING
                        'eva_embeddings': eva_emb,    # [N, 4096] - TARGET
                        'caption': caption,
                        'key': self.current_shard_data.get('keys', [f"sample_{sample_idx}"])[sample_idx] if self.current_shard_data.get('keys') and sample_idx < len(self.current_shard_data.get('keys', [])) else f"sample_{sample_idx}",
                        'shard_idx': self.current_shard_idx - 1,
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                        'normalized': self.current_shard_data.get('normalization_applied', False),
                        'test_type': 'eva_reproduction',
                    }
                    
                    self.current_sample_idx += 1
                    self.total_samples_processed += 1
                    
                    yield item
                    
                except Exception as e:
                    if self.skip_corrupted:
                        logger.warning(f"Skipping corrupted sample {sample_idx} in shard {self.current_shard_idx - 1}: {e}")
                        self.corrupted_samples_skipped += 1
                        self.current_sample_idx += 1
                        continue
                    else:
                        logger.error(f"Error processing sample {sample_idx} in shard {self.current_shard_idx - 1}: {e}")
                        raise
            
            if not self._load_next_shard():
                break
        
        logger.info(f"✅ EVA reproduction dataset iteration completed: "
                   f"{self.total_samples_processed} samples from {self.shards_processed} shards, "
                   f"{self.corrupted_samples_skipped} corrupted samples skipped")


def blip3o_eva_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fixed collate function for EVA reproduction test with robust error handling
    """
    if not batch:
        raise ValueError("Empty batch received")
    
    try:
        # Filter out None items
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            raise ValueError("No valid items in batch")
        
        # Stack tensor data - NOTE THE SWAP FOR EVA REPRODUCTION
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in valid_batch])  # [B, N, 1024] - CONDITIONING
        eva_embeddings = torch.stack([item['eva_embeddings'] for item in valid_batch])    # [B, N, 4096] - TARGET
        
        # Collect metadata
        captions = [item['caption'] for item in valid_batch]
        keys = [item['key'] for item in valid_batch]
        training_mode = valid_batch[0]['training_mode']
        num_tokens = valid_batch[0]['num_tokens']
        
        # Get batch info
        batch_size, seq_len, eva_dim = eva_embeddings.shape
        device = eva_embeddings.device
        dtype = eva_embeddings.dtype
        
        # Ensure proper data types
        clip_embeddings = clip_embeddings.float()
        eva_embeddings = eva_embeddings.float()
        
        # Check and apply L2 normalization if needed
        clip_norm_before = torch.norm(clip_embeddings, dim=-1).mean().item()
        eva_norm_before = torch.norm(eva_embeddings, dim=-1).mean().item()
        
        # Apply L2 normalization with stability epsilon
        eps = 1e-8
        clip_embeddings = F.normalize(clip_embeddings + eps, p=2, dim=-1)
        eva_embeddings = F.normalize(eva_embeddings + eps, p=2, dim=-1)
        
        # Check final norms (should be ~1.0)
        clip_norm_after = torch.norm(clip_embeddings, dim=-1).mean().item()
        eva_norm_after = torch.norm(eva_embeddings, dim=-1).mean().item()
        
        # Log normalization status
        logger.debug(f"EVA Collate L2 normalization:")
        logger.debug(f"  CLIP: {clip_norm_before:.3f} -> {clip_norm_after:.3f}")
        logger.debug(f"  EVA:  {eva_norm_before:.3f} -> {eva_norm_after:.3f}")
        
        # Warn if normalization seems wrong
        if abs(clip_norm_after - 1.0) > 0.1:
            logger.warning(f"CLIP embeddings not properly normalized in collate: norm = {clip_norm_after:.3f}")
        if abs(eva_norm_after - 1.0) > 0.1:
            logger.warning(f"EVA embeddings not properly normalized in collate: norm = {eva_norm_after:.3f}")
        
        # Flow matching setup - Rectified Flow
        # Sample timesteps uniformly with better distribution
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # Create noise (source distribution) - SAME DIMENSION AS TARGET (EVA: 4096)
        noise = torch.randn_like(eva_embeddings, device=device, dtype=dtype)
        # Normalize noise for consistency
        noise = F.normalize(noise + eps, p=2, dim=-1)
        
        # Rectified flow interpolation: x_t = (1-t) * x_0 + t * x_1
        # where x_0 = noise, x_1 = target (normalized EVA embeddings)
        t_expanded = timesteps.view(-1, 1, 1)  # [B, 1, 1]
        noisy_input = (1 - t_expanded) * noise + t_expanded * eva_embeddings
        
        # Velocity target for rectified flow: v = x_1 - x_0 = eva_embeddings - noise
        velocity_target = eva_embeddings - noise
        
        # Validate tensor properties
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP shape: {clip_embeddings.shape}"
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA shape: {eva_embeddings.shape}"
        assert noisy_input.shape == (batch_size, seq_len, 4096), f"Noisy input shape: {noisy_input.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 4096), f"Velocity target shape: {velocity_target.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        assert seq_len in [256, 257], f"Unexpected token count: {seq_len}"
        
        return {
            # Model inputs - SWAPPED FOR EVA REPRODUCTION
            'encoder_hidden_states': clip_embeddings,       # [B, N, 1024] - CLIP conditioning (normalized)
            'hidden_states': noisy_input,                   # [B, N, 4096] - Noisy EVA input for model
            'timestep': timesteps,                          # [B] - Flow matching timesteps
            
            # Training targets - EVA embeddings
            'eva_embeddings': eva_embeddings,               # [B, N, 4096] - Target EVA patches (normalized ~1.0)
            'velocity_target': velocity_target,             # [B, N, 4096] - Velocity target for flow matching
            'noise': noise,                                 # [B, N, 4096] - Original noise
            
            # Metadata
            'captions': captions,                           # List[str] - Text captions
            'keys': keys,                                   # List[str] - Sample keys
            'batch_size': batch_size,                       # int
            'training_mode': training_mode,                 # str - "cls_patch" or "patch_only"
            'num_tokens': num_tokens,                       # int - 256 or 257
            'seq_len': seq_len,                            # int - sequence length
            'test_type': 'eva_reproduction',                # str - test type
            
            # Normalization info
            'clip_embeddings_normalized': True,             # bool
            'eva_embeddings_normalized': True,              # bool
            'clip_norm_mean': clip_norm_after,             # float (~1.0)
            'eva_norm_mean': eva_norm_after,               # float (~1.0)
            'initial_clip_norm': clip_norm_before,          # float (for debugging)
            'initial_eva_norm': eva_norm_before,           # float (for debugging)
        }
        
    except Exception as e:
        logger.error(f"Error in EVA reproduction collate function: {e}")
        logger.error(f"Batch info: {len(batch)} items")
        if batch:
            try:
                first_item = batch[0]
                if first_item is not None:
                    logger.error(f"First item keys: {list(first_item.keys())}")
                    for key, value in first_item.items():
                        if torch.is_tensor(value):
                            logger.error(f"  {key}: {value.shape} {value.dtype}")
                        else:
                            logger.error(f"  {key}: {type(value)} {value}")
            except Exception as debug_e:
                logger.error(f"Failed to debug batch: {debug_e}")
        raise


def create_eva_reproduction_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    eval_split_ratio: float = 0.1,
    normalize_embeddings: bool = True,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    use_same_data_for_eval: bool = False,
    delete_after_use: bool = False,
    num_workers: int = 0,
    pin_memory: bool = None,
    # New parameters for robustness
    skip_corrupted: bool = True,
    validate_shapes: bool = True,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create EVA reproduction test dataloaders with improved robustness
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # Log test configuration
    logger.info(f"Creating EVA reproduction test dataloaders:")
    logger.info(f"  TARGET: EVA embeddings [B, N, 4096] (to reproduce)")
    logger.info(f"  CONDITIONING: CLIP embeddings [B, N, 1024]")
    logger.info(f"  L2 normalization: {normalize_embeddings}")
    logger.info(f"  Skip corrupted: {skip_corrupted}")
    logger.info(f"  Validate shapes: {validate_shapes}")
    
    # Create training dataset
    try:
        train_dataset = BLIP3oEVAReproductionDataset(
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
            skip_corrupted=skip_corrupted,
            validate_shapes=validate_shapes,
        )
        
        logger.info(f"✅ EVA reproduction training dataset created: {len(train_dataset):,} estimated samples")
        
    except Exception as e:
        logger.error(f"❌ Failed to create EVA reproduction training dataset: {e}")
        raise
    
    # Create training dataloader
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=blip3o_eva_collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
            **kwargs
        )
        
        logger.info(f"✅ EVA reproduction training dataloader created")
        
    except Exception as e:
        logger.error(f"❌ Failed to create EVA reproduction training dataloader: {e}")
        raise
    
    # Create evaluation dataloader
    eval_dataloader = None
    if eval_split_ratio > 0 or use_same_data_for_eval:
        try:
            eval_dataset = BLIP3oEVAReproductionDataset(
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
                skip_corrupted=skip_corrupted,
                validate_shapes=validate_shapes,
            )
            
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                num_workers=min(num_workers, 1),
                collate_fn=blip3o_eva_collate_fn,
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=min(num_workers, 1) > 0,
                **kwargs
            )
            
            logger.info(f"✅ EVA reproduction evaluation dataloader created")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create EVA reproduction evaluation dataloader: {e}")
            eval_dataloader = None
    
    return train_dataloader, eval_dataloader