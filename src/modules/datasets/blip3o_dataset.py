#!/usr/bin/env python3
"""
FIXED: BLIP3-o Dataset with Consistent Training/Evaluation Data
Key fixes:
1. Evaluation uses subset of training data to ensure consistent statistics
2. Extensive logging of data statistics to catch norm mismatches
3. Validation that training and evaluation have similar norms
4. Absolutely no hidden normalization differences
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, Subset
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Iterator
from pathlib import Path
import logging
import json
import random
import time
import gc
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BLIP3oCLIPReproductionDataset(IterableDataset):
    """
    Dataset for CLIP reproduction with EVA conditioning
    FIXED: Consistent preprocessing and statistics tracking
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        split: str = "train",
        training_mode: str = "patch_only",
        normalize_embeddings: bool = False,  # Keep disabled for consistency
        max_shards: Optional[int] = None,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        expected_tokens: Optional[int] = None,
        # Error handling
        skip_corrupted: bool = True,
        validate_shapes: bool = True,
        max_retries: int = 3,
        # NEW: Statistics collection and validation
        collect_statistics: bool = True,
        validate_norms: bool = True,
        expected_clip_norm_range: Tuple[float, float] = (20.0, 50.0),
        expected_eva_norm_range: Tuple[float, float] = (20.0, 60.0),
    ):
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.split = split
        self.training_mode = training_mode
        self.normalize_embeddings = normalize_embeddings
        self.max_shards = max_shards
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.skip_corrupted = skip_corrupted
        self.validate_shapes = validate_shapes
        self.max_retries = max_retries
        self.collect_statistics = collect_statistics
        self.validate_norms = validate_norms
        self.expected_clip_norm_range = expected_clip_norm_range
        self.expected_eva_norm_range = expected_eva_norm_range
        
        # Determine expected tokens
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
        # Setup random state
        self.rng = random.Random(42)
        
        # Statistics for norm validation
        self.data_statistics = {
            'clip_norm_mean': 0.0,
            'clip_norm_std': 0.0,
            'clip_std_mean': 0.0,
            'eva_norm_mean': 0.0,
            'eva_norm_std': 0.0,
            'samples_seen': 0,
            'clip_norm_min': float('inf'),
            'clip_norm_max': float('-inf'),
            'eva_norm_min': float('inf'),
            'eva_norm_max': float('-inf'),
        }
        
        # Load manifest and prepare shards
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        # Calculate estimated length
        self._estimate_length()
        
        # NEW: Store all samples for consistent evaluation
        self._all_samples = None
        self._samples_loaded = False
        
        logger.info(f"CLIP Reproduction Dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Split: {self.split}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  TARGET: CLIP embeddings [B, N, 1024]")
        logger.info(f"  CONDITIONING: EVA embeddings [B, N, 4096]")
        logger.info(f"  Normalize: {self.normalize_embeddings}")
        logger.info(f"  Shards: {len(self.shard_files)}")
        logger.info(f"  Estimated samples: {self.estimated_length:,}")
        logger.info(f"  🎯 NORM VALIDATION: CLIP {self.expected_clip_norm_range}, EVA {self.expected_eva_norm_range}")

    def _load_manifest(self):
        """Load embeddings manifest"""
        manifest_path = self.chunked_embeddings_dir / "embeddings_manifest.json"
        
        try:
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    self.manifest = json.load(f)
                logger.info(f"Loaded manifest: {self.manifest.get('total_shards', 0)} shards, {self.manifest.get('total_samples', 0):,} samples")
            else:
                self.manifest = {"total_shards": 0, "total_samples": 0}
                logger.warning(f"No manifest found at {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
            self.manifest = {"total_shards": 0, "total_samples": 0}

    def _prepare_shard_list(self):
        """Prepare list of shard files"""
        mode_suffix = "cls_patch" if self.training_mode == "cls_patch" else "patch_only"
        patterns = [
            f"embeddings_shard_*_{mode_suffix}.pkl",
            f"*_{mode_suffix}.pkl",
            "embeddings_shard_*.pkl",
            "*.pkl"
        ]
        
        shard_files = []
        for pattern in patterns:
            shard_files = list(self.chunked_embeddings_dir.glob(pattern))
            if shard_files:
                logger.info(f"Found {len(shard_files)} files with pattern: {pattern}")
                break
        
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {self.chunked_embeddings_dir}")
        
        # Sort files
        shard_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else 0)
        
        # Apply max shards limit
        if self.max_shards is not None:
            shard_files = shard_files[:self.max_shards]
        
        # Filter existing files
        self.shard_files = [f for f in shard_files if f.exists()]
        
        if self.shuffle_shards:
            self.rng.shuffle(self.shard_files)
        
        logger.info(f"Prepared {len(self.shard_files)} shard files")

    def _estimate_length(self):
        """Estimate total number of samples"""
        try:
            if self.manifest.get('total_samples', 0) > 0:
                manifest_samples = self.manifest['total_samples']
                if self.max_shards is not None and self.manifest.get('total_shards', 0) > 0:
                    ratio = min(self.max_shards / self.manifest['total_shards'], 1.0)
                    self.estimated_length = int(manifest_samples * ratio)
                else:
                    self.estimated_length = manifest_samples
                logger.info(f"Using manifest for length estimation: {self.estimated_length:,} samples")
                return
            
            if self.shard_files:
                try:
                    first_shard = self._load_shard(self.shard_files[0])
                    if first_shard and 'clip_blip3o_embeddings' in first_shard:
                        samples_per_shard = first_shard['clip_blip3o_embeddings'].shape[0]
                        self.estimated_length = samples_per_shard * len(self.shard_files)
                        logger.info(f"Estimated length from first shard: {self.estimated_length:,} samples ({samples_per_shard} per shard)")
                        return
                except Exception as e:
                    logger.warning(f"Could not load first shard for estimation: {e}")
            
            self.estimated_length = len(self.shard_files) * 1000
            logger.warning(f"Using rough length estimate: {self.estimated_length:,} samples")
            
        except Exception as e:
            logger.warning(f"Length estimation failed: {e}")
            self.estimated_length = 1000

    def __len__(self) -> int:
        """Return estimated length for DataLoader compatibility"""
        return self.estimated_length

    def _update_statistics(self, clip_emb: torch.Tensor, eva_emb: torch.Tensor):
        """Update running statistics with detailed norm tracking"""
        if not self.collect_statistics:
            return
            
        with torch.no_grad():
            batch_size = clip_emb.shape[0]
            
            # CLIP statistics
            clip_norms = torch.norm(clip_emb, dim=-1).mean(dim=1)  # [B]
            clip_norm_mean = clip_norms.mean().item()
            clip_norm_std = clip_norms.std().item()
            clip_std_mean = clip_emb.std().item()
            clip_norm_min = clip_norms.min().item()
            clip_norm_max = clip_norms.max().item()
            
            # EVA statistics  
            eva_norms = torch.norm(eva_emb, dim=-1).mean(dim=1)  # [B]
            eva_norm_mean = eva_norms.mean().item()
            eva_norm_std = eva_norms.std().item()
            eva_norm_min = eva_norms.min().item()
            eva_norm_max = eva_norms.max().item()
            
            # Update running averages
            alpha = 0.01
            if self.data_statistics['samples_seen'] == 0:
                # Initialize
                self.data_statistics.update({
                    'clip_norm_mean': clip_norm_mean,
                    'clip_norm_std': clip_norm_std,
                    'clip_std_mean': clip_std_mean,
                    'eva_norm_mean': eva_norm_mean,
                    'eva_norm_std': eva_norm_std,
                    'clip_norm_min': clip_norm_min,
                    'clip_norm_max': clip_norm_max,
                    'eva_norm_min': eva_norm_min,
                    'eva_norm_max': eva_norm_max,
                })
            else:
                # Update with momentum
                self.data_statistics['clip_norm_mean'] = (
                    (1 - alpha) * self.data_statistics['clip_norm_mean'] + alpha * clip_norm_mean
                )
                self.data_statistics['clip_norm_std'] = (
                    (1 - alpha) * self.data_statistics['clip_norm_std'] + alpha * clip_norm_std
                )
                self.data_statistics['clip_std_mean'] = (
                    (1 - alpha) * self.data_statistics['clip_std_mean'] + alpha * clip_std_mean
                )
                self.data_statistics['eva_norm_mean'] = (
                    (1 - alpha) * self.data_statistics['eva_norm_mean'] + alpha * eva_norm_mean
                )
                self.data_statistics['eva_norm_std'] = (
                    (1 - alpha) * self.data_statistics['eva_norm_std'] + alpha * eva_norm_std
                )
                
                # Update min/max
                self.data_statistics['clip_norm_min'] = min(self.data_statistics['clip_norm_min'], clip_norm_min)
                self.data_statistics['clip_norm_max'] = max(self.data_statistics['clip_norm_max'], clip_norm_max)
                self.data_statistics['eva_norm_min'] = min(self.data_statistics['eva_norm_min'], eva_norm_min)
                self.data_statistics['eva_norm_max'] = max(self.data_statistics['eva_norm_max'], eva_norm_max)
            
            self.data_statistics['samples_seen'] += batch_size
            
            # NEW: Validate norms against expected ranges
            if self.validate_norms and self.data_statistics['samples_seen'] % 1000 == 0:
                self._validate_norm_ranges()

    def _validate_norm_ranges(self):
        """Validate that norms are in expected ranges"""
        clip_mean = self.data_statistics['clip_norm_mean']
        eva_mean = self.data_statistics['eva_norm_mean']
        
        # Check CLIP norms
        if not (self.expected_clip_norm_range[0] <= clip_mean <= self.expected_clip_norm_range[1]):
            logger.warning(f"⚠️ CLIP norm {clip_mean:.2f} outside expected range {self.expected_clip_norm_range}")
        
        # Check EVA norms
        if not (self.expected_eva_norm_range[0] <= eva_mean <= self.expected_eva_norm_range[1]):
            logger.warning(f"⚠️ EVA norm {eva_mean:.2f} outside expected range {self.expected_eva_norm_range}")
        
        # Log current statistics
        logger.info(f"📊 Norm validation (samples: {self.data_statistics['samples_seen']:,}):")
        logger.info(f"   CLIP: mean={clip_mean:.2f}, std={self.data_statistics['clip_norm_std']:.2f}, range=[{self.data_statistics['clip_norm_min']:.2f}, {self.data_statistics['clip_norm_max']:.2f}]")
        logger.info(f"   EVA:  mean={eva_mean:.2f}, std={self.data_statistics['eva_norm_std']:.2f}, range=[{self.data_statistics['eva_norm_min']:.2f}, {self.data_statistics['eva_norm_max']:.2f}]")

    def _load_shard(self, shard_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single shard with error handling"""
        for attempt in range(self.max_retries):
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                self._validate_and_process_shard(shard_data, shard_path)
                return shard_data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {shard_path}: {e}")
                if attempt == self.max_retries - 1:
                    if self.skip_corrupted:
                        logger.warning(f"Skipping corrupted shard: {shard_path}")
                        return None
                    else:
                        raise
                time.sleep(0.1)

    def _validate_and_process_shard(self, shard_data: Dict[str, Any], shard_path: Path):
        """Validate and process shard data with extensive logging"""
        # Check required keys
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        # Get embeddings
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Convert to tensors if needed
        if not torch.is_tensor(clip_emb):
            clip_emb = torch.tensor(clip_emb, dtype=torch.float32)
            shard_data['clip_blip3o_embeddings'] = clip_emb
        if not torch.is_tensor(eva_emb):
            eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
            shard_data['eva_blip3o_embeddings'] = eva_emb
        
        # Validate shapes
        if self.validate_shapes:
            if clip_emb.dim() != 3 or eva_emb.dim() != 3:
                raise ValueError(f"Expected 3D tensors, got CLIP: {clip_emb.shape}, EVA: {eva_emb.shape}")
            
            if clip_emb.shape[0] != eva_emb.shape[0]:
                raise ValueError(f"Batch size mismatch: CLIP {clip_emb.shape[0]} vs EVA {eva_emb.shape[0]}")
            
            clip_tokens, eva_tokens = clip_emb.shape[1], eva_emb.shape[1]
            if clip_tokens != eva_tokens:
                raise ValueError(f"Token count mismatch: CLIP {clip_tokens} vs EVA {eva_tokens}")
        
        # Handle token count adaptation
        current_tokens = clip_emb.shape[1]
        if current_tokens != self.expected_tokens:
            logger.debug(f"Adapting from {current_tokens} to {self.expected_tokens} tokens")
            
            if current_tokens == 256 and self.expected_tokens == 257:
                clip_cls = clip_emb.mean(dim=1, keepdim=True)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                shard_data['clip_blip3o_embeddings'] = torch.cat([clip_cls, clip_emb], dim=1)
                shard_data['eva_blip3o_embeddings'] = torch.cat([eva_cls, eva_emb], dim=1)
            elif current_tokens == 257 and self.expected_tokens == 256:
                shard_data['clip_blip3o_embeddings'] = clip_emb[:, 1:, :]
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
            else:
                raise ValueError(f"Cannot adapt from {current_tokens} to {self.expected_tokens} tokens")
        
        # NEW: Log detailed shard statistics
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        clip_norm_mean = torch.norm(clip_emb, dim=-1).mean().item()
        eva_norm_mean = torch.norm(eva_emb, dim=-1).mean().item()
        
        logger.debug(f"Shard {shard_path.name} statistics:")
        logger.debug(f"  CLIP norm: {clip_norm_mean:.2f}")
        logger.debug(f"  EVA norm: {eva_norm_mean:.2f}")
        logger.debug(f"  Samples: {clip_emb.shape[0]}")
        
        # Update statistics BEFORE any normalization
        if self.collect_statistics:
            self._update_statistics(clip_emb, eva_emb)
        
        # Apply normalization if requested (should be disabled)
        if self.normalize_embeddings:
            logger.warning("⚠️ Normalization is enabled - this may cause norm mismatches!")
            shard_data = self._normalize_embeddings(shard_data)

    def _normalize_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply L2 normalization (should be disabled for consistency)"""
        logger.warning("🚨 NORMALIZATION APPLIED - This may cause training/evaluation norm mismatches!")
        
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Check for NaN/Inf
        if torch.isnan(clip_emb).any() or torch.isinf(clip_emb).any():
            logger.warning("Found NaN/Inf in CLIP embeddings")
            clip_emb = torch.nan_to_num(clip_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
            logger.warning("Found NaN/Inf in EVA embeddings")
            eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply L2 normalization
        eps = 1e-8
        clip_normalized = F.normalize(clip_emb + eps, p=2, dim=-1)
        eva_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
        
        shard_data['clip_blip3o_embeddings'] = clip_normalized
        shard_data['eva_blip3o_embeddings'] = eva_normalized
        shard_data['normalization_applied'] = True
        
        return shard_data

    def _load_next_shard(self) -> bool:
        """Load next shard"""
        if self.current_shard_data is not None:
            del self.current_shard_data
            gc.collect()
        
        if self.current_shard_idx >= len(self.shard_files):
            self.current_shard_data = None
            return False
        
        while self.current_shard_idx < len(self.shard_files):
            shard_path = self.shard_files[self.current_shard_idx]
            
            self.current_shard_data = self._load_shard(shard_path)
            
            if self.current_shard_data is not None:
                num_samples = self.current_shard_data['clip_blip3o_embeddings'].shape[0]
                self.current_samples = list(range(num_samples))
                
                if self.shuffle_within_shard:
                    self.rng.shuffle(self.current_samples)
                
                self.current_sample_idx = 0
                
                logger.info(f"Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: {num_samples} samples")
                self.current_shard_idx += 1
                return True
            else:
                self.current_shard_idx += 1
                continue
        
        self.current_shard_data = None
        return False

    def get_data_statistics(self) -> Dict[str, float]:
        """Get collected data statistics"""
        return self.data_statistics.copy()

    def load_all_samples(self) -> List[Dict[str, Any]]:
        """Load all samples into memory for consistent evaluation"""
        if self._samples_loaded:
            return self._all_samples
        
        logger.info("Loading all samples for consistent evaluation...")
        self._all_samples = []
        
        # Reset state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        
        # Load all samples
        total_loaded = 0
        while self._load_next_shard():
            while self.current_sample_idx < len(self.current_samples):
                try:
                    sample_idx = self.current_samples[self.current_sample_idx]
                    
                    clip_emb = self.current_shard_data['clip_blip3o_embeddings'][sample_idx]
                    eva_emb = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Validate sample
                    if self.validate_shapes:
                        if clip_emb.shape != (self.expected_tokens, 1024):
                            raise ValueError(f"Invalid CLIP shape: {clip_emb.shape}")
                        if eva_emb.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {eva_emb.shape}")
                    
                    if torch.isnan(clip_emb).any() or torch.isnan(eva_emb).any():
                        if self.skip_corrupted:
                            self.current_sample_idx += 1
                            continue
                        else:
                            raise ValueError("NaN detected in embeddings")
                    
                    item = {
                        'eva_embeddings': eva_emb,
                        'clip_embeddings': clip_emb,
                        'caption': caption,
                        'key': f"shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                        'normalized': self.current_shard_data.get('normalization_applied', False),
                    }
                    
                    self._all_samples.append(item)
                    total_loaded += 1
                    self.current_sample_idx += 1
                    
                except Exception as e:
                    if self.skip_corrupted:
                        logger.warning(f"Skipping corrupted sample: {e}")
                        self.current_sample_idx += 1
                        continue
                    else:
                        raise
        
        self._samples_loaded = True
        logger.info(f"Loaded {total_loaded:,} samples for consistent evaluation")
        
        # Log statistics of loaded samples
        if self._all_samples:
            clip_norms = []
            eva_norms = []
            for sample in self._all_samples[:1000]:  # Sample first 1000 for statistics
                clip_norm = torch.norm(sample['clip_embeddings'], dim=-1).mean().item()
                eva_norm = torch.norm(sample['eva_embeddings'], dim=-1).mean().item()
                clip_norms.append(clip_norm)
                eva_norms.append(eva_norm)
            
            clip_norm_mean = np.mean(clip_norms)
            eva_norm_mean = np.mean(eva_norms)
            
            logger.info(f"📊 Loaded samples statistics:")
            logger.info(f"   CLIP norm: {clip_norm_mean:.2f} ± {np.std(clip_norms):.2f}")
            logger.info(f"   EVA norm: {eva_norm_mean:.2f} ± {np.std(eva_norms):.2f}")
        
        return self._all_samples

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples"""
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        logger.debug(f"Starting iteration over {len(self.shard_files)} shards")
        
        if not self._load_next_shard():
            return
        
        while self.current_shard_data is not None:
            while self.current_sample_idx < len(self.current_samples):
                try:
                    sample_idx = self.current_samples[self.current_sample_idx]
                    
                    clip_emb = self.current_shard_data['clip_blip3o_embeddings'][sample_idx]
                    eva_emb = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Final validation
                    if self.validate_shapes:
                        if clip_emb.shape != (self.expected_tokens, 1024):
                            raise ValueError(f"Invalid CLIP shape: {clip_emb.shape}")
                        if eva_emb.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {eva_emb.shape}")
                    
                    if torch.isnan(clip_emb).any() or torch.isnan(eva_emb).any():
                        if self.skip_corrupted:
                            self.current_sample_idx += 1
                            continue
                        else:
                            raise ValueError("NaN detected in embeddings")
                    
                    item = {
                        'eva_embeddings': eva_emb,
                        'clip_embeddings': clip_emb,
                        'caption': caption,
                        'key': f"shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                        'normalized': self.current_shard_data.get('normalization_applied', False),
                    }
                    
                    self.current_sample_idx += 1
                    self.total_samples_processed += 1
                    
                    yield item
                    
                except Exception as e:
                    if self.skip_corrupted:
                        logger.warning(f"Skipping corrupted sample {sample_idx}: {e}")
                        self.current_sample_idx += 1
                        continue
                    else:
                        raise
            
            if not self._load_next_shard():
                break
        
        logger.info(f"Iteration completed: {self.total_samples_processed} samples processed")


def clip_reproduction_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function with extensive statistics logging"""
    if not batch:
        raise ValueError("Empty batch")
    
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # Stack embeddings
        eva_embeddings = torch.stack([item['eva_embeddings'] for item in valid_batch])
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in valid_batch])
        
        captions = [item['caption'] for item in valid_batch]
        keys = [item['key'] for item in valid_batch]
        
        batch_size, seq_len, clip_dim = clip_embeddings.shape
        device = clip_embeddings.device
        dtype = clip_embeddings.dtype
        
        # Ensure float32
        eva_embeddings = eva_embeddings.float()
        clip_embeddings = clip_embeddings.float()
        
        # Sample random timesteps
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # Create noise
        noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)
        
        # Linear interpolation for rectified flow
        t_expanded = timesteps.view(batch_size, 1, 1)
        noisy_clip = (1 - t_expanded) * noise + t_expanded * clip_embeddings
        
        # Velocity target
        velocity_target = clip_embeddings - noise
        
        # Validation
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA shape: {eva_embeddings.shape}"
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP shape: {clip_embeddings.shape}"
        assert noisy_clip.shape == (batch_size, seq_len, 1024), f"Noisy CLIP shape: {noisy_clip.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 1024), f"Velocity target shape: {velocity_target.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        # NEW: Compute and log detailed batch statistics
        clip_norm_mean = torch.norm(clip_embeddings, dim=-1).mean().item()
        eva_norm_mean = torch.norm(eva_embeddings, dim=-1).mean().item()
        noise_norm_mean = torch.norm(noise, dim=-1).mean().item()
        
        return {
            # Model inputs
            'encoder_hidden_states': eva_embeddings,
            'hidden_states': noisy_clip,
            'timestep': timesteps,
            
            # Training targets
            'clip_embeddings': clip_embeddings,
            'velocity_target': velocity_target,
            'noise': noise,
            
            # Metadata
            'captions': captions,
            'keys': keys,
            'batch_size': batch_size,
            'training_mode': valid_batch[0]['training_mode'],
            'num_tokens': valid_batch[0]['num_tokens'],
            'seq_len': seq_len,
            
            # Data characteristics
            'eva_embeddings_normalized': False,
            'clip_embeddings_normalized': False,
            'eva_norm_mean': eva_norm_mean,
            'clip_norm_mean': clip_norm_mean,
            'noise_norm_mean': noise_norm_mean,
            'clip_std': clip_embeddings.std().item(),
            'eva_std': eva_embeddings.std().item(),
            
            # NEW: Detailed statistics for debugging
            'clip_norm_std': torch.norm(clip_embeddings, dim=-1).std().item(),
            'eva_norm_std': torch.norm(eva_embeddings, dim=-1).std().item(),
            'clip_norm_min': torch.norm(clip_embeddings, dim=-1).min().item(),
            'clip_norm_max': torch.norm(clip_embeddings, dim=-1).max().item(),
            'eva_norm_min': torch.norm(eva_embeddings, dim=-1).min().item(),
            'eva_norm_max': torch.norm(eva_embeddings, dim=-1).max().item(),
        }
        
    except Exception as e:
        logger.error(f"Error in collate function: {e}")
        logger.error(f"Batch size: {len(batch)}")
        if batch:
            try:
                logger.error(f"First item keys: {list(batch[0].keys())}")
                for key, value in batch[0].items():
                    if torch.is_tensor(value):
                        logger.error(f"  {key}: {value.shape} {value.dtype}")
            except:
                pass
        raise


def create_clip_reproduction_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    normalize_embeddings: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    # NEW: Consistent evaluation parameters
    eval_samples: int = 100,  # Number of samples for evaluation
    eval_from_training: bool = True,  # Use training samples for evaluation
    collect_statistics: bool = True,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create dataloaders with consistent train/eval data"""
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    logger.info(f"Creating CLIP reproduction dataloaders with CONSISTENT evaluation:")
    logger.info(f"  Target: CLIP embeddings [B, N, 1024]")
    logger.info(f"  Conditioning: EVA embeddings [B, N, 4096]")
    logger.info(f"  Normalize: {normalize_embeddings}")
    logger.info(f"  🎯 CONSISTENT EVAL: Use {eval_samples} samples from training data")
    
    # Create training dataset
    train_dataset = BLIP3oCLIPReproductionDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        split="train",
        training_mode=training_mode,
        normalize_embeddings=normalize_embeddings,
        max_shards=max_shards,
        shuffle_shards=True,
        shuffle_within_shard=True,
        collect_statistics=collect_statistics,
        validate_norms=True,
        **kwargs
    )
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=clip_reproduction_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    # NEW: Create evaluation dataset from training samples
    eval_dataloader = None
    if eval_from_training:
        logger.info(f"Creating consistent evaluation dataset from training data...")
        
        # Load all training samples
        all_samples = train_dataset.load_all_samples()
        
        if all_samples and len(all_samples) >= eval_samples:
            # Take first N samples for evaluation (consistent subset)
            eval_subset = all_samples[:eval_samples]
            
            # Create a simple dataset from these samples
            class EvalDataset(Dataset):
                def __init__(self, samples):
                    self.samples = samples
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    return self.samples[idx]
            
            eval_dataset = EvalDataset(eval_subset)
            
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                num_workers=min(num_workers, 1),
                collate_fn=clip_reproduction_collate_fn,
                pin_memory=pin_memory,
                drop_last=False,
                shuffle=False,  # Don't shuffle evaluation
                persistent_workers=min(num_workers, 1) > 0,
            )
            
            # Validate that evaluation data has consistent statistics
            logger.info(f"📊 Validation: Computing evaluation data statistics...")
            eval_clip_norms = []
            eval_eva_norms = []
            
            for sample in eval_subset[:50]:  # Check first 50 samples
                clip_norm = torch.norm(sample['clip_embeddings'], dim=-1).mean().item()
                eva_norm = torch.norm(sample['eva_embeddings'], dim=-1).mean().item()
                eval_clip_norms.append(clip_norm)
                eval_eva_norms.append(eva_norm)
            
            eval_clip_mean = np.mean(eval_clip_norms)
            eval_eva_mean = np.mean(eval_eva_norms)
            
            # Get training statistics for comparison
            train_stats = train_dataset.get_data_statistics()
            train_clip_mean = train_stats.get('clip_norm_mean', 0)
            train_eva_mean = train_stats.get('eva_norm_mean', 0)
            
            logger.info(f"📊 CONSISTENCY CHECK:")
            logger.info(f"   Training CLIP norm: {train_clip_mean:.2f}")
            logger.info(f"   Evaluation CLIP norm: {eval_clip_mean:.2f}")
            logger.info(f"   Training EVA norm: {train_eva_mean:.2f}")
            logger.info(f"   Evaluation EVA norm: {eval_eva_mean:.2f}")
            
            # Check for significant differences
            clip_diff = abs(train_clip_mean - eval_clip_mean)
            eva_diff = abs(train_eva_mean - eval_eva_mean)
            
            if clip_diff > 5.0:
                logger.warning(f"🚨 LARGE CLIP norm difference: {clip_diff:.2f}")
            if eva_diff > 5.0:
                logger.warning(f"🚨 LARGE EVA norm difference: {eva_diff:.2f}")
            
            if clip_diff <= 2.0 and eva_diff <= 2.0:
                logger.info(f"✅ CONSISTENCY VALIDATED: Norm differences are small")
            else:
                logger.warning(f"⚠️ CONSISTENCY ISSUE: Large norm differences detected")
            
        else:
            logger.warning(f"Not enough training samples for evaluation ({len(all_samples)} < {eval_samples})")
    else:
        # Fallback: separate evaluation dataset (not recommended)
        logger.warning("Using separate evaluation dataset - may cause norm mismatches!")
        eval_dataset = BLIP3oCLIPReproductionDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split="eval",
            training_mode=training_mode,
            normalize_embeddings=normalize_embeddings,
            max_shards=max_shards,
            shuffle_shards=False,
            shuffle_within_shard=False,
            collect_statistics=False,
            **kwargs
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            num_workers=min(num_workers, 1),
            collate_fn=clip_reproduction_collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=min(num_workers, 1) > 0,
        )
    
    logger.info(f"CLIP reproduction dataloaders created successfully")
    logger.info(f"  Training dataset length: {len(train_dataset):,}")
    if eval_dataloader:
        try:
            eval_length = len(eval_dataloader.dataset)
            logger.info(f"  Evaluation dataset length: {eval_length:,}")
            logger.info(f"  ✅ CONSISTENT EVALUATION: Uses subset of training data")
        except:
            logger.info(f"  Evaluation dataloader created (length unknown)")
    
    return train_dataloader, eval_dataloader