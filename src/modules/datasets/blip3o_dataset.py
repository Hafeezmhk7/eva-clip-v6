#!/usr/bin/env python3
"""
FIXED: BLIP3-o Dataset with NO Noise Scaling
Key fixes:
1. NO noise scaling applied during data loading (keep raw embeddings)
2. Standard Gaussian noise in collate function (NO SCALING)
3. Consistent data processing between train and eval
4. Enhanced debugging for norm analysis
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
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BLIP3oCLIPReproductionDataset(IterableDataset):
    """
    FIXED: Dataset for CLIP reproduction with NO unwanted normalization
    
    This dataset loads:
    - CLIP embeddings [B, N, 1024] as TARGET (what we want to reproduce) - RAW, no normalization
    - EVA embeddings [B, N, 4096] as CONDITIONING (guidance) - RAW, no normalization
    
    FIXED: NO normalization applied during data loading
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        split: str = "train",
        training_mode: str = "patch_only",
        normalize_embeddings: bool = False,  # FIXED: Always False to prevent normalization
        max_shards: Optional[int] = None,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        expected_tokens: Optional[int] = None,
        # Error handling
        skip_corrupted: bool = True,
        validate_shapes: bool = True,
        max_retries: int = 3,
        # FIXED: Remove statistics collection for consistency
        collect_statistics: bool = False,
    ):
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.split = split
        self.training_mode = training_mode
        # FIXED: Force normalization to False to prevent unwanted normalization
        self.normalize_embeddings = False  # Always False regardless of input
        self.max_shards = max_shards
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.skip_corrupted = skip_corrupted
        self.validate_shapes = validate_shapes
        self.max_retries = max_retries
        self.collect_statistics = False  # Always False
        
        # Log if user tried to enable normalization
        if normalize_embeddings:
            logger.warning("🚫 Normalization was requested but DISABLED to prevent unwanted normalization during training")
        
        # Determine expected tokens
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
        # Setup random state
        self.rng = random.Random(42)
        
        # FIXED: Simple statistics (no adaptive updates)
        self.data_statistics = {
            'clip_norm_mean': 0.0,
            'eva_norm_mean': 0.0,
            'samples_seen': 0,
            'clip_norm_history': [],
            'eva_norm_history': [],
        }
        
        # Load manifest and prepare shards
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        # Calculate estimated length for __len__ method
        self._estimate_length()
        
        logger.info(f"FIXED CLIP Reproduction Dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  TARGET: CLIP embeddings [B, N, 1024] - RAW (no normalization)")
        logger.info(f"  CONDITIONING: EVA embeddings [B, N, 4096] - RAW (no normalization)")
        logger.info(f"  🚫 Normalization: DISABLED (raw embedding space)")
        logger.info(f"  Shards: {len(self.shard_files) if hasattr(self, 'shard_files') else 'Unknown'}")
        logger.info(f"  Statistics collection: {self.collect_statistics}")

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
        # Look for shard files
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
        """Estimate total number of samples for __len__ method"""
        try:
            # If manifest is available, use it
            if self.manifest.get('total_samples', 0) > 0:
                manifest_samples = self.manifest['total_samples']
                # Adjust for max_shards limitation
                if self.max_shards is not None and self.manifest.get('total_shards', 0) > 0:
                    ratio = min(self.max_shards / self.manifest['total_shards'], 1.0)
                    self.estimated_length = int(manifest_samples * ratio)
                else:
                    self.estimated_length = manifest_samples
                logger.info(f"Using manifest for length estimation: {self.estimated_length:,} samples")
                return
            
            # Fallback: try to load first shard to estimate
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
            
            # Final fallback: rough estimate
            self.estimated_length = len(self.shard_files) * 1000  # Assume 1000 samples per shard
            logger.warning(f"Using rough length estimate: {self.estimated_length:,} samples")
            
        except Exception as e:
            logger.warning(f"Length estimation failed: {e}")
            self.estimated_length = 1000  # Very conservative fallback

    def __len__(self) -> int:
        """Return estimated length for DataLoader compatibility"""
        return self.estimated_length

    def _update_statistics(self, clip_emb: torch.Tensor, eva_emb: torch.Tensor):
        """FIXED: Track statistics without normalization"""
        if not self.collect_statistics:
            return
            
        with torch.no_grad():
            batch_size = clip_emb.shape[0]
            
            # FIXED: Simple statistics on RAW embeddings (no normalization)
            clip_norm_mean = torch.norm(clip_emb, dim=-1).mean().item()
            eva_norm_mean = torch.norm(eva_emb, dim=-1).mean().item()
            
            # Store history for analysis
            self.data_statistics['clip_norm_history'].append(clip_norm_mean)
            self.data_statistics['eva_norm_history'].append(eva_norm_mean)
            
            # Simple running average
            if self.data_statistics['samples_seen'] == 0:
                self.data_statistics['clip_norm_mean'] = clip_norm_mean
                self.data_statistics['eva_norm_mean'] = eva_norm_mean
            else:
                # Simple running average
                n = self.data_statistics['samples_seen']
                self.data_statistics['clip_norm_mean'] = (self.data_statistics['clip_norm_mean'] * n + clip_norm_mean * batch_size) / (n + batch_size)
                self.data_statistics['eva_norm_mean'] = (self.data_statistics['eva_norm_mean'] * n + eva_norm_mean * batch_size) / (n + batch_size)
            
            self.data_statistics['samples_seen'] += batch_size

    def _load_shard(self, shard_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single shard with error handling"""
        for attempt in range(self.max_retries):
            try:
                with open(shard_path, 'rb') as f:
                    shard_data = pickle.load(f)
                
                # Validate and process shard
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
        """Validate and process shard data WITHOUT normalization"""
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
                # Add CLS token (average of patches)
                clip_cls = clip_emb.mean(dim=1, keepdim=True)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                shard_data['clip_blip3o_embeddings'] = torch.cat([clip_cls, clip_emb], dim=1)
                shard_data['eva_blip3o_embeddings'] = torch.cat([eva_cls, eva_emb], dim=1)
            elif current_tokens == 257 and self.expected_tokens == 256:
                # Remove CLS token
                shard_data['clip_blip3o_embeddings'] = clip_emb[:, 1:, :]
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
            else:
                raise ValueError(f"Cannot adapt from {current_tokens} to {self.expected_tokens} tokens")
        
        # FIXED: Update statistics BEFORE any processing (if enabled)
        if self.collect_statistics:
            self._update_statistics(
                shard_data['clip_blip3o_embeddings'], 
                shard_data['eva_blip3o_embeddings']
            )
        
        # FIXED: NO normalization applied - keep embeddings in raw space
        # Mark that no normalization was applied
        shard_data['normalization_applied'] = False
        shard_data['raw_embedding_space'] = True
        
        # Log shard statistics for debugging
        if logger.isEnabledFor(logging.DEBUG):
            clip_norm = torch.norm(shard_data['clip_blip3o_embeddings'], dim=-1).mean().item()
            eva_norm = torch.norm(shard_data['eva_blip3o_embeddings'], dim=-1).mean().item()
            logger.debug(f"Shard {shard_path.name}: CLIP norm={clip_norm:.3f}, EVA norm={eva_norm:.3f} (RAW)")

    def _load_next_shard(self) -> bool:
        """Load next shard"""
        # Cleanup previous shard
        if self.current_shard_data is not None:
            del self.current_shard_data
            gc.collect()
        
        # Check if more shards available
        if self.current_shard_idx >= len(self.shard_files):
            self.current_shard_data = None
            return False
        
        # Try to load next shard
        while self.current_shard_idx < len(self.shard_files):
            shard_path = self.shard_files[self.current_shard_idx]
            
            self.current_shard_data = self._load_shard(shard_path)
            
            if self.current_shard_data is not None:
                # Prepare samples
                num_samples = self.current_shard_data['clip_blip3o_embeddings'].shape[0]
                self.current_samples = list(range(num_samples))
                
                if self.shuffle_within_shard:
                    self.rng.shuffle(self.current_samples)
                
                self.current_sample_idx = 0
                
                # FIXED: Log shard loading with norm information
                clip_norm = torch.norm(self.current_shard_data['clip_blip3o_embeddings'], dim=-1).mean().item()
                eva_norm = torch.norm(self.current_shard_data['eva_blip3o_embeddings'], dim=-1).mean().item()
                logger.info(f"Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: {num_samples} samples, CLIP norm={clip_norm:.3f}, EVA norm={eva_norm:.3f}")
                
                self.current_shard_idx += 1
                return True
            else:
                self.current_shard_idx += 1
                continue
        
        self.current_shard_data = None
        return False

    def get_data_statistics(self) -> Dict[str, float]:
        """Get collected data statistics"""
        stats = self.data_statistics.copy()
        
        # Add additional analysis
        if self.data_statistics['clip_norm_history']:
            clip_norms = self.data_statistics['clip_norm_history']
            stats['clip_norm_std'] = float(np.std(clip_norms))
            stats['clip_norm_min'] = float(np.min(clip_norms))
            stats['clip_norm_max'] = float(np.max(clip_norms))
            stats['clip_norm_range'] = stats['clip_norm_max'] - stats['clip_norm_min']
        
        if self.data_statistics['eva_norm_history']:
            eva_norms = self.data_statistics['eva_norm_history']
            stats['eva_norm_std'] = float(np.std(eva_norms))
            stats['eva_norm_min'] = float(np.min(eva_norms))
            stats['eva_norm_max'] = float(np.max(eva_norms))
            stats['eva_norm_range'] = stats['eva_norm_max'] - stats['eva_norm_min']
        
        return stats

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through all samples WITHOUT normalization"""
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        logger.debug(f"Starting iteration over {len(self.shard_files)} shards (NO normalization)")
        
        if not self._load_next_shard():
            return
        
        while self.current_shard_data is not None:
            while self.current_sample_idx < len(self.current_samples):
                try:
                    sample_idx = self.current_samples[self.current_sample_idx]
                    
                    # FIXED: Extract sample data WITHOUT normalization
                    clip_emb = self.current_shard_data['clip_blip3o_embeddings'][sample_idx]
                    eva_emb = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Final validation
                    if self.validate_shapes:
                        if clip_emb.shape != (self.expected_tokens, 1024):
                            raise ValueError(f"Invalid CLIP shape: {clip_emb.shape}")
                        if eva_emb.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {eva_emb.shape}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(clip_emb).any() or torch.isnan(eva_emb).any():
                        if self.skip_corrupted:
                            self.current_sample_idx += 1
                            continue
                        else:
                            raise ValueError("NaN detected in embeddings")
                    
                    # FIXED: Create sample item for CLIP reproduction (RAW embeddings)
                    item = {
                        'eva_embeddings': eva_emb,      # [N, 4096] - CONDITIONING (RAW)
                        'clip_embeddings': clip_emb,    # [N, 1024] - TARGET to reproduce (RAW)
                        'caption': caption,
                        'key': f"shard_{self.current_shard_idx-1}_sample_{sample_idx}",
                        'sample_idx': sample_idx,
                        'training_mode': self.training_mode,
                        'num_tokens': self.expected_tokens,
                        'normalized': False,  # Always False - no normalization applied
                        'raw_embedding_space': True,  # Always True - keep in raw space
                        
                        # FIXED: Embedding statistics for debugging
                        'clip_norm': torch.norm(clip_emb, dim=-1).mean().item(),
                        'eva_norm': torch.norm(eva_emb, dim=-1).mean().item(),
                        'clip_std': clip_emb.std().item(),
                        'eva_std': eva_emb.std().item(),
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
        
        logger.info(f"Iteration completed: {self.total_samples_processed} samples processed (RAW embedding space)")


def clip_reproduction_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    FIXED: Collate function for CLIP reproduction with NO noise scaling
    
    This function:
    1. Takes clean CLIP embeddings as targets (RAW, no normalization)
    2. Uses EVA embeddings for conditioning (RAW, no normalization)
    3. Creates STANDARD GAUSSIAN NOISE (NO SCALING)
    4. Focuses on clean data preparation WITHOUT any scaling
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # FIXED: Stack embeddings WITHOUT normalization
        eva_embeddings = torch.stack([item['eva_embeddings'] for item in valid_batch])     # [B, N, 4096] - RAW
        clip_embeddings = torch.stack([item['clip_embeddings'] for item in valid_batch])   # [B, N, 1024] - RAW
        
        # Collect metadata
        captions = [item['caption'] for item in valid_batch]
        keys = [item['key'] for item in valid_batch]
        
        batch_size, seq_len, clip_dim = clip_embeddings.shape
        device = clip_embeddings.device
        dtype = clip_embeddings.dtype
        
        # Ensure float32 for stability
        eva_embeddings = eva_embeddings.float()
        clip_embeddings = clip_embeddings.float()
        
        # Sample random timesteps for each sample in batch
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # FIXED: Create STANDARD GAUSSIAN NOISE (NO SCALING)
        noise = torch.randn_like(clip_embeddings, device=device, dtype=dtype)
        
        # Linear interpolation for rectified flow: x_t = (1-t) * noise + t * clip_clean (RAW)
        t_expanded = timesteps.view(batch_size, 1, 1)  # [B, 1, 1]
        noisy_clip = (1 - t_expanded) * noise + t_expanded * clip_embeddings
        
        # Velocity target: v = clip_clean - noise (for rectified flow) (RAW)
        velocity_target = clip_embeddings - noise
        
        # Validation
        assert eva_embeddings.shape == (batch_size, seq_len, 4096), f"EVA shape: {eva_embeddings.shape}"
        assert clip_embeddings.shape == (batch_size, seq_len, 1024), f"CLIP shape: {clip_embeddings.shape}"
        assert noisy_clip.shape == (batch_size, seq_len, 1024), f"Noisy CLIP shape: {noisy_clip.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 1024), f"Velocity target shape: {velocity_target.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        # FIXED: Collect embedding statistics for debugging
        clip_norms = [item.get('clip_norm', 0) for item in valid_batch]
        eva_norms = [item.get('eva_norm', 0) for item in valid_batch]
        
        return {
            # Model inputs (RAW embeddings, no normalization)
            'encoder_hidden_states': eva_embeddings,     # [B, N, 4096] - EVA conditioning (RAW)
            'hidden_states': noisy_clip,                 # [B, N, 1024] - Noisy CLIP input (RAW)
            'timestep': timesteps,                       # [B] - Flow matching timesteps
            
            # Training targets (RAW embeddings, no normalization)
            'clip_embeddings': clip_embeddings,          # [B, N, 1024] - Clean CLIP (target) (RAW)
            'velocity_target': velocity_target,          # [B, N, 1024] - Velocity for flow matching (RAW)
            'noise': noise,                              # [B, N, 1024] - STANDARD GAUSSIAN NOISE (NO SCALING)
            
            # Metadata
            'captions': captions,
            'keys': keys,
            'batch_size': batch_size,
            'training_mode': valid_batch[0]['training_mode'],
            'num_tokens': valid_batch[0]['num_tokens'],
            'seq_len': seq_len,
            
            # FIXED: Data characteristics for debugging (NO normalization applied)
            'eva_embeddings_normalized': False,  # Always False
            'clip_embeddings_normalized': False,  # Always False
            'raw_embedding_space': True,  # Always True
            'normalization_applied': False,  # Always False
            'noise_scaling_applied': False,  # NEW: Always False - no noise scaling
            
            # FIXED: Embedding statistics (standard Gaussian noise)
            'eva_norm_mean': torch.norm(eva_embeddings, dim=-1).mean().item(),
            'clip_norm_mean': torch.norm(clip_embeddings, dim=-1).mean().item(),
            'noise_norm_mean': torch.norm(noise, dim=-1).mean().item(),
            'noise_std': noise.std().item(),  # Should be ~1.0 for standard Gaussian
            'noise_mean': noise.mean().item(),  # Should be ~0.0 for standard Gaussian
            
            # Verify standard Gaussian properties
            'noise_is_standard_gaussian': abs(noise.std().item() - 1.0) < 0.2 and abs(noise.mean().item()) < 0.2,
            
            # Individual sample norms for analysis
            'batch_clip_norms': clip_norms,
            'batch_eva_norms': eva_norms,
            'clip_norm_std_in_batch': np.std(clip_norms) if clip_norms else 0.0,
            'eva_norm_std_in_batch': np.std(eva_norms) if eva_norms else 0.0,
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
                    else:
                        logger.error(f"  {key}: {type(value)} = {value}")
            except:
                pass
        raise


def create_clip_reproduction_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    normalize_embeddings: bool = False,  # FIXED: Always forced to False
    num_workers: int = 0,
    pin_memory: bool = False,
    # FIXED: Disable statistics collection by default
    collect_statistics: bool = False,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """FIXED: Create dataloaders for CLIP reproduction with NO unwanted normalization and NO noise scaling"""
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    # FIXED: Force normalization to False
    if normalize_embeddings:
        logger.warning("🚫 Normalization was requested but FORCED TO FALSE to prevent unwanted normalization")
    normalize_embeddings = False
    
    logger.info(f"FIXED: Creating CLIP reproduction dataloaders (NO NOISE SCALING):")
    logger.info(f"  Target: CLIP embeddings [B, N, 1024] - RAW (no normalization)")
    logger.info(f"  Conditioning: EVA embeddings [B, N, 4096] - RAW (no normalization)")
    logger.info(f"  🚫 Normalization: DISABLED (forced)")
    logger.info(f"  🎲 Noise: Standard Gaussian (NO SCALING)")
    logger.info(f"  Collect statistics: {collect_statistics}")
    
    # FIXED: Use identical settings for both train and eval
    dataset_kwargs = {
        'chunked_embeddings_dir': chunked_embeddings_dir,
        'training_mode': training_mode,
        'normalize_embeddings': normalize_embeddings,  # Always False
        'max_shards': max_shards,
        'collect_statistics': collect_statistics,
        **kwargs
    }
    
    # Create training dataset
    train_dataset = BLIP3oCLIPReproductionDataset(
        split="train",
        shuffle_shards=True,
        shuffle_within_shard=True,
        **dataset_kwargs
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
    
    # FIXED: Create evaluation dataset with identical settings but no shuffling
    eval_dataset = BLIP3oCLIPReproductionDataset(
        split="eval",
        shuffle_shards=False,  # No shuffling for consistent evaluation
        shuffle_within_shard=False,
        **dataset_kwargs
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
    
    logger.info(f"FIXED: CLIP reproduction dataloaders created successfully")
    logger.info(f"  Training dataset length: {len(train_dataset):,}")
    logger.info(f"  Evaluation dataset length: {len(eval_dataset):,}")
    logger.info(f"  🚫 NO normalization applied in either dataset")
    logger.info(f"  🎲 Standard Gaussian noise (NO SCALING) in collate function")
    logger.info(f"  Consistent processing: ✅")
    
    return train_dataloader, eval_dataloader