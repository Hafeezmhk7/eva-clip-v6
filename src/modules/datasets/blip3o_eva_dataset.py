#!/usr/bin/env python3
"""
Fixed BLIP3-o Dataset for EVA-CLIP Denoising
Key fixes:
1. EVA → EVA denoising (not CLIP → EVA reproduction)
2. Proper spherical data handling
3. Correct input/output flow
4. Better error handling and normalization
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
import math

logger = logging.getLogger(__name__)


class BLIP3oEVADenoisingDataset(IterableDataset):
    """
    Fixed dataset for EVA-CLIP denoising with proper spherical data handling
    
    This dataset:
    - Takes clean EVA embeddings [B, N, 4096] as TARGET and CONDITIONING
    - Creates noisy versions for INPUT
    - Implements proper spherical noise and interpolation
    """
    
    def __init__(
        self,
        chunked_embeddings_dir: Union[str, Path],
        split: str = "train",
        training_mode: str = "patch_only",
        max_shards: Optional[int] = None,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        expected_tokens: Optional[int] = None,
        # Spherical noise parameters
        noise_schedule: str = "uniform",  # uniform, cosine
        max_noise_level: float = 0.9,  # Maximum noise mixing ratio
        min_noise_level: float = 0.1,   # Minimum noise mixing ratio
        # Error handling
        skip_corrupted: bool = True,
        validate_shapes: bool = True,
        max_retries: int = 3,
    ):
        super().__init__()
        
        self.chunked_embeddings_dir = Path(chunked_embeddings_dir)
        self.split = split
        self.training_mode = training_mode
        self.max_shards = max_shards
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.skip_corrupted = skip_corrupted
        self.validate_shapes = validate_shapes
        self.max_retries = max_retries
        
        # Spherical noise parameters
        self.noise_schedule = noise_schedule
        self.max_noise_level = max_noise_level
        self.min_noise_level = min_noise_level
        
        # Determine expected tokens
        if expected_tokens is None:
            self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        else:
            self.expected_tokens = expected_tokens
        
        # Setup random state
        self.rng = random.Random(42)
        
        # Load manifest and prepare shards
        self._load_manifest()
        self._prepare_shard_list()
        
        # Current state
        self.current_shard_idx = 0
        self.current_shard_data = None
        self.current_sample_idx = 0
        self.total_samples_processed = 0
        
        logger.info(f"EVA Denoising Dataset initialized:")
        logger.info(f"  Directory: {self.chunked_embeddings_dir}")
        logger.info(f"  Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"  TASK: EVA Denoising")
        logger.info(f"  INPUT: Noisy EVA embeddings [B, N, 4096]")
        logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
        logger.info(f"  TARGET: Clean EVA embeddings [B, N, 4096]")
        logger.info(f"  Noise schedule: {self.noise_schedule}")
        logger.info(f"  Noise range: [{self.min_noise_level}, {self.max_noise_level}]")
        logger.info(f"  Shards: {len(self.shard_files) if hasattr(self, 'shard_files') else 'Loading...'}")

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
        """Validate and process shard data"""
        # Check required keys
        required_keys = ['eva_blip3o_embeddings', 'captions']
        for key in required_keys:
            if key not in shard_data:
                raise ValueError(f"Missing key '{key}' in shard {shard_path}")
        
        # Get EVA embeddings (this is our main data)
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Convert to tensors if needed
        if not torch.is_tensor(eva_emb):
            eva_emb = torch.tensor(eva_emb, dtype=torch.float32)
            shard_data['eva_blip3o_embeddings'] = eva_emb
        
        # Validate shapes
        if self.validate_shapes:
            if eva_emb.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got EVA: {eva_emb.shape}")
        
        # Handle token count adaptation
        current_tokens = eva_emb.shape[1]
        if current_tokens != self.expected_tokens:
            logger.debug(f"Adapting from {current_tokens} to {self.expected_tokens} tokens")
            
            if current_tokens == 256 and self.expected_tokens == 257:
                # Add CLS token (average of patches)
                eva_cls = eva_emb.mean(dim=1, keepdim=True)
                shard_data['eva_blip3o_embeddings'] = torch.cat([eva_cls, eva_emb], dim=1)
            elif current_tokens == 257 and self.expected_tokens == 256:
                # Remove CLS token
                shard_data['eva_blip3o_embeddings'] = eva_emb[:, 1:, :]
            else:
                raise ValueError(f"Cannot adapt from {current_tokens} to {self.expected_tokens} tokens")
        
        # Apply normalization - CRITICAL: EVA embeddings must be L2 normalized
        shard_data = self._normalize_embeddings(shard_data)

    def _normalize_embeddings(self, shard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply L2 normalization to EVA embeddings"""
        eva_emb = shard_data['eva_blip3o_embeddings']
        
        # Check for NaN/Inf
        if torch.isnan(eva_emb).any() or torch.isinf(eva_emb).any():
            logger.warning("Found NaN/Inf in EVA embeddings")
            eva_emb = torch.nan_to_num(eva_emb, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply L2 normalization to ensure unit sphere
        eps = 1e-8
        eva_normalized = F.normalize(eva_emb + eps, p=2, dim=-1)
        
        # Verify normalization
        eva_norm = torch.norm(eva_normalized, dim=-1).mean().item()
        
        if abs(eva_norm - 1.0) > 0.1:
            logger.warning(f"EVA normalization may have failed: norm = {eva_norm:.3f}")
        
        shard_data['eva_blip3o_embeddings'] = eva_normalized
        shard_data['normalization_applied'] = True
        
        return shard_data

    def _add_spherical_noise(self, clean_eva: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add spherical noise to EVA embeddings using slerp"""
        device = clean_eva.device
        dtype = clean_eva.dtype
        
        # Generate random noise on sphere
        noise = torch.randn_like(clean_eva, device=device, dtype=dtype)
        noise = F.normalize(noise, p=2, dim=-1)
        
        # Spherical linear interpolation (slerp)
        # noise_level = 0: clean, noise_level = 1: pure noise
        
        # Compute angle between clean and noise
        cos_angle = torch.sum(clean_eva * noise, dim=-1, keepdim=True)
        cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
        angle = torch.acos(cos_angle)
        
        # Avoid division by zero
        sin_angle = torch.sin(angle)
        sin_angle = torch.clamp(sin_angle, min=1e-7)
        
        # Slerp formula: slerp(a, b, t) = (sin((1-t)*θ)/sin(θ)) * a + (sin(t*θ)/sin(θ)) * b
        clean_weight = torch.sin((1 - noise_level) * angle) / sin_angle
        noise_weight = torch.sin(noise_level * angle) / sin_angle
        
        noisy_eva = clean_weight * clean_eva + noise_weight * noise
        
        # Ensure result is on unit sphere
        noisy_eva = F.normalize(noisy_eva, p=2, dim=-1)
        
        return noisy_eva, noise

    def _sample_noise_level(self) -> float:
        """Sample noise level based on schedule"""
        if self.noise_schedule == "uniform":
            return self.rng.uniform(self.min_noise_level, self.max_noise_level)
        elif self.noise_schedule == "cosine":
            # Cosine schedule favors lower noise levels
            u = self.rng.uniform(0, 1)
            t = 0.5 * (1 + math.cos(u * math.pi))  # Cosine decay
            return self.min_noise_level + t * (self.max_noise_level - self.min_noise_level)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

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
                num_samples = self.current_shard_data['eva_blip3o_embeddings'].shape[0]
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

    def __len__(self) -> int:
        """Estimate total number of samples"""
        if hasattr(self, '_estimated_length'):
            return self._estimated_length
        
        # Try to estimate from manifest
        if hasattr(self, 'manifest') and 'total_samples' in self.manifest:
            manifest_samples = self.manifest['total_samples']
            if self.max_shards is not None:
                # Estimate based on max_shards ratio
                total_shards = self.manifest.get('total_shards', len(self.shard_files))
                if total_shards > 0:
                    estimated_samples = int(manifest_samples * self.max_shards / total_shards)
                    self._estimated_length = estimated_samples
                    return estimated_samples
            else:
                self._estimated_length = manifest_samples
                return manifest_samples
        
        # Fallback: estimate based on file count and average samples per shard
        num_shards = len(self.shard_files) if hasattr(self, 'shard_files') else 1
        avg_samples_per_shard = 1000  # Conservative estimate
        
        estimated_samples = num_shards * avg_samples_per_shard
        self._estimated_length = estimated_samples
        
        logger.debug(f"Estimated dataset length: {estimated_samples} samples from {num_shards} shards")
        return estimated_samples

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
                    
                    # Extract clean EVA embeddings
                    clean_eva = self.current_shard_data['eva_blip3o_embeddings'][sample_idx]
                    caption = self.current_shard_data['captions'][sample_idx]
                    
                    # Final validation
                    if self.validate_shapes:
                        if clean_eva.shape != (self.expected_tokens, 4096):
                            raise ValueError(f"Invalid EVA shape: {clean_eva.shape}")
                    
                    # Check for NaN/Inf
                    if torch.isnan(clean_eva).any():
                        if self.skip_corrupted:
                            self.current_sample_idx += 1
                            continue
                        else:
                            raise ValueError("NaN detected in EVA embeddings")
                    
                    # Sample noise level for this sample
                    noise_level = self._sample_noise_level()
                    
                    # Add spherical noise to create noisy version
                    noisy_eva, noise = self._add_spherical_noise(clean_eva, noise_level)
                    
                    # Create sample item for EVA denoising
                    item = {
                        # Model inputs
                        'noisy_eva_embeddings': noisy_eva,      # [N, 4096] - Noisy input
                        'clean_eva_embeddings': clean_eva,      # [N, 4096] - Clean conditioning & target
                        'noise': noise,                         # [N, 4096] - Pure noise used
                        'noise_level': noise_level,             # scalar - Noise mixing ratio
                        'caption': caption,
                        
                        # Metadata
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


def eva_denoising_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for EVA denoising with proper spherical flow matching setup
    
    This function:
    1. Takes clean EVA embeddings as targets and conditioning
    2. Uses pre-computed noisy versions for input
    3. Sets up spherical flow matching with proper timesteps
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Filter valid items
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("No valid items in batch")
    
    try:
        # Stack embeddings
        noisy_eva = torch.stack([item['noisy_eva_embeddings'] for item in valid_batch])    # [B, N, 4096]
        clean_eva = torch.stack([item['clean_eva_embeddings'] for item in valid_batch])    # [B, N, 4096]
        noise = torch.stack([item['noise'] for item in valid_batch])                       # [B, N, 4096]
        noise_levels = torch.tensor([item['noise_level'] for item in valid_batch])         # [B]
        
        # Collect metadata
        captions = [item['caption'] for item in valid_batch]
        keys = [item['key'] for item in valid_batch]
        
        batch_size, seq_len, eva_dim = clean_eva.shape
        device = clean_eva.device
        dtype = clean_eva.dtype
        
        # Ensure float32 for stability
        noisy_eva = noisy_eva.float()
        clean_eva = clean_eva.float()
        noise = noise.float()
        noise_levels = noise_levels.float()
        
        # Ensure L2 normalization
        eps = 1e-8
        noisy_eva = F.normalize(noisy_eva + eps, p=2, dim=-1)
        clean_eva = F.normalize(clean_eva + eps, p=2, dim=-1)
        noise = F.normalize(noise + eps, p=2, dim=-1)
        
        # SPHERICAL FLOW MATCHING SETUP
        # Sample timesteps for flow matching (0 = noise, 1 = clean)
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # For spherical flow, we interpolate on the sphere using slerp
        t_expanded = timesteps.view(batch_size, 1, 1)  # [B, 1, 1]
        
        # Compute angles between clean and noise
        cos_angles = torch.sum(clean_eva * noise, dim=-1, keepdim=True)
        cos_angles = torch.clamp(cos_angles, -1 + 1e-7, 1 - 1e-7)
        angles = torch.acos(cos_angles)
        
        # Avoid division by zero
        sin_angles = torch.sin(angles)
        sin_angles = torch.clamp(sin_angles, min=1e-7)
        
        # Spherical interpolation: x_t = slerp(noise, clean, t)
        clean_weight = torch.sin(t_expanded * angles) / sin_angles
        noise_weight = torch.sin((1 - t_expanded) * angles) / sin_angles
        
        x_t = noise_weight * noise + clean_weight * clean_eva
        x_t = F.normalize(x_t + eps, p=2, dim=-1)
        
        # Spherical velocity (tangent to sphere)
        # For spherical flow: v = d/dt slerp(noise, clean, t)
        velocity_target = (clean_eva - noise_weight / clean_weight * x_t) * angles / sin_angles
        
        # Alternative: Direct velocity from parametric derivative
        # This is more stable for training
        velocity_target = clean_eva - noise
        
        # Validation
        assert noisy_eva.shape == (batch_size, seq_len, 4096), f"Noisy EVA shape: {noisy_eva.shape}"
        assert clean_eva.shape == (batch_size, seq_len, 4096), f"Clean EVA shape: {clean_eva.shape}"
        assert x_t.shape == (batch_size, seq_len, 4096), f"x_t shape: {x_t.shape}"
        assert velocity_target.shape == (batch_size, seq_len, 4096), f"Velocity target shape: {velocity_target.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        return {
            # Model inputs
            'hidden_states': x_t,                        # [B, N, 4096] - Interpolated state
            'encoder_hidden_states': clean_eva,          # [B, N, 4096] - Clean EVA conditioning
            'timestep': timesteps,                       # [B] - Flow matching timesteps
            
            # Training targets
            'clean_eva_embeddings': clean_eva,           # [B, N, 4096] - Clean EVA (target)
            'velocity_target': velocity_target,          # [B, N, 4096] - Velocity for flow matching
            'noise': noise,                              # [B, N, 4096] - Pure noise
            'noisy_eva_embeddings': noisy_eva,           # [B, N, 4096] - Pre-computed noisy version
            
            # Flow matching state
            'x_t': x_t,                                  # [B, N, 4096] - Current flow state
            'noise_levels': noise_levels,                # [B] - Original noise levels
            
            # Metadata
            'captions': captions,
            'keys': keys,
            'batch_size': batch_size,
            'training_mode': valid_batch[0]['training_mode'],
            'num_tokens': valid_batch[0]['num_tokens'],
            'seq_len': seq_len,
            
            # Normalization status
            'eva_embeddings_normalized': True,
            'eva_norm_mean': torch.norm(clean_eva, dim=-1).mean().item(),
            'noisy_eva_norm_mean': torch.norm(noisy_eva, dim=-1).mean().item(),
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


def create_eva_denoising_dataloaders(
    chunked_embeddings_dir: Union[str, Path],
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    training_mode: str = "patch_only",
    max_shards: Optional[int] = None,
    noise_schedule: str = "uniform",
    max_noise_level: float = 0.9,
    min_noise_level: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create dataloaders for EVA denoising"""
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    logger.info(f"Creating EVA denoising dataloaders:")
    logger.info(f"  TASK: EVA-CLIP Denoising")
    logger.info(f"  INPUT: Noisy EVA embeddings [B, N, 4096]")
    logger.info(f"  CONDITIONING: Clean EVA embeddings [B, N, 4096]")
    logger.info(f"  TARGET: Clean EVA embeddings [B, N, 4096]")
    logger.info(f"  Noise schedule: {noise_schedule}")
    logger.info(f"  Noise range: [{min_noise_level}, {max_noise_level}]")
    
    # Create training dataset
    train_dataset = BLIP3oEVADenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        split="train",
        training_mode=training_mode,
        max_shards=max_shards,
        shuffle_shards=True,
        shuffle_within_shard=True,
        noise_schedule=noise_schedule,
        max_noise_level=max_noise_level,
        min_noise_level=min_noise_level,
        **kwargs
    )
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=eva_denoising_collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    
    # Create evaluation dataset (same data, different noise)
    eval_dataset = BLIP3oEVADenoisingDataset(
        chunked_embeddings_dir=chunked_embeddings_dir,
        split="eval",
        training_mode=training_mode,
        max_shards=max_shards,
        shuffle_shards=False,
        shuffle_within_shard=False,
        noise_schedule="uniform",  # Use uniform for consistent eval
        max_noise_level=0.7,  # Less noise for evaluation
        min_noise_level=0.3,
        **kwargs
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=min(num_workers, 1),
        collate_fn=eva_denoising_collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=min(num_workers, 1) > 0,
    )
    
    logger.info(f"EVA denoising dataloaders created successfully")
    
    return train_dataloader, eval_dataloader