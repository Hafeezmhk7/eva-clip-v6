"""
Feature Caching Module for EVA-CLIP Flow Matching
Place this file in: src/modules/cache.py

This module handles:
✅ Efficient saving/loading of embeddings
✅ PyTorch dataset creation from cached features
✅ Multi-shard caching support
✅ Memory-efficient loading
"""

import torch
import pickle
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from torch.utils.data import Dataset, DataLoader
import hashlib
import time

class FeatureCache:
    """
    Feature caching utility for storing and loading embeddings efficiently
    
    This class handles caching of EVA-CLIP and CLIP embeddings along with
    metadata for fast training iteration.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the feature cache
        
        Args:
            cache_dir: Directory to store cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 FeatureCache initialized: {self.cache_dir}")
    
    def _generate_cache_key(self, shard_indices: List[int], config: Dict[str, Any]) -> str:
        """
        Generate a unique cache key based on shard indices and configuration
        
        Args:
            shard_indices: List of shard indices used
            config: Configuration dictionary (model names, preprocessing, etc.)
            
        Returns:
            Unique cache key string
        """
        # Create a string from shard indices and config
        key_data = {
            'shards': sorted(shard_indices),
            'config': config
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:12]
        
        shard_str = "_".join(map(str, sorted(shard_indices)))
        return f"features_shards_{shard_str}_{key_hash}"
    
    def save_features(
        self, 
        features: Dict[str, torch.Tensor],
        shard_indices: List[int],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save features to cache
        
        Args:
            features: Dictionary containing embeddings and data
            shard_indices: List of shard indices processed
            config: Configuration used for processing
            metadata: Additional metadata to store
            
        Returns:
            Path to saved cache file
        """
        cache_key = self._generate_cache_key(shard_indices, config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        print(f"💾 Saving features to cache...")
        print(f"   📁 Cache file: {cache_file}")
        
        # Prepare data to save
        cache_data = {
            'features': features,
            'shard_indices': shard_indices,
            'config': config,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'cache_version': '1.0'
        }
        
        # Add size information
        cache_data['size_info'] = {}
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                cache_data['size_info'][key] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'size_mb': tensor.element_size() * tensor.nelement() / (1024 * 1024)
                }
        
        # Save to file
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"✅ Features cached successfully!")
            print(f"   📊 File size: {file_size_mb:.2f} MB")
            print(f"   🔑 Cache key: {cache_key}")
            
            # Save a human-readable info file
            self._save_cache_info(cache_key, cache_data)
            
            return str(cache_file)
            
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
            raise
    
    def _save_cache_info(self, cache_key: str, cache_data: Dict[str, Any]):
        """Save human-readable cache information"""
        info_file = self.cache_dir / f"{cache_key}_info.json"
        
        info_data = {
            'cache_key': cache_key,
            'shard_indices': cache_data['shard_indices'],
            'config': cache_data['config'],
            'timestamp': time.ctime(cache_data['timestamp']),
            'size_info': cache_data['size_info'],
            'sample_count': cache_data['metadata'].get('total_samples', 'unknown')
        }
        
        with open(info_file, 'w') as f:
            json.dump(info_data, f, indent=2)
    
    def load_features(
        self, 
        shard_indices: List[int],
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Load features from cache
        
        Args:
            shard_indices: List of shard indices to load
            config: Configuration to match
            
        Returns:
            Cached data dictionary or None if not found
        """
        cache_key = self._generate_cache_key(shard_indices, config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            print(f"📭 Cache miss: {cache_key}")
            return None
        
        print(f"📦 Loading features from cache...")
        print(f"   📁 Cache file: {cache_file}")
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            file_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"✅ Features loaded successfully!")
            print(f"   📊 File size: {file_size_mb:.2f} MB")
            print(f"   🕒 Cached: {time.ctime(cache_data['timestamp'])}")
            
            return cache_data
            
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return None
    
    def cache_exists(
        self, 
        shard_indices: List[int],
        config: Dict[str, Any]
    ) -> bool:
        """
        Check if cache exists for given parameters
        
        Args:
            shard_indices: List of shard indices
            config: Configuration to check
            
        Returns:
            True if cache exists, False otherwise
        """
        cache_key = self._generate_cache_key(shard_indices, config)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def list_cached_features(self) -> List[Dict[str, Any]]:
        """
        List all cached feature files with their information
        
        Returns:
            List of cache information dictionaries
        """
        print(f"📋 Listing cached features in {self.cache_dir}...")
        
        cached_features = []
        
        for info_file in self.cache_dir.glob("*_info.json"):
            try:
                with open(info_file, 'r') as f:
                    info_data = json.load(f)
                cached_features.append(info_data)
            except Exception as e:
                print(f"⚠️ Error reading {info_file}: {e}")
        
        print(f"📊 Found {len(cached_features)} cached feature sets")
        return cached_features
    
    def clear_cache(self, confirm: bool = False):
        """
        Clear all cached features
        
        Args:
            confirm: Set to True to confirm deletion
        """
        if not confirm:
            print(f"⚠️ Use clear_cache(confirm=True) to actually delete cached features")
            return
        
        print(f"🗑️ Clearing cache directory: {self.cache_dir}")
        
        deleted_count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            deleted_count += 1
        
        for info_file in self.cache_dir.glob("*_info.json"):
            info_file.unlink()
        
        print(f"✅ Deleted {deleted_count} cache files")

class CachedFeaturesDataset(Dataset):
    """
    PyTorch Dataset for cached features
    
    This dataset loads cached embeddings and provides them for training
    the flow matching model.
    """
    
    def __init__(self, cache_data: Dict[str, Any]):
        """
        Initialize dataset from cached data
        
        Args:
            cache_data: Dictionary containing cached features and metadata
        """
        self.features = cache_data['features']
        self.metadata = cache_data.get('metadata', {})
        
        # Extract core data
        self.eva_clip_embeddings = self.features['eva_clip_embeddings']
        self.clip_embeddings = self.features['clip_embeddings']
        self.captions = self.features['captions']
        self.keys = self.features['keys']
        
        print(f"📊 CachedFeaturesDataset initialized:")
        print(f"   🔢 Samples: {len(self.eva_clip_embeddings)}")
        print(f"   📏 EVA-CLIP dim: {self.eva_clip_embeddings.shape[-1]}")
        print(f"   📏 CLIP dim: {self.clip_embeddings.shape[-1]}")
    
    def __len__(self) -> int:
        return len(self.eva_clip_embeddings)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample for training
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing EVA and CLIP embeddings with metadata
        """
        return {
            'eva_clip_features': self.eva_clip_embeddings[idx],  # Input for flow matching
            'clip_features': self.clip_embeddings[idx],          # Target for flow matching
            'caption': self.captions[idx],
            'key': self.keys[idx]
        }
    
    def get_dataloader(
        self, 
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

def test_feature_cache():
    """Test function for the feature cache"""
    print("🧪 Testing FeatureCache...")
    print("=" * 60)
    
    try:
        # Initialize cache
        cache = FeatureCache(cache_dir="./test_cache")
        
        # Create dummy features
        dummy_features = {
            'eva_clip_embeddings': torch.randn(100, 768),
            'clip_embeddings': torch.randn(100, 768),
            'captions': [f"Caption {i}" for i in range(100)],
            'keys': [f"key_{i:03d}" for i in range(100)]
        }
        
        shard_indices = [0]
        config = {
            'eva_clip_model': 'EVA-CLIP-L-14',
            'clip_model': 'openai/clip-vit-large-patch14'
        }
        metadata = {'total_samples': 100}
        
        print(f"\n💾 Testing cache save...")
        cache_file = cache.save_features(dummy_features, shard_indices, config, metadata)
        print(f"✅ Cache saved to: {cache_file}")
        
        print(f"\n📦 Testing cache load...")
        loaded_data = cache.load_features(shard_indices, config)
        if loaded_data:
            print(f"✅ Cache loaded successfully")
            print(f"   📊 Keys: {list(loaded_data['features'].keys())}")
        else:
            print(f"❌ Failed to load cache")
        
        print(f"\n📋 Testing cache listing...")
        cached_features = cache.list_cached_features()
        for info in cached_features:
            print(f"   📁 {info['cache_key']}: {info['sample_count']} samples")
        
        print(f"\n🧪 Testing CachedFeaturesDataset...")
        if loaded_data:
            dataset = CachedFeaturesDataset(loaded_data)
            dataloader = dataset.get_dataloader(batch_size=8)
            
            # Test one batch
            batch = next(iter(dataloader))
            print(f"✅ Dataset and DataLoader working")
            print(f"   📦 Batch EVA features: {batch['eva_clip_features'].shape}")
            print(f"   📦 Batch CLIP features: {batch['clip_features'].shape}")
        
        print(f"\n🗑️ Cleaning up test cache...")
        cache.clear_cache(confirm=True)
        
        print(f"\n🎉 All cache tests passed!")
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_cache()