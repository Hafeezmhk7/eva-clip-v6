"""
UPDATED Core Dataset Module for BLIP3o WebDataset Loading with Multi-Shard Support
Place this file in: src/data_hand/dataset.py

This module handles:
1. Create WebDataset from multiple TAR files (no extraction)
2. Support for temp directory storage
3. Create DataLoader from WebDataset  
4. Load images through DataLoader

UPDATES:
- Multi-shard support (multiple tar files)
- Temp directory integration
- Better file discovery and validation
- Improved worker/shard handling
"""

import torch
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
import io
import os
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import glob

class BLIP3oWebDataset:
    """
    UPDATED Core WebDataset loader for BLIP3o TAR files with multi-shard support
    
    This class creates a WebDataset from multiple TAR files and provides
    a PyTorch DataLoader for training/inference. Now supports temp directory storage.
    """
    
    def __init__(
        self, 
        tar_paths: Optional[List[str]] = None,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        auto_discover: bool = True,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize BLIP3o WebDataset with multi-shard support
        
        Args:
            tar_paths: List of paths to TAR files. If None, auto-discovers files.
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for DataLoader
            auto_discover: Whether to auto-discover tar files if tar_paths is None
            temp_dir: Temp directory to search for files (auto-detected if None)
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.auto_discover = auto_discover
        self.temp_dir = temp_dir
        
        # Discover or validate tar files
        if tar_paths is None and auto_discover:
            self.tar_paths = self._discover_tar_files()
        elif tar_paths is None:
            raise ValueError("tar_paths is None and auto_discover is False")
        else:
            self.tar_paths = tar_paths
        
        # FIX: Adjust num_workers based on number of shards
        # WebDataset requires num_workers <= number of shards
        max_workers = len(self.tar_paths)
        if num_workers > max_workers:
            print(f"🔧 Reducing num_workers from {num_workers} to {max_workers} (max shards)")
            self.num_workers = max_workers
        else:
            self.num_workers = num_workers
        
        # For single shard, use 0 workers to avoid WebDataset issues
        if len(self.tar_paths) == 1:
            print(f"🔧 Single shard detected, using num_workers=0 for stability")
            self.num_workers = 0
        
        # Verify TAR files exist
        self._verify_tar_files()
        
        # Create WebDataset
        self.dataset = self._create_webdataset()
        
        # Create DataLoader
        self.dataloader = self._create_dataloader()
        
        print(f"✅ BLIP3oWebDataset initialized")
        print(f"    TAR files: {len(self.tar_paths)}")
        print(f"    Shuffle: {self.shuffle}")
        print(f"    Batch size: {self.batch_size}")
        print(f"    Workers: {self.num_workers}")
    
    def _get_temp_directory(self) -> Path:
        """Get the temp directory path for file discovery"""
        if self.temp_dir:
            return Path(self.temp_dir)
        
        # Auto-detect temp directory
        if "TMPDIR" in os.environ:
            return Path(os.environ["TMPDIR"])
        elif "SCRATCH_SHARED" in os.environ:
            return Path(os.environ["SCRATCH_SHARED"])
        elif "BLIP3O_TEMP_DIR" in os.environ:
            return Path(os.environ["BLIP3O_TEMP_DIR"])
        else:
            # Fallback to project data directory
            return Path(__file__).parent.parent.parent / "data"
    
    def _discover_tar_files(self) -> List[str]:
        """Auto-discover TAR files in temp and project directories"""
        discovered_files = []
        
        print("🔍 Auto-discovering TAR files...")
        
        # Search locations in order of preference
        search_locations = []
        
        # 1. Temp directory with blip3o_data subdirectory
        temp_dir = self._get_temp_directory()
        search_locations.append(temp_dir / "blip3o_data")
        search_locations.append(temp_dir / "data")
        search_locations.append(temp_dir)
        
        # 2. Project data directory
        project_root = Path(__file__).parent.parent.parent
        search_locations.append(project_root / "data")
        
        # 3. Current directory
        search_locations.append(Path(".") / "data")
        search_locations.append(Path("."))
        
        # Search each location
        for search_path in search_locations:
            if not search_path.exists():
                continue
                
            print(f"   Searching: {search_path}")
            
            # Look for tar files
            tar_files = list(search_path.glob("*.tar"))
            if tar_files:
                # Sort numerically
                tar_files.sort()
                discovered_files.extend([str(f) for f in tar_files])
                print(f"   ✅ Found {len(tar_files)} files in {search_path}")
                break  # Use first location with files
        
        # Also check for downloaded_shards.txt file
        for search_path in search_locations[:3]:  # Only check temp directories
            shard_list = search_path / "downloaded_shards.txt"
            if shard_list.exists():
                print(f"   📋 Found shard list: {shard_list}")
                try:
                    with open(shard_list, 'r') as f:
                        listed_files = [line.strip() for line in f if line.strip()]
                    
                    # Validate files exist
                    valid_files = [f for f in listed_files if Path(f).exists()]
                    if valid_files and not discovered_files:  # Use if no files found yet
                        discovered_files = valid_files
                        print(f"   ✅ Using {len(valid_files)} files from shard list")
                        break
                except Exception as e:
                    print(f"   ⚠️  Could not read shard list: {e}")
        
        if not discovered_files:
            raise FileNotFoundError(
                "No TAR files found in any search location. "
                "Please run download_data.py first or provide tar_paths explicitly."
            )
        
        print(f"🎯 Auto-discovered {len(discovered_files)} TAR files")
        return discovered_files
    
    def _verify_tar_files(self):
        """Verify that all TAR files exist and show information"""
        missing_files = []
        valid_files = []
        total_size_mb = 0
        
        print(f"🧪 Verifying {len(self.tar_paths)} TAR files...")
        
        for tar_path in self.tar_paths:
            if not os.path.exists(tar_path):
                missing_files.append(tar_path)
                print(f"   ❌ Missing: {Path(tar_path).name}")
            else:
                size_mb = os.path.getsize(tar_path) / (1024 * 1024)
                total_size_mb += size_mb
                valid_files.append(tar_path)
                print(f"   ✅ {Path(tar_path).name}: {size_mb:.1f} MB")
        
        if missing_files:
            print(f"❌ Missing {len(missing_files)} TAR files:")
            for file_path in missing_files:
                print(f"   • {file_path}")
            raise FileNotFoundError(f"Missing {len(missing_files)} TAR files")
        
        # Update tar_paths to only include valid files
        self.tar_paths = valid_files
        
        print(f"📊 Total dataset size: {total_size_mb:.1f} MB")
        print(f"📁 Using {len(valid_files)} valid TAR files")
        
        # Estimate total samples
        estimated_samples = int(total_size_mb * 400)  # Rough estimate: 400 samples per MB
        print(f"📊 Estimated samples: ~{estimated_samples:,}")
    
    def _decode_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decode a single sample from WebDataset
        
        Args:
            sample: Raw sample from WebDataset
            
        Returns:
            Decoded sample with image and caption, or None if decoding fails
        """
        try:
            # Get image data (try different extensions)
            image_data = None
            image_ext = None
            for ext in ['jpg', 'jpeg', 'png', 'webp']:
                if ext in sample:
                    image_data = sample[ext]
                    image_ext = ext
                    break
            
            if image_data is None:
                # Debug: show available keys
                available_keys = list(sample.keys())
                print(f"⚠️  No image found in sample {sample.get('__key__', 'unknown')}. Available keys: {available_keys}")
                return None
            
            # Decode image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Get caption (try different caption formats)
            caption = ""
            for caption_key in ['txt', 'caption', 'text']:
                if caption_key in sample:
                    caption_data = sample[caption_key]
                    if isinstance(caption_data, bytes):
                        caption = caption_data.decode('utf-8').strip()
                    else:
                        caption = str(caption_data).strip()
                    break
            
            # Get sample key
            key = sample.get('__key__', 'unknown')
            
            return {
                'image': image,
                'caption': caption,
                'key': key,
                'image_size': image.size,  # (width, height)
                'image_ext': image_ext
            }
            
        except Exception as e:
            print(f"⚠️  Error decoding sample {sample.get('__key__', 'unknown')}: {e}")
            return None
    
    def _create_webdataset(self) -> wds.WebDataset:
        """
        Create WebDataset from multiple TAR files
        
        Returns:
            Configured WebDataset
        """
        print(f"🔄 Creating WebDataset from {len(self.tar_paths)} TAR files...")
        
        # FIX: Add empty_check=False to prevent shard/worker mismatch error
        if self.shuffle:
            # Shuffle shards and samples
            dataset = (wds.WebDataset(self.tar_paths, shardshuffle=True, empty_check=False)
                      .shuffle(1000)  # Shuffle buffer size
                      .map(self._decode_sample)
                      .select(lambda x: x is not None))  # Filter out failed samples
        else:
            # No shuffling
            dataset = (wds.WebDataset(self.tar_paths, shardshuffle=False, empty_check=False)
                      .map(self._decode_sample)
                      .select(lambda x: x is not None))
        
        print(f"✅ WebDataset created successfully")
        return dataset
    
    def _custom_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to handle PIL Images and mixed data types
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Collated batch with proper handling of PIL Images
        """
        if not batch:
            return {}
        
        # Separate different types of data
        images = [item['image'] for item in batch]
        captions = [item['caption'] for item in batch]
        keys = [item['key'] for item in batch]
        image_sizes = [item['image_size'] for item in batch]
        image_exts = [item['image_ext'] for item in batch]
        
        # Return as lists (don't convert PIL Images to tensors yet)
        # This allows us to handle images in their native format
        return {
            'image': images,           # List of PIL Images
            'caption': captions,       # List of strings
            'key': keys,              # List of strings
            'image_size': image_sizes, # List of tuples
            'image_ext': image_exts   # List of strings
        }
    
    def _create_dataloader(self) -> DataLoader:
        """
        Create PyTorch DataLoader from WebDataset
        
        Returns:
            Configured DataLoader
        """
        print(f"🔄 Creating DataLoader...")
        
        # WebDataset implements IterableDataset, so we can use it directly
        # Use custom collate function to handle PIL Images
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._custom_collate_fn,  # Custom collate for PIL Images
            # Note: shuffle is handled by WebDataset, not DataLoader
        )
        
        print(f"✅ DataLoader created successfully")
        return dataloader
    
    def get_dataloader(self) -> DataLoader:
        """Get the PyTorch DataLoader"""
        return self.dataloader
    
    def sample_data(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Sample a few examples from the dataset for inspection
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of decoded samples
        """
        print(f"🔍 Sampling {num_samples} examples from dataset...")
        
        samples = []
        count = 0
        
        for batch in self.dataloader:
            batch_size = len(batch['image'])
            
            for i in range(batch_size):
                if count >= num_samples:
                    break
                
                sample = {
                    'image': batch['image'][i],
                    'caption': batch['caption'][i],
                    'key': batch['key'][i],
                    'image_size': batch['image_size'][i],
                    'image_ext': batch['image_ext'][i]
                }
                samples.append(sample)
                count += 1
            
            if count >= num_samples:
                break
        
        return samples


def create_multi_shard_dataset(
    tar_paths: Optional[List[str]] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    auto_discover: bool = True,
    temp_dir: Optional[str] = None,
) -> BLIP3oWebDataset:
    """
    Convenience function to create a multi-shard BLIP3o dataset
    
    Args:
        tar_paths: List of TAR file paths. If None, auto-discovers files.
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        auto_discover: Whether to auto-discover files
        temp_dir: Temp directory to search (auto-detected if None)
        
    Returns:
        BLIP3oWebDataset instance
    """
    return BLIP3oWebDataset(
        tar_paths=tar_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        auto_discover=auto_discover,
        temp_dir=temp_dir,
    )


def test_dataset():
    """Test function to verify the dataset works with multi-shard support"""
    print("🧪 Testing BLIP3oWebDataset with multi-shard support...")
    print("=" * 70)
    
    try:
        # Test with auto-discovery
        dataset = BLIP3oWebDataset(
            tar_paths=None,  # Auto-discover
            batch_size=4,
            shuffle=True,
            num_workers=2,  # Will be auto-adjusted
            auto_discover=True,
        )
        
        # Test DataLoader
        print(f"\n🧪 Testing DataLoader...")
        dataloader = dataset.get_dataloader()
        
        # Load one batch
        batch = next(iter(dataloader))
        
        print(f"✅ Batch loaded successfully!")
        print(f"    Batch size: {len(batch['image'])}")
        print(f"    Image type: {type(batch['image'][0])}")
        print(f"    Caption type: {type(batch['caption'][0])}")
        
        # Show sample details
        print(f"\n📋 Sample details:")
        for i in range(min(2, len(batch['image']))):  # Show first 2 samples
            img = batch['image'][i]
            caption = batch['caption'][i]
            key = batch['key'][i]
            size = batch['image_size'][i]
            
            print(f"   Sample {i+1}:")
            print(f"      Key: {key}")
            print(f"      Size: {size}")
            print(f"      PIL Image: {img.size} pixels")
            print(f"      Caption: {caption[:50]}{'...' if len(caption) > 50 else ''}")
        
        # Test sampling function
        print(f"\n🔍 Testing sample function...")
        samples = dataset.sample_data(num_samples=3)
        
        print(f"✅ Sampled {len(samples)} examples")
        for i, sample in enumerate(samples):
            print(f"   Example {i+1}: {sample['key']} - {sample['caption'][:30]}...")
        
        print(f"\n✅ All tests passed!")
        print(f"✅ Multi-shard WebDataset is working correctly")
        print(f"✅ Ready for 256-token embedding extraction")
        print(f"\n📝 Next steps:")
        print(f"   1. Run: python src/modules/extract_embeddings_g.py")
        print(f"   2. This will extract 256-token embeddings from all discovered shards")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔍 Troubleshooting:")
        print(f"   1. Make sure you've downloaded data: python src/data_hand/download_data.py")
        print(f"   2. Check temp directory has tar files")
        print(f"   3. Verify file permissions")

def test_specific_files(tar_paths: List[str]):
    """Test with specific TAR file paths"""
    print(f"🧪 Testing with specific files: {tar_paths}")
    
    try:
        dataset = BLIP3oWebDataset(
            tar_paths=tar_paths,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Safe for testing
        )
        
        dataloader = dataset.get_dataloader()
        batch = next(iter(dataloader))
        
        print(f"✅ Successfully loaded batch from {len(tar_paths)} files")
        print(f"   Batch size: {len(batch['image'])}")
        
    except Exception as e:
        print(f"❌ Test failed with specific files: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test with specific files provided as arguments
        tar_files = sys.argv[1:]
        test_specific_files(tar_files)
    else:
        # Test with auto-discovery
        test_dataset()