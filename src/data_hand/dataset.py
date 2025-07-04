"""
Core Dataset Module for BLIP3o WebDataset Loading
Place this file in: src/data/dataset.py

This module handles:
✅ Create WebDataset from TAR (no extraction)
✅ Create DataLoader from WebDataset  
✅ Load images through DataLoader

Next phase will add: EVA-CLIP and CLIP embedding computation
"""

import torch
from torch.utils.data import DataLoader
import webdataset as wds
from PIL import Image
import io
import os
from typing import Optional, List, Dict, Any

class BLIP3oWebDataset:
    """
    Core WebDataset loader for BLIP3o TAR files
    
    This class creates a WebDataset from TAR files and provides
    a PyTorch DataLoader for training/inference.
    """
    
    def __init__(
        self, 
        tar_paths: List[str], 
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        """
        Initialize BLIP3o WebDataset
        
        Args:
            tar_paths: List of paths to TAR files (e.g., ["./data/00000.tar"])
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for DataLoader
        """
        self.tar_paths = tar_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Verify TAR files exist
        self._verify_tar_files()
        
        # Create WebDataset
        self.dataset = self._create_webdataset()
        
        # Create DataLoader
        self.dataloader = self._create_dataloader()
        
        print(f"✅ BLIP3oWebDataset initialized")
        print(f"   📁 TAR files: {len(self.tar_paths)}")
        print(f"   🔀 Shuffle: {self.shuffle}")
        print(f"   📦 Batch size: {self.batch_size}")
        print(f"   👥 Workers: {self.num_workers}")
    
    def _verify_tar_files(self):
        """Verify that all TAR files exist"""
        missing_files = []
        for tar_path in self.tar_paths:
            if not os.path.exists(tar_path):
                missing_files.append(tar_path)
        
        if missing_files:
            print(f"❌ Missing TAR files:")
            for file_path in missing_files:
                print(f"   • {file_path}")
            raise FileNotFoundError(f"Missing {len(missing_files)} TAR files")
        
        # Show file sizes
        total_size_mb = 0
        for tar_path in self.tar_paths:
            size_mb = os.path.getsize(tar_path) / (1024 * 1024)
            total_size_mb += size_mb
            print(f"📁 {tar_path}: {size_mb:.1f} MB")
        
        print(f"📊 Total dataset size: {total_size_mb:.1f} MB")
    
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
            for ext in ['jpg', 'jpeg', 'png']:
                if ext in sample:
                    image_data = sample[ext]
                    image_ext = ext
                    break
            
            if image_data is None:
                print(f"⚠️ No image found in sample {sample.get('__key__', 'unknown')}")
                return None
            
            # Decode image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Get caption
            caption = ""
            if 'txt' in sample:
                caption = sample['txt'].decode('utf-8').strip()
            
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
            print(f"❌ Error decoding sample {sample.get('__key__', 'unknown')}: {e}")
            return None
    
    def _create_webdataset(self) -> wds.WebDataset:
        """
        Create WebDataset from TAR files
        
        Returns:
            Configured WebDataset
        """
        print(f"🔄 Creating WebDataset from {len(self.tar_paths)} TAR files...")
        
        # Create WebDataset
        if self.shuffle:
            # Shuffle shards and samples
            dataset = (wds.WebDataset(self.tar_paths, shardshuffle=True)
                      .shuffle(1000)  # Shuffle buffer size
                      .map(self._decode_sample)
                      .select(lambda x: x is not None))  # Filter out failed samples
        else:
            # No shuffling
            dataset = (wds.WebDataset(self.tar_paths, shardshuffle=False)
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

def test_dataset():
    """Test function to verify the dataset works"""
    print("🧪 Testing BLIP3oWebDataset...")
    print("=" * 60)
    
    # Test with your downloaded shard
    tar_paths = ["./data/00000.tar"]
    
    try:
        # Create dataset
        dataset = BLIP3oWebDataset(
            tar_paths=tar_paths,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )
        
        # Test DataLoader
        print(f"\n🔄 Testing DataLoader...")
        dataloader = dataset.get_dataloader()
        
        # Load one batch
        batch = next(iter(dataloader))
        
        print(f"✅ Batch loaded successfully!")
        print(f"   📦 Batch size: {len(batch['image'])}")
        print(f"   🖼️ Image type: {type(batch['image'][0])}")
        print(f"   🖼️ Image list type: {type(batch['image'])}")
        print(f"   💬 Caption type: {type(batch['caption'][0])}")
        print(f"   💬 Caption list type: {type(batch['caption'])}")
        
        # Show sample details
        print(f"\n📋 Sample details:")
        for i in range(min(2, len(batch['image']))):  # Show first 2 samples
            img = batch['image'][i]
            caption = batch['caption'][i]
            key = batch['key'][i]
            size = batch['image_size'][i]
            
            print(f"   Sample {i+1}:")
            print(f"     🔑 Key: {key}")
            print(f"     📏 Size: {size}")
            print(f"     🖼️ PIL Image: {img.size} pixels")
            print(f"     💬 Caption: {caption[:50]}{'...' if len(caption) > 50 else ''}")
        
        # Test sampling function
        print(f"\n🔍 Testing sample function...")
        samples = dataset.sample_data(num_samples=3)
        
        print(f"✅ Sampled {len(samples)} examples")
        for i, sample in enumerate(samples):
            print(f"   Example {i+1}: {sample['key']} - {sample['caption'][:30]}...")
        
        print(f"\n🎉 All tests passed!")
        print(f"✅ WebDataset is working correctly")
        print(f"📊 Ready to add embedding computation in next phase")
        print(f"\n💡 Learning note: Fixed PIL Image collate issue!")
        print(f"   PyTorch DataLoader needs custom collate_fn for PIL Images")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()