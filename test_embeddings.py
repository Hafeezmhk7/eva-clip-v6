#!/usr/bin/env python3
"""
Test script to verify BLIP3-o compatible embeddings.
Place this file as: test_embeddings.py

This script verifies that the extracted embeddings are compatible with the BLIP3-o DiT architecture.
"""

import sys
import torch
import pickle
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_embeddings_file(embeddings_path):
    """Test if embeddings file is compatible with BLIP3-o DiT architecture."""
    
    print("🧪 Testing BLIP3-o Embeddings Compatibility")
    print("=" * 60)
    print(f"📁 File: {embeddings_path}")
    
    try:
        # Load embeddings
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        print("✅ File loaded successfully")
        
        # Check required keys
        print("\n🔑 Checking required keys...")
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings']
        for key in required_keys:
            if key in data:
                print(f"   ✅ {key}")
            else:
                print(f"   ❌ Missing: {key}")
                return False
        
        # Check optional keys
        optional_keys = ['captions', 'keys', 'total_samples', 'config']
        for key in optional_keys:
            if key in data:
                print(f"   ✅ {key} (optional)")
            else:
                print(f"   ⚠️  Missing: {key} (optional)")
        
        # Extract embeddings
        clip_embeddings = data['clip_blip3o_embeddings']
        eva_embeddings = data['eva_blip3o_embeddings']
        
        # Check types
        print(f"\n📐 Checking tensor types...")
        print(f"   CLIP type: {type(clip_embeddings)}")
        print(f"   EVA type: {type(eva_embeddings)}")
        
        # Convert to torch tensors if needed
        if isinstance(clip_embeddings, np.ndarray):
            clip_embeddings = torch.from_numpy(clip_embeddings)
        if isinstance(eva_embeddings, np.ndarray):
            eva_embeddings = torch.from_numpy(eva_embeddings)
        
        # Check shapes
        print(f"\n📏 Checking shapes...")
        print(f"   CLIP shape: {clip_embeddings.shape}")
        print(f"   EVA shape: {eva_embeddings.shape}")
        
        # Validate BLIP3-o requirements
        print(f"\n🎯 Validating BLIP3-o requirements...")
        
        # Check dimensions
        if len(clip_embeddings.shape) != 3:
            print(f"   ❌ CLIP should be 3D, got {len(clip_embeddings.shape)}D")
            return False
        print(f"   ✅ CLIP is 3D tensor")
        
        if len(eva_embeddings.shape) != 3:
            print(f"   ❌ EVA should be 3D, got {len(eva_embeddings.shape)}D")
            return False
        print(f"   ✅ EVA is 3D tensor")
        
        # Check token count (64 tokens for 8x8 grid)
        if clip_embeddings.shape[1] != 64:
            print(f"   ❌ CLIP tokens should be 64, got {clip_embeddings.shape[1]}")
            return False
        print(f"   ✅ CLIP has 64 tokens")
        
        if eva_embeddings.shape[1] != 64:
            print(f"   ❌ EVA tokens should be 64, got {eva_embeddings.shape[1]}")
            return False
        print(f"   ✅ EVA has 64 tokens")
        
        # Check feature dimensions
        if clip_embeddings.shape[2] != 1024:
            print(f"   ❌ CLIP dim should be 1024 (ViT-L/14), got {clip_embeddings.shape[2]}")
            return False
        print(f"   ✅ CLIP dimension: 1024 (ViT-L/14)")
        
        if eva_embeddings.shape[2] != 4096:
            print(f"   ❌ EVA dim should be 4096 (EVA-CLIP-8B), got {eva_embeddings.shape[2]}")
            return False
        print(f"   ✅ EVA dimension: 4096 (EVA-CLIP-8B)")
        
        # Check batch consistency
        if clip_embeddings.shape[0] != eva_embeddings.shape[0]:
            print(f"   ❌ Batch size mismatch: CLIP {clip_embeddings.shape[0]} vs EVA {eva_embeddings.shape[0]}")
            return False
        print(f"   ✅ Batch consistency: {clip_embeddings.shape[0]} samples")
        
        # Check data types
        print(f"\n🔢 Checking data types...")
        print(f"   CLIP dtype: {clip_embeddings.dtype}")
        print(f"   EVA dtype: {eva_embeddings.dtype}")
        
        # Check for NaN or Inf values
        print(f"\n🧮 Checking for invalid values...")
        clip_nan = torch.isnan(clip_embeddings).any()
        clip_inf = torch.isinf(clip_embeddings).any()
        eva_nan = torch.isnan(eva_embeddings).any()
        eva_inf = torch.isinf(eva_embeddings).any()
        
        if clip_nan or clip_inf:
            print(f"   ❌ CLIP contains NaN: {clip_nan}, Inf: {clip_inf}")
            return False
        print(f"   ✅ CLIP values are valid")
        
        if eva_nan or eva_inf:
            print(f"   ❌ EVA contains NaN: {eva_nan}, Inf: {eva_inf}")
            return False
        print(f"   ✅ EVA values are valid")
        
        # Statistical summary
        print(f"\n📊 Statistical summary...")
        print(f"   CLIP - Mean: {clip_embeddings.mean():.4f}, Std: {clip_embeddings.std():.4f}")
        print(f"   CLIP - Min: {clip_embeddings.min():.4f}, Max: {clip_embeddings.max():.4f}")
        print(f"   EVA  - Mean: {eva_embeddings.mean():.4f}, Std: {eva_embeddings.std():.4f}")
        print(f"   EVA  - Min: {eva_embeddings.min():.4f}, Max: {eva_embeddings.max():.4f}")
        
        # Check embedding norms
        clip_norms = torch.norm(clip_embeddings, dim=-1).mean()
        eva_norms = torch.norm(eva_embeddings, dim=-1).mean()
        print(f"   CLIP avg norm: {clip_norms:.4f}")
        print(f"   EVA avg norm: {eva_norms:.4f}")
        
        # Memory usage
        clip_memory = clip_embeddings.numel() * clip_embeddings.element_size() / (1024 * 1024)
        eva_memory = eva_embeddings.numel() * eva_embeddings.element_size() / (1024 * 1024)
        total_memory = clip_memory + eva_memory
        
        print(f"\n💾 Memory usage...")
        print(f"   CLIP: {clip_memory:.1f} MB")
        print(f"   EVA: {eva_memory:.1f} MB")
        print(f"   Total: {total_memory:.1f} MB")
        
        # File size
        file_size = Path(embeddings_path).stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size:.1f} MB")
        
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"🎉 Embeddings are compatible with BLIP3-o DiT architecture!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading(embeddings_path):
    """Test loading embeddings with BLIP3-o dataset class."""
    
    print(f"\n🗂️ Testing Dataset Loading...")
    print("=" * 40)
    
    try:
        from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset, test_blip3o_dataset
        
        # Test dataset creation
        print("📦 Creating BLIP3oEmbeddingDataset...")
        dataset = BLIP3oEmbeddingDataset(
            embeddings_path=embeddings_path,
            subset_size=10,  # Small subset for testing
            normalize_embeddings=True,
            split="all"
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test single item
        print("🔍 Testing single item access...")
        item = dataset[0]
        
        print(f"   EVA embeddings shape: {item['eva_embeddings'].shape}")
        print(f"   CLIP embeddings shape: {item['clip_embeddings'].shape}")
        print(f"   Caption: {item['caption'][:50]}...")
        print(f"   Key: {item['key']}")
        
        # Test batch loading
        print("📦 Testing batch loading...")
        from src.modules.datasets.blip3o_dataset import create_blip3o_dataloader
        
        dataloader = create_blip3o_dataloader(
            embeddings_path=embeddings_path,
            batch_size=4,
            subset_size=10,
            num_workers=0,
            split="all"
        )
        
        batch = next(iter(dataloader))
        print(f"   Batch EVA shape: {batch['eva_embeddings'].shape}")
        print(f"   Batch CLIP shape: {batch['clip_embeddings'].shape}")
        print(f"   Batch captions: {len(batch['captions'])}")
        
        print(f"✅ Dataset loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    
    # Check for embeddings file
    embeddings_path = "embeddings/blip3o_grid_embeddings.pkl"
    
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings file not found: {embeddings_path}")
        print(f"Please run: python src/modules/extract_embeddings_g.py")
        return 1
    
    # Test embeddings file
    if not test_embeddings_file(embeddings_path):
        print(f"\n❌ Embeddings compatibility test failed!")
        return 1
    
    # Test dataset loading
    if not test_dataset_loading(embeddings_path):
        print(f"\n❌ Dataset loading test failed!")
        return 1
    
    print(f"\n" + "=" * 60)
    print(f"🎉 ALL TESTS PASSED!")
    print(f"✅ Embeddings are ready for BLIP3-o DiT training!")
    print(f"")
    print(f"🚀 Next steps:")
    print(f"1. Start training:")
    print(f"   python train_blip3o_dit.py --embeddings_path {embeddings_path} --output_dir ./checkpoints/blip3o-dit")
    print(f"2. Or use job script:")
    print(f"   sbatch job_scripts/train_flow_match_en.job")
    print(f"=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)