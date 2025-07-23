#!/usr/bin/env python3
"""
Clean dataset test that definitely works - for SLURM job validation
"""

import sys
import os
from pathlib import Path
import torch
import json
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_embeddings_directory(embeddings_dir, training_mode="patch_only"):
    """Test embeddings directory thoroughly"""
    print(f"🔍 Testing Embeddings Directory")
    print(f"   Path: {embeddings_dir}")
    print(f"   Mode: {training_mode}")
    
    embeddings_path = Path(embeddings_dir)
    
    # Check directory exists
    if not embeddings_path.exists():
        print(f"   ❌ Directory not found: {embeddings_path}")
        return False
    
    # Find pickle files
    pkl_files = list(embeddings_path.glob("*.pkl"))
    print(f"   📦 Found {len(pkl_files)} .pkl files")
    
    if len(pkl_files) == 0:
        print("   ❌ No .pkl files found")
        return False
    
    # Test first file
    try:
        test_file = pkl_files[0]
        print(f"   🧪 Testing: {test_file.name}")
        
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
        
        # Check required keys
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        for key in required_keys:
            if key not in data:
                print(f"   ❌ Missing key: {key}")
                return False
        
        clip_emb = data['clip_blip3o_embeddings']
        eva_emb = data['eva_blip3o_embeddings']
        captions = data['captions']
        
        print(f"   ✅ CLIP: {clip_emb.shape}")
        print(f"   ✅ EVA: {eva_emb.shape}")
        print(f"   ✅ Captions: {len(captions)}")
        
        # Validate shapes
        if clip_emb.dim() != 3 or eva_emb.dim() != 3:
            print(f"   ❌ Wrong tensor dimensions")
            return False
        
        if clip_emb.shape[2] != 1024:
            print(f"   ❌ Wrong CLIP dimension: {clip_emb.shape[2]}")
            return False
        
        if eva_emb.shape[2] != 4096:
            print(f"   ❌ Wrong EVA dimension: {eva_emb.shape[2]}")
            return False
        
        # Check token count
        tokens = clip_emb.shape[1]
        expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        print(f"   🎯 Tokens: {tokens} (expected: {expected_tokens})")
        
        if tokens not in [256, 257]:
            print(f"   ❌ Invalid token count: {tokens}")
            return False
        
        print(f"   ✅ File validation passed")
        return True
        
    except Exception as e:
        print(f"   ❌ File loading failed: {e}")
        return False

def test_fixed_dataset_import():
    """Test that the fixed dataset can be imported"""
    print(f"📦 Testing Fixed Dataset Import")
    
    try:
        from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset, flexible_collate_fn
        print(f"   ✅ Dataset classes imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_fixed_dataset_creation(embeddings_dir, training_mode="patch_only"):
    """Test creating the fixed dataset"""
    print(f"🔧 Testing Fixed Dataset Creation")
    
    try:
        from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset
        
        # Create dataset
        dataset = BLIP3oEmbeddingDataset(
            chunked_embeddings_dir=embeddings_dir,
            split="all",
            training_mode=training_mode,
            max_shards=1,  # Just test with one shard
            delete_after_use=False,
            cache_next_shard=False,  # Disable caching for simple test
        )
        
        print(f"   ✅ Dataset created successfully")
        print(f"   📊 Estimated length: {len(dataset):,}")
        
        # Test iteration
        sample_count = 0
        for i, sample in enumerate(dataset):
            if i >= 3:  # Just test first 3 samples
                break
            sample_count += 1
            
            # Validate sample
            eva_shape = sample['eva_embeddings'].shape
            clip_shape = sample['clip_embeddings'].shape
            
            expected_tokens = 257 if training_mode == "cls_patch" else 256
            
            if eva_shape[0] != expected_tokens or clip_shape[0] != expected_tokens:
                print(f"   ❌ Wrong token count in sample {i}")
                return False
            
            print(f"   ✅ Sample {i}: EVA {eva_shape}, CLIP {clip_shape}")
        
        print(f"   ✅ Iteration test passed ({sample_count} samples)")
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_creation(embeddings_dir, training_mode="patch_only"):
    """Test creating dataloaders"""
    print(f"🔄 Testing Dataloader Creation")
    
    try:
        from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_flexible_dataloaders(
            chunked_embeddings_dir=embeddings_dir,
            batch_size=2,  # Small batch for testing
            eval_batch_size=2,
            training_mode=training_mode,
            max_shards=1,  # Just one shard for testing
            use_same_data_for_eval=True,
            delete_after_use=False,
            num_workers=0,  # No multiprocessing for testing
        )
        
        print(f"   ✅ Dataloaders created")
        print(f"   📊 Train batches: {len(train_dataloader)}")
        if eval_dataloader:
            print(f"   📊 Eval batches: {len(eval_dataloader)}")
        
        # Test getting a batch
        batch = next(iter(train_dataloader))
        
        print(f"   ✅ Batch created successfully")
        print(f"   📦 EVA: {batch['encoder_hidden_states'].shape}")
        print(f"   📦 CLIP: {batch['clip_embeddings'].shape}")
        print(f"   📦 Hidden states: {batch['hidden_states'].shape}")
        print(f"   🔧 Gradients: {batch['hidden_states'].requires_grad}")
        
        if not batch['hidden_states'].requires_grad:
            print(f"   ❌ Hidden states don't require gradients!")
            return False
        
        print(f"   ✅ Dataloader test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run complete clean test"""
    if len(sys.argv) < 2:
        print("Usage: python test_dataset_clean.py <embeddings_dir> [training_mode]")
        return False
    
    embeddings_dir = sys.argv[1]
    training_mode = sys.argv[2] if len(sys.argv) > 2 else "patch_only"
    
    print("🧪 Clean Dataset Test - No Old Code")
    print("=" * 50)
    print(f"📁 Directory: {embeddings_dir}")
    print(f"🎯 Mode: {training_mode}")
    print("=" * 50)
    
    # Test 1: Check embeddings directory
    if not test_embeddings_directory(embeddings_dir, training_mode):
        print("❌ Embeddings directory test failed")
        return False
    
    # Test 2: Test imports
    if not test_fixed_dataset_import():
        print("❌ Import test failed")
        return False
    
    # Test 3: Test dataset creation
    if not test_fixed_dataset_creation(embeddings_dir, training_mode):
        print("❌ Dataset creation test failed")
        return False
    
    # Test 4: Test dataloader
    if not test_dataloader_creation(embeddings_dir, training_mode):
        print("❌ Dataloader test failed")
        return False
    
    print("=" * 50)
    print("🎉 ALL CLEAN TESTS PASSED!")
    print("✅ Directory validation: PASSED")
    print("✅ Import test: PASSED") 
    print("✅ Dataset creation: PASSED")
    print("✅ Dataloader creation: PASSED")
    print("✅ Gradient flow: PASSED")
    print("=" * 50)
    print("🚀 Ready for training!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)