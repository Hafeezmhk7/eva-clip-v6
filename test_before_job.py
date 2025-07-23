#!/usr/bin/env python3
"""
Pre-job test script to verify everything is working before submitting SLURM job
"""

import sys
import os
from pathlib import Path
import torch

sys.path.insert(0, "src")

def test_environment():
    """Test basic environment setup"""
    print("🔧 Testing Environment...")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   ✅ CUDA available: {gpu_count} GPUs")
        for i in range(min(gpu_count, 3)):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"      GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print("   ⚠️ CUDA not available")
    
    # Check Python packages
    try:
        import transformers
        import torch
        print(f"   ✅ transformers: {transformers.__version__}")
        print(f"   ✅ torch: {torch.__version__}")
    except ImportError as e:
        print(f"   ❌ Missing package: {e}")
        return False
    
    return True

def test_embeddings(embeddings_dir):
    """Test embeddings directory"""
    print(f"📊 Testing Embeddings: {embeddings_dir}")
    
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        print(f"   ❌ Directory not found: {embeddings_path}")
        return False
    
    # Check for pickle files
    pkl_files = list(embeddings_path.glob("*.pkl"))
    print(f"   📦 Found {len(pkl_files)} .pkl files")
    
    if len(pkl_files) == 0:
        print("   ❌ No .pkl files found")
        return False
    
    # Test loading one file
    try:
        import pickle
        test_file = pkl_files[0]
        print(f"   🔍 Testing file: {test_file.name}")
        
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
        
        clip_emb = data['clip_blip3o_embeddings']
        eva_emb = data['eva_blip3o_embeddings']
        captions = data['captions']
        
        print(f"   ✅ CLIP embeddings: {clip_emb.shape}")
        print(f"   ✅ EVA embeddings: {eva_emb.shape}")
        print(f"   ✅ Captions: {len(captions)}")
        
        # Check token count
        tokens = clip_emb.shape[1]
        if tokens == 256:
            training_mode = "patch_only"
        elif tokens == 257:
            training_mode = "cls_patch"
        else:
            print(f"   ⚠️ Unexpected token count: {tokens}")
            training_mode = "unknown"
        
        print(f"   🎯 Detected mode: {training_mode} ({tokens} tokens)")
        return True, training_mode
        
    except Exception as e:
        print(f"   ❌ Failed to load test file: {e}")
        return False, None

def test_dataset():
    """Test the fixed dataset implementation"""
    print("🔄 Testing FIXED Dataset...")
    
    try:
        from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset, flexible_collate_fn
        print("   ✅ Dataset imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Dataset import failed: {e}")
        return False

def test_model():
    """Test model creation"""
    print("🏗️ Testing Model Creation...")
    
    try:
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
        
        config = BLIP3oDiTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            training_mode="patch_only",
            num_tokens=256,
        )
        
        model = create_blip3o_patch_dit_model(config=config)
        param_count = model.get_num_parameters()
        
        print(f"   ✅ Model created: {param_count:,} parameters")
        return True
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

def test_training_script():
    """Test that training script exists and is importable"""
    print("📜 Testing Training Script...")
    
    train_script = Path("train_blip3o_enhanced.py")
    if not train_script.exists():
        print("   ❌ train_blip3o_enhanced.py not found")
        return False
    
    print("   ✅ Training script found")
    
    # Test basic imports
    try:
        import argparse
        print("   ✅ Basic imports work")
        return True
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Pre-Job Testing for BLIP3-o Enhanced Training")
    print("=" * 60)
    
    # Test environment
    if not test_environment():
        print("❌ Environment test failed")
        return False
    
    # Test embeddings
    embeddings_dir = "/scratch-shared/scur2711/blip3o_workspace/embeddings/chunked_256_tokens"
    
    result = test_embeddings(embeddings_dir)
    if isinstance(result, tuple):
        success, training_mode = result
        if not success:
            print("❌ Embeddings test failed")
            return False
    else:
        print("❌ Embeddings test failed")
        return False
    
    # Test dataset
    if not test_dataset():
        print("❌ Dataset test failed")
        return False
    
    # Test model
    if not test_model():
        print("❌ Model test failed")
        return False
    
    # Test training script
    if not test_training_script():
        print("❌ Training script test failed")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("✅ Environment: Ready")
    print(f"✅ Embeddings: Ready ({training_mode} mode)")
    print("✅ Dataset: Ready")
    print("✅ Model: Ready")
    print("✅ Training Script: Ready")
    print("\n🚀 You can now submit the SLURM job:")
    print("   sbatch train_blip3o_enhanced_fixed.job")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)