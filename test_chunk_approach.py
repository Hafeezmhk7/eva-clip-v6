#!/usr/bin/env python3
"""
Test script for chunked BLIP3-o approach
Tests extraction and training pipeline with chunked embeddings
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_chunked_extraction():
    """Test chunked extraction on a small dataset"""
    print("🧪 Testing chunked extraction approach...")
    
    # This would test with 2-3 TAR files instead of 30
    test_command = """
    python src/modules/extract_embeddings_chunked.py
    """
    
    print("Test command for chunked extraction:")
    print(test_command)
    print("\nThis will process each TAR file separately and save individual .pkl files")

def test_chunked_dataset():
    """Test chunked dataset loading"""
    print("\n🧪 Testing chunked dataset loading...")
    
    chunked_dir = input("Enter path to chunked embeddings directory: ").strip()
    
    if not chunked_dir:
        print("Skipping dataset test (no path provided)")
        return
    
    try:
        from src.modules.datasets.blip3o_chunked_dataset import test_chunked_dataset
        test_chunked_dataset(chunked_dir)
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")

def test_chunked_training():
    """Test chunked training approach"""
    print("\n🧪 Testing chunked training...")
    
    chunked_dir = input("Enter path to chunked embeddings directory: ").strip()
    
    if not chunked_dir:
        print("Skipping training test (no path provided)")
        return
    
    # Test training command
    test_command = f"""
    python train_blip3o_dit.py \\
      --chunked_embeddings_dir "{chunked_dir}" \\
      --output_dir ./test_checkpoints \\
      --num_epochs 1 \\
      --batch_size 8 \\
      --learning_rate 1e-4 \\
      --debug \\
      --no_wandb
    """
    
    print("Test command for chunked training:")
    print(test_command)
    print("\nThis will load chunks sequentially and train on each one")

def compare_approaches():
    """Compare single-file vs chunked approaches"""
    print("\n📊 Comparing Single-File vs Chunked Approaches")
    print("=" * 60)
    
    comparison = """
    SINGLE-FILE APPROACH (Original):
    ✅ Simple implementation
    ✅ Fast loading once in memory
    ❌ Requires large disk space (12GB+ files)
    ❌ Disk quota issues
    ❌ Memory usage scales with dataset size
    ❌ Limited to ~10 TAR files due to disk constraints
    
    CHUNKED APPROACH (New):
    ✅ Small individual files (~500MB each)
    ✅ No disk quota issues
    ✅ Scales to 30+ TAR files (~100k samples)
    ✅ Memory usage constant (one chunk at a time)
    ✅ Automatic cleanup saves disk space
    ❌ Slightly more complex implementation
    ❌ Small overhead from sequential loading
    
    RECOMMENDATION:
    Use CHUNKED approach for:
    - Large datasets (30+ TAR files)
    - Systems with disk quota limits
    - Scaling to 100k+ samples
    
    Use SINGLE-FILE approach for:
    - Small datasets (<10 TAR files)
    - Systems with unlimited disk space
    - Maximum training speed
    """
    
    print(comparison)

def show_usage_examples():
    """Show example usage for chunked approach"""
    print("\n📖 Usage Examples for Chunked Approach")
    print("=" * 50)
    
    examples = """
    1. DOWNLOAD MORE SHARDS (30 for ~100k samples):
    python src/data_hand/download_data.py --shards $(seq -s ' ' 0 29)
    
    2. EXTRACT CHUNKED EMBEDDINGS:
    python src/modules/extract_embeddings_chunked.py
    
    3. TRAIN WITH CHUNKED DATASET:
    python train_blip3o_dit.py \\
      --chunked_embeddings_dir /path/to/chunked_embeddings \\
      --output_dir ./checkpoints/blip3o-dit-100k \\
      --num_epochs 10 \\
      --batch_size 32 \\
      --learning_rate 1e-4
    
    4. JOB SCRIPT FOR CLUSTER:
    sbatch job_scripts/extract_chunked.job
    
    EXPECTED OUTPUT:
    - Multiple files: embeddings_shard_00000.pkl, embeddings_shard_00001.pkl, ...
    - Manifest: embeddings_manifest.json
    - Total samples: ~100,000
    - Individual file size: ~500MB each
    - Total size: ~15-20GB spread across many small files
    """
    
    print(examples)

def main():
    """Main test function"""
    print("🚀 BLIP3-o Chunked Approach Testing")
    print("=" * 40)
    
    print("\nThis script helps test the new chunked approach for BLIP3-o training")
    print("that solves disk quota issues and enables scaling to 100k+ samples.\n")
    
    while True:
        print("\nSelect a test:")
        print("1. Compare approaches (single-file vs chunked)")
        print("2. Show usage examples")
        print("3. Test chunked extraction (requires TAR files)")
        print("4. Test chunked dataset loading")
        print("5. Test chunked training")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            compare_approaches()
        elif choice == "2":
            show_usage_examples()
        elif choice == "3":
            test_chunked_extraction()
        elif choice == "4":
            test_chunked_dataset()
        elif choice == "5":
            test_chunked_training()
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()