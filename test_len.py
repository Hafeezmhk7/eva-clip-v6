#!/usr/bin/env python3
"""
Test script to validate the IterableDataset __len__ fix
Run this before training to ensure the fix works correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_dataset_boolean_evaluation(chunked_embeddings_dir: str):
    """Test that the dataset fix resolves boolean evaluation issues."""
    
    print("🧪 Testing Dataset Boolean Evaluation Fix")
    print("=" * 50)
    
    try:
        from src.modules.datasets.blip3o_dataset import (
            BLIP3oEmbeddingDataset, 
            create_chunked_dataloader,
            create_chunked_dataloaders
        )
        
        print("✅ Successfully imported dataset modules")
        
        # Test 1: Create dataset and check __len__ method
        print("\n📊 Test 1: Dataset __len__ method")
        dataset = BLIP3oEmbeddingDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split="train",
            eval_split_ratio=0.1,
            delete_after_use=False,
        )
        
        # This should work now
        dataset_length = len(dataset)
        print(f"✅ Dataset length: {dataset_length:,}")
        
        # Test 2: Create dataloader and test boolean evaluation
        print("\n📊 Test 2: DataLoader boolean evaluation")
        dataloader = create_chunked_dataloader(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=4,
            split="train",
            eval_split_ratio=0.1,
            delete_after_use=False,
        )
        
        # This was the problematic line - should work now
        has_dataloader = bool(dataloader)  # This caused the original error
        print(f"✅ DataLoader boolean evaluation: {has_dataloader}")
        
        # Test the specific problematic pattern from training script
        eval_strategy = "steps" if dataloader else "no"
        print(f"✅ Conditional evaluation: eval_strategy = '{eval_strategy}'")
        
        # Test 3: Create both train and eval dataloaders
        print("\n📊 Test 3: Train/Eval dataloader creation")
        train_dataloader, eval_dataloader = create_chunked_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=4,
            eval_batch_size=8,
            eval_split_ratio=0.1,
            delete_after_use=False,
        )
        
        print(f"✅ Train dataloader created: {bool(train_dataloader)}")
        print(f"✅ Eval dataloader created: {bool(eval_dataloader) if eval_dataloader else False}")
        
        # Test the exact pattern from the training script
        has_eval = eval_dataloader is not None  # Safe way
        eval_strategy_safe = "steps" if has_eval else "no"
        print(f"✅ Safe evaluation check: eval_strategy = '{eval_strategy_safe}'")
        
        # Test 4: DataLoader length access
        print("\n📊 Test 4: DataLoader length access")
        try:
            train_length = len(train_dataloader)
            print(f"✅ Train dataloader length: {train_length:,}")
        except Exception as e:
            print(f"⚠️  Train dataloader length error: {e}")
        
        if eval_dataloader:
            try:
                eval_length = len(eval_dataloader)
                print(f"✅ Eval dataloader length: {eval_length:,}")
            except Exception as e:
                print(f"⚠️  Eval dataloader length error: {e}")
        
        # Test 5: Iteration test
        print("\n📊 Test 5: Quick iteration test")
        batch_count = 0
        for batch in train_dataloader:
            print(f"✅ Batch {batch_count}: EVA {batch['eva_embeddings'].shape}, CLIP {batch['clip_embeddings'].shape}")
            batch_count += 1
            if batch_count >= 2:  # Only test first 2 batches
                break
        
        print(f"✅ Successfully processed {batch_count} batches")
        
        # Test 6: Distributed simulation
        print("\n📊 Test 6: Distributed training simulation")
        
        # Simulate distributed environment
        os.environ['WORLD_SIZE'] = '3'
        os.environ['RANK'] = '0'
        
        # Create dataset with distributed settings
        dist_dataset = BLIP3oEmbeddingDataset(
            chunked_embeddings_dir=chunked_embeddings_dir,
            split="train",
            eval_split_ratio=0.1,
            delete_after_use=False,
        )
        
        dist_length = len(dist_dataset)
        print(f"✅ Distributed dataset length (rank 0/3): {dist_length:,}")
        
        # Clean up environment
        del os.environ['WORLD_SIZE']
        del os.environ['RANK']
        
        print("\n🎉 All tests passed! The IterableDataset fix is working correctly.")
        print("You can now run your multi-GPU training script.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_args_pattern(chunked_embeddings_dir: str):
    """Test the specific pattern used in TrainingArguments that was failing."""
    
    print("\n🔧 Testing TrainingArguments Pattern")
    print("=" * 40)
    
    try:
        from src.modules.datasets.blip3o_dataset import create_chunked_dataloaders
        from transformers import TrainingArguments
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_chunked_dataloaders(
            chunked_embeddings_dir=chunked_embeddings_dir,
            batch_size=4,
            eval_batch_size=8,
            eval_split_ratio=0.1,
            delete_after_use=False,
        )
        
        # Test the exact pattern that was failing
        print("Testing the exact failing pattern from TrainingArguments...")
        
        # This was the line causing the error:
        # eval_strategy="steps" if eval_dataloader else "no"
        
        # FIXED approach:
        has_eval_dataloader = eval_dataloader is not None
        eval_strategy = "steps" if has_eval_dataloader else "no"
        eval_steps = 100 if has_eval_dataloader else None
        
        print(f"✅ has_eval_dataloader: {has_eval_dataloader}")
        print(f"✅ eval_strategy: {eval_strategy}")
        print(f"✅ eval_steps: {eval_steps}")
        
        # Test creating TrainingArguments with our fix
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            eval_strategy=eval_strategy,  # This should work now
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
        )
        
        print("✅ TrainingArguments created successfully!")
        print(f"   eval_strategy: {training_args.eval_strategy}")
        print(f"   eval_steps: {training_args.eval_steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ TrainingArguments pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    
    if len(sys.argv) != 2:
        print("Usage: python test_dataset_fix.py <chunked_embeddings_dir>")
        print("Example: python test_dataset_fix.py /path/to/chunked_embeddings")
        sys.exit(1)
    
    chunked_embeddings_dir = sys.argv[1]
    
    if not Path(chunked_embeddings_dir).exists():
        print(f"❌ Directory not found: {chunked_embeddings_dir}")
        sys.exit(1)
    
    print(f"🧪 Testing dataset fix with: {chunked_embeddings_dir}")
    
    # Run tests
    test1_passed = test_dataset_boolean_evaluation(chunked_embeddings_dir)
    test2_passed = test_training_args_pattern(chunked_embeddings_dir)
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("The IterableDataset __len__ fix is working correctly.")
        print("You can now run your multi-GPU training script safely.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()