#!/usr/bin/env python3
"""
Quick test to check if training can start
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def quick_test():
    """Quick test of essential components"""
    print("🚀 Quick BLIP3-o Test")
    print("=" * 30)
    
    try:
        # Test 1: Model
        print("1. Testing model...")
        from src.modules.models.blip3o_dit import BLIP3oPatchDiTModel, create_blip3o_patch_dit_model
        from src.modules.config.blip3o_config import get_blip3o_patch_config
        
        config = get_blip3o_patch_config("base")
        model = create_blip3o_patch_dit_model(config=config)
        print(f"   ✅ Model: {model.get_num_parameters():,} params")
        
        # Test 2: Loss
        print("2. Testing loss...")
        from modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        loss_fn = create_blip3o_flow_matching_loss(enhanced=True)
        print("   ✅ Loss function created")
        
        # Test 3: Trainer
        print("3. Testing trainer...")
        from modules.trainers.blip3o_patch_trainer import BLIP3oPatchTrainer
        print("   ✅ Trainer class imported")
        
        # Test 4: Dataset
        print("4. Testing dataset...")
        from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset
        print("   ✅ Dataset class imported")
        
        print("\n🎉 Essential components working!")
        print("🚀 Ready to start training!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if not success:
        print("\n🔧 Fix the issues above before training")
        sys.exit(1)
    else:
        print("\n✅ All good! You can start training!")
        sys.exit(0)