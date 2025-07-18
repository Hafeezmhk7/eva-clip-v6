#!/usr/bin/env python3
"""
Quick import verification script for BLIP3-o project
Tests all major imports to ensure everything is working correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all major imports"""
    print("🧪 Testing BLIP3-o imports...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 6
    
    # Test 1: Config imports
    try:
        from src.modules.config.blip3o_config import (
            BLIP3oDiTConfig, 
            FlowMatchingConfig, 
            TrainingConfig,
            get_default_blip3o_config,
            get_default_flow_matching_config,
            get_default_training_config
        )
        print("✅ Config imports successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Config imports failed: {e}")
    
    # Test 2: Model imports
    try:
        from src.modules.models.blip3o_dit import BLIP3oDiTModel, create_blip3o_dit_model
        print("✅ Standard model imports successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Standard model imports failed: {e}")
    
    # Test 3: Dual supervision model imports
    try:
        from src.modules.models.dual_supervision_blip3o_dit import (
            DualSupervisionBLIP3oDiTModel, 
            create_blip3o_dit_model as create_dual_model
        )
        print("✅ Dual supervision model imports successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Dual supervision model imports failed: {e}")
    
    # Test 4: Loss imports
    try:
        from src.modules.losses.flow_matching_loss import BLIP3oFlowMatchingLoss, create_blip3o_flow_matching_loss
        from src.modules.losses.dual_supervision_flow_matching_loss import DualSupervisionFlowMatchingLoss, create_dual_supervision_loss
        print("✅ Loss function imports successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Loss function imports failed: {e}")
    
    # Test 5: Dataset imports
    try:
        from src.modules.datasets.blip3o_dataset import (
            BLIP3oEmbeddingDataset,
            create_chunked_dataloader,
            create_chunked_dataloaders
        )
        print("✅ Dataset imports successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Dataset imports failed: {e}")
    
    # Test 6: Trainer imports
    try:
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer, create_blip3o_training_args
        from src.modules.trainers.dual_supervision_blip3o_trainer import DualSupervisionBLIP3oTrainer
        print("✅ Trainer imports successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Trainer imports failed: {e}")
    
    # Test src package import
    try:
        import src
        print("✅ Main src package import successful")
    except Exception as e:
        print(f"❌ Main src package import failed: {e}")
    
    print("=" * 50)
    print(f"📊 Results: {success_count}/{total_tests} import groups successful")
    
    if success_count == total_tests:
        print("🎉 ALL IMPORTS WORKING!")
        print("✅ Ready to run test_single_example.py")
        return True
    else:
        print(f"⚠️ {total_tests - success_count} import groups failed")
        print("🔧 Fix the import issues before running tests")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)