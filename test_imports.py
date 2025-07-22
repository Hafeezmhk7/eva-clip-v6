#!/usr/bin/env python3
"""
Test script to verify all BLIP3-o imports work correctly
Run this to debug import issues
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing BLIP3-o Import Structure")
    print("=" * 40)
    
    # Test 1: Config imports
    print("\n1. Testing config imports...")
    try:
        from src.modules.config.blip3o_config import BLIP3oDiTConfig, FlowMatchingConfig, TrainingConfig
        print("   ✅ Direct config imports work")
        
        from src.modules.config import get_recommended_config, create_training_config
        print("   ✅ Config factory functions work")
        
    except ImportError as e:
        print(f"   ❌ Config import failed: {e}")
        return False
    
    # Test 2: Model imports  
    print("\n2. Testing model imports...")
    try:
        from src.modules.models.blip3o_dit import BLIP3oPatchDiTModel
        print("   ✅ Direct model import works")
        
        from src.modules.models import create_blip3o_dit_model, BLIP3oDiTModel
        print("   ✅ Model factory functions work")
        
    except ImportError as e:
        print(f"   ❌ Model import failed: {e}")
        return False
    
    # Test 3: Loss imports
    print("\n3. Testing loss imports...")
    try:
        from modules.losses.blip3o_flow_matching_loss import BLIP3oFlowMatchingLoss
        print("   ✅ Direct loss import works")
        
        from src.modules.losses import create_blip3o_flow_matching_loss, get_loss_function
        print("   ✅ Loss factory functions work")
        
    except ImportError as e:
        print(f"   ❌ Loss import failed: {e}")
        return False
    
    # Test 4: Trainer imports
    print("\n4. Testing trainer imports...")
    try:
        from modules.trainers.blip3o_patch_trainer import BLIP3oPatchTrainer
        print("   ✅ Direct trainer import works")
        
        from src.modules.trainers import BLIP3oTrainer, create_training_args
        print("   ✅ Trainer factory functions work")
        
    except ImportError as e:
        print(f"   ❌ Trainer import failed: {e}")
        return False
    
    # Test 5: Dataset imports
    print("\n5. Testing dataset imports...")
    try:
        from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset
        print("   ✅ Direct dataset import works")
        
        from src.modules.datasets import create_dataloaders
        print("   ✅ Dataset factory functions work")
        
    except ImportError as e:
        print(f"   ❌ Dataset import failed: {e}")
        return False
    
    # Test 6: Evaluation imports
    print("\n6. Testing evaluation imports...")
    try:
        from src.modules.evaluation.blip3o_recall_evaluator import BLIP3oRecallEvaluator
        print("   ✅ Direct evaluator import works")
        
        from src.modules import get_evaluator_class
        print("   ✅ Evaluator factory functions work")
        
    except ImportError as e:
        print(f"   ❌ Evaluation import failed: {e}")
        print(f"   💡 This is optional - training will still work")
    
    # Test 7: Complete pipeline test
    print("\n7. Testing complete pipeline...")
    try:
        # Create config
        config = get_recommended_config(model_size="small")
        print(f"   ✅ Config created: {config.hidden_size}D model")
        
        # Create model
        model = create_blip3o_dit_model(config=config)
        print(f"   ✅ Model created: {model.get_num_parameters():,} parameters")
        
        # Create loss
        loss_fn = get_loss_function(enhanced=True)
        print("   ✅ Loss function created")
        
        # Create training args
        training_args = create_training_config()
        print("   ✅ Training config created")
        
        print("   🎉 Complete pipeline test passed!")
        
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("✅ ALL IMPORTS WORKING CORRECTLY!")
    print("🚀 Ready for BLIP3-o patch-level training!")
    return True

def test_training_script_imports():
    """Test imports specifically needed by training script"""
    print("\n🎯 Testing Training Script Imports")
    print("=" * 40)
    
    try:
        # Test the exact imports used in train_blip3o_patch_gpu.py
        from src.modules.config.blip3o_config import get_blip3o_patch_config
        from src.modules.models.blip3o_dit import create_blip3o_patch_dit_model
        from modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        from modules.trainers.blip3o_patch_trainer import BLIP3oPatchTrainer, create_blip3o_patch_training_args
        from src.modules.datasets import create_dataloaders
        
        print("✅ All training script imports successful!")
        
        # Test creating objects
        config = get_blip3o_patch_config("base")
        model = create_blip3o_patch_dit_model(config=config)
        loss_fn = create_blip3o_flow_matching_loss(enhanced=True)
        
        print("✅ Training objects created successfully!")
        print(f"   Model: {model.get_num_parameters():,} parameters")
        print(f"   Config: {config.hidden_size}D, {config.num_hidden_layers}L, {config.num_attention_heads}H")
        
        return True
        
    except Exception as e:
        print(f"❌ Training script imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 BLIP3-o Import Testing")
    print("=" * 50)
    
    # Test basic imports
    success1 = test_imports()
    
    # Test training script specific imports  
    success2 = test_training_script_imports()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 BLIP3-o training pipeline is ready!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔧 Please fix the import issues above")
        sys.exit(1)