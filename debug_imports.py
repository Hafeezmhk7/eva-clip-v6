#!/usr/bin/env python3
"""
Debug script to test imports step by step
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_step_by_step():
    """Test imports step by step to isolate the issue"""
    
    print("🔧 Testing imports step by step...")
    print("=" * 50)
    
    # Step 1: Test config
    try:
        print("Step 1: Testing config import...")
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        print("✅ Config import successful")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    # Step 2: Test standard flow matching loss
    try:
        print("Step 2: Testing standard flow matching loss...")
        from src.modules.losses.flow_matching_loss import BLIP3oFlowMatchingLoss
        print("✅ Standard flow matching loss import successful")
    except Exception as e:
        print(f"❌ Standard flow matching loss import failed: {e}")
        return False
    
    # Step 3: Test dual supervision loss DIRECTLY
    try:
        print("Step 3: Testing dual supervision loss directly...")
        
        # Check if file exists
        loss_file = Path("src/modules/losses/dual_supervision_flow_matching_loss.py")
        print(f"   File exists: {loss_file.exists()}")
        
        if loss_file.exists():
            # Try to read the file and check what classes are defined
            with open(loss_file, 'r') as f:
                content = f.read()
                
            print("   Classes found in file:")
            if "class DualSupervisionFlowMatchingLoss" in content:
                print("   ✅ DualSupervisionFlowMatchingLoss")
            else:
                print("   ❌ DualSupervisionFlowMatchingLoss")
                
            if "def create_dual_supervision_loss" in content:
                print("   ✅ create_dual_supervision_loss")
            else:
                print("   ❌ create_dual_supervision_loss")
        
        # Now try the import
        from src.modules.losses.dual_supervision_flow_matching_loss import DualSupervisionFlowMatchingLoss
        print("✅ DualSupervisionFlowMatchingLoss import successful")
        
        from src.modules.losses.dual_supervision_flow_matching_loss import create_dual_supervision_loss
        print("✅ create_dual_supervision_loss import successful")
        
    except Exception as e:
        print(f"❌ Dual supervision loss import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test losses __init__
    try:
        print("Step 4: Testing losses __init__...")
        from src.modules.losses import DUAL_SUPERVISION_AVAILABLE
        print(f"✅ Losses init successful, dual supervision available: {DUAL_SUPERVISION_AVAILABLE}")
    except Exception as e:
        print(f"❌ Losses init failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't return False here, continue testing
    
    # Step 5: Test standard model
    try:
        print("Step 5: Testing standard model...")
        from src.modules.models.blip3o_dit import BLIP3oDiTModel
        print("✅ Standard model import successful")
    except Exception as e:
        print(f"❌ Standard model import failed: {e}")
        return False
    
    # Step 6: Test dual supervision model DIRECTLY
    try:
        print("Step 6: Testing dual supervision model directly...")
        
        # Check if file exists
        model_file = Path("src/modules/models/dual_supervision_blip3o_dit.py")
        print(f"   File exists: {model_file.exists()}")
        
        if model_file.exists():
            # Try to read the file and check what classes are defined
            with open(model_file, 'r') as f:
                content = f.read()
                
            print("   Classes found in file:")
            if "class DualSupervisionBLIP3oDiTModel" in content:
                print("   ✅ DualSupervisionBLIP3oDiTModel")
            else:
                print("   ❌ DualSupervisionBLIP3oDiTModel")
                
            if "def create_blip3o_dit_model" in content:
                print("   ✅ create_blip3o_dit_model")
            else:
                print("   ❌ create_blip3o_dit_model")
        
        # Now try the import
        from src.modules.models.dual_supervision_blip3o_dit import DualSupervisionBLIP3oDiTModel
        print("✅ DualSupervisionBLIP3oDiTModel import successful")
        
        from src.modules.models.dual_supervision_blip3o_dit import create_blip3o_dit_model
        print("✅ create_blip3o_dit_model import successful")
        
    except Exception as e:
        print(f"❌ Dual supervision model import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Test models __init__
    try:
        print("Step 7: Testing models __init__...")
        from src.modules.models import DUAL_SUPERVISION_MODEL_AVAILABLE
        print(f"✅ Models init successful, dual supervision available: {DUAL_SUPERVISION_MODEL_AVAILABLE}")
    except Exception as e:
        print(f"❌ Models init failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't return False here, continue testing
    
    print("\n🎉 All step-by-step tests passed!")
    return True

def test_final_integration():
    """Test the final integration that was failing"""
    print("\n🧪 Testing final integration...")
    
    try:
        # This is what the original test was trying to do
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.losses.dual_supervision_flow_matching_loss import (
            DualSupervisionFlowMatchingLoss,
            create_dual_supervision_loss
        )
        from src.modules.models.dual_supervision_blip3o_dit import (
            DualSupervisionBLIP3oDiTModel,
            create_blip3o_dit_model
        )
        
        print("✅ Final integration test successful!")
        
        # Test creating a model
        config = BLIP3oDiTConfig()
        model = create_blip3o_dit_model(config=config, load_clip_projection=False)
        print("✅ Model creation successful!")
        
        # Check key attributes
        has_global_velocity_proj = hasattr(model, 'global_velocity_proj')
        print(f"✅ Has global_velocity_proj: {has_global_velocity_proj}")
        
        return True
        
    except Exception as e:
        print(f"❌ Final integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 BLIP3-o Import Debug Testing")
    print("=" * 60)
    
    step_success = test_step_by_step()
    final_success = test_final_integration()
    
    print("\n" + "=" * 60)
    print("📊 DEBUG RESULTS:")
    print(f"   Step-by-step: {'✅ PASS' if step_success else '❌ FAIL'}")
    print(f"   Final integration: {'✅ PASS' if final_success else '❌ FAIL'}")
    
    if step_success and final_success:
        print("\n🎉 ALL DEBUG TESTS PASSED!")
        print("✅ Imports are working correctly")
    else:
        print("\n❌ DEBUG TESTS FAILED!")
        print("💡 Check the specific error messages above")
    
    print("=" * 60)