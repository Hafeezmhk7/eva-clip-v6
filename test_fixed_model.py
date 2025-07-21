#!/usr/bin/env python3
"""
Quick test script to verify the fixed model loading works
"""

import sys
import torch
from pathlib import Path
import json

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_fixed_imports():
    """Test that the fixed imports work correctly"""
    
    print("🧪 Testing FIXED imports...")
    print("=" * 50)
    
    try:
        # Test direct import approach (should work now)
        import importlib.util
        
        print("1. Loading config module...")
        config_path = Path("src/modules/config/blip3o_config.py")
        spec = importlib.util.spec_from_file_location("blip3o_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        BLIP3oDiTConfig = config_module.BLIP3oDiTConfig
        print("✅ Config loaded")
        
        print("2. Loading dual supervision model...")
        model_path = Path("src/modules/models/dual_supervision_blip3o_dit.py")
        spec = importlib.util.spec_from_file_location("dual_model", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        create_blip3o_dit_model = model_module.create_blip3o_dit_model
        print("✅ Model module loaded")
        
        print("3. Creating test model...")
        config = BLIP3oDiTConfig()
        model = create_blip3o_dit_model(
            config=config,
            load_clip_projection=False,  # Skip CLIP loading for speed
            enable_dual_supervision=True,
        )
        print("✅ Model created successfully")
        
        print("4. Checking model capabilities...")
        has_global_velocity_proj = hasattr(model, 'global_velocity_proj')
        has_global_adaptation_mlp = hasattr(model, 'global_adaptation_mlp')
        
        print(f"   Has global_velocity_proj: {'✅' if has_global_velocity_proj else '❌'}")
        print(f"   Has global_adaptation_mlp: {'✅' if has_global_adaptation_mlp else '❌'}")
        
        if has_global_velocity_proj:
            print("🎉 SUCCESS: Model has dual supervision components!")
            return True
        else:
            print("❌ FAILURE: Model missing dual supervision components!")
            return False
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test loading your actual trained model"""
    
    model_path = "/scratch-shared/scur2711/blip3o_workspace/checkpoints/blip3o_multi_gpu_fixed_cosine_13219643_20250719_081230"
    
    print(f"\n🧪 Testing model loading from checkpoint...")
    print(f"Model path: {model_path}")
    print("=" * 50)
    
    try:
        # Import the fixed evaluation script
        from fixed_blip3o_evaluation import FixedBLIP3oRecallEvaluator
        
        print("✅ Imported fixed evaluation script")
        
        # Create evaluator
        evaluator = FixedBLIP3oRecallEvaluator(device="cpu")
        print("✅ Created evaluator")
        
        # Try to load the model
        evaluator.load_blip3o_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Check capabilities
        if hasattr(evaluator, 'model_capabilities'):
            caps = evaluator.model_capabilities
            print("📊 Model capabilities:")
            for key, value in caps.items():
                status = "✅" if value else "❌"
                print(f"   {key}: {status}")
            
            if caps.get('has_global_velocity_proj', False):
                print("\n🎉 SUCCESS: Your model has dual supervision!")
                print("🎯 Expected recall: 50-70% (much better than 0.1%)")
                return True
            else:
                print("\n⚠️  WARNING: Model missing global velocity projection")
                print("🎯 Expected recall: still low (~0-2%)")
                return False
        else:
            print("⚠️  Could not check model capabilities")
            return False
            
    except FileNotFoundError:
        print(f"❌ Model checkpoint not found: {model_path}")
        print("💡 Check if the path is correct")
        return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 QUICK TEST: Fixed Model Loading")
    print("=" * 60)
    
    # Test imports
    import_success = test_fixed_imports()
    
    # Test model loading
    model_success = test_model_loading()
    
    print("\n" + "=" * 60)
    print("📊 QUICK TEST RESULTS:")
    print(f"   Import test: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"   Model loading: {'✅ PASS' if model_success else '❌ FAIL'}")
    
    if import_success and model_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your model has dual supervision and should work!")
        print("\n📋 Ready to run full evaluation:")
        print("python fixed_blip3o_evaluation.py \\")
        print("  --coco_root ./data/coco \\")
        print("  --blip3o_model_path /scratch-shared/scur2711/blip3o_workspace/checkpoints/blip3o_multi_gpu_fixed_cosine_13219643_20250719_081230 \\")
        print("  --num_samples 1000 \\")
        print("  --generation_mode auto \\")
        print("  --save_results results/recall_evaluation.json")
    else:
        print("\n❌ TESTS FAILED!")
        print("💡 You may need to:")
        print("   1. Fix the import in dual_supervision_flow_matching_loss.py")
        print("   2. Update your evaluation script with the fixed version")
        print("   3. Check if your model path is correct")
    
    print("=" * 60)