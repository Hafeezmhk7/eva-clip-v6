#!/usr/bin/env python3
"""
Minimal test that imports modules directly without __init__.py
"""

import sys
import importlib.util
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def load_module_directly(file_path, module_name):
    """Load a module directly from file path"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, None
    except Exception as e:
        return None, str(e)

def test_direct_loading():
    """Test loading modules directly"""
    print("🔧 Testing direct module loading...")
    print("=" * 50)
    
    # Test config
    print("1. Testing config...")
    config_module, error = load_module_directly(
        "src/modules/config/blip3o_config.py", 
        "blip3o_config"
    )
    if config_module:
        print("✅ Config loaded directly")
        print(f"   Has BLIP3oDiTConfig: {hasattr(config_module, 'BLIP3oDiTConfig')}")
    else:
        print(f"❌ Config failed: {error}")
        return False
    
    # Test standard flow matching loss
    print("2. Testing standard flow matching loss...")
    flow_loss_module, error = load_module_directly(
        "src/modules/losses/flow_matching_loss.py", 
        "flow_matching_loss"
    )
    if flow_loss_module:
        print("✅ Standard flow matching loss loaded directly")
        print(f"   Has BLIP3oFlowMatchingLoss: {hasattr(flow_loss_module, 'BLIP3oFlowMatchingLoss')}")
    else:
        print(f"❌ Standard flow matching loss failed: {error}")
        return False
    
    # Test dual supervision loss
    print("3. Testing dual supervision loss...")
    dual_loss_module, error = load_module_directly(
        "src/modules/losses/dual_supervision_flow_matching_loss.py", 
        "dual_supervision_flow_matching_loss"
    )
    if dual_loss_module:
        print("✅ Dual supervision loss loaded directly")
        print(f"   Has DualSupervisionFlowMatchingLoss: {hasattr(dual_loss_module, 'DualSupervisionFlowMatchingLoss')}")
        print(f"   Has create_dual_supervision_loss: {hasattr(dual_loss_module, 'create_dual_supervision_loss')}")
    else:
        print(f"❌ Dual supervision loss failed: {error}")
        return False
    
    # Test standard model
    print("4. Testing standard model...")
    std_model_module, error = load_module_directly(
        "src/modules/models/blip3o_dit.py", 
        "blip3o_dit"
    )
    if std_model_module:
        print("✅ Standard model loaded directly")
        print(f"   Has BLIP3oDiTModel: {hasattr(std_model_module, 'BLIP3oDiTModel')}")
    else:
        print(f"❌ Standard model failed: {error}")
        return False
    
    # Test dual supervision model
    print("5. Testing dual supervision model...")
    dual_model_module, error = load_module_directly(
        "src/modules/models/dual_supervision_blip3o_dit.py", 
        "dual_supervision_blip3o_dit"
    )
    if dual_model_module:
        print("✅ Dual supervision model loaded directly")
        print(f"   Has DualSupervisionBLIP3oDiTModel: {hasattr(dual_model_module, 'DualSupervisionBLIP3oDiTModel')}")
        print(f"   Has create_blip3o_dit_model: {hasattr(dual_model_module, 'create_blip3o_dit_model')}")
    else:
        print(f"❌ Dual supervision model failed: {error}")
        return False
    
    # Test creating a model instance
    print("6. Testing model creation...")
    try:
        # Create config
        config = config_module.BLIP3oDiTConfig()
        print("✅ Config created")
        
        # Create model
        model = dual_model_module.create_blip3o_dit_model(
            config=config, 
            load_clip_projection=False,
            enable_dual_supervision=False  # Avoid CLIP loading issues
        )
        print("✅ Model created")
        
        # Check key attributes
        has_global_velocity_proj = hasattr(model, 'global_velocity_proj')
        has_global_adaptation_mlp = hasattr(model, 'global_adaptation_mlp')
        
        print(f"   Has global_velocity_proj: {has_global_velocity_proj}")
        print(f"   Has global_adaptation_mlp: {has_global_adaptation_mlp}")
        
        if has_global_velocity_proj:
            print("🎉 SUCCESS: Model has dual supervision components!")
        else:
            print("❌ PROBLEM: Model missing dual supervision components!")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All direct loading tests passed!")
    return True

def test_with_your_checkpoint():
    """Test with your actual checkpoint"""
    print("\n🧪 Testing with your checkpoint...")
    
    model_path = "/scratch-shared/scur2711/blip3o_workspace/checkpoints/blip3o_multi_gpu_fixed_cosine_13219643_20250719_081230"
    
    if not Path(model_path).exists():
        print(f"❌ Checkpoint not found: {model_path}")
        return False
    
    try:
        # Load modules directly
        config_module, _ = load_module_directly("src/modules/config/blip3o_config.py", "config")
        dual_model_module, _ = load_module_directly("src/modules/models/dual_supervision_blip3o_dit.py", "dual_model")
        
        # Load config
        import json
        config_file = Path(model_path) / "config.json"
        if not config_file.exists():
            config_file = Path(model_path) / "blip3o_model_config.json"
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = config_module.BLIP3oDiTConfig(**config_dict)
        print("✅ Loaded config from checkpoint")
        
        # Create model
        model = dual_model_module.create_blip3o_dit_model(
            config=config,
            load_clip_projection=False,  # Skip CLIP loading for now
            enable_dual_supervision=False
        )
        print("✅ Created model")
        
        # Check for checkpoint weights
        model_file = Path(model_path) / "model.safetensors"
        if model_file.exists():
            print(f"✅ Checkpoint file exists: {model_file}")
            
            # Try to load weights
            from safetensors.torch import load_file
            state_dict = load_file(str(model_file))
            print(f"✅ Loaded state dict with {len(state_dict)} keys")
            
            # Check for dual supervision keys
            dual_keys = [k for k in state_dict.keys() if 'global_velocity_proj' in k]
            if dual_keys:
                print(f"✅ Found dual supervision keys: {dual_keys}")
                print("🎉 Your checkpoint has dual supervision!")
                return True
            else:
                print("❌ No dual supervision keys in checkpoint")
                print("⚠️  Your model was not trained with dual supervision")
                return False
        else:
            print(f"❌ No model file found in {model_path}")
            return False
            
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Minimal Direct Import Testing")
    print("=" * 60)
    
    direct_success = test_direct_loading()
    checkpoint_success = test_with_your_checkpoint()
    
    print("\n" + "=" * 60)
    print("📊 MINIMAL TEST RESULTS:")
    print(f"   Direct loading: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"   Checkpoint test: {'✅ PASS' if checkpoint_success else '❌ FAIL'}")
    
    if direct_success and checkpoint_success:
        print("\n🎉 MINIMAL TESTS PASSED!")
        print("✅ Your modules and checkpoint are working")
        print("💡 The issue might be in the __init__.py files")
    else:
        print("\n❌ MINIMAL TESTS FAILED!")
        print("💡 There are fundamental issues with the modules")
    
    print("=" * 60)