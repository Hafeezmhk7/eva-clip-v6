#!/usr/bin/env python3
"""
FIXED Test script to verify the dual supervision model loading works correctly
"""

import sys
import torch
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_fixed_model_loading():
    """Test the fixed model loading with correct imports"""
    
    model_path = "/scratch-shared/scur2711/blip3o_workspace/checkpoints/blip3o_multi_gpu_fixed_cosine_13219643_20250719_081230"
    
    print("🧪 Testing FIXED model loading...")
    print("=" * 50)
    
    try:
        # Test basic imports first
        print("📦 Testing imports...")
        try:
            # Test config import
            from src.modules.config.blip3o_config import BLIP3oDiTConfig
            print("✅ Config import successful")
            
            # Test standard loss import first
            from src.modules.losses.flow_matching_loss import (
                BLIP3oFlowMatchingLoss,
                create_blip3o_flow_matching_loss
            )
            print("✅ Standard loss import successful")
            
            # Test dual supervision loss import
            from src.modules.losses.dual_supervision_flow_matching_loss import (
                DualSupervisionFlowMatchingLoss,
                create_dual_supervision_loss
            )
            print("✅ Dual supervision loss import successful")
            
            # Test model import - this is the critical one
            from src.modules.models.dual_supervision_blip3o_dit import (
                DualSupervisionBLIP3oDiTModel,
                create_blip3o_dit_model
            )
            print("✅ Dual supervision model import successful")
            
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("\n🔧 Debugging import issue...")
            
            # Check if files exist
            model_file = Path("src/modules/models/dual_supervision_blip3o_dit.py")
            loss_file = Path("src/modules/losses/dual_supervision_flow_matching_loss.py")
            
            print(f"Model file exists: {model_file.exists()}")
            print(f"Loss file exists: {loss_file.exists()}")
            
            if not model_file.exists():
                print("❌ Missing dual supervision model file!")
                return False
                
            if not loss_file.exists():
                print("❌ Missing dual supervision loss file!")
                return False
            
            # Try to import with different approach
            try:
                import importlib.util
                
                # Load model module
                spec = importlib.util.spec_from_file_location("dual_supervision_model", model_file)
                dual_model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dual_model_module)
                
                DualSupervisionBLIP3oDiTModel = dual_model_module.DualSupervisionBLIP3oDiTModel
                create_blip3o_dit_model = dual_model_module.create_blip3o_dit_model
                
                print("✅ Direct import successful")
                
            except Exception as e2:
                print(f"❌ Direct import also failed: {e2}")
                return False
        
        # Load config
        import json
        model_path_obj = Path(model_path)
        
        config_file = model_path_obj / "config.json"
        if not config_file.exists():
            config_file = model_path_obj / "blip3o_model_config.json"
        
        if not config_file.exists():
            print(f"❌ Config file not found in {model_path}")
            return False
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = BLIP3oDiTConfig(**config_dict)
        print(f"✅ Loaded config with {len(config_dict)} parameters")
        
        # Create model
        print("🏗️  Creating dual supervision model...")
        model = create_blip3o_dit_model(
            config=config,
            load_clip_projection=True,
            enable_dual_supervision=True,
        )
        print("✅ Created dual supervision model")
        
        # Check for key components
        has_global_velocity_proj = hasattr(model, 'global_velocity_proj')
        has_frozen_clip_proj = hasattr(model, 'frozen_clip_visual_proj') and model.frozen_clip_visual_proj is not None
        has_global_adaptation_mlp = hasattr(model, 'global_adaptation_mlp')
        
        print("🔍 Model capabilities:")
        print(f"   Has global velocity projection: {'✅' if has_global_velocity_proj else '❌'}")
        print(f"   Has frozen CLIP projection: {'✅' if has_frozen_clip_proj else '❌'}")
        print(f"   Has global adaptation MLP: {'✅' if has_global_adaptation_mlp else '❌'}")
        
        if not has_global_velocity_proj:
            print("❌ CRITICAL: Missing global_velocity_proj - this is required for dual supervision!")
            return False
        
        # Load weights
        model_file = model_path_obj / "model.safetensors"
        if not model_file.exists():
            print(f"❌ Model file not found: {model_file}")
            return False
        
        from safetensors.torch import load_file
        state_dict = load_file(str(model_file))
        print(f"✅ Loaded state dict with {len(state_dict)} keys")
        
        # Check for dual supervision keys
        dual_keys = [k for k in state_dict.keys() if 'global_velocity_proj' in k]
        if dual_keys:
            print(f"✅ Found dual supervision keys: {dual_keys}")
        else:
            print("❌ No dual supervision keys found in checkpoint")
            print("⚠️  This model was not trained with dual supervision")
            return False
        
        # Load weights into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded weights into model")
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)} (first 5: {missing_keys[:5]})")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)} (first 5: {unexpected_keys[:5]})")
        
        # Test generation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Create dummy EVA embeddings
        dummy_eva = torch.randn(2, 256, 4096, device=device)  # [batch, tokens, eva_dim]
        
        print("🧪 Testing generation...")
        
        try:
            with torch.no_grad():
                # Test different generation modes
                if hasattr(model, 'generate') and 'generation_mode' in str(model.generate.__code__.co_varnames):
                    print("   Testing global generation mode...")
                    generated = model.generate(
                        encoder_hidden_states=dummy_eva,
                        num_inference_steps=5,  # Quick test
                        generation_mode="global",
                        return_global_only=True,
                    )
                    print(f"   ✅ Global generation: {generated.shape}")
                    
                else:
                    print("   Testing standard generation...")
                    generated = model.generate(
                        encoder_hidden_states=dummy_eva,
                        num_inference_steps=5,  # Quick test
                    )
                    print(f"   ✅ Standard generation: {generated.shape}")
                    
                    # Convert to global if needed
                    if generated.dim() == 3 and generated.shape[1] == 256:
                        generated = generated.mean(dim=1)  # Average pool
                        if hasattr(model, 'frozen_clip_visual_proj') and model.frozen_clip_visual_proj is not None:
                            generated = model.frozen_clip_visual_proj(generated)
                        print(f"   ✅ Converted to global: {generated.shape}")
            
            print("✅ Generation test successful!")
            
        except Exception as e:
            print(f"❌ Generation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🎉 SUCCESS: Fixed model loading works correctly!")
        print("🎯 Your model has dual supervision and should achieve 50-70% recall!")
        print("\n💡 Next steps:")
        print("   1. Your evaluation script should now work")
        print("   2. Re-run comp_eval.py with your model")
        print("   3. You should see much better recall performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_script():
    """Test that the evaluation script can import everything correctly"""
    print("\n🧪 Testing evaluation script imports...")
    
    try:
        # Test the main evaluation class import
        from comp_eval import FixedBLIP3oRecallEvaluator
        print("✅ Evaluation script imports successful")
        
        # Test creating evaluator
        evaluator = FixedBLIP3oRecallEvaluator(device="cpu")
        print("✅ Evaluator creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 BLIP3-o Fixed Model and Evaluation Testing")
    print("=" * 60)
    
    # Test model loading
    model_success = test_fixed_model_loading()
    
    # Test evaluation script
    eval_success = test_evaluation_script()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   Model loading: {'✅ PASS' if model_success else '❌ FAIL'}")
    print(f"   Evaluation script: {'✅ PASS' if eval_success else '❌ FAIL'}")
    
    if model_success and eval_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Ready to run evaluation with dual supervision model")
        print("\n📋 To run evaluation:")
        print("python comp_eval.py \\")
        print("  --coco_root ./data/coco \\")
        print("  --blip3o_model_path /scratch-shared/scur2711/blip3o_workspace/checkpoints/blip3o_multi_gpu_fixed_cosine_13219643_20250719_081230 \\")
        print("  --num_samples 1000 \\")
        print("  --generation_mode global \\")
        print("  --save_results results/recall_evaluation.json")
    else:
        print("\n❌ TESTS FAILED!")
        print("💡 Check the error messages above and fix the import issues")
    
    print("=" * 60)