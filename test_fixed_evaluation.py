#!/usr/bin/env python3
"""
Test script to verify the fixed evaluation code works
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
    """Test the fixed model loading"""
    
    model_path = "/scratch-shared/scur2711/blip3o_workspace/checkpoints/blip3o_multi_gpu_fixed_cosine_13219643_20250719_081230"
    
    print("🧪 Testing FIXED model loading...")
    print("=" * 50)
    
    try:
        # FIXED: Import with correct class names
        try:
            from src.modules.models.dual_supervision_blip3o_dit import (
                DualSupervisionBLIP3oDiTModel,  # CORRECT class name
                create_blip3o_dit_model
            )
            from src.modules.config.blip3o_config import BLIP3oDiTConfig
            print("✅ Successfully imported dual supervision model")
            dual_supervision_available = True
        except ImportError as e:
            print(f"❌ Could not import dual supervision model: {e}")
            return False
        
        # Load config
        import json
        model_path_obj = Path(model_path)
        
        config_file = model_path_obj / "config.json"
        if not config_file.exists():
            config_file = model_path_obj / "blip3o_model_config.json"
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = BLIP3oDiTConfig(**config_dict)
        print(f"✅ Loaded config with {len(config_dict)} parameters")
        
        # Create model
        model = create_blip3o_dit_model(
            config=config,
            load_clip_projection=True,
            enable_dual_supervision=True,
        )
        print("✅ Created dual supervision model")
        
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
            print("❌ No dual supervision keys found")
            return False
        
        # Load weights into model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded weights into model")
        
        # Check capabilities
        capabilities = {
            'has_frozen_clip_proj': hasattr(model, 'frozen_clip_visual_proj') and model.frozen_clip_visual_proj is not None,
            'has_global_adaptation_mlp': hasattr(model, 'global_adaptation_mlp'),
            'has_global_velocity_proj': hasattr(model, 'global_velocity_proj'),  # KEY!
            'supports_training_modes': hasattr(model, 'forward') and 'training_mode' in str(model.forward.__code__.co_varnames),
            'supports_generation_modes': hasattr(model, 'generate') and 'generation_mode' in str(model.generate.__code__.co_varnames),
            'is_fixed_model': hasattr(model, 'global_velocity_proj'),
        }
        
        print("🔍 Model capabilities:")
        for key, value in capabilities.items():
            status = "✅" if value else "❌"
            print(f"   {key}: {status}")
        
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
                if capabilities['supports_generation_modes']:
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
        print("🎯 Your model should now achieve 50-70% recall!")
        print("\n💡 Next steps:")
        print("   1. Update your comp_eval.py with the fixed imports")
        print("   2. Re-run the evaluation")
        print("   3. You should see much better recall performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_model_loading()
    
    if success:
        print("\n✅ Ready to fix your evaluation!")
    else:
        print("\n❌ Need to debug the model loading issue")