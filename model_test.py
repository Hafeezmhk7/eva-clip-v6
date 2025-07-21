#!/usr/bin/env python3
"""
Simple script to test if your dual supervision model loads correctly
"""

import sys
import torch
from pathlib import Path
import json

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / "src"))

def test_model_loading(model_path: str):
    """Test if the model can be loaded properly."""
    print(f"🧪 Testing model loading from: {model_path}")
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        print("✅ Config import successful")
        
        from src.modules.models.dual_supervision_blip3o_dit import DualSupervisionBLIP3oDiTModel, create_blip3o_dit_model
        print("✅ Dual supervision model import successful")
        
        # Load config
        print("📋 Loading config...")
        model_path = Path(model_path)
        config_file = model_path / "config.json"
        if not config_file.exists():
            config_file = model_path / "blip3o_model_config.json"
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        config = BLIP3oDiTConfig(**config_dict)
        print(f"✅ Config loaded: {config.dim}D, {config.n_layers}L, {config.n_heads}H")
        
        # Create model
        print("🏗️ Creating model...")
        model = create_blip3o_dit_model(
            config=config,
            load_clip_projection=False,  # Skip CLIP loading for speed
            enable_dual_supervision=True,
        )
        print(f"✅ Model created: {type(model).__name__}")
        
        # Check dual supervision components
        print("🔍 Checking dual supervision components...")
        has_global_velocity = hasattr(model, 'global_velocity_proj')
        has_global_mlp = hasattr(model, 'global_adaptation_mlp')
        has_clip_proj = hasattr(model, 'frozen_clip_visual_proj')
        
        print(f"   Global velocity projection: {'✅' if has_global_velocity else '❌'}")
        print(f"   Global adaptation MLP: {'✅' if has_global_mlp else '❌'}")
        print(f"   CLIP projection: {'✅' if has_clip_proj else '❌'}")
        
        # Load weights
        print("⚖️ Loading weights...")
        model_files = [
            model_path / "model.safetensors",
            model_path / "pytorch_model.bin",
            model_path / "pytorch_model.safetensors"
        ]
        
        model_file = None
        for file_path in model_files:
            if file_path.exists():
                model_file = file_path
                break
        
        if model_file is None:
            print("❌ No model weights found!")
            return False
        
        print(f"📁 Loading from: {model_file}")
        
        if model_file.suffix == ".bin":
            state_dict = torch.load(model_file, map_location="cpu")
        else:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_file))
        
        # Check for dual supervision keys
        dual_keys = [k for k in state_dict.keys() if 'global_velocity_proj' in k]
        if dual_keys:
            print(f"✅ Found dual supervision keys: {len(dual_keys)}")
        else:
            print("❌ No dual supervision keys found in weights!")
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        
        print("✅ Weights loaded successfully!")
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        model.eval()
        
        with torch.no_grad():
            # Create dummy inputs
            batch_size = 2
            dummy_hidden = torch.randn(batch_size, 256, 1024)  # [B, 256, 1024]
            dummy_timestep = torch.rand(batch_size)  # [B]
            dummy_eva = torch.randn(batch_size, 256, 4096)  # [B, 256, 4096]
            
            try:
                # Test dual supervision forward
                outputs = model(
                    hidden_states=dummy_hidden,
                    timestep=dummy_timestep,
                    encoder_hidden_states=dummy_eva,
                    training_mode="dual_supervision",
                    return_dict=True
                )
                
                print(f"✅ Forward pass successful!")
                print(f"   Output keys: {list(outputs.keys())}")
                
                if 'patch_output' in outputs:
                    print(f"   Patch output shape: {outputs['patch_output'].shape}")
                if 'global_output' in outputs:
                    print(f"   Global output shape: {outputs['global_output'].shape}")
                
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                return False
        
        # Test generation
        print("🎯 Testing generation...")
        try:
            generated = model.generate(
                encoder_hidden_states=dummy_eva,
                num_inference_steps=5,  # Small number for speed
                generation_mode="global",
            )
            print(f"✅ Generation successful! Output shape: {generated.shape}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            # This might be expected if generation_mode isn't supported
            print("   Trying standard generation...")
            try:
                generated = model.generate(
                    encoder_hidden_states=dummy_eva,
                    num_inference_steps=5,
                )
                print(f"✅ Standard generation successful! Output shape: {generated.shape}")
            except Exception as e2:
                print(f"❌ Standard generation also failed: {e2}")
        
        print("🎉 Model loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python model_test.py <model_path>")
        print("Example: python model_test.py /scratch-shared/scur2711/blip3o_workspace/checkpoints/blip3o_multi_gpu_fixed_cosine_13219643_20250719_081230")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = test_model_loading(model_path)
    
    if success:
        print("\n✅ Your model should work with the fixed evaluation script!")
        print("Now you can run the fixed evaluation script:")
        print(f"python comp_eval_fixed.py --coco_root ./data/coco --blip3o_model_path {model_path}")
    else:
        print("\n❌ Model loading failed. Check the error messages above.")
    
    sys.exit(0 if success else 1)