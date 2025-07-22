#!/usr/bin/env python3
"""
Test script to verify that the global training fixes work correctly
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_parameter_compatibility():
    """Test that the model accepts both old and new parameter names"""
    print("🧪 Testing model parameter compatibility...")
    
    try:
        # Import fixed model
        from src.modules.config.blip3o_config import get_global_blip3o_config
        from src.modules.models.global_blip3o_dit import GlobalBLIP3oDiTModel
        
        # Create test model
        config = get_global_blip3o_config("small")
        model = GlobalBLIP3oDiTModel(config)
        model.eval()
        
        print("✅ Model created successfully")
        
        # Test data
        batch_size = 2
        device = "cpu"  # Use CPU for testing
        
        # Test with standard transformer parameter names (trainer uses these)
        hidden_states = torch.randn(batch_size, 768)  # Global input
        timestep = torch.rand(batch_size)
        encoder_hidden_states = torch.randn(batch_size, 256, 4096)  # EVA features
        
        print("🔄 Testing standard parameter names...")
        with torch.no_grad():
            output1 = model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )
        
        print(f"✅ Standard parameters work! Output shape: {output1.shape}")
        
        # Test with legacy parameter names
        print("🔄 Testing legacy parameter names...")
        with torch.no_grad():
            output2 = model(
                noisy_global_features=hidden_states,
                timestep=timestep,
                eva_features=encoder_hidden_states,
                return_dict=False
            )
        
        print(f"✅ Legacy parameters work! Output shape: {output2.shape}")
        
        # Test with patch inputs (should auto-convert)
        print("🔄 Testing patch input conversion...")
        patch_input = torch.randn(batch_size, 256, 1024)  # Patch format
        with torch.no_grad():
            output3 = model(
                hidden_states=patch_input,  # Should auto-convert to global
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )
        
        print(f"✅ Patch input conversion works! Output shape: {output3.shape}")
        
        # Verify outputs are reasonable
        assert output1.shape == (batch_size, 768), f"Wrong output shape: {output1.shape}"
        assert output2.shape == (batch_size, 768), f"Wrong output shape: {output2.shape}"  
        assert output3.shape == (batch_size, 768), f"Wrong output shape: {output3.shape}"
        
        print("✅ All model parameter compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model parameter compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_compatibility():
    """Test that the trainer can call the model correctly"""
    print("\n🧪 Testing trainer compatibility...")
    
    # Check for accelerate dependency first
    try:
        import accelerate
        print(f"✅ Accelerate found: {accelerate.__version__}")
    except ImportError:
        print("❌ Missing accelerate dependency")
        print("💡 Run: pip install 'accelerate>=0.26.0'")
        return False
    
    try:
        from src.modules.trainers.global_blip3o_trainer import EnhancedBLIP3oTrainer
        from src.modules.losses.global_flow_matching_loss import GlobalFlowMatchingLoss
        
        # Create dummy training args
        from transformers import TrainingArguments
        args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=1000,
        )
        
        # Create model and loss
        from src.modules.config.blip3o_config import get_global_blip3o_config
        from src.modules.models.global_blip3o_dit import GlobalBLIP3oDiTModel
        
        config = get_global_blip3o_config("small")
        model = GlobalBLIP3oDiTModel(config)
        
        # Create flow matching loss
        flow_loss = GlobalFlowMatchingLoss()
        
        # Create trainer
        trainer = EnhancedBLIP3oTrainer(
            model=model,
            args=args,
            flow_matching_loss=flow_loss,
        )
        
        print("✅ Trainer created successfully")
        
        # Test compute_loss with sample inputs
        batch_size = 2
        sample_inputs = {
            'eva_embeddings': torch.randn(batch_size, 256, 4096),
            'clip_embeddings': torch.randn(batch_size, 256, 1024),
        }
        
        print("🔄 Testing trainer compute_loss...")
        with torch.no_grad():
            loss = trainer.compute_loss(model, sample_inputs)
        
        print(f"✅ Trainer compute_loss works! Loss: {loss.item():.4f}")
        
        # Test with return_outputs=True
        print("🔄 Testing trainer compute_loss with outputs...")
        with torch.no_grad():
            loss, outputs = trainer.compute_loss(model, sample_inputs, return_outputs=True)
        
        print(f"✅ Trainer compute_loss with outputs works! Loss: {loss.item():.4f}")
        print(f"   Outputs keys: {list(outputs.keys()) if outputs else 'None'}")
        
        print("✅ All trainer compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Trainer compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_global_flow_matching_loss():
    """Test that the global flow matching loss works correctly"""
    print("\n🧪 Testing global flow matching loss...")
    
    try:
        from src.modules.losses.global_flow_matching_loss import GlobalFlowMatchingLoss
        
        # Create loss function
        loss_fn = GlobalFlowMatchingLoss()
        print("✅ Global flow matching loss created")
        
        # Test data
        batch_size = 2
        predicted_global = torch.randn(batch_size, 768)  # Model predictions
        clip_patches = torch.randn(batch_size, 256, 1024)  # CLIP patch targets
        timesteps = torch.rand(batch_size)
        
        print("🔄 Testing loss computation...")
        loss, metrics = loss_fn(
            predicted_global=predicted_global,
            clip_patches=clip_patches,
            timesteps=timesteps,
            return_metrics=True
        )
        
        print(f"✅ Loss computation works! Loss: {loss.item():.4f}")
        print(f"   Metrics: {list(metrics.keys()) if metrics else 'None'}")
        
        # Test target global features computation
        print("🔄 Testing target global features computation...")
        target_global = loss_fn.compute_target_global_features(clip_patches)
        
        print(f"✅ Target global computation works! Shape: {target_global.shape}")
        assert target_global.shape == (batch_size, 768), f"Wrong target shape: {target_global.shape}"
        
        print("✅ All global flow matching loss tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Global flow matching loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Running Global Training Fixes Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Test model compatibility
    if not test_model_parameter_compatibility():
        all_passed = False
    
    # Test trainer compatibility  
    if not test_trainer_compatibility():
        all_passed = False
        
    # Test loss function
    if not test_global_flow_matching_loss():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your global training fixes are working correctly")
        print("🚀 You can now run training with:")
        print("   python train_global_blip3o_multi_gpu.py --chunked_embeddings_dir <path> --output_dir <path>")
    else:
        print("❌ SOME TESTS FAILED!")
        print("💡 Please check the error messages above and ensure the fixes are applied correctly")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)