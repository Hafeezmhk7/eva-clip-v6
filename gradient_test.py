#!/usr/bin/env python3
"""
Test gradient flow after applying fixes
Run this after updating the compute_loss method
"""

import torch
import sys
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_gradient_flow():
    """Test if gradient flow is working with the fixes"""
    
    print("🧪 Testing gradient flow with fixes...")
    
    try:
        # Import your modules
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model
        from src.modules.config.blip3o_config import get_blip3o_patch_config
        from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        
        # Create small test model
        config = get_blip3o_patch_config("tiny")
        model = create_blip3o_patch_dit_model(config)
        model.train()  # CRITICAL: Set to training mode
        
        print(f"✅ Model created with {model.get_num_parameters():,} parameters")
        
        # Create loss function
        loss_fn = create_blip3o_flow_matching_loss(enhanced=True)
        
        # Create dummy inputs (similar to real training)
        batch_size = 2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # EVA embeddings (conditioning) - these don't need gradients
        eva_embeddings = torch.randn(batch_size, 256, 4096, device=device, requires_grad=False)
        
        # CLIP targets - these should not have gradients
        clip_targets = torch.randn(batch_size, 256, 1024, device=device, requires_grad=False)
        
        print(f"✅ Inputs created on device: {device}")
        
        # Test the flow matching process
        timesteps = loss_fn.sample_timesteps(batch_size, device)
        x_0 = torch.randn_like(clip_targets, requires_grad=True)  # Source needs gradients
        noise = torch.randn_like(clip_targets) * 0.1
        
        # Test interpolation (this should preserve gradients)
        noisy_clip = loss_fn.interpolate_data(
            x_0=x_0, 
            x_1=clip_targets, 
            t=timesteps, 
            noise=noise
        )
        
        print(f"✅ Interpolation: noisy_clip requires_grad = {noisy_clip.requires_grad}")
        if not noisy_clip.requires_grad:
            print("❌ GRADIENT FLOW BROKEN at interpolation!")
            return False
        
        # Test model forward pass
        model_output = model(
            hidden_states=noisy_clip,
            timestep=timesteps,
            encoder_hidden_states=eva_embeddings,
            return_dict=False
        )
        
        print(f"✅ Model forward: output requires_grad = {model_output.requires_grad}")
        if not model_output.requires_grad:
            print("❌ GRADIENT FLOW BROKEN at model forward!")
            return False
        
        # Test loss computation
        loss, metrics = loss_fn(
            model_output=model_output,
            target_samples=clip_targets,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            noise=noise,
            return_metrics=True
        )
        
        print(f"✅ Loss computation: loss requires_grad = {loss.requires_grad}")
        print(f"✅ Loss value: {loss.item():.6f}")
        
        if not loss.requires_grad:
            print("❌ GRADIENT FLOW BROKEN at loss computation!")
            return False
        
        # Test backward pass
        loss.backward()
        
        # Check gradients
        grad_params = 0
        total_params = 0
        max_grad = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad_params += 1
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)
                if grad_params <= 3:  # Log first few
                    print(f"   Parameter {name}: grad_norm = {grad_norm:.6f}")
        
        print(f"✅ Gradient check: {grad_params}/{total_params} parameters have gradients")
        print(f"✅ Max gradient norm: {max_grad:.6f}")
        
        if grad_params == 0:
            print("❌ NO PARAMETERS HAVE GRADIENTS!")
            return False
        
        if max_grad < 1e-8:
            print("❌ GRADIENTS ARE TOO SMALL!")
            return False
        
        # Success!
        print("\n🎉 SUCCESS! Gradient flow is working correctly!")
        print("✅ All checks passed:")
        print("   • Model output requires gradients")
        print("   • Loss requires gradients")
        print("   • Backward pass works")
        print("   • Parameters receive gradients")
        print("   • Gradient magnitudes are reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_gradient_flow():
    """Test gradient flow using the actual trainer"""
    
    print("\n🧪 Testing trainer gradient flow...")
    
    try:
        from src.modules.trainers.blip3o_patch_trainer import BLIP3oPatchTrainer
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model
        from src.modules.config.blip3o_config import get_blip3o_patch_config
        from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        from transformers import TrainingArguments
        
        # Create components
        config = get_blip3o_patch_config("tiny")
        model = create_blip3o_patch_dit_model(config)
        loss_fn = create_blip3o_flow_matching_loss(enhanced=True)
        
        # Create minimal training args
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=2,
            num_train_epochs=1,
            logging_steps=1,
            save_steps=1000,
        )
        
        # Create trainer
        trainer = BLIP3oPatchTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=loss_fn,
        )
        
        # Create test input batch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        test_inputs = {
            'eva_embeddings': torch.randn(2, 256, 4096, device=device),
            'clip_embeddings': torch.randn(2, 256, 1024, device=device),
            'captions': ['test caption 1', 'test caption 2']
        }
        
        # Test compute_loss method
        loss = trainer.compute_loss(model, test_inputs)
        
        print(f"✅ Trainer compute_loss: loss = {loss.item():.6f}")
        print(f"✅ Trainer compute_loss: requires_grad = {loss.requires_grad}")
        
        if not loss.requires_grad:
            print("❌ TRAINER GRADIENT FLOW BROKEN!")
            return False
        
        print("✅ Trainer gradient flow working!")
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 BLIP3-o Gradient Flow Test (After Fixes)")
    print("=" * 50)
    
    # Test basic gradient flow
    basic_test = test_gradient_flow()
    
    # Test trainer gradient flow
    trainer_test = test_trainer_gradient_flow()
    
    print("\n" + "=" * 50)
    if basic_test and trainer_test:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your gradient flow fixes are working correctly")
        print("✅ You can now restart training safely")
        print("\nNext steps:")
        print("1. Stop your current training")
        print("2. Apply these fixes to your code")
        print("3. Restart training")
        print("4. The warnings should disappear")
    else:
        print("❌ SOME TESTS FAILED")
        print("❌ Review the error messages above")
        print("❌ Check your model and loss implementations")