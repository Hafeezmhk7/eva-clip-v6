#!/usr/bin/env python3
"""
Gradient Flow Test Script for BLIP3-o Patch-Level Training
gradient_flow_test.py

This script tests that all gradient flow fixes are working correctly.
Run this BEFORE starting training to verify everything is fixed.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import traceback
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_gradient_flow():
    """Test gradient flow through the complete BLIP3-o pipeline"""
    
    print("🧪 Testing BLIP3-o Gradient Flow")
    print("=" * 50)
    
    try:
        # Import fixed modules
        print("1. Testing module imports...")
        from src.modules.config.blip3o_config import get_blip3o_patch_config
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model
        from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        print("   ✅ All modules imported successfully")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   📱 Using device: {device}")
        
        # Create model
        print("\n2. Testing model creation...")
        config = get_blip3o_patch_config(model_size="tiny")  # Use tiny for testing
        model = create_blip3o_patch_dit_model(config=config)
        model = model.to(device)
        model.train()  # CRITICAL: Set to training mode
        print(f"   ✅ Model created with {model.get_num_parameters():,} parameters")
        print(f"   📊 Model training mode: {model.training}")
        
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   📈 Trainable parameters: {trainable_params:,}/{total_params:,}")
        
        if trainable_params == 0:
            raise RuntimeError("❌ No trainable parameters!")
        
        # Create loss function
        print("\n3. Testing loss function creation...")
        loss_fn = create_blip3o_flow_matching_loss(
            enhanced=True,
            use_contrastive_loss=True
        )
        print("   ✅ Loss function created")
        
        # Create test data
        print("\n4. Creating test data...")
        batch_size = 2
        eva_embeddings = torch.randn(batch_size, 256, 4096, device=device)  # EVA conditioning
        clip_embeddings = torch.randn(batch_size, 256, 1024, device=device)  # CLIP targets
        timesteps = torch.rand(batch_size, device=device)
        
        # Create noisy input with gradients
        x_0 = torch.randn_like(clip_embeddings, device=device, requires_grad=True)
        noise = torch.randn_like(clip_embeddings, device=device) * 0.1
        
        # Test interpolation
        print("\n5. Testing data interpolation...")
        noisy_clip = loss_fn.interpolate_data(
            x_0=x_0,
            x_1=clip_embeddings.detach(),
            t=timesteps,
            noise=noise
        )
        print(f"   ✅ Interpolation successful")
        print(f"   📊 Noisy input requires_grad: {noisy_clip.requires_grad}")
        
        if not noisy_clip.requires_grad:
            raise RuntimeError("❌ Interpolated input doesn't require gradients!")
        
        # Test model forward pass
        print("\n6. Testing model forward pass...")
        model_output = model(
            hidden_states=noisy_clip,
            timestep=timesteps,
            encoder_hidden_states=eva_embeddings.detach(),
            return_dict=False
        )
        print(f"   ✅ Model forward pass successful")
        print(f"   📊 Model output shape: {model_output.shape}")
        print(f"   📊 Model output requires_grad: {model_output.requires_grad}")
        print(f"   📊 Model output grad_fn: {model_output.grad_fn}")
        
        if not model_output.requires_grad:
            raise RuntimeError("❌ Model output doesn't require gradients!")
        
        # Test loss computation
        print("\n7. Testing loss computation...")
        loss, metrics = loss_fn(
            model_output=model_output,
            target_samples=clip_embeddings.detach(),
            timesteps=timesteps,
            eva_conditioning=eva_embeddings.detach(),
            noise=noise,
            return_metrics=True
        )
        print(f"   ✅ Loss computation successful")
        print(f"   📊 Loss value: {loss.item():.4f}")
        print(f"   📊 Loss requires_grad: {loss.requires_grad}")
        print(f"   📊 Loss grad_fn: {loss.grad_fn}")
        
        if not loss.requires_grad:
            raise RuntimeError("❌ Loss doesn't require gradients!")
        
        # Test backward pass
        print("\n8. Testing backward pass...")
        loss.backward()
        print("   ✅ Backward pass successful")
        
        # Check gradients
        print("\n9. Checking parameter gradients...")
        params_with_grad = 0
        params_without_grad = 0
        total_grad_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    total_grad_norm += param.grad.norm().item()
                else:
                    params_without_grad += 1
        
        print(f"   📊 Parameters with gradients: {params_with_grad}")
        print(f"   📊 Parameters without gradients: {params_without_grad}")
        print(f"   📊 Total gradient norm: {total_grad_norm:.6f}")
        
        if params_with_grad == 0:
            raise RuntimeError("❌ No parameters received gradients!")
        
        if total_grad_norm == 0:
            raise RuntimeError("❌ All gradients are zero!")
        
        # Test metrics
        print("\n10. Testing metrics...")
        if metrics:
            print(f"   📊 Available metrics: {list(metrics.keys())}")
            print(f"   📊 Gradient flow status: {metrics.get('gradient_flow_ok', 'unknown')}")
            
            key_metrics = ['flow_matching_loss', 'velocity_cosine_sim', 'global_cosine_sim']
            for metric in key_metrics:
                if metric in metrics:
                    print(f"   📊 {metric}: {metrics[metric]:.4f}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL GRADIENT FLOW TESTS PASSED!")
        print("✅ Model is ready for training")
        print("✅ Gradients flow correctly through entire pipeline")
        print("✅ Loss computation works properly")
        print("✅ Parameters will be updated during training")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ GRADIENT FLOW TEST FAILED!")
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\n" + "=" * 50)
        print("🔧 FIXES NEEDED:")
        print("1. Check that all fixed files are properly saved")
        print("2. Verify module imports are working")
        print("3. Check model is in training mode")
        print("4. Ensure parameters require gradients")
        print("=" * 50)
        
        return False

def test_trainer_integration():
    """Test integration with the trainer"""
    
    print("\n🧪 Testing Trainer Integration")
    print("=" * 50)
    
    try:
        # Import trainer
        from src.modules.trainers.blip3o_patch_trainer import BLIP3oPatchTrainer, create_blip3o_patch_training_args
        from transformers import TrainingArguments
        print("   ✅ Trainer imported successfully")
        
        # Create minimal training args for testing
        training_args = TrainingArguments(
            output_dir="./test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=1000,
            remove_unused_columns=False,
            report_to=[],
        )
        print("   ✅ Training arguments created")
        
        # Create model and loss
        from src.modules.config.blip3o_config import get_blip3o_patch_config
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model
        from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        
        config = get_blip3o_patch_config(model_size="tiny")
        model = create_blip3o_patch_dit_model(config=config)
        loss_fn = create_blip3o_flow_matching_loss()
        
        # Create trainer
        trainer = BLIP3oPatchTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=loss_fn,
            enable_recall_evaluation=False,  # Disable for testing
        )
        print("   ✅ Trainer created successfully")
        
        # Test compute_loss with sample data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        sample_inputs = {
            'eva_embeddings': torch.randn(1, 256, 4096, device=device),
            'clip_embeddings': torch.randn(1, 256, 1024, device=device),
            'captions': ['test caption']
        }
        
        print("   🧪 Testing trainer compute_loss...")
        loss, outputs = trainer.compute_loss(model, sample_inputs, return_outputs=True)
        
        print(f"   ✅ Trainer compute_loss successful")
        print(f"   📊 Loss: {loss.item():.4f}")
        print(f"   📊 Loss requires_grad: {loss.requires_grad}")
        
        if outputs:
            print(f"   📊 Gradient flow status: {outputs.get('gradient_flow_status', 'unknown')}")
            print(f"   📊 Model diagnostics available: {bool(outputs.get('model_diagnostics'))}")
        
        if not loss.requires_grad:
            raise RuntimeError("❌ Trainer loss doesn't require gradients!")
        
        print("   🎉 Trainer integration test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Trainer integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 BLIP3-o Gradient Flow Testing Suite")
    print("=" * 60)
    
    # Test 1: Basic gradient flow
    test1_passed = test_gradient_flow()
    
    # Test 2: Trainer integration
    test2_passed = test_trainer_integration()
    
    print("\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"✅ Gradient Flow Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Trainer Integration: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Ready to start BLIP3-o training!")
        print("\nNext steps:")
        print("1. Run your training script")
        print("2. Monitor for gradient flow warnings (should be none)")
        print("3. Check that loss decreases over time")
        print("4. Verify training metrics improve")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔧 Fix the issues before starting training")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)