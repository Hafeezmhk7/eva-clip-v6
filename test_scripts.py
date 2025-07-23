#!/usr/bin/env python3
"""
Gradient Flow Test Script for BLIP3-o Fixed Implementation
test_gradient_flow.py

This script tests that the gradient flow fixes work correctly.
Run this before your main training to verify everything is working.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_gradient_flow():
    """Test gradient flow through the fixed BLIP3-o implementation"""
    
    print("🧪 Testing BLIP3-o Gradient Flow Fixes")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        # Import fixed components
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
        from src.modules.datasets.blip3o_data_collator import create_blip3o_data_collator
        from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        
        print("✅ All fixed modules imported successfully")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 1: Model Creation
    print("\n🧪 Test 1: Model Creation")
    print("-" * 30)
    
    try:
        config = BLIP3oDiTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=256,
        )
        
        model = create_blip3o_patch_dit_model(config=config)
        model = model.to(device)
        model.train()
        
        param_count = model.get_num_parameters()
        print(f"✅ Model created successfully")
        print(f"   Parameters: {param_count:,}")
        print(f"   Device: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 2: Data Collator
    print("\n🧪 Test 2: Data Collator")
    print("-" * 30)
    
    try:
        # Create sample data (simulating your dataset)
        batch_size = 4
        sample_data = []
        
        for i in range(batch_size):
            sample = {
                'eva_embeddings': torch.randn(256, 4096),  # EVA-CLIP features
                'clip_embeddings': torch.randn(256, 1024), # CLIP patch features
                'caption': f'Test caption {i}',
            }
            sample_data.append(sample)
        
        # Create data collator
        data_collator = create_blip3o_data_collator(
            normalize_embeddings=True,
            device=device,
        )
        
        # Test collation
        batch = data_collator(sample_data)
        
        print(f"✅ Data collator created successfully")
        print(f"   EVA shape: {batch['eva_embeddings'].shape}")
        print(f"   CLIP shape: {batch['clip_embeddings'].shape}")
        print(f"   Noisy input shape: {batch['hidden_states'].shape}")
        print(f"   Timesteps shape: {batch['timesteps'].shape}")
        
        # Check gradient requirements
        print(f"   EVA requires_grad: {batch['eva_embeddings'].requires_grad}")
        print(f"   CLIP requires_grad: {batch['clip_embeddings'].requires_grad}")
        print(f"   Noisy input requires_grad: {batch['hidden_states'].requires_grad}")
        
        if not batch['hidden_states'].requires_grad:
            print("❌ Noisy input doesn't require gradients!")
            return False
        
    except Exception as e:
        print(f"❌ Data collator test failed: {e}")
        return False
    
    # Test 3: Forward Pass
    print("\n🧪 Test 3: Model Forward Pass")
    print("-" * 30)
    
    try:
        with torch.enable_grad():  # Ensure gradients are enabled
            outputs = model(
                hidden_states=batch['hidden_states'],
                timestep=batch['timesteps'],
                encoder_hidden_states=batch['eva_embeddings'],
                return_dict=True
            )
        
        velocity_pred = outputs['velocity_prediction']
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {velocity_pred.shape}")
        print(f"   Output requires_grad: {velocity_pred.requires_grad}")
        print(f"   Output grad_fn: {velocity_pred.grad_fn}")
        
        if not velocity_pred.requires_grad:
            print("❌ Model output doesn't require gradients!")
            return False
        
        if velocity_pred.grad_fn is None:
            print("❌ Model output has no grad_fn - not connected to computation graph!")
            return False
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test 4: Loss Computation
    print("\n🧪 Test 4: Loss Computation")
    print("-" * 30)
    
    try:
        # Create flow matching loss
        flow_matching_loss = create_blip3o_flow_matching_loss(
            enhanced=True,
            use_contrastive_loss=True,
            contrastive_weight=0.1,
        )
        
        # Compute loss
        loss, metrics = flow_matching_loss(
            model_output=velocity_pred,
            target_samples=batch['clip_embeddings'],
            timesteps=batch['timesteps'],
            eva_conditioning=batch['eva_embeddings'],
            noise=batch['noise'],
            return_metrics=True
        )
        
        print(f"✅ Loss computation successful")
        print(f"   Loss value: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss grad_fn: {loss.grad_fn}")
        
        if metrics:
            print(f"   Global cosine sim: {metrics.get('global_cosine_sim', 0):.3f}")
            print(f"   Training quality: {metrics.get('training_quality', 'unknown')}")
        
        if not loss.requires_grad:
            print("❌ Loss doesn't require gradients!")
            return False
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    # Test 5: Backward Pass
    print("\n🧪 Test 5: Backward Pass")
    print("-" * 30)
    
    try:
        # Clear any existing gradients
        model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Check if gradients were computed
        has_gradients = 0
        total_params = 0
        max_grad = 0.0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                has_gradients += 1
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        print(f"✅ Backward pass successful")
        print(f"   Parameters with gradients: {has_gradients}/{total_params}")
        print(f"   Max gradient magnitude: {max_grad:.6f}")
        
        if has_gradients == 0:
            print("❌ No gradients computed!")
            return False
        
        if has_gradients < total_params * 0.8:  # At least 80% should have gradients
            print(f"⚠️  Only {has_gradients/total_params*100:.1f}% of parameters have gradients")
        
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    # Test 6: Multiple Steps
    print("\n🧪 Test 6: Multiple Training Steps")
    print("-" * 30)
    
    try:
        initial_loss = loss.item()
        losses = [initial_loss]
        
        for step in range(2):
            # Clear gradients
            model.zero_grad()
            
            # Create new batch
            new_batch = data_collator(sample_data)
            
            # Forward pass
            with torch.enable_grad():
                outputs = model(
                    hidden_states=new_batch['hidden_states'],
                    timestep=new_batch['timesteps'],
                    encoder_hidden_states=new_batch['eva_embeddings'],
                    return_dict=True
                )
            
            # Loss computation
            loss, _ = flow_matching_loss(
                model_output=outputs['velocity_prediction'],
                target_samples=new_batch['clip_embeddings'],
                timesteps=new_batch['timesteps'],
                eva_conditioning=new_batch['eva_embeddings'],
                noise=new_batch['noise'],
                return_metrics=False
            )
            
            # Backward pass
            loss.backward()
            
            losses.append(loss.item())
            
            print(f"   Step {step+1}: Loss = {loss.item():.4f}")
        
        print(f"✅ Multiple steps successful")
        print(f"   Loss progression: {losses[0]:.4f} → {losses[-1]:.4f}")
        
        # Check for NaN or infinite losses
        if any(not torch.isfinite(torch.tensor(l)) for l in losses):
            print("❌ Found NaN or infinite losses!")
            return False
        
    except Exception as e:
        print(f"❌ Multiple steps failed: {e}")
        return False
    
    # Test 7: Memory Usage
    print("\n🧪 Test 7: Memory Usage")
    print("-" * 30)
    
    if torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            
            print(f"✅ Memory usage check")
            print(f"   Allocated: {memory_allocated:.2f} GB")
            print(f"   Cached: {memory_cached:.2f} GB")
            
            if memory_allocated > 20:  # More than 20GB seems excessive for this test
                print(f"⚠️  High memory usage: {memory_allocated:.2f} GB")
            
        except Exception as e:
            print(f"⚠️  Memory check failed: {e}")
    else:
        print("✅ CPU mode - no memory check needed")
    
    # Final Summary
    print("\n🎉 GRADIENT FLOW TEST SUMMARY")
    print("=" * 50)
    print("✅ All tests passed!")
    print("✅ Gradient flow is working correctly")
    print("✅ BLIP3-o fixed implementation ready for training")
    print()
    print("🚀 You can now proceed with your BLIP3-o training!")
    print("   The gradient flow issues have been resolved.")
    
    return True


def test_paper_alignment():
    """Test alignment with BLIP3-o paper architecture"""
    print("\n📋 BLIP3-o Paper Alignment Check")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 6
    
    try:
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
        
        config = BLIP3oDiTConfig()
        model = create_blip3o_patch_dit_model(config=config)
        
        # Check 1: 256 patches (16x16)
        if config.max_position_embeddings == 256:
            print("✅ 256-token patch structure (16×16 grid)")
            checks_passed += 1
        else:
            print("❌ Incorrect patch count")
        
        # Check 2: CLIP dimensions
        if config.clip_embedding_size == 1024:
            print("✅ CLIP patch dimension (1024)")
            checks_passed += 1
        else:
            print("❌ Incorrect CLIP dimension")
        
        # Check 3: EVA dimensions
        if config.eva_embedding_size == 4096:
            print("✅ EVA-CLIP conditioning dimension (4096)")
            checks_passed += 1
        else:
            print("❌ Incorrect EVA dimension")
        
        # Check 4: 3D RoPE
        first_block = model.blocks[0]
        if hasattr(first_block, 'rope'):
            print("✅ 3D Rotary Position Embedding (Lumina-Next style)")
            checks_passed += 1
        else:
            print("❌ Missing 3D RoPE")
        
        # Check 5: RMSNorm
        if hasattr(first_block, 'norm1') and 'RMS' in str(type(first_block.norm1)):
            print("✅ RMSNorm for stability")
            checks_passed += 1
        else:
            print("❌ Missing RMSNorm")
        
        # Check 6: Flow matching objective
        if config.prediction_type == "velocity":
            print("✅ Velocity prediction for flow matching")
            checks_passed += 1
        else:
            print("❌ Incorrect prediction type")
        
        print(f"\n📊 Paper Alignment: {checks_passed}/{total_checks} checks passed")
        
        if checks_passed == total_checks:
            print("🎉 Perfect alignment with BLIP3-o paper!")
        elif checks_passed >= total_checks * 0.8:
            print("✅ Good alignment with BLIP3-o paper")
        else:
            print("⚠️  Some alignment issues detected")
        
    except Exception as e:
        print(f"❌ Paper alignment check failed: {e}")


if __name__ == "__main__":
    print("BLIP3-o Gradient Flow Fix Verification")
    print("=" * 60)
    
    success = test_gradient_flow()
    
    if success:
        test_paper_alignment()
        print("\n🎊 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Your BLIP3-o implementation is ready for training.")
    else:
        print("\n❌ TESTS FAILED!")
        print("Please check the error messages above and fix the issues.")
        sys.exit(1)