#!/usr/bin/env python3
"""
Quick test script to verify EVA reproduction implementation
Run this locally before submitting cluster jobs
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_forward():
    """Test model forward pass"""
    print("Testing model forward pass...")
    
    from src.modules.models.blip3o_eva_dit import create_eva_reproduction_model
    
    # Create small model
    model = create_eva_reproduction_model(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        training_mode="patch_only",
        zero_init_output=True,
    )
    
    # Create dummy inputs
    batch_size = 2
    num_tokens = 256
    eva_dim = 4096
    clip_dim = 1024
    
    noisy_eva = torch.randn(batch_size, num_tokens, eva_dim)
    clip_cond = torch.randn(batch_size, num_tokens, clip_dim)
    timesteps = torch.rand(batch_size)
    
    # Forward pass
    output = model(
        hidden_states=noisy_eva,
        timestep=timesteps,
        encoder_hidden_states=clip_cond,
    )
    
    if isinstance(output, dict):
        velocity = output['velocity_prediction']
    else:
        velocity = output
    
    print(f"✅ Model forward pass successful")
    print(f"   Input shape: {noisy_eva.shape}")
    print(f"   Output shape: {velocity.shape}")
    print(f"   Output norm: {torch.norm(velocity, dim=-1).mean():.3f}")
    
    # Test generation
    with torch.no_grad():
        generated = model.generate(
            clip_features=clip_cond,
            num_inference_steps=10,
            normalize_output=True,
        )
    
    print(f"✅ Generation successful")
    print(f"   Generated shape: {generated.shape}")
    print(f"   Generated norm: {torch.norm(generated, dim=-1).mean():.3f}")
    
    return True

def test_loss_function():
    """Test loss computation"""
    print("\nTesting loss function...")
    
    from src.modules.losses.blip3o_eva_loss import create_eva_reproduction_loss
    
    # Create loss function
    loss_fn = create_eva_reproduction_loss(
        loss_scale=100.0,
        debug_mode=True,
    )
    
    # Create dummy data
    batch_size = 2
    num_tokens = 256
    eva_dim = 4096
    clip_dim = 1024
    
    # Normalized inputs
    model_output = F.normalize(torch.randn(batch_size, num_tokens, eva_dim), p=2, dim=-1)
    target_eva = F.normalize(torch.randn(batch_size, num_tokens, eva_dim), p=2, dim=-1)
    clip_cond = F.normalize(torch.randn(batch_size, num_tokens, clip_dim), p=2, dim=-1)
    timesteps = torch.rand(batch_size)
    
    # Compute loss
    loss, metrics = loss_fn(
        model_output=model_output,
        target_samples=target_eva,
        timesteps=timesteps,
        clip_conditioning=clip_cond,
        return_metrics=True,
    )
    
    print(f"✅ Loss computation successful")
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Scaled loss: {metrics['scaled_loss']:.6f}")
    print(f"   Velocity similarity: {metrics['velocity_similarity']:.4f}")
    print(f"   Prediction norm: {metrics['pred_norm']:.3f}")
    print(f"   Target norm: {metrics['velocity_norm']:.3f}")
    
    # Check gradients
    if loss.requires_grad:
        loss.backward()
        print(f"✅ Backward pass successful")
    
    return True

def test_overfitting_capability():
    """Test if model can overfit on single sample"""
    print("\nTesting overfitting capability...")
    
    from src.modules.models.blip3o_eva_dit import create_eva_reproduction_model
    from src.modules.losses.blip3o_eva_loss import create_eva_reproduction_loss
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create small model
    model = create_eva_reproduction_model(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        zero_init_output=True,
    ).to(device)
    
    # Create loss
    loss_fn = create_eva_reproduction_loss(loss_scale=100.0)
    
    # Create single sample
    eva_target = F.normalize(torch.randn(1, 256, 4096), p=2, dim=-1).to(device)
    clip_cond = F.normalize(torch.randn(1, 256, 1024), p=2, dim=-1).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    print("Training on single sample for 100 steps...")
    model.train()
    
    losses = []
    similarities = []
    
    for step in range(100):
        # Random timestep
        t = torch.rand(1).to(device)
        
        # Create noisy input
        noise = F.normalize(torch.randn_like(eva_target), p=2, dim=-1)
        noisy_eva = (1 - t.view(1, 1, 1)) * noise + t.view(1, 1, 1) * eva_target
        
        # Forward pass
        velocity_pred = model(
            hidden_states=noisy_eva,
            timestep=t,
            encoder_hidden_states=clip_cond,
            return_dict=False
        )
        
        # Compute loss
        loss, metrics = loss_fn(
            model_output=velocity_pred,
            target_samples=eva_target,
            timesteps=t,
            clip_conditioning=clip_cond,
            noise=noise,
            return_metrics=True,
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        similarities.append(metrics['velocity_similarity'])
        
        if step % 20 == 0:
            print(f"   Step {step}: Loss={loss.item():.4f}, Similarity={metrics['velocity_similarity']:.4f}")
    
    # Check improvement
    initial_loss = losses[0]
    final_loss = losses[-1]
    initial_sim = similarities[0]
    final_sim = similarities[-1]
    
    print(f"\n📊 Overfitting Test Results:")
    print(f"   Initial Loss: {initial_loss:.4f} → Final Loss: {final_loss:.4f}")
    print(f"   Initial Sim: {initial_sim:.4f} → Final Sim: {final_sim:.4f}")
    
    if final_loss < initial_loss * 0.5 and final_sim > 0.3:
        print(f"✅ Model can learn! Architecture seems correct.")
        return True
    else:
        print(f"⚠️ Model struggling to learn. Check implementation.")
        return False

def main():
    """Run all tests"""
    print("🔬 EVA Reproduction Implementation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Model forward pass
    try:
        if test_model_forward():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Loss function
    try:
        if test_loss_function():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Overfitting capability
    try:
        if torch.cuda.is_available():
            if test_overfitting_capability():
                tests_passed += 1
        else:
            print("\n⚠️ Skipping overfitting test (no GPU available)")
            total_tests -= 1
    except Exception as e:
        print(f"❌ Overfitting test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Ready for training.")
    else:
        print("⚠️ Some tests failed. Fix issues before training.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()