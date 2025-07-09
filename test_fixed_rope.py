#!/usr/bin/env python3
"""
Test script for the FIXED BLIP3-o implementation with proper 3D RoPE.
This script validates that the 3D RoPE implementation works correctly.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_3d_rope_utilities():
    """Test the 3D RoPE utility functions."""
    print("🧪 Testing 3D RoPE utilities...")
    
    # Import the utility functions
    from src.modules.models.blip3o_dit import get_3d_rotary_pos_embed, apply_rotary_pos_emb
    
    # Test 3D RoPE creation - FIXED: use head_dim (64), not full model dim
    head_dim = 64   # Head dimension, must be divisible by 4
    grid_size = 8   # 8x8 = 64 tokens
    
    cos_emb, sin_emb = get_3d_rotary_pos_embed(head_dim, grid_size)
    
    print(f"   ✅ Created 3D RoPE embeddings:")
    print(f"      cos_emb shape: {cos_emb.shape}")  # Should be [1, 64, 32] (where 32 = head_dim//2)
    print(f"      sin_emb shape: {sin_emb.shape}")  # Should be [1, 64, 32]
    
    # Test RoPE application
    batch_size = 2
    seq_len = 64
    num_heads = 8
    head_dim = 64
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)
    
    print(f"   ✅ Applied RoPE to Q, K:")
    print(f"      q_rot shape: {q_rot.shape}")  # Should be [2, 64, 8, 64]
    print(f"      k_rot shape: {k_rot.shape}")  # Should be [2, 64, 8, 64]
    
    # Verify no dimension mismatches
    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"
    
    print(f"   ✅ Dimension validation passed")
    
    return True

def test_fixed_model():
    """Test the fixed BLIP3-o model with proper 3D RoPE."""
    print("🧪 Testing FIXED BLIP3-o model...")
    
    from src.modules.config.blip3o_config import BLIP3oDiTConfig
    from src.modules.models.blip3o_dit import BLIP3oDiTModel
    
    # Create a configuration that ensures 3D RoPE compatibility
    config = BLIP3oDiTConfig(
        input_size=8,                    # 8x8 = 64 tokens
        patch_size=1,                    # Pre-tokenized
        in_channels=1024,                # CLIP dimension
        dim=512,                         # Hidden dimension (divisible by heads)
        eva_embedding_size=4096,         # EVA-CLIP dimension
        n_layers=4,                      # Fewer layers for testing
        n_heads=8,                       # 8 heads -> head_dim = 64 (divisible by 4)
        n_kv_heads=8,                    # Same as n_heads
        multiple_of=256,                 # FFN multiple
        norm_eps=1e-5,                   # Normalization
        qk_norm=True,                    # Query-key norm
        learn_sigma=False,               # Flow matching
        _gradient_checkpointing=False,   # Disable for testing
    )
    
    print(f"   📐 Model config:")
    print(f"      dim: {config.dim}")
    print(f"      n_heads: {config.n_heads}")
    print(f"      head_dim: {config.dim // config.n_heads}")
    print(f"      head_dim % 4: {(config.dim // config.n_heads) % 4}")
    
    # Create model
    model = BLIP3oDiTModel(config)
    
    print(f"   ✅ Model created successfully")
    print(f"      Parameters: {model.get_num_parameters():,}")
    print(f"      Head dim: {model.head_dim} (divisible by 4: {model.head_dim % 4 == 0})")
    
    return model, config

def test_fixed_forward_pass():
    """Test that the fixed forward pass works without tensor shape errors."""
    print("🧪 Testing FIXED forward pass...")
    
    # Get model
    model, config = test_fixed_model()
    
    device = torch.device("cpu")  # Use CPU for testing
    model = model.to(device)
    model.eval()
    
    # Create test inputs
    batch_size = 2
    eva_embeddings = torch.randn(batch_size, 64, 4096, device=device)  # EVA-CLIP
    clip_embeddings = torch.randn(batch_size, 64, 1024, device=device) # CLIP targets
    timesteps = torch.rand(batch_size, device=device)                  # Flow matching times
    
    print(f"   📊 Input shapes:")
    print(f"      EVA embeddings: {eva_embeddings.shape}")
    print(f"      CLIP embeddings: {clip_embeddings.shape}")
    print(f"      Timesteps: {timesteps.shape}")
    
    try:
        # Forward pass
        with torch.no_grad():
            output = model(
                hidden_states=clip_embeddings,
                timestep=timesteps,
                encoder_hidden_states=eva_embeddings,
                return_dict=False
            )
        
        print(f"   ✅ Forward pass successful!")
        print(f"      Output shape: {output.shape}")
        print(f"      Expected shape: {clip_embeddings.shape}")
        
        # Verify output shape
        assert output.shape == clip_embeddings.shape, f"Output shape mismatch: {output.shape} vs {clip_embeddings.shape}"
        
        # Test that output is reasonable
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        print(f"   ✅ Output validation passed")
        print(f"      Output mean: {output.mean().item():.4f}")
        print(f"      Output std: {output.std().item():.4f}")
        print(f"      Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation():
    """Test generation functionality."""
    print("🧪 Testing generation...")
    
    # Get model
    model, config = test_fixed_model()
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Create test inputs
    batch_size = 1
    eva_embeddings = torch.randn(batch_size, 64, 4096, device=device)
    
    print(f"   📊 Generation inputs:")
    print(f"      EVA embeddings: {eva_embeddings.shape}")
    
    try:
        # Generate samples
        with torch.no_grad():
            generated = model.generate(
                encoder_hidden_states=eva_embeddings,
                num_inference_steps=10,  # Few steps for testing
                return_intermediate=False
            )
        
        print(f"   ✅ Generation successful!")
        print(f"      Generated shape: {generated.shape}")
        print(f"      Expected shape: [{batch_size}, 64, {config.in_channels}]")
        
        # Verify generation
        expected_shape = (batch_size, 64, config.in_channels)
        assert generated.shape == expected_shape, f"Generated shape mismatch: {generated.shape} vs {expected_shape}"
        
        # Test reasonableness
        assert not torch.isnan(generated).any(), "Generated output contains NaN"
        assert torch.isfinite(generated).all(), "Generated output contains infinite values"
        
        print(f"   ✅ Generation validation passed")
        print(f"      Generated mean: {generated.mean().item():.4f}")
        print(f"      Generated std: {generated.std().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_loss():
    """Test model with loss computation."""
    print("🧪 Testing with loss computation...")
    
    from src.modules.losses.flow_matching_loss import create_blip3o_flow_matching_loss
    
    # Get model
    model, config = test_fixed_model()
    
    device = torch.device("cpu")
    model = model.to(device)
    model.train()  # Set to training mode
    
    # Create loss function
    flow_loss = create_blip3o_flow_matching_loss()
    
    # Create test inputs
    batch_size = 2
    eva_embeddings = torch.randn(batch_size, 64, 4096, device=device)
    clip_embeddings = torch.randn(batch_size, 64, 1024, device=device)
    timesteps = torch.rand(batch_size, device=device)
    noise = torch.randn_like(clip_embeddings)
    
    print(f"   📊 Training setup:")
    print(f"      Batch size: {batch_size}")
    print(f"      Model parameters: {model.get_num_parameters():,}")
    
    try:
        # Forward pass
        output = model(
            hidden_states=clip_embeddings,
            timestep=timesteps,
            encoder_hidden_states=eva_embeddings,
            return_dict=False
        )
        
        # Compute loss
        loss, metrics = flow_loss(
            model_output=output,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            noise=noise,
            return_metrics=True
        )
        
        print(f"   ✅ Loss computation successful!")
        print(f"      Loss value: {loss.item():.4f}")
        
        if metrics:
            print(f"      Available metrics: {list(metrics.keys())}")
            if 'cosine_similarity' in metrics:
                print(f"      Cosine similarity: {metrics['cosine_similarity']:.4f}")
            if 'snr_db' in metrics:
                print(f"      SNR: {metrics['snr_db']:.1f} dB")
        
        # Test backpropagation
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"   ✅ Backpropagation successful!")
        print(f"      Total gradient norm: {total_grad_norm:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 BLIP3-o FIXED Implementation Test (v2)")
    print("=" * 60)
    
    tests = [
        ("3D RoPE Utilities", test_3d_rope_utilities),
        ("Fixed Model Creation", lambda: test_fixed_model() is not None),
        ("Fixed Forward Pass", test_fixed_forward_pass),
        ("Generation", test_generation),
        ("Loss Computation", test_with_loss),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            if success:
                print(f"   ✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"   ❌ {test_name} FAILED")
        except Exception as e:
            print(f"   ❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The FIXED BLIP3-o implementation with proper 3D RoPE is working correctly")
        print("\n📋 Next steps:")
        print("   1. Replace your blip3o_dit.py with the fixed version")
        print("   2. Run: python train_blip3o_dit.py --debug")
        print("   3. The 3D RoPE should now work without tensor shape errors")
        print("   4. No more fallback needed - proper 3D RoPE is implemented!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\n🔧 If you're still seeing dimension errors:")
        print("   1. Make sure head_dim is divisible by 4")
        print("   2. Check that RoPE embed_dim matches head_dim") 
        print("   3. Verify tensor shapes in apply_rotary_pos_emb")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)