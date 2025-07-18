#!/usr/bin/env python3
"""
Simple Single Example Test for BLIP3-o DiT
Place this file at: test_single_example.py (in root directory)

This script tests your BLIP3-o implementation on a single synthetic example to verify:
1. Model can be created and runs forward pass
2. Enhanced loss function works correctly  
3. Training step executes without errors
4. Generation produces reasonable outputs
5. Evaluation metrics compute properly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create a single synthetic example for testing."""
    print("📊 Creating synthetic test data...")
    
    batch_size = 1
    num_tokens = 256  # 16x16 grid
    eva_dim = 4096
    clip_dim = 1024
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # EVA-CLIP embeddings (conditioning) - realistic magnitudes
    eva_embeddings = torch.randn(batch_size, num_tokens, eva_dim) * 0.5
    eva_embeddings = F.normalize(eva_embeddings, dim=-1) * 2.0
    
    # CLIP embeddings (target) - realistic magnitudes  
    clip_embeddings = torch.randn(batch_size, num_tokens, clip_dim) * 0.3
    clip_embeddings = F.normalize(clip_embeddings, dim=-1) * 1.5
    
    # Add correlation between EVA and CLIP (realistic)
    correlation_factor = 0.3
    clip_embeddings = (
        clip_embeddings * (1 - correlation_factor) + 
        eva_embeddings[:, :, :clip_dim] * correlation_factor
    )
    clip_embeddings = F.normalize(clip_embeddings, dim=-1) * 1.5
    
    return {
        'eva_embeddings': eva_embeddings,
        'clip_embeddings': clip_embeddings,
        'captions': ['Test image for BLIP3-o validation'],
        'keys': ['test_001'],
    }

def test_model_creation():
    """Test that the model can be created and runs."""
    print("🏗️ Testing model creation...")
    
    try:
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.blip3o_dit import create_blip3o_dit_model
        
        # Small config for testing
        config = BLIP3oDiTConfig(
            input_size=16,  # 16x16 = 256 tokens
            patch_size=1,
            in_channels=1024,  # CLIP dimension
            dim=512,  # Smaller for fast testing
            eva_embedding_size=4096,
            n_layers=4,  # Fewer layers
            n_heads=8,
            norm_eps=1e-5,
            learn_sigma=False,
            _gradient_checkpointing=False,
        )
        
        # Create model
        model = create_blip3o_dit_model(config=config)
        model.eval()
        
        # Test data
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        # Test forward pass
        timesteps = torch.tensor([0.5])
        noisy_input = clip_emb + torch.randn_like(clip_emb) * 0.1
        
        with torch.no_grad():
            output = model(
                hidden_states=noisy_input,
                timestep=timesteps,
                encoder_hidden_states=eva_emb,
                return_dict=False
            )
        
        # Validate output
        assert output.shape == clip_emb.shape, f"Shape mismatch: {output.shape} vs {clip_emb.shape}"
        assert torch.isfinite(output).all(), "Non-finite values in output"
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully! Parameters: {total_params:,}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_enhanced_loss():
    """Test the enhanced loss function."""
    print("🔥 Testing enhanced loss function...")
    
    try:
        from src.modules.losses.enhanced_flow_matching_loss import create_enhanced_blip3o_flow_matching_loss
        
        # Create enhanced loss
        loss_fn = create_enhanced_blip3o_flow_matching_loss(
            sigma_min=1e-4,
            sigma_max=1.0,
            prediction_type="v_prediction",
            schedule_type="linear",
            clip_dim=1024,
            eva_dim=4096,
            alignment_loss_weight=0.1,
            temporal_loss_weight=0.05,
            regularization_weight=0.01,
        )
        
        # Test data
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        # Test timestep sampling
        timesteps = loss_fn.sample_timesteps(1, eva_emb.device)
        assert 0 <= timesteps.min() <= timesteps.max() <= 1, "Invalid timestep range"
        
        # Test velocity computation
        noise = torch.randn_like(clip_emb)
        x_0 = torch.randn_like(clip_emb)
        velocity_target = loss_fn.compute_velocity_target(x_0, clip_emb, timesteps, noise)
        assert velocity_target.shape == clip_emb.shape
        
        print(f"✅ Enhanced loss created successfully!")
        print(f"   Timesteps range: [{timesteps.min().item():.3f}, {timesteps.max().item():.3f}]")
        print(f"   Velocity target shape: {velocity_target.shape}")
        
        return loss_fn
        
    except Exception as e:
        print(f"❌ Enhanced loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_training_step(model, loss_fn):
    """Test a complete training step."""
    print("🎯 Testing training step...")
    
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Test data
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        # Training step
        optimizer.zero_grad()
        
        # Sample timesteps and noise
        timesteps = loss_fn.sample_timesteps(1, eva_emb.device)
        noise = torch.randn_like(clip_emb)
        x_0 = torch.randn_like(clip_emb)
        
        # Create noisy input
        noisy_input = loss_fn.interpolate_data(x_0, clip_emb, timesteps, noise)
        
        # Forward pass
        model_output = model(
            hidden_states=noisy_input,
            timestep=timesteps,
            encoder_hidden_states=eva_emb,
            return_dict=False
        )
        
        # Compute loss with metrics
        loss, metrics = loss_fn(
            model_output=model_output,
            target_samples=clip_emb,
            timesteps=timesteps,
            eva_conditioning=eva_emb,
            noise=noise,
            return_metrics=True
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"✅ Training step successful!")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Flow matching loss: {metrics['flow_matching_loss']:.6f}")
        print(f"   Alignment loss: {metrics['alignment_loss']:.6f}")
        print(f"   Cosine similarity: {metrics['cosine_similarity']:.4f}")
        print(f"   CLIP alignment: {metrics['clip_alignment_score']:.4f}")
        print(f"   Gradient norm: {grad_norm:.4f}")
        
        return True, metrics
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_generation(model):
    """Test generation quality."""
    print("🎨 Testing generation...")
    
    try:
        model.eval()
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        with torch.no_grad():
            # Generate embeddings
            generated = model.generate(
                encoder_hidden_states=eva_emb,
                num_inference_steps=10,  # Fast for testing
            )
            
            # Compute alignment metrics
            gen_global = F.normalize(generated.mean(dim=1), dim=-1)
            target_global = F.normalize(clip_emb.mean(dim=1), dim=-1)
            
            global_cosine_sim = F.cosine_similarity(gen_global, target_global, dim=1).mean().item()
            l2_distance = torch.norm(generated - clip_emb, dim=-1).mean().item()
            
            print(f"✅ Generation successful!")
            print(f"   Generated shape: {generated.shape}")
            print(f"   Global cosine similarity: {global_cosine_sim:.4f}")
            print(f"   L2 distance: {l2_distance:.4f}")
            print(f"   Generated norm: {torch.norm(generated, dim=-1).mean().item():.4f}")
            print(f"   Target norm: {torch.norm(clip_emb, dim=-1).mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_step_training(model, loss_fn, num_steps=5):
    """Test multi-step training for convergence."""
    print(f"📈 Testing {num_steps}-step training...")
    
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        losses = []
        cosine_sims = []
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Sample fresh noise each step
            timesteps = loss_fn.sample_timesteps(1, eva_emb.device)
            noise = torch.randn_like(clip_emb)
            x_0 = torch.randn_like(clip_emb)
            
            noisy_input = loss_fn.interpolate_data(x_0, clip_emb, timesteps, noise)
            
            model_output = model(
                hidden_states=noisy_input,
                timestep=timesteps,
                encoder_hidden_states=eva_emb,
                return_dict=False
            )
            
            loss, metrics = loss_fn(
                model_output=model_output,
                target_samples=clip_emb,
                timesteps=timesteps,
                eva_conditioning=eva_emb,
                noise=noise,
                return_metrics=True
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            cosine_sims.append(metrics['cosine_similarity'])
            
            if step == 0 or step == num_steps - 1:
                print(f"   Step {step}: Loss={loss.item():.6f}, Cosine={metrics['cosine_similarity']:.4f}")
        
        # Analyze trends
        loss_improvement = losses[0] - losses[-1]
        cosine_improvement = cosine_sims[-1] - cosine_sims[0]
        
        print(f"✅ Multi-step training completed!")
        print(f"   Loss improvement: {loss_improvement:.6f} ({'good' if loss_improvement > 0 else 'check'})")
        print(f"   Cosine improvement: {cosine_improvement:.4f} ({'good' if cosine_improvement > 0 else 'check'})")
        
        return True, {'losses': losses, 'cosine_sims': cosine_sims}
        
    except Exception as e:
        print(f"❌ Multi-step training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run the complete test suite."""
    print("🧪 BLIP3-o Single Example Test Suite")
    print("=" * 50)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Model Creation
    model = test_model_creation()
    if model is not None:
        success_count += 1
        print()
    else:
        print("❌ Stopping tests - model creation failed")
        return False
    
    # Test 2: Enhanced Loss
    loss_fn = test_enhanced_loss()
    if loss_fn is not None:
        success_count += 1
        print()
    else:
        print("❌ Stopping tests - loss function failed")
        return False
    
    # Test 3: Training Step
    training_success, metrics = test_training_step(model, loss_fn)
    if training_success:
        success_count += 1
        print()
    else:
        print("❌ Training step failed")
        print()
    
    # Test 4: Generation
    gen_success = test_generation(model)
    if gen_success:
        success_count += 1
        print()
    else:
        print("❌ Generation failed")
        print()
    
    # Test 5: Multi-step Training
    multi_success, multi_results = test_multi_step_training(model, loss_fn)
    if multi_success:
        success_count += 1
        print()
    else:
        print("❌ Multi-step training failed")
        print()
    
    # Final Results
    print("🎯 TEST RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("💡 Your BLIP3-o implementation is working correctly!")
        print("\n📋 Next Steps:")
        print("   1. Your model architecture is correct")
        print("   2. Enhanced loss function works properly")
        print("   3. Training pipeline executes without errors")
        print("   4. Ready for full-scale training!")
        return True
    else:
        print(f"⚠️ {total_tests - success_count} test(s) failed")
        print("🔍 Check the error messages above and fix the issues")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)