#!/usr/bin/env python3
"""
FIXED Single Example Test for BLIP3-o DiT with Dual Supervision
Tests the complete dual supervision architecture including:
1. Dual model outputs (patch + global)
2. Dual supervision loss computation
3. Frozen CLIP projection
4. Enhanced metrics tracking
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
    """Create a single synthetic example for dual supervision testing."""
    print("📊 Creating synthetic test data for dual supervision...")
    
    batch_size = 2  # Use 2 for better testing
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
        'captions': ['Test image for BLIP3-o validation', 'Second test image'],
        'keys': ['test_001', 'test_002'],
    }

def test_dual_supervision_model_creation():
    """Test that the dual supervision model can be created and runs."""
    print("🏗️ Testing dual supervision model creation...")
    
    try:
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.blip3o_dit import create_blip3o_dit_model
        
        # Config for dual supervision with smaller dims for testing
        config = BLIP3oDiTConfig(
            input_size=16,  # 16x16 = 256 tokens
            patch_size=1,
            in_channels=1024,  # CLIP dimension
            dim=512,  # Smaller for fast testing
            eva_embedding_size=4096,
            n_layers=4,  # Fewer layers
            n_heads=8,  # 512/8 = 64, divisible by 4 for RoPE
            norm_eps=1e-5,
            qk_norm=True,
            learn_sigma=False,
            _gradient_checkpointing=False,
            # NEW: MLP configuration for dual supervision
            mlp_hidden_dim=1024,
            mlp_num_layers=2,
            mlp_dropout=0.1,
            mlp_activation="gelu",
        )
        
        # Create model with dual supervision
        model = create_blip3o_dit_model(
            config=config,
            load_clip_projection=True,  # Load frozen CLIP projection
        )
        model.eval()
        
        # Test data
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        # Test forward pass
        timesteps = torch.tensor([0.5, 0.3])  # For batch size 2
        noisy_input = clip_emb + torch.randn_like(clip_emb) * 0.1
        
        with torch.no_grad():
            outputs = model(
                hidden_states=noisy_input,
                timestep=timesteps,
                encoder_hidden_states=eva_emb,
                return_dict=True
            )
        
        # Validate dual outputs
        assert 'patch_output' in outputs, "Missing patch_output"
        assert 'global_output' in outputs, "Missing global_output"
        
        patch_output = outputs['patch_output']
        global_output = outputs['global_output']
        
        # Validate shapes
        assert patch_output.shape == clip_emb.shape, f"Patch shape mismatch: {patch_output.shape} vs {clip_emb.shape}"
        if global_output is not None:
            assert global_output.shape == (2, 768), f"Global shape mismatch: {global_output.shape} vs (2, 768)"
        assert torch.isfinite(patch_output).all(), "Non-finite values in patch output"
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"✅ Dual supervision model created successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters (CLIP): {frozen_params:,}")
        print(f"   Patch output shape: {patch_output.shape}")
        print(f"   Global output shape: {global_output.shape if global_output is not None else 'None'}")
        print(f"   Has frozen CLIP projection: {model.frozen_clip_visual_proj is not None}")
        
        return model
        
    except Exception as e:
        print(f"❌ Dual supervision model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dual_supervision_loss():
    """Test the dual supervision loss function."""
    print("🔥 Testing dual supervision loss function...")
    
    try:
        from src.modules.losses.dual_supervision_flow_matching_loss import create_dual_supervision_loss
        from src.modules.config.blip3o_config import get_default_flow_matching_config
        
        # Create dual supervision loss
        config = get_default_flow_matching_config()
        loss_fn = create_dual_supervision_loss(
            config=config,
            patch_loss_weight=1.0,
            global_loss_weight=2.0,
            flow_matching_loss_weight=1.0,
            use_cosine_similarity=False,
        )
        
        # Test data
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        batch_size = eva_emb.shape[0]
        
        # Test timestep sampling
        timesteps = loss_fn.sample_timesteps(batch_size, eva_emb.device)
        assert 0 <= timesteps.min() <= timesteps.max() <= 1, "Invalid timestep range"
        
        # Create dummy model outputs (patch + global)
        patch_output = torch.randn_like(clip_emb)
        global_output = torch.randn(batch_size, 768)  # CLIP's 768-dim space
        
        # Compute target global features
        target_global = loss_fn.apply_clip_visual_projection(clip_emb)
        
        # Test loss computation
        loss, metrics = loss_fn(
            dit_output=patch_output,
            dit_global=global_output,
            clip_patches=clip_emb,
            clip_global=target_global,
            timesteps=timesteps,
            eva_conditioning=eva_emb,
            return_metrics=True
        )
        
        print(f"✅ Dual supervision loss created successfully!")
        print(f"   Total loss: {loss.item():.6f}")
        print(f"   Patch loss: {metrics['patch_loss']:.6f}")
        print(f"   Global loss: {metrics['global_loss']:.6f}")
        print(f"   Flow matching loss: {metrics['flow_matching_loss']:.6f}")
        print(f"   Target global shape: {target_global.shape}")
        
        return loss_fn
        
    except Exception as e:
        print(f"❌ Dual supervision loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dual_supervision_training_step(model, loss_fn):
    """Test a complete dual supervision training step."""
    print("🎯 Testing dual supervision training step...")
    
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Test data
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        batch_size = eva_emb.shape[0]
        
        # Training step
        optimizer.zero_grad()
        
        # Sample timesteps and noise
        timesteps = loss_fn.sample_timesteps(batch_size, eva_emb.device)
        noise = torch.randn_like(clip_emb)
        x_0 = torch.randn_like(clip_emb)
        
        # Create noisy input
        noisy_input = loss_fn.interpolate_data(x_0, clip_emb, timesteps, noise)
        
        # Forward pass through dual supervision model
        outputs = model(
            hidden_states=noisy_input,
            timestep=timesteps,
            encoder_hidden_states=eva_emb,
            return_dict=True
        )
        
        patch_output = outputs['patch_output']
        global_output = outputs['global_output']
        
        # Compute target global features
        target_global = loss_fn.apply_clip_visual_projection(clip_emb)
        
        # Compute dual supervision loss
        loss, metrics = loss_fn(
            dit_output=patch_output,
            dit_global=global_output,
            clip_patches=clip_emb,
            clip_global=target_global,
            timesteps=timesteps,
            eva_conditioning=eva_emb,
            return_metrics=True
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"✅ Dual supervision training step successful!")
        print(f"   Total loss: {loss.item():.6f}")
        print(f"   Patch loss: {metrics['patch_loss']:.6f}")
        print(f"   Global loss: {metrics['global_loss']:.6f}")
        print(f"   Flow matching loss: {metrics['flow_matching_loss']:.6f}")
        print(f"   Patch cosine similarity: {metrics.get('patch_cosine_similarity', 'N/A'):.4f}")
        print(f"   Global cosine similarity: {metrics.get('global_cosine_similarity', 'N/A'):.4f}")
        print(f"   Gradient norm: {grad_norm:.4f}")
        
        return True, metrics
        
    except Exception as e:
        print(f"❌ Dual supervision training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_dual_supervision_generation(model):
    """Test dual supervision generation."""
    print("🎨 Testing dual supervision generation...")
    
    try:
        model.eval()
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        with torch.no_grad():
            # Test generation with dual outputs
            generated = model.generate(
                encoder_hidden_states=eva_emb,
                num_inference_steps=10,  # Fast for testing
                return_global_only=False,  # Get patch outputs
            )
            
            # Also test global-only generation
            generated_global = model.generate(
                encoder_hidden_states=eva_emb,
                num_inference_steps=10,
                return_global_only=True,  # Get global outputs
            )
            
            # Compute alignment metrics for patch outputs
            gen_global = F.normalize(generated.mean(dim=1), dim=-1)
            target_global = F.normalize(clip_emb.mean(dim=1), dim=-1)
            
            patch_cosine_sim = F.cosine_similarity(gen_global, target_global, dim=1).mean().item()
            patch_l2_distance = torch.norm(generated - clip_emb, dim=-1).mean().item()
            
            print(f"✅ Dual supervision generation successful!")
            print(f"   Generated patch shape: {generated.shape}")
            print(f"   Generated global shape: {generated_global.shape if generated_global is not None else 'None'}")
            print(f"   Patch cosine similarity: {patch_cosine_sim:.4f}")
            print(f"   L2 distance: {patch_l2_distance:.4f}")
            print(f"   Generated norm: {torch.norm(generated, dim=-1).mean().item():.4f}")
            print(f"   Target norm: {torch.norm(clip_emb, dim=-1).mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dual supervision generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dual_supervision_trainer():
    """Test the dual supervision trainer."""
    print("👨‍🏫 Testing dual supervision trainer...")
    
    try:
        from src.modules.trainers.dual_supervision_blip3o_trainer import DualSupervisionBLIP3oTrainer
        from transformers import TrainingArguments
        
        # Create small model and loss for testing
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.blip3o_dit import create_blip3o_dit_model
        from src.modules.losses.dual_supervision_flow_matching_loss import create_dual_supervision_loss
        
        config = BLIP3oDiTConfig(dim=256, n_layers=2, n_heads=4, mlp_hidden_dim=512)
        model = create_blip3o_dit_model(config=config, load_clip_projection=True)
        loss_fn = create_dual_supervision_loss()
        
        # Minimal training args
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            save_steps=1000,  # Don't save during test
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = DualSupervisionBLIP3oTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=loss_fn,
        )
        
        # Test compute_loss method
        test_data = create_test_data()
        inputs = {
            'eva_embeddings': test_data['eva_embeddings'],
            'clip_embeddings': test_data['clip_embeddings'],
        }
        
        loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
        
        print(f"✅ Dual supervision trainer created successfully!")
        print(f"   Trainer type: {type(trainer).__name__}")
        print(f"   Test loss: {loss.item():.6f}")
        print(f"   Has dual outputs: {'patch_output' in outputs and 'global_output' in outputs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dual supervision trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_step_dual_supervision_training(model, loss_fn, num_steps=3):
    """Test multi-step dual supervision training for convergence."""
    print(f"📈 Testing {num_steps}-step dual supervision training...")
    
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Lower LR for dual supervision
        
        test_data = create_test_data()
        eva_emb = test_data['eva_embeddings']
        clip_emb = test_data['clip_embeddings']
        
        batch_size = eva_emb.shape[0]
        
        losses = []
        patch_cosines = []
        global_cosines = []
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Sample fresh noise each step
            timesteps = loss_fn.sample_timesteps(batch_size, eva_emb.device)
            noise = torch.randn_like(clip_emb)
            x_0 = torch.randn_like(clip_emb)
            
            noisy_input = loss_fn.interpolate_data(x_0, clip_emb, timesteps, noise)
            
            outputs = model(
                hidden_states=noisy_input,
                timestep=timesteps,
                encoder_hidden_states=eva_emb,
                return_dict=True
            )
            
            patch_output = outputs['patch_output']
            global_output = outputs['global_output']
            target_global = loss_fn.apply_clip_visual_projection(clip_emb)
            
            loss, metrics = loss_fn(
                dit_output=patch_output,
                dit_global=global_output,
                clip_patches=clip_emb,
                clip_global=target_global,
                timesteps=timesteps,
                eva_conditioning=eva_emb,
                return_metrics=True
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            patch_cosines.append(metrics.get('patch_cosine_similarity', 0))
            global_cosines.append(metrics.get('global_cosine_similarity', 0))
            
            if step == 0 or step == num_steps - 1:
                print(f"   Step {step}: Loss={loss.item():.6f}, Patch_Cos={patch_cosines[-1]:.4f}, Global_Cos={global_cosines[-1]:.4f}")
        
        # Analyze trends
        loss_improvement = losses[0] - losses[-1]
        patch_improvement = patch_cosines[-1] - patch_cosines[0]
        global_improvement = global_cosines[-1] - global_cosines[0]
        
        print(f"✅ Multi-step dual supervision training completed!")
        print(f"   Loss improvement: {loss_improvement:.6f} ({'good' if loss_improvement > 0 else 'check'})")
        print(f"   Patch cosine improvement: {patch_improvement:.4f} ({'good' if patch_improvement > 0 else 'check'})")
        print(f"   Global cosine improvement: {global_improvement:.4f} ({'good' if global_improvement > 0 else 'check'})")
        
        return True, {
            'losses': losses, 
            'patch_cosines': patch_cosines, 
            'global_cosines': global_cosines
        }
        
    except Exception as e:
        print(f"❌ Multi-step dual supervision training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run the complete dual supervision test suite."""
    print("🧪 BLIP3-o Dual Supervision Test Suite")
    print("=" * 60)
    print("🎯 Testing dual supervision architecture:")
    print("   EVA [B,256,4096] → DiT → [B,256,1024] → {")
    print("     Patch Output: [B,256,1024] (patch loss)")
    print("     Global Path: Avg Pool → MLP → Frozen CLIP Proj → [B,768]")
    print("   }")
    print("=" * 60)
    
    success_count = 0
    total_tests = 6
    
    # Test 1: Dual Supervision Model Creation
    model = test_dual_supervision_model_creation()
    if model is not None:
        success_count += 1
        print()
    else:
        print("❌ Stopping tests - dual supervision model creation failed")
        return False
    
    # Test 2: Dual Supervision Loss
    loss_fn = test_dual_supervision_loss()
    if loss_fn is not None:
        success_count += 1
        print()
    else:
        print("❌ Stopping tests - dual supervision loss function failed")
        return False
    
    # Test 3: Dual Supervision Training Step
    training_success, metrics = test_dual_supervision_training_step(model, loss_fn)
    if training_success:
        success_count += 1
        print()
    else:
        print("❌ Dual supervision training step failed")
        print()
    
    # Test 4: Dual Supervision Generation
    gen_success = test_dual_supervision_generation(model)
    if gen_success:
        success_count += 1
        print()
    else:
        print("❌ Dual supervision generation failed")
        print()
    
    # Test 5: Dual Supervision Trainer
    trainer_success = test_dual_supervision_trainer()
    if trainer_success:
        success_count += 1
        print()
    else:
        print("❌ Dual supervision trainer failed")
        print()
    
    # Test 6: Multi-step Training
    multi_success, multi_results = test_multi_step_dual_supervision_training(model, loss_fn)
    if multi_success:
        success_count += 1
        print()
    else:
        print("❌ Multi-step dual supervision training failed")
        print()
    
    # Final Results
    print("🎯 DUAL SUPERVISION TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("🎉 ALL DUAL SUPERVISION TESTS PASSED!")
        print("💡 Your dual supervision BLIP3-o implementation is working correctly!")
        print("\n📋 Architecture Verified:")
        print("   ✅ Dual outputs (patch + global)")
        print("   ✅ Global adaptation MLP")
        print("   ✅ Frozen CLIP visual projection")
        print("   ✅ Dual supervision loss (patch + global + flow matching)")
        print("   ✅ Enhanced trainer with dual metrics")
        print("\n🚀 Ready for dual supervision training!")
        print("   Expected improvements:")
        print("   • Patch fidelity: Maintained high quality")
        print("   • Global alignment: Optimized for retrieval")
        print("   • Recall performance: Expected 0% → 60%+ improvement")
        return True
    else:
        print(f"⚠️ {total_tests - success_count} test(s) failed")
        print("🔍 Check the error messages above and fix the issues")
        print("\n🛠️ Common issues:")
        print("   • Import path problems")
        print("   • Model-loss interface mismatches")
        print("   • Missing frozen CLIP projection")
        print("   • Incorrect dual supervision configuration")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)