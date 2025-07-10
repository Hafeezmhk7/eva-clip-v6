#!/usr/bin/env python3
"""
Simple test training script to verify BLIP3-o setup is working.
This will do a minimal training run to test everything is connected properly.
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from src.modules.config.blip3o_config import get_default_blip3o_config
        print("✅ Config import successful")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from src.modules.models.blip3o_dit import create_blip3o_dit_model
        print("✅ Model import successful")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from src.modules.losses.flow_matching_loss import create_blip3o_flow_matching_loss
        print("✅ Loss import successful")
    except Exception as e:
        print(f"❌ Loss import failed: {e}")
        return False
    
    try:
        from src.modules.datasets.blip3o_dataset import test_blip3o_dataset
        print("✅ Dataset import successful")
    except Exception as e:
        print(f"❌ Dataset import failed: {e}")
        return False
    
    try:
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer
        print("✅ Trainer import successful")
    except Exception as e:
        print(f"❌ Trainer import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if model can be created."""
    print("\n🧪 Testing model creation...")
    
    try:
        from src.modules.config.blip3o_config import get_default_blip3o_config
        from src.modules.models.blip3o_dit import create_blip3o_dit_model
        
        # Create small model for testing
        config = get_default_blip3o_config()
        config.dim = 512  # Smaller for testing
        config.n_layers = 4  # Fewer layers
        config.n_heads = 8   # Fewer heads
        
        model = create_blip3o_dit_model(config)
        print(f"✅ Model created successfully")
        print(f"   Parameters: {model.get_num_parameters():,}")
        print(f"   Memory: {model.get_memory_footprint()}")
        
        return True, model, config
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_dataset_loading(embeddings_path):
    """Test if dataset can be loaded."""
    print(f"\n🧪 Testing dataset loading from: {embeddings_path}")
    
    try:
        from src.modules.datasets.blip3o_dataset import test_blip3o_dataset
        
        # Test dataset loading
        test_blip3o_dataset(embeddings_path)
        print("✅ Dataset test completed")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass(model, config):
    """Test a forward pass through the model."""
    print("\n🧪 Testing model forward pass...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        eva_embeddings = torch.randn(batch_size, 64, 1280, device=device)
        clip_embeddings = torch.randn(batch_size, 64, 768, device=device)
        timesteps = torch.rand(batch_size, device=device)
        
        print(f"   Using device: {device}")
        print(f"   EVA input shape: {eva_embeddings.shape}")
        print(f"   CLIP input shape: {clip_embeddings.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(
                hidden_states=clip_embeddings,
                timestep=timesteps,
                encoder_hidden_states=eva_embeddings,
                return_dict=False
            )
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation."""
    print("\n🧪 Testing loss computation...")
    
    try:
        from src.modules.losses.flow_matching_loss import create_blip3o_flow_matching_loss
        
        # Create loss function
        loss_fn = create_blip3o_flow_matching_loss()
        
        # Create dummy data
        batch_size = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_output = torch.randn(batch_size, 64, 768, device=device)
        target_samples = torch.randn(batch_size, 64, 768, device=device)
        timesteps = torch.rand(batch_size, device=device)
        eva_conditioning = torch.randn(batch_size, 64, 1280, device=device)
        
        # Compute loss
        loss, metrics = loss_fn(
            model_output=model_output,
            target_samples=target_samples,
            timesteps=timesteps,
            eva_conditioning=eva_conditioning,
            return_metrics=True
        )
        
        print(f"✅ Loss computation successful")
        print(f"   Loss value: {loss.item():.4f}")
        if metrics:
            print(f"   Metrics: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 BLIP3-o Setup Test")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import test failed. Please check your file structure and dependencies.")
        return False
    
    # Test 2: Model creation
    success, model, config = test_model_creation()
    if not success:
        print("❌ Model creation failed.")
        return False
    
    # Test 3: Forward pass
    if not test_forward_pass(model, config):
        print("❌ Forward pass failed.")
        return False
    
    # Test 4: Loss computation
    if not test_loss_computation():
        print("❌ Loss computation failed.")
        return False
    
    # Test 5: Dataset (if embeddings file exists)
    possible_embeddings_paths = [
        "embeddings/fixed_grid_embeddings.pkl",
        "data/embeddings/fixed_grid_embeddings.pkl",
        "embeddings/blip3o_grid_embeddings.pkl", 
        "data/embeddings/blip3o_grid_embeddings.pkl"
    ]
    
    embeddings_path = None
    for path in possible_embeddings_paths:
        if os.path.exists(path):
            embeddings_path = path
            break
    
    if embeddings_path:
        print(f"📁 Found embeddings file: {embeddings_path}")
        if not test_dataset_loading(embeddings_path):
            print("❌ Dataset loading failed.")
            return False
    else:
        print(f"⚠️  No embeddings file found in common locations:")
        for path in possible_embeddings_paths:
            print(f"     - {path}")
        print("   Please run the embeddings test first")
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Your BLIP3-o setup is working correctly")
    print("\n📋 Next steps:")
    print("   1. Run: python test_embeddings.py")
    print("   2. If embeddings are good, run: python train_blip3o_dit.py --debug")
    print("   3. For full training, remove --debug flag")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("   - Check if all files are in the right locations")
        print("   - Make sure you have all dependencies: pip install -r requirements.txt")
        print("   - Verify your Python path includes the src directory")
        sys.exit(1)