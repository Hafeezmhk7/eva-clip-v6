#!/usr/bin/env python3
"""
Simple Multi-GPU Test Script for BLIP3-o
Tests if the multi-GPU setup works with a tiny model and dummy data
"""

import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_distributed():
    """Setup distributed training environment"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"Rank {global_rank}: Setting up distributed training...")
    print(f"  Local rank: {local_rank}")
    print(f"  Global rank: {global_rank}")
    print(f"  World size: {world_size}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    return local_rank, global_rank, world_size, device

def test_model_creation():
    """Test creating a tiny BLIP3-o model"""
    try:
        from src.modules.config.memory_optimized_config import get_memory_optimized_model_configs
        from src.modules.models.blip3o_dit import create_blip3o_dit_model
        from src.modules.losses.flow_matching_loss import create_blip3o_flow_matching_loss
        
        # Get tiny model config
        configs = get_memory_optimized_model_configs()
        tiny_config = configs['tiny']
        
        print(f"Creating tiny model: dim={tiny_config.dim}, layers={tiny_config.n_layers}")
        
        # Create model
        model = create_blip3o_dit_model(config=tiny_config)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model created successfully: {param_count:,} parameters")
        
        # Create loss
        loss_fn = create_blip3o_flow_matching_loss()
        print(f"✅ Loss function created successfully")
        
        return model, loss_fn, tiny_config
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_dummy_forward_pass(model, loss_fn, config, device):
    """Test a dummy forward pass"""
    try:
        batch_size = 2
        seq_len = 256  # 16x16 tokens
        
        # Create dummy inputs
        eva_embeddings = torch.randn(batch_size, seq_len, 4096).to(device)  # EVA-CLIP
        clip_embeddings = torch.randn(batch_size, seq_len, 1024).to(device)  # CLIP target
        timesteps = torch.rand(batch_size).to(device)
        
        print(f"Testing forward pass with batch_size={batch_size}")
        print(f"  EVA shape: {eva_embeddings.shape}")
        print(f"  CLIP shape: {clip_embeddings.shape}")
        print(f"  Timesteps shape: {timesteps.shape}")
        
        # Move model to device
        model = model.to(device)
        model.train()
        
        # Sample noise and create noisy input
        noise = torch.randn_like(clip_embeddings)
        x_0 = torch.randn_like(clip_embeddings)
        noisy_clip = loss_fn.interpolate_data(x_0, clip_embeddings, timesteps, noise)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=True):
            model_output = model(
                hidden_states=noisy_clip,
                timestep=timesteps,
                encoder_hidden_states=eva_embeddings,
                return_dict=False
            )
        
        print(f"✅ Forward pass successful")
        print(f"  Output shape: {model_output.shape}")
        
        # Test loss computation
        loss, metrics = loss_fn(
            model_output=model_output,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            noise=noise,
            return_metrics=True
        )
        
        print(f"✅ Loss computation successful")
        print(f"  Loss: {loss.item():.4f}")
        if metrics:
            print(f"  Cosine similarity: {metrics.get('cosine_similarity', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_creation(model, loss_fn):
    """Test creating a trainer"""
    try:
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer
        from transformers import TrainingArguments
        
        # Create minimal training arguments
        training_args = TrainingArguments(
            output_dir="./test_output",
            max_steps=10,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            logging_steps=5,
            save_steps=10,
            ddp_find_unused_parameters=False,  # FIXED
            remove_unused_columns=False,
            report_to=[],
        )
        
        # Create dummy dataset
        class DummyDataset:
            def __len__(self):
                return 20
        
        # Create trainer
        trainer = BLIP3oTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=loss_fn,
            train_dataset=DummyDataset(),
        )
        
        print(f"✅ Trainer created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 Testing Multi-GPU BLIP3-o Setup")
    print("=" * 50)
    
    # Setup distributed
    local_rank, global_rank, world_size, device = setup_distributed()
    
    # Only print detailed info from rank 0
    if local_rank == 0:
        print(f"🔧 Environment Info:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device: {device}")
        
        if torch.cuda.is_available():
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test 1: Model creation
    print(f"\n🧪 Test 1: Model Creation (Rank {global_rank})")
    model, loss_fn, config = test_model_creation()
    if model is None:
        print("❌ Test failed - exiting")
        return 1
    
    # Test 2: Forward pass
    print(f"\n🧪 Test 2: Forward Pass (Rank {global_rank})")
    forward_success = test_dummy_forward_pass(model, loss_fn, config, device)
    if not forward_success:
        print("❌ Test failed - exiting")
        return 1
    
    # Test 3: Trainer creation (only on rank 0 to avoid conflicts)
    if local_rank == 0:
        print(f"\n🧪 Test 3: Trainer Creation")
        trainer_success = test_trainer_creation(model, loss_fn)
        if not trainer_success:
            print("❌ Test failed")
            return 1
    
    # Synchronize all processes
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        dist.barrier()
    
    if local_rank == 0:
        print(f"\n✅ All tests passed! Multi-GPU setup is working correctly.")
        print(f"🚀 Ready for full training with the fixed script.")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)