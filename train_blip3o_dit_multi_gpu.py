#!/usr/bin/env python3
"""
Multi-GPU Training script for BLIP3-o DiT using PyTorch DDP
Optimized for Snellius H100 nodes with proper distributed setup
"""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
from pathlib import Path
import json
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_distributed_training():
    """Setup distributed training environment"""
    # Get environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"🌐 Distributed setup: local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}")
    
    # Set the device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=global_rank
        )
        print(f"✅ Process group initialized for rank {global_rank}")
    
    return local_rank, global_rank, world_size, device

def setup_logging(local_rank):
    """Setup logging - only rank 0 should print most messages"""
    if local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for multi-GPU training."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU BLIP3-o DiT training with DDP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument(
        "--chunked_embeddings_dir", type=str, required=True,
        help="Path to directory containing chunked embedding files"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for checkpoints"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_dim", type=int, default=768,
                           help="Model hidden dimension")
    model_group.add_argument("--num_layers", type=int, default=16,
                           help="Number of transformer layers")
    model_group.add_argument("--num_heads", type=int, default=12,
                           help="Number of attention heads")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--num_epochs", type=int, default=5,
                           help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=8,
                           help="Training batch size per GPU")
    train_group.add_argument("--eval_batch_size", type=int, default=4,
                           help="Evaluation batch size per GPU")
    train_group.add_argument("--learning_rate", type=float, default=1e-4,
                           help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                           help="Weight decay")
    train_group.add_argument("--warmup_steps", type=int, default=100,
                           help="Number of warmup steps")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=4,
                           help="Gradient accumulation steps")
    
    # Hardware configuration
    hw_group = parser.add_argument_group("Hardware Configuration")
    hw_group.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (fp16)")
    hw_group.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers per GPU")
    
    return parser.parse_args()

def create_multi_gpu_dataloaders(chunked_dir, batch_size, eval_batch_size, num_workers):
    """Create dataloaders optimized for multi-GPU training"""
    from src.modules.datasets.blip3o_dataset import create_chunked_dataloader
    
    print(f"🔄 Creating multi-GPU dataloaders...")
    print(f"   Batch size per GPU: {batch_size}")
    print(f"   Eval batch size per GPU: {eval_batch_size}")
    print(f"   Workers per GPU: {num_workers}")
    
    # Create training dataloader
    train_dataloader = create_chunked_dataloader(
        chunked_embeddings_dir=chunked_dir,
        batch_size=batch_size,
        split="train",
        eval_split_ratio=0.1,
        normalize_embeddings=True,
        shuffle_shards=True,
        shuffle_within_shard=True,
        delete_after_use=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Critical for multi-GPU training
        persistent_workers=True
    )
    
    # Create evaluation dataloader
    eval_dataloader = create_chunked_dataloader(
        chunked_embeddings_dir=chunked_dir,
        batch_size=eval_batch_size,
        split="eval",
        eval_split_ratio=0.1,
        normalize_embeddings=True,
        shuffle_shards=False,
        shuffle_within_shard=False,
        delete_after_use=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    return train_dataloader, eval_dataloader

def main():
    """Main multi-GPU training function."""
    # Setup distributed training
    local_rank, global_rank, world_size, device = setup_distributed_training()
    
    # Setup logging
    logger = setup_logging(local_rank)
    
    if local_rank == 0:
        print("🚀 Starting Multi-GPU BLIP3-o DiT Training")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        print(f"Global rank: {global_rank}")
        print(f"Device: {device}")
        print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    if local_rank == 0:
        print(f"📊 Multi-GPU Training Configuration:")
        print(f"   Total GPUs: {world_size}")
        print(f"   Batch size per GPU: {args.batch_size}")
        print(f"   Total effective batch: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"   Model dimension: {args.model_dim}")
        print(f"   Layers: {args.num_layers}")
        print(f"   Attention heads: {args.num_heads}")
    
    try:
        # Import modules
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.blip3o_dit import create_blip3o_dit_model
        from src.modules.losses.flow_matching_loss import create_blip3o_flow_matching_loss
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer, create_blip3o_training_args
        
        # Load manifest
        manifest_path = Path(args.chunked_embeddings_dir) / "embeddings_manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if local_rank == 0:
            print(f"📊 Dataset info:")
            print(f"   Total shards: {manifest['total_shards']}")
            print(f"   Total samples: {manifest['total_samples']:,}")
            print(f"   Samples per GPU: {manifest['total_samples'] // world_size:,}")
        
        # Create model config
        model_config = BLIP3oDiTConfig(
            input_size=16,
            patch_size=1,
            in_channels=1024,
            dim=args.model_dim,
            eva_embedding_size=4096,
            n_layers=args.num_layers,
            n_heads=args.num_heads,
            norm_eps=1e-5,
            qk_norm=True,
            learn_sigma=False,
            _gradient_checkpointing=True,
        )
        
        if local_rank == 0:
            print(f"🏗️  Creating model on rank {global_rank}...")
        
        # Create model and move to device
        model = create_blip3o_dit_model(config=model_config)
        model = model.to(device)
        
        # Enable gradient checkpointing
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
        if local_rank == 0:
            print(f"📊 Model parameters: {model.get_num_parameters():,}")
            print(f"💾 Memory per GPU: ~{model.get_num_parameters() * 4 / (1024**3):.1f} GB")
        
        # Create flow matching loss
        flow_matching_loss = create_blip3o_flow_matching_loss()
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_multi_gpu_dataloaders(
            chunked_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.dataloader_num_workers,
        )
        
        # Create dummy datasets for trainer
        class DummyDataset:
            def __init__(self, length):
                self.length = length
            def __len__(self):
                return self.length
        
        train_dataset = DummyDataset(manifest['total_samples'])
        eval_dataset = DummyDataset(manifest['total_samples'] // 10)
        
        # Calculate training steps
        samples_per_gpu = manifest['total_samples'] // world_size
        steps_per_epoch = (samples_per_gpu + args.batch_size - 1) // args.batch_size
        max_steps = (steps_per_epoch * args.num_epochs) // args.gradient_accumulation_steps
        
        if local_rank == 0:
            print(f"📈 Training schedule:")
            print(f"   Steps per epoch per GPU: {steps_per_epoch}")
            print(f"   Max steps: {max_steps}")
            print(f"   Total epochs: {args.num_epochs}")
        
        # Create training arguments
        training_args = create_blip3o_training_args(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            logging_steps=20,
            save_steps=max(100, max_steps // 5),
            eval_steps=max(50, max_steps // 10),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=False,
            load_best_model_at_end=False,
            local_rank=local_rank,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=True,
            save_on_each_node=False,
        )
        
        if local_rank == 0:
            print("🔧 Creating trainer...")
        
        # Create trainer
        trainer = BLIP3oTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Override dataloader methods
        trainer.get_train_dataloader = lambda: train_dataloader
        trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
        
        # Synchronize before training
        if world_size > 1:
            dist.barrier()
        
        if local_rank == 0:
            print("🚀 Starting multi-GPU training...")
        
        # Start training
        trainer.train()
        
        # Save final model
        if local_rank == 0:
            print("💾 Saving final model...")
            trainer.save_model()
            print("✅ Multi-GPU training completed successfully!")
        
        # Wait for all processes
        if world_size > 1:
            dist.barrier()
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed on rank {global_rank}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)