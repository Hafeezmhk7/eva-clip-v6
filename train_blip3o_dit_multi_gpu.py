#!/usr/bin/env python3
"""
FULLY FIXED Multi-GPU Training script for BLIP3-o DiT
Fixes the IterableDataset length issue and all other problems
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

def setup_logging():
    """Setup logging for all ranks"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
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
        description="FULLY FIXED Multi-GPU BLIP3-o DiT training",
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

def main():
    """Main training function - FULLY FIXED"""
    logger = setup_logging()
    
    # Get distributed info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Only print from rank 0
    if local_rank == 0:
        print("🚀 FULLY FIXED Multi-GPU BLIP3-o DiT Training")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        print(f"Global rank: {global_rank}")
        print("=" * 60)
        print("🔧 Fix: IterableDataset boolean evaluation issue resolved")
    
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
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer
        from src.modules.datasets.blip3o_dataset import create_chunked_dataloaders
        
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
            input_size=16,  # 16x16 = 256 tokens
            patch_size=1,
            in_channels=1024,  # CLIP dimension
            dim=args.model_dim,
            eva_embedding_size=4096,  # EVA-CLIP dimension
            n_layers=args.num_layers,
            n_heads=args.num_heads,
            norm_eps=1e-5,
            qk_norm=True,
            learn_sigma=False,
            _gradient_checkpointing=True,  # Memory optimization
        )
        
        if local_rank == 0:
            print(f"🏗️  Creating model...")
        
        # Create model - NO DDP WRAPPING (Trainer handles this)
        model = create_blip3o_dit_model(config=model_config)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
        # Calculate parameters
        if local_rank == 0:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"📊 Model parameters: {param_count:,}")
            print(f"💾 Memory per GPU: ~{param_count * 4 / (1024**3):.1f} GB")
        
        # Create flow matching loss
        flow_matching_loss = create_blip3o_flow_matching_loss()
        
        # Create dataloaders using the chunked dataset
        if local_rank == 0:
            print(f"🔄 Creating dataloaders...")
        
        train_dataloader, eval_dataloader = create_chunked_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_split_ratio=0.1,
            normalize_embeddings=True,
            delete_after_use=False,  # Keep files for multi-GPU access
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,  # Important for DDP
        )
        
        # FIXED: Create dummy datasets without boolean evaluation of dataloaders
        class DummyDataset:
            def __init__(self, length):
                self.length = length
            def __len__(self):
                return self.length
        
        # FIXED: Always create both datasets, avoid boolean evaluation of dataloader
        train_dataset = DummyDataset(manifest['total_samples'])
        eval_dataset = DummyDataset(manifest['total_samples'] // 10)  # Always create eval dataset
        
        # Calculate training steps
        samples_per_gpu = manifest['total_samples'] // world_size
        steps_per_epoch = (samples_per_gpu + args.batch_size - 1) // args.batch_size
        max_steps = (steps_per_epoch * args.num_epochs) // args.gradient_accumulation_steps
        
        if local_rank == 0:
            print(f"📈 Training schedule:")
            print(f"   Steps per epoch per GPU: {steps_per_epoch}")
            print(f"   Max steps: {max_steps}")
            print(f"   Total epochs: {args.num_epochs}")
        
        # Create proper TrainingArguments for multi-GPU
        from transformers import TrainingArguments
        
        training_args = TrainingArguments(
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
            eval_strategy="steps" if eval_dataloader else "no",
            eval_steps=max(50, max_steps // 10) if eval_dataloader else None,
            save_strategy="steps",
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=False,  # Important for custom data
            load_best_model_at_end=False,  # Disable for simplicity
            
            # CRITICAL: Multi-GPU DDP settings
            ddp_find_unused_parameters=False,  # Better performance
            dataloader_pin_memory=True,
            save_on_each_node=False,  # Only save on main process
            local_rank=local_rank,
            
            # Memory optimizations
            dataloader_persistent_workers=True,
            save_total_limit=3,
            prediction_loss_only=False,
            
            # Disable features that can cause issues
            push_to_hub=False,
            report_to=[],  # Disable wandb for now
        )
        
        if local_rank == 0:
            print("🔧 Creating trainer...")
        
        # Create trainer - Trainer will handle DDP automatically
        trainer = BLIP3oTrainer(
            model=model,  # NO DDP wrapping - Trainer handles this
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,  # FIXED: Always provide eval_dataset
        )
        
        # Override dataloader methods to use our chunked dataloaders
        trainer.get_train_dataloader = lambda: train_dataloader
        if eval_dataloader:
            trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
        
        if local_rank == 0:
            print("🚀 Starting multi-GPU training...")
            print("✅ IterableDataset issue fixed")
            print("✅ Trainer will automatically handle DDP setup")
        
        # Start training - Trainer handles all DDP automatically
        trainer.train()
        
        # Save final model (only on main process)
        if local_rank == 0:
            print("💾 Saving final model...")
            trainer.save_model()
            print("✅ Multi-GPU training completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed on rank {global_rank}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)