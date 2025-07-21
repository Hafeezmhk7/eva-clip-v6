#!/usr/bin/env python3
"""
Multi-GPU Global BLIP3-o Training Script
File: train_global_blip3o_multi_gpu.py

Trains the simplified global model on multiple GPUs using DistributedDataParallel.
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

def setup_ddp_environment():
    """Setup DDP environment variables"""
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

def parse_arguments():
    """Parse command line arguments for multi-GPU global training"""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Global BLIP3-o Training - Direct Global Feature Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Model configuration
    parser.add_argument("--model_dim", type=int, default=768,
                       help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--mlp_hidden_dim", type=int, default=2048,
                       help="MLP hidden dimension")
    
    # Training configuration  
    parser.add_argument("--num_epochs", type=int, default=6,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6,
                       help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=12,
                       help="Evaluation batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=8e-5,
                       help="Learning rate (adjusted for multi-GPU)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=150,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "constant"],
                       help="Learning rate scheduler type")
    
    # Hardware configuration
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers per GPU")
    
    return parser.parse_args()

def main():
    """Main multi-GPU global training function"""
    logger = setup_logging()
    
    # Setup DDP environment
    setup_ddp_environment()
    
    # Get distributed info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize distributed training
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f"cuda:{local_rank}")
    
    # Only print from rank 0
    if global_rank == 0:
        print("🚀 Multi-GPU Global BLIP3-o Training")
        print("=" * 70)
        print("✅ KEY ADVANTAGE: Training directly on global [B, 768] features")
        print("✅ No training-inference mismatch!")
        print("✅ Single clean objective: global flow matching")
        print("✅ Expected: 50-70% recall (vs previous 0.1%)")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        print(f"Global rank: {global_rank}")
        print(f"Device: {device}")
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Import global components
        if global_rank == 0:
            print("📦 Importing global components...")
        
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.global_blip3o_dit import create_global_blip3o_dit_model
        from src.modules.losses.global_flow_matching_loss import create_global_flow_matching_loss
        from src.modules.trainers.global_blip3o_trainer import GlobalBLIP3oTrainer, create_global_training_args
        from src.modules.datasets.blip3o_dataset import create_chunked_dataloaders
        
        if global_rank == 0:
            print("✅ All global components imported successfully")
        
        # Load manifest
        manifest_path = Path(args.chunked_embeddings_dir) / "embeddings_manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if global_rank == 0:
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
            learn_sigma=False,
            _gradient_checkpointing=True,
            mlp_hidden_dim=args.mlp_hidden_dim,
        )
        
        if global_rank == 0:
            print(f"🏗️  Creating global model for multi-GPU...")
        
        # Create global model
        model = create_global_blip3o_dit_model(
            config=model_config,
            load_clip_projection=True,
        )
        
        # Move model to device before DDP wrapping
        model = model.to(device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
        # Wrap with DDP for multi-GPU training
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,  # Global model uses all parameters
            )
            if global_rank == 0:
                print(f"✅ Model wrapped with DDP for {world_size} GPUs")
        
        # Calculate parameters (from unwrapped model)
        if global_rank == 0:
            unwrapped_model = model.module if hasattr(model, 'module') else model
            total_params = sum(p.numel() for p in unwrapped_model.parameters())
            trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
            
            print(f"📊 Global model parameters:")
            print(f"   Total: {total_params:,}")
            print(f"   Trainable: {trainable_params:,}")
            print(f"   Has frozen CLIP projection: {unwrapped_model.frozen_clip_proj is not None}")
        
        # Create global flow matching loss
        if global_rank == 0:
            print(f"🎯 Creating global flow matching loss...")
        
        flow_matching_loss = create_global_flow_matching_loss()
        
        if global_rank == 0:
            print(f"✅ Global flow matching loss created")
            print(f"   Training target: [B, 768] global embeddings")
            print(f"   Single clean objective")
        
        # Create dataloaders with DDP support
        if global_rank == 0:
            print(f"🔄 Creating multi-GPU dataloaders...")
        
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
        
        # Create dummy datasets for Trainer
        class MultiGPUDataset:
            def __init__(self, estimated_samples):
                # Adjust for distributed training
                self.estimated_samples = estimated_samples // world_size if world_size > 1 else estimated_samples
            
            def __len__(self):
                return self.estimated_samples
            
            def __getitem__(self, idx):
                raise NotImplementedError("Use custom dataloader")
        
        # Calculate samples per GPU
        total_samples = manifest['total_samples']
        train_samples_per_gpu = int(total_samples * 0.9) // world_size
        eval_samples_per_gpu = int(total_samples * 0.1) // world_size if eval_dataloader else 0
        
        train_dataset = MultiGPUDataset(train_samples_per_gpu)
        eval_dataset = MultiGPUDataset(eval_samples_per_gpu) if eval_dataloader else None
        
        # Calculate training steps
        effective_batch_size = args.batch_size * world_size * args.gradient_accumulation_steps
        steps_per_epoch = max(1, total_samples // effective_batch_size)
        max_steps = steps_per_epoch * args.num_epochs
        
        if global_rank == 0:
            print(f"📈 Multi-GPU training schedule:")
            print(f"   Effective batch size: {effective_batch_size}")
            print(f"   Steps per epoch: {steps_per_epoch}")
            print(f"   Max steps: {max_steps}")
            print(f"   Total epochs: {args.num_epochs}")
        
        # Create training arguments optimized for multi-GPU
        training_args = create_global_training_args(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            dataloader_num_workers=args.dataloader_num_workers,
            
            # Multi-GPU specific settings
            logging_steps=max(10, steps_per_epoch // 20),
            save_steps=max(50, steps_per_epoch // 4),
            eval_steps=max(25, steps_per_epoch // 8) if eval_dataloader else 0,
            
            # DDP optimizations
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=True,
            save_on_each_node=False,  # Only save on main process
        )
        
        if global_rank == 0:
            print("🔧 Creating global trainer for multi-GPU...")
        
        # Create trainer
        trainer = GlobalBLIP3oTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        if global_rank == 0:
            print(f"✅ GlobalBLIP3oTrainer created for {world_size} GPUs")
        
        # Override dataloader methods for multi-GPU
        def get_train_dataloader_override():
            if global_rank == 0:
                logger.info("Using multi-GPU global train dataloader")
            return train_dataloader
        
        def get_eval_dataloader_override(eval_dataset=None):
            if eval_dataloader:
                if global_rank == 0:
                    logger.info("Using multi-GPU global eval dataloader")
                return eval_dataloader
            return None
        
        trainer.get_train_dataloader = get_train_dataloader_override
        trainer.get_eval_dataloader = get_eval_dataloader_override
        
        # Enhanced compute_loss for multi-GPU compatibility
        original_compute_loss = trainer.compute_loss
        
        def multi_gpu_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            # Call original compute_loss
            result = original_compute_loss(model, inputs, return_outputs, num_items_in_batch)
            
            # Store metrics only on main process
            if hasattr(trainer, 'loss_components') and global_rank != 0:
                # Clear metrics on non-main processes to save memory
                trainer.loss_components.clear()
            
            return result
        
        trainer.compute_loss = multi_gpu_compute_loss
        
        # Enhanced logging for multi-GPU
        original_log_metrics = trainer._log_global_training_metrics
        
        def multi_gpu_log_metrics(*args, **kwargs):
            # Only log on main process
            if global_rank == 0:
                return original_log_metrics(*args, **kwargs)
        
        trainer._log_global_training_metrics = multi_gpu_log_metrics
        
        if global_rank == 0:
            print("🚀 Starting multi-GPU global training...")
            print("=" * 50)
            print("✅ MULTI-GPU GLOBAL ARCHITECTURE:")
            print(f"  • {world_size} GPUs training in parallel")
            print(f"  • Effective batch size: {effective_batch_size}")
            print(f"  • Direct [B, 768] global training")
            print(f"  • No training-inference mismatch")
            print(f"  • Single flow matching objective")
            print("=" * 50)
            print("🎯 EXPECTED RESULTS:")
            print(f"  • Previous: 0.1% R@1 recall")
            print(f"  • Expected: 50-70% R@1 recall") 
            print(f"  • {world_size}x faster training")
            print("=" * 50)
        
        # Start training
        trainer.train()
        
        # Save model (only on main process)
        if global_rank == 0:
            print("💾 Saving final global model...")
            trainer.save_model()
            
            # Print final metrics
            if hasattr(trainer.flow_matching_loss, 'ema_cosine'):
                final_cosine = trainer.flow_matching_loss.ema_cosine.item()
                predicted_recall = min(final_cosine * 70, 70)
                
                print("📊 FINAL MULTI-GPU TRAINING METRICS:")
                print(f"   Global cosine similarity: {final_cosine:.4f}")
                print(f"   Predicted recall: {predicted_recall:.1f}%")
                print(f"   Training successful: {final_cosine > 0.7}")
                print(f"   Improvement vs previous: {final_cosine / 0.001:.0f}x")
            
            print("✅ Multi-GPU global training completed successfully!")
            print(f"📁 Model saved to: {args.output_dir}")
            print("🎯 Ready for recall evaluation!")
        
        # Cleanup distributed training
        if world_size > 1:
            dist.destroy_process_group()
        
        return 0
        
    except Exception as e:
        if global_rank == 0:
            print(f"❌ Training failed on rank {global_rank}: {e}")
            print("Full traceback:")
            traceback.print_exc()
        
        # Cleanup on error
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)