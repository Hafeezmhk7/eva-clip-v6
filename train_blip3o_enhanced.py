#!/usr/bin/env python3
"""
UPDATED: Enhanced BLIP3-o Training Script - Training Only (No Evaluation)
train_blip3o_enhanced.py

CHANGES:
1. Disabled all evaluation during training
2. Only reports loss, learning rate, and training metrics
3. Simplified for single shard training
4. Post-training evaluation done separately
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
import time
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging(local_rank=0, log_dir=None):
    """Setup comprehensive logging"""
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'blip3o_training_rank_{local_rank}.log'
    else:
        log_file = f'blip3o_training_rank_{local_rank}.log'
    
    log_level = logging.INFO if local_rank == 0 else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format=f'[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments - Training only mode"""
    parser = argparse.ArgumentParser(
        description="BLIP3-o Training - No Evaluation During Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output paths
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints and results")
    
    # Training mode selection
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["cls_patch", "patch_only"],
                       help="Training mode: cls_patch (257 tokens) or patch_only (256 tokens)")
    
    # Single shard training
    parser.add_argument("--shard_index", type=int, default=0,
                       help="Index of the shard to train on (0-based)")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size configuration")
    parser.add_argument("--hidden_size", type=int, default=768,
                       help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads")
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       choices=["linear", "cosine", "constant"],
                       help="Learning rate scheduler")
    
    # Loss configuration
    parser.add_argument("--normalize_targets", action="store_true", default=True,
                       help="Normalize target embeddings")
    parser.add_argument("--prediction_type", type=str, default="velocity",
                       choices=["velocity", "epsilon"],
                       help="Flow matching prediction type")
    parser.add_argument("--flow_type", type=str, default="rectified",
                       choices=["rectified", "standard"],
                       help="Flow matching type (rectified for BLIP3-o)")
    
    # Hardware configuration
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                       help="Enable gradient checkpointing")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                       help="Number of dataloader workers")
    
    # Logging and saving - NO EVALUATION
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=200,
                       help="Model saving frequency")
    
    # Debugging
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()

def validate_training_setup(args, logger):
    """Validate training setup"""
    logger.info("🔍 Validating training setup...")
    
    # Check embeddings directory
    embeddings_path = Path(args.chunked_embeddings_dir)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_path}")
    
    manifest_path = embeddings_path / "embeddings_manifest.json" 
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        logger.info(f"Loaded manifest: {manifest.get('total_shards', 0)} shards, {manifest.get('total_samples', 0):,} samples")
    else:
        logger.warning("Embeddings manifest not found, proceeding anyway")
        manifest = {}
    
    # Validate mode
    expected_tokens = 257 if args.training_mode == "cls_patch" else 256
    tokens_per_sample = manifest.get('tokens_per_sample', expected_tokens)
    
    if tokens_per_sample != expected_tokens:
        logger.warning(f"Token mismatch: training expects {expected_tokens}, embeddings have {tokens_per_sample}")
        logger.warning("The dataset will handle this automatically")
    
    logger.info("✅ Training setup validated")
    logger.info(f"   📊 Dataset: {manifest.get('total_shards', 0)} total shards")
    logger.info(f"   🎯 Training mode: {args.training_mode} ({expected_tokens} tokens)")
    logger.info(f"   📦 Training on shard: {args.shard_index}")
    logger.info(f"   🚫 Evaluation during training: DISABLED")
    
    return manifest

def setup_device_and_distributed():
    """Setup device and distributed training"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    is_distributed = world_size > 1
    is_main_process = global_rank == 0
    
    if is_distributed:
        try:
            if not dist.is_initialized():
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                dist.init_process_group(backend=backend, init_method='env://')
            
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
            else:
                device = torch.device("cpu")
        except Exception as e:
            print(f"⚠️ Distributed setup failed: {e}, using single GPU")
            is_distributed = False
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device, is_distributed, is_main_process, local_rank, global_rank, world_size

def create_model_safely(args, device, logger):
    """Create model"""
    logger.info("🏗️ Creating model...")
    
    try:
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
        
        # Size configurations
        size_configs = {
            "tiny": {"hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 6},
            "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8}, 
            "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
            "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16},
        }
        
        base_config = size_configs.get(args.model_size, size_configs["base"])
        
        config = BLIP3oDiTConfig(
            hidden_size=args.hidden_size or base_config["hidden_size"],
            num_hidden_layers=args.num_layers or base_config["num_hidden_layers"], 
            num_attention_heads=args.num_heads or base_config["num_attention_heads"],
            training_mode=args.training_mode,
            num_tokens=257 if args.training_mode == "cls_patch" else 256,
            max_position_embeddings=257,
            use_gradient_checkpointing=False,
        )
        
        model = create_blip3o_patch_dit_model(config=config)
        model = model.to(device)
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing and device.type != "cpu":
            try:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logger.info("✅ Gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"⚠️ Gradient checkpointing failed: {e}")
        
        logger.info(f"✅ BLIP3-o model created successfully")
        logger.info(f"   Parameters: {model.get_num_parameters():,}")
        logger.info(f"   Mode: {args.training_mode} ({config.num_tokens} tokens)")
        logger.info(f"   Device: {device}")
        
        return model, config
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise

def main():
    """Main training function - Training only, no evaluation"""
    args = parse_arguments()
    
    # Setup device and distributed training
    device, is_distributed, is_main_process, local_rank, global_rank, world_size = setup_device_and_distributed()
    
    # Setup logging
    log_dir = Path(args.output_dir) / "logs" if is_main_process else None
    logger = setup_logging(local_rank, log_dir)
    
    if is_main_process:
        print("🚀 BLIP3-o Training - No Evaluation Mode")
        print("=" * 50)
        print(f"  ✅ Training mode: {args.training_mode}")
        print(f"  ✅ Training on shard: {args.shard_index}")
        print(f"  🚫 Evaluation during training: DISABLED")
        print(f"  📊 Will report: Loss, Learning Rate, Training Metrics")
        print("=" * 50)
    
    try:
        # 1. Validate setup
        manifest = validate_training_setup(args, logger)
        
        # 2. Load modules
        from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        from src.modules.trainers.blip3o_training_only_trainer import BLIP3oTrainingOnlyTrainer, create_training_only_args
        from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
        
        # 3. Create model
        model, config = create_model_safely(args, device, logger)
        
        # 4. Wrap with DDP if needed
        if is_distributed and device.type != "cpu":
            try:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,
                )
                if is_main_process:
                    logger.info("✅ Model wrapped with DDP")
            except Exception as e:
                logger.warning(f"⚠️ DDP wrapping failed: {e}")
                raise
        
        # 5. Create loss function
        flow_matching_loss = create_blip3o_flow_matching_loss(
            prediction_type=args.prediction_type,
            normalize_targets=args.normalize_targets,
            flow_type=args.flow_type,
        )
        
        logger.info("✅ Flow matching loss created")
        
        # 6. Create dataloaders - ONLY training data
        logger.info("🔄 Creating training dataloader...")
        
        train_dataloader, _ = create_flexible_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            eval_batch_size=None,  # No evaluation
            eval_split_ratio=0.0,  # No evaluation split
            normalize_embeddings=True,
            training_mode=args.training_mode,
            max_shards=1,  # Single shard
            use_same_data_for_eval=False,  # No evaluation
            delete_after_use=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=torch.cuda.is_available() and device.type != "cpu",
        )
        
        logger.info(f"✅ Training dataloader created for single shard")
        
        # 7. Create training arguments - NO EVALUATION
        training_args = create_training_only_args(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16 and device.type != "cpu",
            dataloader_num_workers=args.dataloader_num_workers,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
        )
        
        # 8. Create trainer - NO EVALUATION
        trainer = BLIP3oTrainingOnlyTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=train_dataloader.dataset if hasattr(train_dataloader, 'dataset') else None,
            training_mode=args.training_mode,
        )
        
        # Override dataloader
        trainer.get_train_dataloader = lambda: train_dataloader
        
        logger.info("✅ Training-only trainer created")
        logger.info(f"   Mode: {args.training_mode}")
        logger.info(f"   Will report: Loss, Learning Rate, Training Metrics")
        logger.info(f"   No evaluation during training")
        
        # 9. Start training
        logger.info("🚀 Starting training...")
        logger.info(f"   Training for {args.num_epochs} epochs")
        logger.info(f"   Logging every {args.logging_steps} steps")
        logger.info(f"   Saving every {args.save_steps} steps")
        
        # Start training
        train_result = trainer.train()
        
        # 10. Save results
        if is_main_process:
            logger.info("💾 Saving model and results...")
            trainer.save_model()
            
            # Save training info
            training_info = {
                'training_completed': True,
                'training_mode': args.training_mode,
                'expected_tokens': config.num_tokens,
                'shard_index': args.shard_index,
                'evaluation_during_training': False,
                'total_epochs': args.num_epochs,
                'learning_rate': args.learning_rate,
                'final_loss': trainer.state.log_history[-1].get('train_loss', 'unknown') if trainer.state.log_history else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'next_steps': [
                    'Run evaluation script: python eval_blip3o_patch_similarity.py',
                    f'Use model path: {args.output_dir}',
                    f'Use embeddings: {args.chunked_embeddings_dir}',
                    f'Use training mode: {args.training_mode}'
                ]
            }
            
            with open(Path(args.output_dir) / 'training_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"   Model saved to: {args.output_dir}")
            logger.info(f"   Training mode: {args.training_mode}")
            logger.info(f"   Next step: Run evaluation script")
        
        # Cleanup
        if is_distributed:
            dist.destroy_process_group()
        
        return 0
        
    except Exception as e:
        if is_main_process:
            logger.error(f"❌ Training failed: {e}")
            if args.debug:
                traceback.print_exc()
        
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)