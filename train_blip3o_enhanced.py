#!/usr/bin/env python3
"""
FIXED: BLIP3-o Training Script - Training Only (No Evaluation During Training)
train_blip3o_fixed.py

FIXES:
1. Fixed TrainingArguments parameter issue (eval_strategy vs evaluation_strategy)
2. Proper gradient flow handling
3. Clean single-shard training
4. Aligned with fixed trainer implementation
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FIXED BLIP3-o Training - Single Shard Training",
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
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Training batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    
    # Hardware configuration
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                       help="Enable gradient checkpointing")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=5,
                       help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100,
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
    
    # Look for shard files
    shard_patterns = [
        f"embeddings_shard_{args.shard_index:05d}_*.pkl",
        f"embeddings_shard_{args.shard_index}_*.pkl",
        "embeddings_shard_*.pkl"
    ]
    
    shard_file = None
    for pattern in shard_patterns:
        files = list(embeddings_path.glob(pattern))
        if files:
            shard_file = files[0]
            break
    
    if not shard_file:
        raise FileNotFoundError(f"No shard file found for index {args.shard_index} in {embeddings_path}")
    
    logger.info(f"Found shard file: {shard_file}")
    
    # Check manifest
    manifest_path = embeddings_path / "embeddings_manifest.json" 
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        logger.info(f"Loaded manifest: {manifest.get('total_shards', 0)} shards, {manifest.get('total_samples', 0):,} samples")
    else:
        logger.warning("Embeddings manifest not found, proceeding anyway")
    
    logger.info("✅ Training setup validated")
    return manifest, shard_file

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

def create_model_and_config(args, device, logger):
    """Create BLIP3-o model with proper configuration"""
    logger.info("🏗️ Creating BLIP3-o model...")
    
    try:
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
        
        # Model size configurations
        size_configs = {
            "tiny": {"hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 6, "intermediate_size": 1536},
            "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048}, 
            "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
            "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16, "intermediate_size": 4096},
        }
        
        base_config = size_configs.get(args.model_size, size_configs["base"])
        
        # Create configuration
        config = BLIP3oDiTConfig(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_hidden_layers"], 
            num_attention_heads=base_config["num_attention_heads"],
            intermediate_size=base_config["intermediate_size"],
            training_mode=args.training_mode,
            num_tokens=257 if args.training_mode == "cls_patch" else 256,
            max_position_embeddings=257,  # Support both modes
            use_gradient_checkpointing=False,  # Will be enabled later if requested
        )
        
        # Create model
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
        logger.info(f"   Hidden size: {config.hidden_size}")
        logger.info(f"   Layers: {config.num_hidden_layers}")
        logger.info(f"   Heads: {config.num_attention_heads}")
        logger.info(f"   Device: {device}")
        
        return model, config
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise

def create_loss_function(args, logger):
    """Create flow matching loss function"""
    logger.info("🔧 Creating flow matching loss...")
    
    try:
        from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        
        flow_matching_loss = create_blip3o_flow_matching_loss(
            prediction_type="velocity",  # BLIP3-o uses velocity prediction
            normalize_targets=True,      # Normalize targets for consistency
            flow_type="rectified",       # Rectified flow as per paper
        )
        
        logger.info("✅ FIXED BLIP3-o Flow Matching Loss initialized")
        logger.info("   Flow type: rectified")
        logger.info("   Prediction type: velocity")
        logger.info("   Normalize targets: True")
        
        return flow_matching_loss
        
    except Exception as e:
        logger.error(f"❌ Loss creation failed: {e}")
        raise

def create_dataloader(args, logger):
    """Create single-shard training dataloader"""
    logger.info("🔄 Creating training dataloader...")
    
    try:
        from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
        
        # Create dataloaders for single shard training
        train_dataloader, _ = create_flexible_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            eval_batch_size=None,  # No evaluation
            eval_split_ratio=0.0,  # No evaluation split
            normalize_embeddings=False,  # Loss function handles normalization
            training_mode=args.training_mode,
            max_shards=1,  # Single shard training
            use_same_data_for_eval=False,  # No evaluation
            delete_after_use=False,
            num_workers=0,  # FIXED: Avoid multiprocessing issues
            pin_memory=torch.cuda.is_available(),
        )
        
        logger.info(f"✅ Training dataloader created for single shard")
        logger.info(f"   Training batches: {len(train_dataloader):,}")
        
        return train_dataloader
        
    except Exception as e:
        logger.error(f"❌ Dataloader creation failed: {e}")
        raise

def create_training_args(args):
    """Create training arguments using fixed implementation"""
    from src.modules.trainers.blip3o_training_only_trainer import create_training_only_args
    
    return create_training_only_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=0,  # FIXED: Avoid multiprocessing
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

def create_trainer(model, training_args, flow_matching_loss, train_dataloader, args, logger):
    """Create trainer using fixed implementation"""
    from src.modules.trainers.blip3o_training_only_trainer import BLIP3oTrainingOnlyTrainer
    
    # Create trainer
    trainer = BLIP3oTrainingOnlyTrainer(
        model=model,
        args=training_args,
        flow_matching_loss=flow_matching_loss,
        train_dataset=None,  # We'll override dataloader
        training_mode=args.training_mode,
    )
    
    # Override dataloader to use our specific dataloader
    trainer.get_train_dataloader = lambda: train_dataloader
    
    logger.info("✅ FIXED BLIP3oTrainingOnlyTrainer created")
    logger.info(f"   Mode: {args.training_mode}")
    logger.info(f"   Evaluation during training: DISABLED")
    
    return trainer

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup device and distributed training
    device, is_distributed, is_main_process, local_rank, global_rank, world_size = setup_device_and_distributed()
    
    # Setup logging
    log_dir = Path(args.output_dir) / "logs" if is_main_process else None
    logger = setup_logging(local_rank, log_dir)
    
    if is_main_process:
        print("🚀 Clean BLIP3-o Training - No Evaluation During Training")
        print("🚀 Starting BLIP3-o Training-Only Job")
        print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'local')}")
        print(f"Node: {os.environ.get('HOSTNAME', 'local')}")
        print(f"Time: {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}")
        print("=" * 60)
        print(f"  ✅ Training mode: {args.training_mode}")
        print(f"  ✅ Training on shard: {args.shard_index}")
        print(f"  🚫 Evaluation during training: DISABLED (clean separation)")
        print(f"  📊 Will report: Loss, Learning Rate, Training Metrics Only")
        print(f"  🔧 Post-training evaluation: Run separately for clean separation")
        print("=" * 60)
    
    try:
        # 1. Validate setup
        manifest, shard_file = validate_training_setup(args, logger)
        
        # 2. Create model
        model, config = create_model_and_config(args, device, logger)
        
        # 3. Wrap with DDP if needed
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
        
        # 4. Create loss function
        flow_matching_loss = create_loss_function(args, logger)
        
        # 5. Create dataloader
        train_dataloader = create_dataloader(args, logger)
        
        # 6. Create training arguments
        training_args = create_training_args(args)
        
        # 7. Create trainer
        trainer = create_trainer(model, training_args, flow_matching_loss, train_dataloader, args, logger)
        
        # 8. Start training
        logger.info("🚀 Starting Training-Only Process...")
        logger.info("🚀 Clean BLIP3-o Training - No Evaluation During Training")
        logger.info(f"   Training for {args.num_epochs} epochs")
        logger.info(f"   Logging every {args.logging_steps} steps")
        logger.info(f"   Saving every {args.save_steps} steps")
        
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        # 9. Save results
        if is_main_process:
            logger.info("💾 Saving model and results...")
            trainer.save_model()
            
            # Save training info
            training_stats = trainer.get_training_statistics()
            training_info = {
                'training_completed': True,
                'training_mode': args.training_mode,
                'expected_tokens': config.num_tokens,
                'shard_index': args.shard_index,
                'shard_file': str(shard_file),
                'evaluation_during_training': False,
                'total_epochs': args.num_epochs,
                'total_steps': training_stats.get('total_steps', 0),
                'learning_rate': args.learning_rate,
                'training_time_seconds': training_time,
                'loss_statistics': training_stats.get('loss_statistics', {}),
                'timestamp': datetime.now().isoformat(),
                'trainer_used': 'BLIP3oTrainingOnlyTrainer',
                'fixed_version': True,
                'model_config': {
                    'hidden_size': config.hidden_size,
                    'num_layers': config.num_hidden_layers,
                    'num_heads': config.num_attention_heads,
                    'num_tokens': config.num_tokens,
                },
                'next_steps': [
                    'Run evaluation script separately',
                    f'Use model from: {args.output_dir}',
                    f'Use embeddings from: {args.chunked_embeddings_dir}',
                ]
            }
            
            with open(Path(args.output_dir) / 'training_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"   Model saved to: {args.output_dir}")
            logger.info(f"   Training mode: {args.training_mode}")
            logger.info(f"   Training time: {training_time:.1f} seconds")
            final_loss = training_stats.get('loss_statistics', {}).get('current_loss', 'unknown')
            logger.info(f"   Final loss: {final_loss}")
            logger.info(f"   Trainer used: BLIP3oTrainingOnlyTrainer (FIXED)")
            
            print("\n✅ Training completed successfully!")
            print(f"   Model saved to: {args.output_dir}")
            print(f"   Training mode: {args.training_mode} ({config.num_tokens} tokens)")
            print(f"   Final loss: {final_loss}")
            print(f"   Training time: {training_time:.1f} seconds")
            
            # Show next steps
            print("\n🎯 Next steps:")
            print("1. Run evaluation script to test your trained model")
            print("2. Check training logs in the output directory")
            print("3. Inspect model checkpoints")
        
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