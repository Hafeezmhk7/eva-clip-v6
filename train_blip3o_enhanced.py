#!/usr/bin/env python3
"""
FIXED: BLIP3-o Training Script with Proper Scaling and Loss Function
Addresses the scale mismatch and normalization issues
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
        log_file = log_dir / f'blip3o_training_fixed_rank_{local_rank}.log'
    else:
        log_file = f'blip3o_training_fixed_rank_{local_rank}.log'
    
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
        description="FIXED BLIP3-o Training with Proper Scaling",
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
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size configuration")
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    
    # FIXED: Scaling parameters
    parser.add_argument("--velocity_scale", type=float, default=0.1,
                       help="Velocity scaling factor for flow matching")
    parser.add_argument("--output_scale", type=float, default=0.1,
                       help="Output scaling factor for model")
    parser.add_argument("--target_norm_scale", type=float, default=1.0,
                       help="Target normalization scaling factor")
    
    # Hardware configuration
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                       help="Enable gradient checkpointing")
    
    # Logging and saving
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
    
    # Look for shard files
    shard_files = list(embeddings_path.glob("embeddings_shard_*.pkl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {embeddings_path}")
    
    logger.info(f"Found {len(shard_files)} shard files")
    
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
    return manifest, shard_files

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
    logger.info("🏗️ Creating FIXED BLIP3-o model...")
    
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
        
        # Create configuration with FIXED scaling
        config = BLIP3oDiTConfig(
            hidden_size=base_config["hidden_size"],
            num_hidden_layers=base_config["num_hidden_layers"], 
            num_attention_heads=base_config["num_attention_heads"],
            intermediate_size=base_config["intermediate_size"],
            training_mode=args.training_mode,
            num_tokens=257 if args.training_mode == "cls_patch" else 256,
            max_position_embeddings=257,
            use_gradient_checkpointing=False,
            output_scale=args.output_scale,  # FIXED: Add output scaling
        )
        
        # Create model with proper scaling
        model = create_blip3o_patch_dit_model(
            config=config,
            output_scale=args.output_scale,
        )
        model = model.to(device)
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing and device.type != "cpu":
            try:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logger.info("✅ Gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"⚠️ Gradient checkpointing failed: {e}")
        
        logger.info(f"✅ FIXED BLIP3-o model created successfully")
        logger.info(f"   Parameters: {model.get_num_parameters():,}")
        logger.info(f"   Mode: {args.training_mode} ({config.num_tokens} tokens)")
        logger.info(f"   Hidden size: {config.hidden_size}")
        logger.info(f"   Output scale: {config.output_scale}")
        logger.info(f"   Device: {device}")
        
        return model, config
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise

def create_loss_function(args, logger):
    """Create FIXED flow matching loss function"""
    logger.info("🔧 Creating FIXED flow matching loss...")
    
    try:
        from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
        
        flow_matching_loss = create_blip3o_flow_matching_loss(
            prediction_type="velocity",
            normalize_targets=True,
            flow_type="rectified",
            velocity_scale=args.velocity_scale,      # FIXED: Configurable velocity scaling
            target_norm_scale=args.target_norm_scale, # FIXED: Configurable target scaling
        )
        
        logger.info("✅ FIXED BLIP3-o Flow Matching Loss initialized")
        logger.info(f"   Flow type: rectified")
        logger.info(f"   Prediction type: velocity")
        logger.info(f"   Velocity scale: {args.velocity_scale}")
        logger.info(f"   Target norm scale: {args.target_norm_scale}")
        
        return flow_matching_loss
        
    except Exception as e:
        logger.error(f"❌ Loss creation failed: {e}")
        raise

def create_dataloader(args, logger):
    """Create training dataloader"""
    logger.info("🔄 Creating training dataloader...")
    
    try:
        from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
        
        # Create dataloaders
        train_dataloader, _ = create_flexible_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            eval_batch_size=None,
            eval_split_ratio=0.0,
            normalize_embeddings=False,  # Loss function handles normalization
            training_mode=args.training_mode,
            max_shards=1,  # Single shard for overfitting test
            use_same_data_for_eval=False,
            delete_after_use=False,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=torch.cuda.is_available(),
        )
        
        logger.info(f"✅ Training dataloader created")
        logger.info(f"   Training batches: {len(train_dataloader):,}")
        
        return train_dataloader
        
    except Exception as e:
        logger.error(f"❌ Dataloader creation failed: {e}")
        raise

def create_training_args(args):
    """Create training arguments"""
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
        dataloader_num_workers=0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

def create_trainer(model, training_args, flow_matching_loss, train_dataloader, args, logger):
    """Create FIXED trainer"""
    from src.modules.trainers.blip3o_training_only_trainer import BLIP3oTrainingOnlyTrainer
    
    # Create trainer
    trainer = BLIP3oTrainingOnlyTrainer(
        model=model,
        args=training_args,
        flow_matching_loss=flow_matching_loss,
        train_dataset=None,
        training_mode=args.training_mode,
    )
    
    # Override dataloader
    trainer.get_train_dataloader = lambda: train_dataloader
    
    logger.info("✅ FIXED BLIP3oTrainingOnlyTrainer created")
    logger.info(f"   Mode: {args.training_mode}")
    logger.info(f"   Velocity scale: {args.velocity_scale}")
    logger.info(f"   Output scale: {args.output_scale}")
    
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
        print("🚀 FIXED BLIP3-o Training with Proper Scaling")
        print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'local')}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print("🔧 FIXES APPLIED:")
        print("  ✅ Velocity scaling to address norm mismatch")
        print("  ✅ Output scaling in DiT model")
        print("  ✅ Adaptive scaling in loss function")
        print("  ✅ Proper normalization handling")
        print("  ✅ Improved generation timestep schedule")
        print("=" * 60)
        print(f"  ✅ Training mode: {args.training_mode}")
        print(f"  ✅ Velocity scale: {args.velocity_scale}")
        print(f"  ✅ Output scale: {args.output_scale}")
        print(f"  ✅ Target norm scale: {args.target_norm_scale}")
        print("=" * 60)
    
    try:
        # 1. Validate setup
        manifest, shard_files = validate_training_setup(args, logger)
        
        # 2. Create FIXED model
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
        
        # 4. Create FIXED loss function
        flow_matching_loss = create_loss_function(args, logger)
        
        # 5. Create dataloader
        train_dataloader = create_dataloader(args, logger)
        
        # 6. Create training arguments
        training_args = create_training_args(args)
        
        # 7. Create FIXED trainer
        trainer = create_trainer(model, training_args, flow_matching_loss, train_dataloader, args, logger)
        
        # 8. Start training
        logger.info("🚀 Starting FIXED Training Process...")
        logger.info(f"   Training for {args.num_epochs} epochs")
        logger.info(f"   Expected improvements:")
        logger.info(f"     • Better norm alignment (velocity_scale={args.velocity_scale})")
        logger.info(f"     • Proper output scaling (output_scale={args.output_scale})")
        logger.info(f"     • Adaptive loss scaling")
        logger.info(f"     • Should see cosine similarity > 0.1 quickly")
        
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        # 9. Save results
        if is_main_process:
            logger.info("💾 Saving FIXED model and results...")
            trainer.save_model()
            
            # Save training info with fixes
            training_stats = trainer.get_training_statistics()
            training_info = {
                'training_completed': True,
                'training_mode': args.training_mode,
                'expected_tokens': config.num_tokens,
                'total_epochs': args.num_epochs,
                'total_steps': training_stats.get('total_steps', 0),
                'training_time_seconds': training_time,
                'loss_statistics': training_stats.get('loss_statistics', {}),
                'timestamp': datetime.now().isoformat(),
                
                # FIXED parameters
                'fixes_applied': {
                    'velocity_scale': args.velocity_scale,
                    'output_scale': args.output_scale,
                    'target_norm_scale': args.target_norm_scale,
                    'adaptive_scaling': True,
                    'proper_normalization': True,
                    'improved_generation': True,
                },
                'trainer_used': 'BLIP3oTrainingOnlyTrainer (FIXED)',
                'fixed_version': True,
                'model_config': {
                    'hidden_size': config.hidden_size,
                    'num_layers': config.num_hidden_layers,
                    'num_heads': config.num_attention_heads,
                    'num_tokens': config.num_tokens,
                    'output_scale': config.output_scale,
                },
                'expected_improvements': [
                    'Prediction norms should match target norms better',
                    'Cosine similarity should improve significantly',
                    'Loss should decrease more effectively',
                    'Evaluation should show much higher similarities',
                ]
            }
            
            with open(Path(args.output_dir) / 'training_info_fixed.json', 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info("✅ FIXED Training completed successfully!")
            logger.info(f"   Model saved to: {args.output_dir}")
            logger.info(f"   Training mode: {args.training_mode}")
            logger.info(f"   Training time: {training_time:.1f} seconds")
            final_loss = training_stats.get('loss_statistics', {}).get('current_loss', 'unknown')
            logger.info(f"   Final loss: {final_loss}")
            logger.info(f"   Fixes applied: velocity_scale={args.velocity_scale}, output_scale={args.output_scale}")
            
            print("\n✅ FIXED Training completed successfully!")
            print(f"   Model saved to: {args.output_dir}")
            print(f"   Training mode: {args.training_mode} ({config.num_tokens} tokens)")
            print(f"   Final loss: {final_loss}")
            print(f"   Training time: {training_time:.1f} seconds")
            print(f"   Fixes applied: ✅ Scaling, ✅ Normalization, ✅ Generation")
            
            print("\n🎯 Next steps:")
            print("1. Run the FIXED evaluation script to test improvements")
            print("2. Expect much higher cosine similarities (>0.3)")
            print("3. Check that prediction norms now match target norms")
        
        # Cleanup
        if is_distributed:
            dist.destroy_process_group()
        
        return 0
        
    except Exception as e:
        if is_main_process:
            logger.error(f"❌ FIXED Training failed: {e}")
            if args.debug:
                traceback.print_exc()
        
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)