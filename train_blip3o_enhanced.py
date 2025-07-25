#!/usr/bin/env python3
"""
FIXED: BLIP3-o Enhanced Training Script
train_blip3o_enhanced.py

Simple, working training script for BLIP3-o DiT with flow matching
"""

import os
import sys
import argparse
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Setup CUDA environment
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BLIP3-o Enhanced Training")
    
    # Required arguments
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Training configuration
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=200,
                       help="Save frequency")
    
    # Scaling parameters (CRITICAL FIXES)
    parser.add_argument("--velocity_scale", type=float, default=0.1,
                       help="Velocity scaling factor")
    parser.add_argument("--output_scale", type=float, default=0.1,
                       help="Output scaling factor")
    parser.add_argument("--target_norm_scale", type=float, default=1.0,
                       help="Target norm scaling")
    
    # Options
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--max_training_shards", type=int, default=1,
                       help="Maximum training shards")
    
    return parser.parse_args()

def setup_device(logger):
    """Setup device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def create_model(args, logger):
    """Create BLIP3-o model"""
    from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
    
    logger.info(f"Creating {args.model_size} model for {args.training_mode} mode...")
    
    # Model configurations
    size_configs = {
        "tiny": {"hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 6},
        "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16},
    }
    
    config_params = size_configs[args.model_size].copy()
    config_params.update({
        "num_tokens": 257 if args.training_mode == "cls_patch" else 256,
        "training_mode": args.training_mode,
        "output_scale": args.output_scale,
        "use_gradient_checkpointing": False,
    })
    
    config = BLIP3oDiTConfig(**config_params)
    model = create_blip3o_patch_dit_model(config=config)
    
    logger.info(f"✅ Model created: {model.get_num_parameters():,} parameters")
    return model

def create_loss_function(args, logger):
    """Create flow matching loss"""
    from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
    
    logger.info("Creating flow matching loss with scaling fixes...")
    
    loss_fn = create_blip3o_flow_matching_loss(
        velocity_scale=args.velocity_scale,
        target_norm_scale=args.target_norm_scale,
        adaptive_scaling=False,
        prediction_type="velocity",
        normalize_targets=True,
        flow_type="rectified",
    )
    
    logger.info(f"✅ Loss created with velocity_scale={args.velocity_scale}")
    return loss_fn

def create_dataloaders(args, logger):
    """Create data loaders"""
    from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
    
    logger.info("Creating dataloaders...")
    
    train_dataloader, eval_dataloader = create_flexible_dataloaders(
        chunked_embeddings_dir=args.chunked_embeddings_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        eval_split_ratio=0.0,  # No evaluation during training
        normalize_embeddings=False,
        training_mode=args.training_mode,
        max_shards=args.max_training_shards,
        use_same_data_for_eval=True,
        delete_after_use=False,
        num_workers=0,
        pin_memory=False,
    )
    
    logger.info(f"✅ Dataloaders created: {len(train_dataloader)} batches")
    return train_dataloader, eval_dataloader

def create_trainer(model, loss_fn, train_dataloader, args, logger):
    """Create trainer"""
    from src.modules.trainers.blip3o_trainer import BLIP3oTrainer, create_training_args
    
    logger.info("Creating trainer...")
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        dataloader_num_workers=0,
    )
    
    # Create trainer
    trainer = BLIP3oTrainer(
        model=model,
        args=training_args,
        flow_matching_loss=loss_fn,
        train_dataset=None,  # We use custom dataloader
        training_mode=args.training_mode,
    )
    
    # Override dataloader
    trainer.get_train_dataloader = lambda: train_dataloader
    
    logger.info("✅ Trainer created")
    return trainer

def save_training_info(args, output_dir, logger):
    """Save training configuration"""
    training_info = {
        'training_completed': True,
        'timestamp': datetime.now().isoformat(),
        'training_mode': args.training_mode,
        'model_size': args.model_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'scaling_fixes': {
            'velocity_scale': args.velocity_scale,
            'output_scale': args.output_scale,
            'target_norm_scale': args.target_norm_scale,
        },
        'embeddings_dir': args.chunked_embeddings_dir,
        'output_dir': args.output_dir,
        'max_training_shards': args.max_training_shards,
    }
    
    info_file = Path(output_dir) / "training_info.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"Training info saved to: {info_file}")

def main():
    """Main training function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 Starting BLIP3-o Enhanced Training")
    logger.info("=" * 50)
    logger.info(f"Training mode: {args.training_mode}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Embeddings: {args.chunked_embeddings_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Velocity scale: {args.velocity_scale}")
    logger.info(f"Output scale: {args.output_scale}")
    logger.info("=" * 50)
    
    try:
        # Setup
        device = setup_device(logger)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create components
        model = create_model(args, logger)
        loss_fn = create_loss_function(args, logger)
        train_dataloader, _ = create_dataloaders(args, logger)
        
        # Move model to device
        model = model.to(device)
        
        # Create trainer
        trainer = create_trainer(model, loss_fn, train_dataloader, args, logger)
        
        # Start training
        logger.info("🚀 Starting training...")
        start_time = datetime.now()
        
        trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save model
        trainer.save_model()
        
        # Save training info
        save_training_info(args, args.output_dir, logger)
        
        logger.info("=" * 50)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)