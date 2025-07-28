#!/usr/bin/env python3
"""
Fixed EVA-CLIP Reproduction Training Script
Main training script with comprehensive fixes for gradient flow and architecture issues
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

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('eva_reproduction_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="EVA-CLIP Reproduction with BLIP3-o DiT")
    
    # Required arguments
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size")
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=500,
                       help="Number of samples for evaluation")
    
    # Debugging and testing
    parser.add_argument("--overfit_test_size", type=int, default=None,
                       help="Size for overfitting test (None to disable)")
    parser.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--max_shards", type=int, default=1,
                       help="Maximum number of shards to use")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    return parser.parse_args()

def setup_device_and_model(args, logger):
    """Setup device and create model"""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Clear any cached memory
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Import and create model
    try:
        from src.modules.models.blip3o_eva_dit import create_eva_reproduction_model, BLIP3oEVADiTConfig
    except ImportError:
        logger.error("Could not import fixed model. Make sure fixed_model.py is in the same directory.")
        raise
    
    logger.info(f"Creating {args.model_size} model for {args.training_mode} mode...")
    
    model = create_eva_reproduction_model(
        model_size=args.model_size,
        training_mode=args.training_mode
    )
    
    model = model.to(device)
    
    logger.info(f"Model created with {model.get_num_parameters():,} parameters")
    logger.info(f"Model moved to {device}")
    
    return device, model

def create_loss_function(args, logger):
    """Create loss function"""
    try:
        from src.modules.losses.blip3o_eva_loss import create_eva_reproduction_loss
    except ImportError:
        logger.error("Could not import fixed loss. Make sure fixed_loss.py is in the same directory.")
        raise
    
    logger.info("Creating flow matching loss...")
    
    loss_fn = create_eva_reproduction_loss(
        prediction_type="velocity",
        flow_type="rectified",
        loss_weight=1.0,
        debug_mode=args.debug_mode
    )
    
    logger.info("Flow matching loss created")
    return loss_fn

def get_dataloader_length_safe(dataloader):
    """Safely get dataloader length, handling IterableDataset"""
    try:
        return len(dataloader)
    except TypeError:
        # For IterableDataset, try to get estimated length from dataset
        try:
            return len(dataloader.dataset)
        except:
            # Final fallback - return "unknown"
            return "unknown"

def create_dataloaders(args, logger):
    """Create data loaders"""
    try:
        from src.modules.datasets.blip3o_eva_dataset import create_eva_reproduction_dataloaders
    except ImportError:
        logger.error("Could not import fixed dataset. Make sure fixed_dataset.py is in the same directory.")
        raise
    
    logger.info("Creating dataloaders...")
    
    train_dataloader, eval_dataloader = create_eva_reproduction_dataloaders(
        chunked_embeddings_dir=args.chunked_embeddings_dir,
        batch_size=args.batch_size,
        training_mode=args.training_mode,
        max_shards=args.max_shards,
        normalize_embeddings=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Dataloaders created")
    
    # Safely get lengths
    train_length = get_dataloader_length_safe(train_dataloader)
    if train_length != "unknown":
        logger.info(f"  Training batches: {train_length:,}")
    else:
        logger.info(f"  Training batches: Estimated from IterableDataset")
    
    logger.info(f"  Evaluation available: {eval_dataloader is not None}")
    
    return train_dataloader, eval_dataloader

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create trainer"""
    try:
        from src.modules.trainers.blip3o_eva_trainer import create_eva_trainer
    except ImportError:
        logger.error("Could not import fixed trainer. Make sure fixed_trainer.py is in the same directory.")
        raise
    
    logger.info("Creating trainer...")
    
    trainer = create_eva_trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        eval_every_n_steps=args.eval_every_n_steps,
        eval_num_samples=args.eval_num_samples,
        debug_mode=args.debug_mode,
        overfit_test_size=args.overfit_test_size,
        output_dir=args.output_dir,
        device=device
    )
    
    logger.info("Trainer created")
    return trainer

def main():
    """Main training function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 Starting EVA-CLIP Reproduction with Fixed BLIP3-o DiT")
    logger.info("=" * 80)
    logger.info("EXPERIMENT DETAILS:")
    logger.info("  📋 Task: Reproduce clean EVA embeddings from noisy EVA embeddings")
    logger.info("  🧠 Model: BLIP3-o DiT with 3D RoPE and Grouped-Query Attention")
    logger.info("  🎯 Target: EVA embeddings [B, N, 4096]")
    logger.info("  🎮 Conditioning: CLIP embeddings [B, N, 1024]")
    logger.info("  🌊 Method: Rectified Flow Matching")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model size: {args.model_size}")
    logger.info(f"  Training mode: {args.training_mode}")
    logger.info(f"  Embeddings dir: {args.chunked_embeddings_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Max shards: {args.max_shards}")
    if args.overfit_test_size:
        logger.info(f"  🧪 OVERFITTING TEST: {args.overfit_test_size} samples")
    logger.info(f"  Debug mode: {args.debug_mode}")
    logger.info("=" * 80)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device and model
        device, model = setup_device_and_model(args, logger)
        
        # Create loss function
        loss_fn = create_loss_function(args, logger)
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Create trainer
        trainer = create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        config = {
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model, 'config') else {},
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'eva_reproduction',
            'fixes_applied': [
                'gradient_flow_improvements',
                'proper_initialization',
                'correct_data_flow',
                'numerical_stability',
                'overfitting_test_capability',
                'iterable_dataset_length_fix'
            ]
        }
        
        config_path = output_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
        # Start training
        logger.info("\n🚀 Starting training...")
        logger.info("Expected behavior:")
        logger.info("  • Loss should decrease steadily")
        logger.info("  • Velocity similarity should increase")
        logger.info("  • EVA similarity should improve during evaluation")
        logger.info("  • Gradients should be non-zero and stable")
        
        if args.overfit_test_size:
            logger.info(f"  • OVERFITTING TEST: Should achieve >0.8 similarity on {args.overfit_test_size} samples")
        
        logger.info("")
        
        start_time = datetime.now()
        
        # Run training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📊 RESULTS SUMMARY:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best EVA similarity: {summary.get('best_eval_similarity', 0):.4f}")
        
        # Evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 FINAL EVALUATION:")
            logger.info(f"  EVA similarity: {final_eval.get('eval_eva_similarity', 0):.4f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
            logger.info(f"  Samples evaluated: {final_eval.get('eval_samples', 0)}")
        
        # Overfitting test results
        if args.overfit_test_size:
            overfit_success = summary.get('overfit_success', False)
            logger.info(f"🧪 OVERFITTING TEST: {'✅ PASSED' if overfit_success else '❌ FAILED'}")
            if overfit_success:
                logger.info("   ✅ Model can learn and memorize - architecture is working!")
            else:
                logger.info("   ⚠️  Model struggles to overfit - check architecture/loss")
        
        # Architecture assessment
        best_sim = summary.get('best_eval_similarity', 0)
        logger.info(f"🏗️  ARCHITECTURE ASSESSMENT:")
        if best_sim > 0.7:
            logger.info("   🎉 EXCELLENT: BLIP3-o DiT architecture working perfectly!")
        elif best_sim > 0.4:
            logger.info("   ✅ GOOD: BLIP3-o DiT architecture shows strong capability!")
        elif best_sim > 0.1:
            logger.info("   📈 FAIR: BLIP3-o DiT architecture is functional!")
        else:
            logger.info("   ⚠️  NEEDS WORK: Architecture may need tuning!")
        
        # Save final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        
        summary_path = output_dir / 'final_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📁 Final summary saved to {summary_path}")
        logger.info(f"📁 Model checkpoints saved to {output_dir}")
        
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        traceback.print_exc()
        return 1
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)