#!/usr/bin/env python3
"""
FIXED: CLIP Reproduction Training Script - train_dit_fixed.py
Updated training script that uses all the FIXED components for consistent training and evaluation.

Key updates:
1. Uses FIXED loss function with disabled adaptive noise scaling
2. Uses FIXED dataset with consistent processing
3. Uses FIXED model with proper generation
4. Uses FIXED trainer with consistent evaluation
5. Better debugging and monitoring

Usage:
    python train_dit_fixed.py --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints
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
            logging.FileHandler('fixed_clip_reproduction_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FIXED CLIP Reproduction from EVA Embeddings")
    
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
    
    # BLIP3-o architecture features
    parser.add_argument("--use_3d_rope", action="store_true", default=True,
                       help="Use 3D Rotary Position Embedding")
    parser.add_argument("--use_sandwich_norm", action="store_true", default=True,
                       help="Use sandwich normalization")
    parser.add_argument("--no_3d_rope", action="store_true",
                       help="Disable 3D RoPE")
    parser.add_argument("--no_sandwich_norm", action="store_true",
                       help="Disable sandwich normalization")
    
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
    
    # FIXED: Noise scaling options
    parser.add_argument("--use_adaptive_noise_scaling", action="store_true",
                       help="Enable adaptive noise scaling (not recommended)")
    parser.add_argument("--fixed_noise_scale", type=float, default=1.0,
                       help="Fixed noise scale to use")
    
    # Evaluation
    parser.add_argument("--eval_every_n_steps", type=int, default=50,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=20,
                       help="Number of inference steps for evaluation")
    
    # Debugging and testing
    parser.add_argument("--overfit_test_size", type=int, default=None,
                       help="Size for overfitting test (None to disable)")
    parser.add_argument("--debug_mode", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--max_shards", type=int, default=2,
                       help="Maximum number of shards to use")
    
    # System
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    # WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Enable WandB logging")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="fixed-blip3o-clip-reproduction",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    return parser.parse_args()

def setup_device_and_model(args, logger):
    """Setup device and create FIXED model"""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Process architecture arguments
    use_3d_rope = args.use_3d_rope and not args.no_3d_rope
    use_sandwich_norm = args.use_sandwich_norm and not args.no_sandwich_norm
    
    logger.info("🏗️ FIXED BLIP3-o Architecture Configuration:")
    logger.info(f"  3D Rotary Position Embedding: {'✅ Enabled' if use_3d_rope else '❌ Disabled'}")
    logger.info(f"  Sandwich Normalization: {'✅ Enabled' if use_sandwich_norm else '❌ Disabled'}")
    
    # Import and create FIXED model
    try:
        from src.modules.models.blip3o_dit import create_clip_reproduction_model, BLIP3oCLIPDiTConfig
        logger.info("✅ Imported FIXED model")
        
        model = create_clip_reproduction_model(
            model_size=args.model_size,
            training_mode=args.training_mode,
            use_3d_rope=use_3d_rope,
            use_sandwich_norm=use_sandwich_norm,
        )
        
    except ImportError as e:
        logger.error(f"❌ Could not import FIXED model: {e}")
        logger.error("Make sure blip3o_dit.py is in the current directory")
        raise
    
    model = model.to(device)
    logger.info(f"FIXED model created with {model.get_num_parameters():,} parameters")
    logger.info(f"Model moved to {device}")
    
    return device, model

def create_loss_function(args, logger):
    """Create FIXED loss function"""
    try:
        from src.modules.losses.blip3o_fm_loss import create_clip_reproduction_loss
        logger.info("✅ Imported FIXED loss function")
        
        # FIXED: Use consistent noise scaling settings
        loss_fn = create_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            loss_weight=1.0,
            use_adaptive_noise_scaling=args.use_adaptive_noise_scaling,  # Usually False
            fixed_noise_scale=args.fixed_noise_scale,
            debug_mode=args.debug_mode
        )
        
        logger.info(f"FIXED loss function created:")
        logger.info(f"  Adaptive noise scaling: {args.use_adaptive_noise_scaling}")
        logger.info(f"  Fixed noise scale: {args.fixed_noise_scale}")
        
    except ImportError as e:
        logger.error(f"❌ Could not import FIXED loss function: {e}")
        logger.error("Make sure blip3o_fm_loss.py is in the current directory")
        raise
    
    return loss_fn

def create_dataloaders(args, logger):
    """Create FIXED data loaders"""
    try:
        from src.modules.datasets.blip3o_dataset import create_clip_reproduction_dataloaders
        logger.info("✅ Imported FIXED dataset")
        
        # FIXED: Use consistent settings
        train_dataloader, eval_dataloader = create_clip_reproduction_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            normalize_embeddings=False,  # FIXED: No normalization
            collect_statistics=False,    # FIXED: No statistics collection
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"FIXED dataloaders created:")
        logger.info(f"  Normalization: Disabled")
        logger.info(f"  Statistics collection: Disabled")
        logger.info(f"  Max shards: {args.max_shards}")
        
    except ImportError as e:
        logger.error(f"❌ Could not import FIXED dataset: {e}")
        logger.error("Make sure blip3o_dataset.py is in the current directory")
        raise
    
    return train_dataloader, eval_dataloader

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create FIXED trainer"""
    try:
        from src.modules.trainers.blip3o_trainer import create_clip_trainer
        logger.info("✅ Imported FIXED trainer")
        
        # Create run name if not provided
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None and args.use_wandb and not args.no_wandb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            arch_features = []
            if getattr(model.config, 'use_3d_rope', False):
                arch_features.append("3drope")
            if getattr(model.config, 'use_sandwich_norm', False):
                arch_features.append("sandwich")
            arch_str = "_".join(arch_features) if arch_features else "standard"
            wandb_run_name = f"fixed_blip3o_{args.model_size}_{args.training_mode}_{arch_str}_{timestamp}"
        
        # WandB configuration
        wandb_config = {
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "use_3d_rope": getattr(model.config, 'use_3d_rope', False),
            "use_sandwich_norm": getattr(model.config, 'use_sandwich_norm', False),
            "batch_size": args.batch_size,
            "max_shards": args.max_shards,
            "fixed_version": True,
            "adaptive_noise_scaling": args.use_adaptive_noise_scaling,
            "fixed_noise_scale": args.fixed_noise_scale,
            "experiment_version": "fixed_v1_consistent_scaling",
        }
        
        trainer = create_clip_trainer(
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
            device=device,
            # WandB parameters
            use_wandb=args.use_wandb and not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config,
        )
        
        logger.info("FIXED trainer created:")
        logger.info(f"  Evaluation: Every {args.eval_every_n_steps} steps")
        logger.info(f"  Inference steps: {args.eval_inference_steps}")
        logger.info(f"  WandB enabled: {args.use_wandb and not args.no_wandb}")
        
    except ImportError as e:
        logger.error(f"❌ Could not import FIXED trainer: {e}")
        logger.error("Make sure blip3o_trainer.py is in the current directory")
        raise
    
    return trainer

def main():
    """Main FIXED training function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 FIXED CLIP Reproduction Training with BLIP3-o DiT")
    logger.info("=" * 80)
    logger.info("🔧 FIXES APPLIED:")
    logger.info("  ✅ Disabled adaptive noise scaling for consistency")
    logger.info("  ✅ Fixed data processing (no statistics collection)")
    logger.info("  ✅ Consistent evaluation with reference statistics")
    logger.info("  ✅ Proper noise scaling in generation")
    logger.info("  ✅ Simplified trainer without complex synchronization")
    logger.info("=" * 80)
    logger.info("EXPERIMENT DETAILS:")
    logger.info("  📋 Task: Reproduce clean CLIP embeddings from EVA embeddings")
    logger.info("  🧠 Model: FIXED BLIP3-o DiT with consistent scaling")
    logger.info("  🎯 Target: CLIP embeddings [B, N, 1024]")
    logger.info("  🎮 Conditioning: EVA embeddings [B, N, 4096]")
    logger.info("  🌊 Method: Rectified Flow Matching with FIXED noise handling")
    logger.info("  🚫 Normalization: MINIMAL (only for evaluation similarity)")
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
    logger.info(f"  Adaptive noise scaling: {args.use_adaptive_noise_scaling}")
    logger.info(f"  Fixed noise scale: {args.fixed_noise_scale}")
    if args.overfit_test_size:
        logger.info(f"  🧪 OVERFITTING TEST: {args.overfit_test_size} samples")
    logger.info(f"  Debug mode: {args.debug_mode}")
    if args.use_wandb and not args.no_wandb:
        logger.info(f"  📊 WandB project: {args.wandb_project}")
    logger.info("=" * 80)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device and model
        device, model = setup_device_and_model(args, logger)
        
        # Create FIXED loss function
        loss_fn = create_loss_function(args, logger)
        
        # Create FIXED dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Create FIXED trainer
        trainer = create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        config = {
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            'model_params': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'fixed_clip_reproduction_blip3o',
            'normalization_approach': 'minimal_with_fixed_noise_scaling',
            'fixes_applied': [
                'disabled_adaptive_noise_scaling',
                'fixed_data_processing_consistency',
                'consistent_evaluation_pipeline',
                'reference_based_generation_scaling',
                'simplified_trainer_logic',
                'removed_complex_statistics_tracking',
                'better_debugging_and_monitoring'
            ],
            'architecture_features': {
                '3d_rope': getattr(model.config, 'use_3d_rope', False),
                'sandwich_normalization': getattr(model.config, 'use_sandwich_norm', False),
                'grouped_query_attention': True,
                'minimal_normalization': True,
                'fixed_noise_scaling': True,
            },
            'noise_scaling': {
                'adaptive': args.use_adaptive_noise_scaling,
                'fixed_scale': args.fixed_noise_scale,
                'method': 'target_based' if not args.use_adaptive_noise_scaling else 'adaptive'
            },
            'wandb_config': {
                'enabled': args.use_wandb and not args.no_wandb,
                'project': args.wandb_project,
                'run_name': args.wandb_run_name,
            }
        }
        
        config_path = output_dir / 'fixed_experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"FIXED configuration saved to {config_path}")
        
        # Start training
        logger.info("\n🚀 Starting FIXED BLIP3-o training...")
        logger.info("Expected behavior with FIXES:")
        logger.info("  • Consistent noise scaling between training and evaluation")
        logger.info("  • Much better norm consistency (target vs generated)")
        logger.info("  • Higher cosine similarity (should reach >0.8 for overfitting)")
        logger.info("  • No more adaptive noise scale drift")
        logger.info("  • Stable and reproducible evaluation results")
        logger.info("  • Better debugging information")
        
        if args.overfit_test_size:
            logger.info(f"  • OVERFITTING TEST: Should achieve >0.8 similarity on {args.overfit_test_size} samples")
        
        logger.info("")
        
        start_time = datetime.now()
        
        # Run FIXED training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 FIXED BLIP3-o TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📊 RESULTS SUMMARY:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        logger.info(f"  🔧 FIXED version with consistent scaling")
        
        # Compare with expected improvements
        best_sim = summary.get('best_eval_similarity', 0)
        if best_sim > 0.8:
            logger.info(f"  🎉 EXCELLENT: Similarity >0.8 - Fixes worked perfectly!")
        elif best_sim > 0.5:
            logger.info(f"  ✅ GOOD: Similarity >0.5 - Significant improvement!")
        elif best_sim > 0.3:
            logger.info(f"  📈 BETTER: Similarity >0.3 - Some improvement seen")
        else:
            logger.info(f"  ⚠️  STILL ISSUES: Similarity <0.3 - May need more fixes")
        
        # Evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 FINAL FIXED EVALUATION:")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  Generated norm: {final_eval.get('eval_generated_norm_mean', 0):.3f}")
            logger.info(f"  Target norm: {final_eval.get('eval_target_norm_mean', 0):.3f}")
            logger.info(f"  Norm ratio: {final_eval.get('eval_norm_ratio', 0):.3f}")
            logger.info(f"  Norm consistency: {final_eval.get('eval_norm_consistency', 0):.3f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
            
            # Assess norm consistency improvement
            norm_ratio = final_eval.get('eval_norm_ratio', 0)
            if 0.9 <= norm_ratio <= 1.1:
                logger.info(f"  🎉 EXCELLENT norm consistency!")
            elif 0.8 <= norm_ratio <= 1.2:
                logger.info(f"  ✅ GOOD norm consistency!")
            else:
                logger.info(f"  ⚠️  Norm consistency still needs work")
        
        # Overfitting test results
        if args.overfit_test_size:
            overfit_success = summary.get('overfit_success', False)
            logger.info(f"🧪 OVERFITTING TEST: {'✅ PASSED' if overfit_success else '❌ FAILED'}")
            if overfit_success:
                logger.info("   ✅ FIXED model can learn and memorize effectively!")
            else:
                logger.info("   ⚠️  Model still struggles - may need more fixes")
        
        # WandB information
        if summary.get('wandb_enabled', False):
            logger.info(f"📊 WandB Dashboard: Check your {args.wandb_project} project for detailed metrics")
        
        # Save final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        summary['fixes_applied'] = config['fixes_applied']
        
        summary_path = output_dir / 'fixed_final_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📁 FIXED summary saved to {summary_path}")
        logger.info(f"📁 Model checkpoints saved to {output_dir}")
        
        logger.info("=" * 80)
        logger.info("🔧 COMPARISON TIPS:")
        logger.info("  • Compare this run with your previous results")
        logger.info("  • Look for improved norm consistency in logs")
        logger.info("  • Check if cosine similarity is much higher")
        logger.info("  • Verify evaluation is more stable/consistent")
        logger.info("  • Run debug script: python debug_clip_issues.py <embeddings_dir>")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ FIXED training failed with error: {e}")
        traceback.print_exc()
        return 1
    
    except KeyboardInterrupt:
        logger.info("FIXED training interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)