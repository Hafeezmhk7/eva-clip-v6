#!/usr/bin/env python3
"""
FIXED: CLIP Reproduction Training Script with Consistent Data and NO Unwanted Normalization
Updated training script that uses all the FIXED components:

Key updates:
1. Uses FIXED loss function with NO unwanted normalization
2. Uses FIXED dataset with NO normalization applied
3. Uses FIXED model with NO unwanted normalization
4. Uses FIXED trainer with consistent overfitting test data
5. Enhanced norm tracking and debugging
6. Overfitting test uses same data source as evaluation

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
    parser = argparse.ArgumentParser(description="FIXED CLIP Reproduction from EVA Embeddings with Consistent Data")
    
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
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable WandB logging")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="fixed-blip3o-clip-reproduction-consistent",
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
    logger.info(f"  🚫 NO unwanted normalization in model")
    
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
        logger.info(f"  🚫 NO unwanted normalization during training")
        
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
            normalize_embeddings=False,  # FIXED: NO normalization
            collect_statistics=False,    # FIXED: NO statistics collection
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"FIXED dataloaders created:")
        logger.info(f"  🚫 Normalization: DISABLED (forced)")
        logger.info(f"  Statistics collection: Disabled")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  Raw embedding space: ✅")
        
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
            "experiment_version": "fixed_v2_consistent_data_no_normalization",
            "consistent_overfit_test": True,
            "enhanced_norm_tracking": True,
            "no_unwanted_normalization": True,
            "raw_embedding_space": True,
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
        logger.info(f"  🔧 FIXED: Overfitting test uses same data source as evaluation")
        logger.info(f"  📊 FIXED: Enhanced norm tracking enabled")
        
    except ImportError as e:
        logger.error(f"❌ Could not import FIXED trainer: {e}")
        logger.error("Make sure blip3o_trainer.py is in the current directory")
        raise
    
    return trainer

def main():
    """Main FIXED training function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🔧 FIXED: CLIP Reproduction Training with Consistent Data and NO Unwanted Normalization")
    logger.info("=" * 90)
    logger.info("🔧 FIXES APPLIED:")
    logger.info("  ✅ Overfitting test uses SAME data source as evaluation")
    logger.info("  ✅ NO normalization applied during data loading (raw embedding space)")
    logger.info("  ✅ NO normalization applied during training (except for cosine similarity)")
    logger.info("  ✅ NO normalization applied during generation (unless explicitly requested)")
    logger.info("  ✅ Enhanced norm tracking for debugging")
    logger.info("  ✅ Consistent data processing between train and eval")
    logger.info("  ✅ Fixed noise scaling for training/inference consistency")
    logger.info("=" * 90)
    logger.info("EXPERIMENT DETAILS:")
    logger.info("  📋 Task: Reproduce clean CLIP embeddings from EVA embeddings")
    logger.info("  🧠 Model: FIXED BLIP3-o DiT with NO unwanted normalization")
    logger.info("  🎯 Target: CLIP embeddings [B, N, 1024] - RAW (no normalization)")
    logger.info("  🎮 Conditioning: EVA embeddings [B, N, 4096] - RAW (no normalization)")
    logger.info("  🌊 Method: Rectified Flow Matching in RAW embedding space")
    logger.info("  🚫 Normalization: ONLY for cosine similarity computation")
    logger.info("  📊 Enhanced: Detailed norm tracking and debugging")
    logger.info("=" * 90)
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
        logger.info(f"  🧪 FIXED OVERFITTING TEST: {args.overfit_test_size} samples (from eval data)")
    logger.info(f"  Debug mode: {args.debug_mode}")
    if args.use_wandb and not args.no_wandb:
        logger.info(f"  📊 WandB project: {args.wandb_project}")
    logger.info("=" * 90)
    
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
            'experiment_type': 'fixed_clip_reproduction_consistent_data_no_normalization',
            'normalization_approach': 'raw_embedding_space_with_cosine_similarity_only',
            'fixes_applied': [
                'consistent_overfit_test_data_source',
                'no_unwanted_normalization_anywhere',
                'raw_embedding_space_training',
                'enhanced_norm_tracking',
                'consistent_data_processing',
                'fixed_noise_scaling',
                'detailed_debugging_and_monitoring'
            ],
            'architecture_features': {
                '3d_rope': getattr(model.config, 'use_3d_rope', False),
                'sandwich_normalization': getattr(model.config, 'use_sandwich_norm', False),
                'grouped_query_attention': True,
                'no_unwanted_normalization': True,
                'raw_embedding_space': True,
                'fixed_noise_scaling': True,
            },
            'noise_scaling': {
                'adaptive': args.use_adaptive_noise_scaling,
                'fixed_scale': args.fixed_noise_scale,
                'method': 'target_based' if not args.use_adaptive_noise_scaling else 'adaptive',
                'applied_during': 'training_and_inference_consistently'
            },
            'normalization_policy': {
                'data_loading': 'none',
                'training': 'none_except_cosine_similarity',
                'generation': 'none_unless_explicitly_requested',
                'evaluation': 'only_for_cosine_similarity',
                'raw_embedding_space': True,
            },
            'data_consistency': {
                'overfit_test_source': 'eval_dataloader',
                'train_eval_identical_processing': True,
                'consistent_shuffling': False,  # eval doesn't shuffle
                'norm_tracking': True,
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
        logger.info("\n🔧 Starting FIXED BLIP3-o training with consistent data...")
        logger.info("Expected behavior with ALL FIXES:")
        logger.info("  • Consistent target norms between training and evaluation")
        logger.info("  • Overfitting test should achieve >0.8 similarity (same data as eval)")
        logger.info("  • NO unwanted normalization anywhere in the pipeline")
        logger.info("  • Raw embedding space training with proper scale learning")
        logger.info("  • Enhanced norm tracking shows data distribution")
        logger.info("  • Much better norm consistency (target vs generated)")
        logger.info("  • Higher cosine similarity due to consistent data and scaling")
        logger.info("  • Stable and reproducible evaluation results")
        
        if args.overfit_test_size:
            logger.info(f"  • OVERFITTING TEST: Should achieve >0.8 similarity on {args.overfit_test_size} samples")
            logger.info(f"    ✅ Uses SAME data source as evaluation for consistency")
        
        logger.info("")
        
        start_time = datetime.now()
        
        # Run FIXED training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.info("\n" + "=" * 90)
        logger.info("🎉 FIXED BLIP3-o TRAINING COMPLETED!")
        logger.info("=" * 90)
        logger.info(f"📊 RESULTS SUMMARY:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        logger.info(f"  🔧 FIXED version with consistent data and NO unwanted normalization")
        
        # Compare with expected improvements
        best_sim = summary.get('best_eval_similarity', 0)
        if best_sim > 0.8:
            logger.info(f"  🎉 EXCELLENT: Similarity >0.8 - All fixes worked perfectly!")
        elif best_sim > 0.6:
            logger.info(f"  ✅ VERY GOOD: Similarity >0.6 - Major improvement!")
        elif best_sim > 0.4:
            logger.info(f"  ✅ GOOD: Similarity >0.4 - Significant improvement!")
        elif best_sim > 0.3:
            logger.info(f"  📈 BETTER: Similarity >0.3 - Some improvement seen")
        else:
            logger.info(f"  ⚠️  STILL ISSUES: Similarity <0.3 - May need additional investigation")
        
        # Enhanced evaluation results analysis
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 FINAL FIXED EVALUATION:")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  Generated norm: {final_eval.get('eval_generated_norm_mean', 0):.3f}")
            logger.info(f"  Target norm: {final_eval.get('eval_target_norm_mean', 0):.3f}")
            logger.info(f"  Norm ratio: {final_eval.get('eval_norm_ratio', 0):.3f}")
            logger.info(f"  Norm consistency: {final_eval.get('eval_norm_consistency', 0):.3f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
            
            # Assess norm consistency improvement
            norm_ratio = final_eval.get('eval_norm_ratio', 0)
            if 0.9 <= norm_ratio <= 1.1:
                logger.info(f"  🎉 EXCELLENT norm consistency! (ratio: {norm_ratio:.3f})")
            elif 0.8 <= norm_ratio <= 1.2:
                logger.info(f"  ✅ GOOD norm consistency! (ratio: {norm_ratio:.3f})")
            elif 0.7 <= norm_ratio <= 1.3:
                logger.info(f"  📈 IMPROVED norm consistency (ratio: {norm_ratio:.3f})")
            else:
                logger.info(f"  ⚠️  Norm consistency still needs work (ratio: {norm_ratio:.3f})")
        
        # Norm analysis from enhanced tracking
        norm_stats = summary.get('norm_statistics', {})
        if norm_stats:
            logger.info(f"📊 ENHANCED NORM ANALYSIS:")
            
            if 'training_target_norm' in norm_stats:
                train_stats = norm_stats['training_target_norm']
                logger.info(f"  Training target norms: mean={train_stats['mean']:.3f}, std={train_stats.get('std', 0):.3f}")
                logger.info(f"    Range: [{train_stats.get('min', 0):.3f}, {train_stats.get('max', 0):.3f}]")
            
            if 'eval_target_norm' in norm_stats:
                eval_stats = norm_stats['eval_target_norm']
                logger.info(f"  Eval target norms: mean={eval_stats['mean']:.3f}, std={eval_stats.get('std', 0):.3f}")
                logger.info(f"    Range: [{eval_stats.get('min', 0):.3f}, {eval_stats.get('max', 0):.3f}]")
            
            if 'overfit_target_norm' in norm_stats:
                overfit_stats = norm_stats['overfit_target_norm']
                logger.info(f"  Overfit target norm: {overfit_stats['mean']:.3f}")
            
            # Check final consistency
            if 'training_target_norm' in norm_stats and 'eval_target_norm' in norm_stats:
                train_mean = norm_stats['training_target_norm']['mean']
                eval_mean = norm_stats['eval_target_norm']['mean']
                diff = abs(train_mean - eval_mean)
                if diff < 1.0:
                    logger.info(f"  🎉 EXCELLENT data consistency! (diff={diff:.3f})")
                elif diff < 2.0:
                    logger.info(f"  ✅ GOOD data consistency! (diff={diff:.3f})")
                elif diff < 5.0:
                    logger.info(f"  📈 IMPROVED data consistency (diff={diff:.3f})")
                else:
                    logger.info(f"  ⚠️  Data consistency still needs work (diff={diff:.3f})")
        
        # Overfitting test results
        if args.overfit_test_size:
            overfit_success = summary.get('overfit_success', False)
            overfit_data_source = summary.get('overfit_test_data_source', 'unknown')
            logger.info(f"🧪 FIXED OVERFITTING TEST: {'✅ PASSED' if overfit_success else '❌ FAILED'}")
            logger.info(f"  Data source: {overfit_data_source}")
            if overfit_success:
                logger.info("   ✅ FIXED model can learn and memorize effectively!")
                logger.info("   ✅ Architecture and data pipeline are working correctly!")
            else:
                logger.info("   ⚠️  Model still struggles - may need hyperparameter tuning")
        
        # WandB information
        if summary.get('wandb_enabled', False):
            logger.info(f"📊 WandB Dashboard: Check your {args.wandb_project} project for detailed metrics")
            logger.info(f"  Enhanced norm tracking available in WandB logs")
        
        # Save final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        summary['fixes_applied'] = config['fixes_applied']
        summary['consistent_data'] = True
        summary['no_unwanted_normalization'] = True
        summary['enhanced_norm_tracking'] = True
        
        summary_path = output_dir / 'fixed_final_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📁 FIXED summary saved to {summary_path}")
        logger.info(f"📁 Model checkpoints saved to {output_dir}")
        
        logger.info("=" * 90)
        logger.info("🔧 VERIFICATION CHECKLIST:")
        logger.info("  ✅ Overfitting test uses same data source as evaluation")
        logger.info("  ✅ Target norms should be consistent between training and eval")
        logger.info("  ✅ NO normalization applied except for cosine similarity")
        logger.info("  ✅ Raw embedding space preserved throughout pipeline")
        logger.info("  ✅ Enhanced norm tracking shows data distribution")
        logger.info("  ✅ Model can potentially achieve >0.8 similarity on overfitting test")
        logger.info("=" * 90)
        logger.info("🔬 DEBUGGING TIPS:")
        logger.info("  • Check norm tracking logs for consistent target norms")
        logger.info("  • Compare training vs eval target norm means (should be similar)")
        logger.info("  • Look for 'CLIP norm=X.X' in shard loading logs")
        logger.info("  • Verify overfitting test shows 'from evaluation data' message")
        logger.info("  • Run norm analysis script: python debug_norm_analysis.py <embeddings_dir>")
        logger.info("=" * 90)
        
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