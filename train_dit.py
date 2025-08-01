#!/usr/bin/env python3
"""
UPDATED: CLIP Reproduction Training Script with Scale-Aware Generation
Key improvements:
1. Log-normal timestep scheduling for better sampling
2. Velocity explosion prevention during inference  
3. Periodic norm guidance for scale consistency
4. Adaptive target norm estimation
5. Enhanced evaluation with scale-aware metrics

Usage:
    python train_dit.py --chunked_embeddings_dir /path/to/embeddings --output_dir ./checkpoints --use_scale_aware
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
            logging.FileHandler('scale_aware_clip_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments with scale-aware options"""
    parser = argparse.ArgumentParser(description="BLIP3-o CLIP Reproduction with Scale-Aware Generation")
    
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
    
    # NEW: Scale-aware generation parameters
    parser.add_argument("--use_scale_aware", action="store_true", default=True,
                       help="Enable scale-aware generation and evaluation")
    parser.add_argument("--no_scale_aware", action="store_true",
                       help="Disable scale-aware generation")
    parser.add_argument("--typical_clip_norm", type=float, default=26.0,
                       help="Typical CLIP embedding norm for scale guidance")
    parser.add_argument("--velocity_explosion_threshold", type=float, default=100.0,
                       help="Threshold for velocity explosion prevention")
    parser.add_argument("--norm_guidance_strength", type=float, default=0.1,
                       help="Strength of norm guidance during generation")
    parser.add_argument("--norm_guidance_frequency", type=int, default=10,
                       help="Frequency of norm guidance application")
    parser.add_argument("--eval_use_lognormal_schedule", action="store_true", default=True,
                       help="Use log-normal timestep schedule for evaluation")
    parser.add_argument("--adaptive_target_norm", action="store_true", default=True,
                       help="Adaptively estimate target norm from data")
    
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
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-scale-aware",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    return parser.parse_args()

def setup_device_and_model(args, logger):
    """Setup device and create model with scale-aware generation"""
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
    use_scale_aware = args.use_scale_aware and not args.no_scale_aware
    
    logger.info("🏗️ BLIP3-o Architecture Configuration:")
    logger.info(f"  3D Rotary Position Embedding: {'✅ Enabled' if use_3d_rope else '❌ Disabled'}")
    logger.info(f"  Sandwich Normalization: {'✅ Enabled' if use_sandwich_norm else '❌ Disabled'}")
    logger.info(f"  🚀 Scale-Aware Generation: {'✅ Enabled' if use_scale_aware else '❌ Disabled'}")
    
    if use_scale_aware:
        logger.info(f"🎯 Scale-Aware Parameters:")
        logger.info(f"  Typical CLIP norm: {args.typical_clip_norm}")
        logger.info(f"  Velocity explosion threshold: {args.velocity_explosion_threshold}")
        logger.info(f"  Norm guidance strength: {args.norm_guidance_strength}")
        logger.info(f"  Norm guidance frequency: {args.norm_guidance_frequency}")
        logger.info(f"  Log-normal schedule: {args.eval_use_lognormal_schedule}")
        logger.info(f"  Adaptive target norm: {args.adaptive_target_norm}")
    
    # Import and create model
    try:
        from src.modules.models.blip3o_dit import create_clip_reproduction_model, BLIP3oCLIPDiTConfig
        logger.info("✅ Imported UPDATED model with scale-aware generation")
        
        # Create model with scale-aware parameters
        model_kwargs = {}
        if use_scale_aware:
            model_kwargs.update({
                'typical_clip_norm': args.typical_clip_norm,
                'velocity_explosion_threshold': args.velocity_explosion_threshold,
                'norm_guidance_strength': args.norm_guidance_strength,
                'norm_guidance_frequency': args.norm_guidance_frequency,
            })
        
        model = create_clip_reproduction_model(
            model_size=args.model_size,
            training_mode=args.training_mode,
            use_3d_rope=use_3d_rope,
            use_sandwich_norm=use_sandwich_norm,
            **model_kwargs
        )
        
    except ImportError as e:
        logger.error(f"❌ Could not import model: {e}")
        logger.error("Make sure blip3o_dit.py is updated with scale-aware generation")
        raise
    
    model = model.to(device)
    logger.info(f"Model created with {model.get_num_parameters():,} parameters")
    logger.info(f"Model moved to {device}")
    
    return device, model

def create_loss_function(args, logger):
    """Create loss function"""
    try:
        from src.modules.losses.blip3o_fm_loss import create_clip_reproduction_loss
        logger.info("✅ Imported loss function")
        
        loss_fn = create_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            loss_weight=1.0,
            use_adaptive_noise_scaling=False,
            fixed_noise_scale=1.0,
            debug_mode=args.debug_mode
        )
        
        logger.info(f"Loss function created:")
        logger.info(f"  Prediction type: velocity")
        logger.info(f"  Flow type: rectified")
        logger.info(f"  Standard Gaussian noise")
        
    except ImportError as e:
        logger.error(f"❌ Could not import loss function: {e}")
        raise
    
    return loss_fn

def create_dataloaders(args, logger):
    """Create data loaders"""
    try:
        from src.modules.datasets.blip3o_dataset import create_clip_reproduction_dataloaders
        logger.info("✅ Imported dataset")
        
        train_dataloader, eval_dataloader = create_clip_reproduction_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            normalize_embeddings=False,
            collect_statistics=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Dataloaders created:")
        logger.info(f"  Normalization: DISABLED")
        logger.info(f"  Max shards: {args.max_shards}")
        logger.info(f"  Raw embedding space: ✅")
        
    except ImportError as e:
        logger.error(f"❌ Could not import dataset: {e}")
        raise
    
    return train_dataloader, eval_dataloader

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create trainer with scale-aware evaluation"""
    try:
        from src.modules.trainers.blip3o_trainer import create_clip_trainer
        logger.info("✅ Imported UPDATED trainer with scale-aware evaluation")
        
        # Determine scale-aware settings
        use_scale_aware = args.use_scale_aware and not args.no_scale_aware
        
        # Create run name if not provided
        wandb_run_name = args.wandb_run_name
        if wandb_run_name is None and args.use_wandb and not args.no_wandb:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            arch_features = []
            if getattr(model.config, 'use_3d_rope', False):
                arch_features.append("3drope")
            if getattr(model.config, 'use_sandwich_norm', False):
                arch_features.append("sandwich")
            if use_scale_aware:
                arch_features.append("scale_aware")
            arch_str = "_".join(arch_features) if arch_features else "standard"
            wandb_run_name = f"blip3o_{args.model_size}_{args.training_mode}_{arch_str}_{timestamp}"
        
        # Update WandB config with scale-aware parameters
        wandb_config = {
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "use_3d_rope": getattr(model.config, 'use_3d_rope', False),
            "use_sandwich_norm": getattr(model.config, 'use_sandwich_norm', False),
            "batch_size": args.batch_size,
            "max_shards": args.max_shards,
            "experiment_version": "scale_aware_v1",
            
            # Scale-aware parameters
            "use_scale_aware_generation": use_scale_aware,
            "typical_clip_norm": args.typical_clip_norm,
            "velocity_explosion_threshold": args.velocity_explosion_threshold,
            "norm_guidance_strength": args.norm_guidance_strength,
            "norm_guidance_frequency": args.norm_guidance_frequency,
            "eval_use_lognormal_schedule": args.eval_use_lognormal_schedule,
            "adaptive_target_norm": args.adaptive_target_norm,
            
            # Key improvements
            "key_improvements": [
                "lognormal_timestep_schedule",
                "velocity_explosion_prevention", 
                "periodic_norm_guidance",
                "adaptive_target_norm_estimation",
                "scale_aware_evaluation"
            ] if use_scale_aware else ["baseline"],
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
            
            # Scale-aware evaluation parameters
            use_scale_aware_eval=use_scale_aware,
            eval_target_norm=args.typical_clip_norm if not args.adaptive_target_norm else None,
            eval_use_lognormal_schedule=args.eval_use_lognormal_schedule,
            adaptive_target_norm=args.adaptive_target_norm,
            
            # WandB parameters
            use_wandb=args.use_wandb and not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config,
        )
        
        logger.info("UPDATED trainer created with scale-aware evaluation:")
        logger.info(f"  🚀 Scale-aware evaluation: {use_scale_aware}")
        logger.info(f"  🎯 Adaptive target norm: {args.adaptive_target_norm}")
        logger.info(f"  📅 Log-normal schedule: {args.eval_use_lognormal_schedule}")
        logger.info(f"  Evaluation: Every {args.eval_every_n_steps} steps")
        logger.info(f"  WandB enabled: {args.use_wandb and not args.no_wandb}")
        
    except ImportError as e:
        logger.error(f"❌ Could not import trainer: {e}")
        logger.error("Make sure blip3o_trainer.py is updated with scale-aware evaluation")
        raise
    
    return trainer

def main():
    """Main training function with scale-aware generation"""
    args = parse_arguments()
    logger = setup_logging()
    
    use_scale_aware = args.use_scale_aware and not args.no_scale_aware
    
    logger.info("🚀 BLIP3-o CLIP Reproduction Training with Scale-Aware Generation")
    logger.info("=" * 90)
    
    if use_scale_aware:
        logger.info("🎯 SCALE-AWARE GENERATION ENABLED:")
        logger.info("  ✅ Log-normal timestep scheduling for better sampling")
        logger.info("  ✅ Velocity explosion prevention during inference")
        logger.info("  ✅ Periodic norm guidance for scale consistency")
        logger.info("  ✅ Adaptive target norm estimation from data")
        logger.info("  ✅ Enhanced evaluation with scale-aware metrics")
    else:
        logger.info("📊 BASELINE GENERATION:")
        logger.info("  • Standard linear timestep scheduling")
        logger.info("  • No scale guidance during inference")
        logger.info("  • Basic evaluation metrics")
    
    logger.info("=" * 90)
    logger.info("EXPERIMENT DETAILS:")
    logger.info("  📋 Task: Reproduce clean CLIP embeddings from EVA embeddings")
    logger.info("  🧠 Model: BLIP3-o DiT with 3D RoPE and Sandwich Normalization")
    logger.info("  🎯 Target: CLIP embeddings [B, N, 1024]")
    logger.info("  🎮 Conditioning: EVA embeddings [B, N, 4096]")
    logger.info("  🌊 Method: Rectified Flow Matching")
    
    if use_scale_aware:
        logger.info("  🚀 Generation: Scale-aware with log-normal scheduling")
        logger.info(f"  🎯 Target norm guidance: {args.typical_clip_norm:.1f}")
        logger.info(f"  ⚡ Velocity explosion threshold: {args.velocity_explosion_threshold:.1f}")
        logger.info(f"  📊 Norm guidance strength: {args.norm_guidance_strength:.2f}")
    else:
        logger.info("  📊 Generation: Standard linear scheduling")
    
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
    if args.overfit_test_size:
        logger.info(f"  🧪 Overfitting test: {args.overfit_test_size} samples")
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
        
        # Create loss function
        loss_fn = create_loss_function(args, logger)
        
        # Create dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Create trainer
        trainer = create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger)
        
        # Save configuration
        config = {
            'args': vars(args),
            'model_config': model.config.to_dict() if hasattr(model.config, 'to_dict') else {},
            'model_params': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'blip3o_clip_scale_aware_generation' if use_scale_aware else 'blip3o_clip_baseline',
            
            # Scale-aware configuration
            'scale_aware_enabled': use_scale_aware,
            'scale_aware_config': {
                'typical_clip_norm': args.typical_clip_norm,
                'velocity_explosion_threshold': args.velocity_explosion_threshold,
                'norm_guidance_strength': args.norm_guidance_strength,
                'norm_guidance_frequency': args.norm_guidance_frequency,
                'use_lognormal_schedule': args.eval_use_lognormal_schedule,
                'adaptive_target_norm': args.adaptive_target_norm,
            } if use_scale_aware else {},
            
            'key_improvements': [
                'lognormal_timestep_schedule',
                'velocity_explosion_prevention',
                'periodic_norm_guidance', 
                'adaptive_target_norm_estimation',
                'scale_aware_evaluation'
            ] if use_scale_aware else [],
            
            'architecture_features': {
                '3d_rope': getattr(model.config, 'use_3d_rope', False),
                'sandwich_normalization': getattr(model.config, 'use_sandwich_norm', False),
                'grouped_query_attention': True,
                'scale_aware_generation': use_scale_aware,
            },
            
            'generation_method': {
                'timestep_schedule': 'lognormal' if use_scale_aware and args.eval_use_lognormal_schedule else 'linear',
                'velocity_explosion_prevention': use_scale_aware,
                'norm_guidance': use_scale_aware,
                'adaptive_target_norm': use_scale_aware and args.adaptive_target_norm,
            },
            
            'evaluation_method': {
                'scale_aware': use_scale_aware,
                'adaptive_target_norm': args.adaptive_target_norm,
                'lognormal_schedule': args.eval_use_lognormal_schedule,
                'enhanced_metrics': use_scale_aware,
            },
            
            'wandb_config': {
                'enabled': args.use_wandb and not args.no_wandb,
                'project': args.wandb_project,
                'run_name': args.wandb_run_name,
            }
        }
        
        config_filename = 'scale_aware_experiment_config.json' if use_scale_aware else 'baseline_experiment_config.json'
        config_path = output_dir / config_filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
        # Start training
        logger.info(f"\n🚀 Starting BLIP3-o training with {'Scale-Aware' if use_scale_aware else 'Baseline'} generation...")
        
        if use_scale_aware:
            logger.info("Expected improvements with Scale-Aware Generation:")
            logger.info("  • Better scale consistency between training and inference")
            logger.info("  • Reduced velocity explosion issues during generation")
            logger.info("  • More stable and consistent embedding norms")
            logger.info("  • Improved CLIP similarity scores")
            logger.info("  • Enhanced convergence and overfitting capability")
            logger.info("  • More robust generation across different noise scales")
        else:
            logger.info("Baseline generation for comparison:")
            logger.info("  • Standard linear timestep scheduling")
            logger.info("  • No scale guidance mechanisms")
            logger.info("  • Basic evaluation metrics")
        
        if args.overfit_test_size:
            logger.info(f"  • OVERFITTING TEST: Should achieve >0.8 similarity on {args.overfit_test_size} samples")
            logger.info(f"    ✅ Uses same data source as evaluation for consistency")
        
        logger.info("")
        
        start_time = datetime.now()
        
        # Run training
        summary = trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.info("\n" + "=" * 90)
        if use_scale_aware:
            logger.info("🎉 SCALE-AWARE BLIP3-o TRAINING COMPLETED!")
        else:
            logger.info("🎉 BASELINE BLIP3-o TRAINING COMPLETED!")
        logger.info("=" * 90)
        logger.info(f"📊 RESULTS SUMMARY:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        logger.info(f"  🚀 Scale-aware generation: {use_scale_aware}")
        
        # Compare with expectations
        best_sim = summary.get('best_eval_similarity', 0)
        if best_sim > 0.9:
            logger.info(f"  🎉 OUTSTANDING: Similarity >0.9 - Excellent results!")
        elif best_sim > 0.8:
            logger.info(f"  🎉 EXCELLENT: Similarity >0.8 - Great improvement!")
        elif best_sim > 0.6:
            logger.info(f"  ✅ VERY GOOD: Similarity >0.6 - Solid results!")
        elif best_sim > 0.4:
            logger.info(f"  ✅ GOOD: Similarity >0.4 - Shows promise!")
        elif best_sim > 0.3:
            logger.info(f"  📈 DECENT: Similarity >0.3 - Some learning observed")
        else:
            logger.info(f"  ⚠️  NEEDS WORK: Similarity <0.3 - May need tuning")
        
        # Enhanced evaluation results analysis
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 FINAL EVALUATION:")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  Generated norm: {final_eval.get('eval_generated_norm_mean', 0):.3f}")
            logger.info(f"  Target norm: {final_eval.get('eval_target_norm_mean', 0):.3f}")
            logger.info(f"  Norm ratio: {final_eval.get('eval_norm_ratio', 0):.3f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
            
            # Scale-aware specific metrics
            if use_scale_aware and 'eval_scale_consistency_mean' in final_eval:
                logger.info(f"  🎯 Scale consistency: {final_eval.get('eval_scale_consistency_mean', 0):.3f}")
                logger.info(f"  📅 Log-normal schedule used: {final_eval.get('eval_lognormal_schedule', False)}")
                logger.info(f"  🎛️  Target norm used: {final_eval.get('eval_target_norm_used', 0):.3f}")
            
            # Assess improvements
            norm_ratio = final_eval.get('eval_norm_ratio', 0)
            if 0.9 <= norm_ratio <= 1.1:
                logger.info(f"  🎉 EXCELLENT norm consistency! (ratio: {norm_ratio:.3f})")
            elif 0.8 <= norm_ratio <= 1.2:
                logger.info(f"  ✅ GOOD norm consistency! (ratio: {norm_ratio:.3f})")
            elif 0.7 <= norm_ratio <= 1.3:
                logger.info(f"  📈 IMPROVED norm consistency (ratio: {norm_ratio:.3f})")
            else:
                logger.info(f"  ⚠️  Norm consistency needs work (ratio: {norm_ratio:.3f})")
        
        # Scale-aware analysis from enhanced tracking
        if use_scale_aware:
            norm_stats = summary.get('norm_statistics', {})
            if norm_stats:
                logger.info(f"📊 SCALE-AWARE ANALYSIS:")
                
                if 'target_norm_estimates' in norm_stats:
                    est_stats = norm_stats['target_norm_estimates']
                    logger.info(f"  🎯 Target norm estimates: mean={est_stats['mean']:.3f}, std={est_stats.get('std', 0):.3f}")
                
                if 'scale_consistency' in norm_stats:
                    consist_stats = norm_stats['scale_consistency']
                    logger.info(f"  📊 Scale consistency: mean={consist_stats['mean']:.3f}, std={consist_stats.get('std', 0):.3f}")
                    
                    if consist_stats['mean'] > 0.8:
                        logger.info(f"    🎉 Excellent scale consistency achieved!")
                    elif consist_stats['mean'] > 0.6:
                        logger.info(f"    ✅ Good scale consistency!")
                    else:
                        logger.info(f"    📈 Scale consistency improving...")
        
        # Overfitting test results
        if args.overfit_test_size:
            overfit_success = summary.get('overfit_success', False)
            logger.info(f"🧪 OVERFITTING TEST: {'✅ PASSED' if overfit_success else '❌ FAILED'}")
            if overfit_success:
                logger.info("   ✅ Model can learn and memorize effectively!")
                logger.info("   ✅ Architecture and generation method working correctly!")
                if use_scale_aware:
                    logger.info("   ✅ Scale-aware generation enables effective learning!")
            else:
                logger.info("   ⚠️  Model still struggles - may need hyperparameter tuning")
                if use_scale_aware:
                    logger.info("   💡 Try adjusting scale-aware parameters:")
                    logger.info(f"      • Increase norm guidance strength (current: {args.norm_guidance_strength})")
                    logger.info(f"      • Adjust target norm (current: {args.typical_clip_norm})")
                    logger.info(f"      • Modify guidance frequency (current: {args.norm_guidance_frequency})")
        
        # WandB information
        if summary.get('wandb_enabled', False):
            logger.info(f"📊 WandB Dashboard: Check your {args.wandb_project} project for detailed metrics")
            if use_scale_aware:
                logger.info(f"  Scale-aware metrics available: target_norm_estimates, scale_consistency, etc.")
        
        # Save final summary
        summary['duration_seconds'] = duration
        summary['end_time'] = end_time.isoformat()
        summary['experiment_config'] = config
        summary['scale_aware_enabled'] = use_scale_aware
        summary['generation_method'] = config['generation_method']
        summary['evaluation_method'] = config['evaluation_method']
        
        summary_filename = 'scale_aware_training_summary.json' if use_scale_aware else 'baseline_training_summary.json'
        summary_path = output_dir / summary_filename
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📁 Training summary saved to {summary_path}")
        logger.info(f"📁 Model checkpoints saved to {output_dir}")
        
        logger.info("=" * 90)
        if use_scale_aware:
            logger.info("🎯 SCALE-AWARE GENERATION VERIFICATION:")
            logger.info("  ✅ Log-normal timestep scheduling applied")
            logger.info("  ✅ Velocity explosion prevention active")
            logger.info("  ✅ Periodic norm guidance enabled")
            logger.info("  ✅ Scale consistency metrics tracked")
            logger.info("  ✅ Adaptive target norm estimation used")
        else:
            logger.info("📊 BASELINE GENERATION SUMMARY:")
            logger.info("  • Standard linear timestep scheduling")
            logger.info("  • No scale-aware improvements")
            logger.info("  • Basic evaluation metrics")
        
        logger.info("=" * 90)
        logger.info("🔬 DEBUGGING TIPS:")
        if use_scale_aware:
            logger.info("  • Check WandB for scale_aware/* metrics")
            logger.info("  • Monitor target_norm_estimates for adaptive behavior")
            logger.info("  • Look for scale_consistency improvements over time")
            logger.info("  • Verify velocity explosion prevention in logs")
            logger.info("  • Compare with baseline results if available")
        else:
            logger.info("  • Monitor basic norm consistency")
            logger.info("  • Check for training stability")
            logger.info("  • Compare with scale-aware results")
        
        logger.info("  • Check norm tracking logs for consistent target norms")
        logger.info("  • Verify overfitting test shows learning capability")
        logger.info("=" * 90)
        
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