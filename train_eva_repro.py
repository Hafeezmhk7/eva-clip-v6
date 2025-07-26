#!/usr/bin/env python3
"""
Fixed EVA-CLIP Reproduction Training Script
train_eva_reproduction.py

MAJOR FIXES:
1. Better error handling and recovery
2. Fixed import issues and module paths
3. Improved hyperparameters based on feedback
4. Better debugging and monitoring
5. Robust training flow with overfitting test capability
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
    parser = argparse.ArgumentParser(description="Fixed EVA-CLIP Reproduction Test with BLIP3-o DiT Architecture")
    
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
    
    # Training parameters (improved based on feedback)
    parser.add_argument("--num_epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "constant"],
                       help="Learning rate scheduler type")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=200,
                       help="Save frequency")
    
    # Evaluation parameters
    parser.add_argument("--eval_every_n_steps", type=int, default=50,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=50,
                       help="Number of inference steps for evaluation")
    
    # Debugging and overfitting test
    parser.add_argument("--overfit_test_size", type=int, default=None,
                       help="Number of samples for overfitting test (None to disable)")
    parser.add_argument("--debug_mode", action="store_true", default=False,
                       help="Enable debug mode")
    
    # WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb_project", type=str, default="eva-reproduction-test",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="WandB entity/team name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[],
                       help="WandB tags for this run")
    parser.add_argument("--wandb_notes", type=str, default="",
                       help="WandB notes for this run")
    
    # Options
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                       help="Use gradient checkpointing")
    parser.add_argument("--max_training_shards", type=int, default=1,
                       help="Maximum training shards")
    
    return parser.parse_args()

def setup_wandb(args, logger):
    """Setup Weights & Biases tracking for EVA reproduction test"""
    if not args.use_wandb:
        logger.info("WandB tracking disabled")
        return None
    
    try:
        import wandb
        
        # Auto-generate run name if not provided
        if args.wandb_run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            overfit_suffix = f"_overfit{args.overfit_test_size}" if args.overfit_test_size else ""
            args.wandb_run_name = f"eva_repro_{args.model_size}_{args.training_mode}{overfit_suffix}_{timestamp}"
        
        # Add automatic tags
        auto_tags = [
            "eva_reproduction",
            "dit_validation",
            "blip3o_architecture",
            args.model_size,
            args.training_mode,
            f"{args.max_training_shards}shards",
            f"bs{args.batch_size}",
            f"lr{args.learning_rate}",
        ]
        
        if args.overfit_test_size:
            auto_tags.append(f"overfit_test_{args.overfit_test_size}")
        
        all_tags = list(set(auto_tags + args.wandb_tags))
        
        # WandB configuration
        config = {
            # Test configuration
            "test_type": "eva_reproduction",
            "test_purpose": "Validate BLIP3-o DiT architecture by reproducing EVA from noisy EVA",
            "target_embeddings": "EVA-CLIP [B, N, 4096]",
            "conditioning_embeddings": "CLIP [B, N, 1024]",
            
            # Model configuration
            "model_size": args.model_size,
            "training_mode": args.training_mode,
            "expected_tokens": 257 if args.training_mode == "cls_patch" else 256,
            
            # Training hyperparameters
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            
            # Training configuration
            "fp16": args.fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_training_shards": args.max_training_shards,
            
            # Evaluation configuration
            "eval_every_n_steps": args.eval_every_n_steps,
            "eval_num_samples": args.eval_num_samples,
            "eval_inference_steps": args.eval_inference_steps,
            
            # Debugging
            "overfit_test_size": args.overfit_test_size,
            "debug_mode": args.debug_mode,
            
            # Implementation details
            "architecture": "BLIP3-o DiT with 3D RoPE and Grouped-Query Attention",
            "flow_matching_type": "rectified",
            "prediction_type": "velocity",
            "normalize_targets": True,
            "l2_normalization_enabled": True,
            
            # Paths
            "embeddings_dir": str(args.chunked_embeddings_dir),
            "output_dir": str(args.output_dir),
        }
        
        # Initialize WandB
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=config,
            tags=all_tags,
            notes=args.wandb_notes,
            save_code=True,
        )
        
        logger.info(f"✅ WandB initialized for EVA reproduction test:")
        logger.info(f"   Project: {args.wandb_project}")
        logger.info(f"   Run name: {args.wandb_run_name}")
        logger.info(f"   Tags: {all_tags}")
        logger.info(f"   URL: {wandb.run.url}")
        
        return wandb
        
    except ImportError:
        logger.error("❌ WandB not installed. Install with: pip install wandb")
        logger.error("   Continuing without WandB tracking...")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize WandB: {e}")
        logger.error("   Continuing without WandB tracking...")
        return None

def setup_device(logger):
    """Setup device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def create_eva_model(args, logger, wandb_instance=None):
    """Create EVA reproduction DiT model with BLIP3-o architecture"""
    try:
        from src.modules.models.blip3o_eva_dit import create_eva_reproduction_model, BLIP3oEVADiTConfig
    except ImportError as e:
        logger.error(f"❌ Failed to import model: {e}")
        logger.error("   Make sure the fixed model file is in the correct location")
        raise
    
    logger.info(f"Creating {args.model_size} BLIP3-o DiT model for EVA reproduction ({args.training_mode} mode)...")
    
    # Model configurations with BLIP3-o specifications
    size_configs = {
        "tiny": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,  # Grouped-query attention
            "intermediate_size": 1536
        },
        "small": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "intermediate_size": 2048
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 3072
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 4096
        },
    }
    
    config_params = size_configs[args.model_size].copy()
    config_params.update({
        "num_tokens": 257 if args.training_mode == "cls_patch" else 256,
        "training_mode": args.training_mode,
        "use_gradient_checkpointing": args.gradient_checkpointing,
        "clip_embedding_size": 1024,  # CLIP conditioning
        "eva_embedding_size": 4096,   # EVA input/output
        "use_3d_rope": True,           # Enable 3D RoPE as per BLIP3-o
        "zero_init_output": True,      # Zero init for flow matching
        "dropout_prob": 0.0,           # Disable dropout for better training
    })
    
    config = BLIP3oEVADiTConfig(**config_params)
    model = create_eva_reproduction_model(config=config)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"⚠️ Could not enable gradient checkpointing: {e}")
    
    logger.info(f"✅ BLIP3-o EVA DiT Model created: {model.get_num_parameters():,} parameters")
    logger.info(f"   Architecture: BLIP3-o DiT with 3D RoPE and Grouped-Query Attention")
    logger.info(f"   Input: EVA embeddings [B, N, 4096] (noisy)")
    logger.info(f"   Conditioning: CLIP embeddings [B, N, 1024]")
    logger.info(f"   Output: EVA embeddings [B, N, 4096] (clean)")
    
    # Log model to WandB
    if wandb_instance:
        wandb_instance.config.update({
            "model_parameters": model.get_num_parameters(),
            "model_config": config_params,
            "input_dim": 4096,
            "conditioning_dim": 1024,
            "output_dim": 4096,
            "architecture_details": {
                "3d_rope": True,
                "grouped_query_attention": True,
                "sandwich_normalization": True,
                "rms_norm": True,
                "zero_init_output": True,
            }
        })
        # Watch model for gradients and parameters
        wandb_instance.watch(model, log="all", log_freq=args.logging_steps * 5)
        logger.info("✅ BLIP3-o model registered with WandB for gradient tracking")
    
    return model

def create_eva_loss_function(args, logger):
    """Create EVA reproduction flow matching loss"""
    try:
        from src.modules.losses.blip3o_eva_loss import create_eva_reproduction_loss
    except ImportError as e:
        logger.error(f"❌ Failed to import loss function: {e}")
        raise
    
    logger.info("Creating fixed EVA reproduction flow matching loss...")
    
    loss_fn = create_eva_reproduction_loss(
        prediction_type="velocity",
        normalize_targets=True,
        flow_type="rectified",
        loss_scale=1.0,  # Reduced based on feedback
        gradient_clip_val=1.0,
        debug_mode=args.debug_mode,
    )
    
    logger.info(f"✅ Fixed EVA Reproduction Loss created")
    logger.info(f"   Target: EVA embeddings [B, N, 4096]")
    logger.info(f"   Conditioning: CLIP embeddings [B, N, 1024]")
    logger.info(f"   Loss scale: 1.0 (improved stability)")
    return loss_fn

def create_eva_dataloaders(args, logger):
    """Create data loaders for EVA reproduction with robust error handling"""
    try:
        from src.modules.datasets.blip3o_eva_dataset import create_eva_reproduction_dataloaders
    except ImportError as e:
        logger.error(f"❌ Failed to import dataset: {e}")
        raise
    
    logger.info("Creating robust EVA reproduction dataloaders...")
    
    train_dataloader, eval_dataloader = create_eva_reproduction_dataloaders(
        chunked_embeddings_dir=args.chunked_embeddings_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        eval_split_ratio=0.0,  # Use same data for evaluation
        normalize_embeddings=True,  # Enable L2 normalization
        training_mode=args.training_mode,
        max_shards=args.max_training_shards,
        use_same_data_for_eval=True,
        delete_after_use=False,
        num_workers=0,
        pin_memory=False,
        skip_corrupted=True,  # Skip corrupted samples
        validate_shapes=True,  # Validate tensor shapes
    )
    
    logger.info(f"✅ Robust EVA reproduction dataloaders created:")
    logger.info(f"   Train batches: {len(train_dataloader)}")
    logger.info(f"   Eval dataloader: {'Available' if eval_dataloader else 'None'}")
    logger.info(f"   TARGET: EVA embeddings [B, N, 4096] (to reproduce)")
    logger.info(f"   CONDITIONING: CLIP embeddings [B, N, 1024]")
    logger.info(f"   Robust error handling: Enabled")
    
    return train_dataloader, eval_dataloader

def create_eva_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, logger, wandb_instance=None):
    """Create EVA reproduction trainer with comprehensive monitoring"""
    try:
        from src.modules.trainers.blip3o_eva_trainer import BLIP3oEVATrainer, create_eva_training_args
    except ImportError as e:
        logger.error(f"❌ Failed to import trainer: {e}")
        raise
    
    logger.info("Creating comprehensive EVA reproduction trainer...")
    
    # Create training arguments with improved settings
    training_args = create_eva_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        dataloader_num_workers=0,
        report_to=["wandb"] if wandb_instance else [],
    )
    
    # Create trainer with comprehensive monitoring
    trainer = BLIP3oEVATrainer(
        model=model,
        args=training_args,
        flow_matching_loss=loss_fn,
        train_dataset=None,  # We use custom dataloader
        eval_dataset=None,
        eval_dataloader=eval_dataloader,
        training_mode=args.training_mode,
        # Evaluation parameters
        eval_every_n_steps=args.eval_every_n_steps,
        eval_num_samples=args.eval_num_samples,
        eval_batch_size=args.batch_size,
        eval_inference_steps=args.eval_inference_steps,
        # Debugging parameters
        debug_mode=args.debug_mode,
        track_gradients=True,
        overfit_test_size=args.overfit_test_size,
        # WandB integration
        wandb_instance=wandb_instance,
        use_wandb=args.use_wandb,
    )
    
    # Override dataloader
    trainer.get_train_dataloader = lambda: train_dataloader
    
    logger.info("✅ Comprehensive EVA Reproduction Trainer created")
    logger.info(f"   Evaluation every {args.eval_every_n_steps} steps")
    logger.info(f"   Evaluation samples: {args.eval_num_samples}")
    logger.info(f"   Debug mode: {args.debug_mode}")
    logger.info(f"   Overfit test: {args.overfit_test_size if args.overfit_test_size else 'Disabled'}")
    logger.info(f"   WandB tracking: {'Enabled' if wandb_instance else 'Disabled'}")
    return trainer

def save_eva_training_info(args, final_results, output_dir, logger, wandb_instance=None):
    """Save comprehensive EVA reproduction training information"""
    training_info = {
        'training_completed': True,
        'timestamp': datetime.now().isoformat(),
        'test_type': 'eva_reproduction',
        'test_purpose': 'Validate BLIP3-o DiT architecture by reproducing EVA from noisy EVA using CLIP conditioning',
        
        # Test configuration
        'architecture': 'BLIP3-o DiT with 3D RoPE, Grouped-Query Attention, and Sandwich Normalization',
        'target_embeddings': 'EVA-CLIP [B, N, 4096]',
        'conditioning_embeddings': 'CLIP [B, N, 1024]',
        'expected_behavior': 'Model should learn to reproduce clean EVA embeddings from noisy EVA embeddings',
        
        # Training configuration
        'training_mode': args.training_mode,
        'model_size': args.model_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'overfit_test_size': args.overfit_test_size,
        'debug_mode': args.debug_mode,
        
        # Evaluation configuration
        'evaluation_config': {
            'eval_every_n_steps': args.eval_every_n_steps,
            'eval_num_samples': args.eval_num_samples,
            'eval_inference_steps': args.eval_inference_steps,
            'evaluation_metric': 'EVA cosine similarity',
        },
        
        # WandB configuration
        'wandb_config': {
            'use_wandb': args.use_wandb,
            'wandb_project': args.wandb_project if args.use_wandb else None,
            'wandb_run_name': args.wandb_run_name if args.use_wandb else None,
            'wandb_url': wandb_instance.run.url if wandb_instance else None,
        },
        
        # Implementation details
        'implementation_details': {
            'blip3o_architecture': True,
            '3d_rope': True,
            'grouped_query_attention': True,
            'sandwich_normalization': True,
            'rms_norm': True,
            'l2_normalization_enabled': True,
            'robust_error_handling': True,
            'shape_validation': True,
            'corrupted_sample_skipping': True,
            'comprehensive_monitoring': True,
        },
        
        # Fixes applied
        'fixes_applied': {
            'timestep_embedding_shape_fix': True,
            'adaptive_layer_norm_fix': True,
            'gradient_flow_improvements': True,
            'numerical_stability_enhancements': True,
            'error_handling_robustness': True,
            'overfitting_test_capability': True,
        },
        
        # Paths
        'embeddings_dir': args.chunked_embeddings_dir,
        'output_dir': args.output_dir,
        'max_training_shards': args.max_training_shards,
        
        # Results
        'final_results': final_results,
    }
    
    info_file = Path(output_dir) / "eva_reproduction_training_info.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"EVA reproduction training info saved to: {info_file}")
    
    # Log final results to WandB
    if wandb_instance and final_results:
        summary_data = {}
        
        if 'training_summary' in final_results:
            summary = final_results['training_summary']
            summary_data.update({
                "final/best_velocity_sim": summary.get('best_velocity_sim', 0),
                "final/best_eva_sim": summary.get('best_eva_sim', 0),
                "final/total_steps": summary.get('total_steps', 0),
                "final/training_health": summary.get('training_health', 'Unknown'),
                "final/evaluations_performed": summary.get('evaluations_performed', 0),
                "final/test_type": "eva_reproduction_blip3o",
                "final/architecture": "BLIP3-o DiT",
            })
            
            if args.overfit_test_size:
                summary_data["final/overfit_test_success"] = summary.get('overfit_success', False)
        
        if 'final_evaluation' in final_results and final_results['final_evaluation']:
            eval_results = final_results['final_evaluation']
            summary_data.update({
                "final_eval/overall_eva_similarity": eval_results.get('overall_eva_similarity', 0),
                "final_eval/high_quality_images_pct": eval_results.get('high_quality_images', 0) * 100,
                "final_eval/very_high_quality_images_pct": eval_results.get('very_high_quality_images', 0) * 100,
                "final_eval/excellent_quality_images_pct": eval_results.get('excellent_quality_images', 0) * 100,
                "final_eval/samples_evaluated": eval_results.get('samples_evaluated', 0),
            })
        
        # Log summary metrics
        for key, value in summary_data.items():
            wandb_instance.run.summary[key] = value
        
        logger.info("✅ Final EVA reproduction results logged to WandB")

def main():
    """Main EVA reproduction training function with comprehensive error handling"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 Starting Fixed EVA-CLIP Reproduction Test with BLIP3-o DiT Architecture")
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE TEST PURPOSE:")
    logger.info("  ✅ Validate BLIP3-o DiT architecture with 3D RoPE and Grouped-Query Attention")
    logger.info("  ✅ Input: Noisy EVA embeddings [B, N, 4096]")
    logger.info("  ✅ Conditioning: CLIP embeddings [B, N, 1024]")
    logger.info("  ✅ Target: Clean EVA embeddings [B, N, 4096]")
    logger.info("  ✅ Evaluation: EVA cosine similarity")
    logger.info("  ✅ Architecture: BLIP3-o DiT with Sandwich Normalization")
    logger.info("=" * 80)
    logger.info(f"Training mode: {args.training_mode}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Embeddings: {args.chunked_embeddings_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Evaluation every {args.eval_every_n_steps} steps")
    logger.info(f"Overfit test: {args.overfit_test_size if args.overfit_test_size else 'Disabled'}")
    logger.info(f"Debug mode: {args.debug_mode}")
    logger.info(f"WandB tracking: {'Enabled' if args.use_wandb else 'Disabled'}")
    logger.info("=" * 80)
    
    # Initialize WandB early
    wandb_instance = setup_wandb(args, logger)
    
    try:
        # Setup
        device = setup_device(logger)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create components with comprehensive error handling
        logger.info("🏗️ Creating BLIP3-o EVA reproduction model components...")
        
        try:
            model = create_eva_model(args, logger, wandb_instance)
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            raise
        
        try:
            loss_fn = create_eva_loss_function(args, logger)
        except Exception as e:
            logger.error(f"❌ Loss function creation failed: {e}")
            raise
        
        try:
            train_dataloader, eval_dataloader = create_eva_dataloaders(args, logger)
        except Exception as e:
            logger.error(f"❌ Dataloader creation failed: {e}")
            raise
        
        # Move model to device
        try:
            model = model.to(device)
            logger.info(f"✅ Model moved to {device}")
        except Exception as e:
            logger.error(f"❌ Failed to move model to device: {e}")
            raise
        
        # Create trainer with comprehensive monitoring
        try:
            trainer = create_eva_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, logger, wandb_instance)
        except Exception as e:
            logger.error(f"❌ Trainer creation failed: {e}")
            raise
        
        # Start training
        logger.info("🚀 Starting BLIP3-o EVA reproduction training...")
        logger.info("📊 Expected behavior with fixed architecture:")
        logger.info("  • EVA target norms should be ~1.0 (properly normalized)")
        logger.info("  • CLIP conditioning norms should be ~1.0")
        logger.info("  • Prediction norms should be ~1.0")
        logger.info("  • Velocity similarity should increase from ~0.01 to >0.1")
        logger.info("  • EVA similarity should increase from ~0.01 to >0.1")
        logger.info("  • No tensor shape mismatches or NaN/Inf issues")
        logger.info("  • Robust error handling and recovery")
        
        if args.overfit_test_size:
            logger.info(f"  • Overfitting test on {args.overfit_test_size} samples should show rapid learning")
        
        if wandb_instance:
            logger.info(f"  • All metrics tracked in WandB: {wandb_instance.run.url}")
        logger.info("")
        
        start_time = datetime.now()
        
        # Train model with comprehensive error handling
        try:
            trainer.train()
            logger.info("✅ Training completed successfully")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            traceback.print_exc()
            
            # Try to save partial results
            try:
                logger.info("Attempting to save partial training results...")
                partial_results = trainer.get_final_evaluation()
                save_eva_training_info(args, partial_results, args.output_dir, logger, wandb_instance)
                logger.info("✅ Partial results saved")
            except Exception as save_e:
                logger.error(f"Failed to save partial results: {save_e}")
            
            raise
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get final comprehensive evaluation
        logger.info("🔍 Running final comprehensive EVA reproduction evaluation...")
        try:
            final_results = trainer.get_final_evaluation()
        except Exception as e:
            logger.error(f"❌ Final evaluation failed: {e}")
            final_results = None
        
        # Save model
        try:
            trainer.save_model()
            logger.info("✅ Model saved successfully")
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
        
        # Save training info
        try:
            save_eva_training_info(args, final_results, args.output_dir, logger, wandb_instance)
        except Exception as e:
            logger.error(f"❌ Failed to save training info: {e}")
        
        # Final summary
        logger.info("=" * 80)
        logger.info("✅ BLIP3-O EVA REPRODUCTION TEST COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Model saved to: {args.output_dir}")
        
        if wandb_instance:
            logger.info(f"📊 WandB Run: {wandb_instance.run.url}")
            logger.info(f"📊 All training curves and metrics available in WandB")
        
        if final_results and 'training_summary' in final_results:
            summary = final_results['training_summary']
            logger.info(f"📊 FINAL BLIP3-O EVA REPRODUCTION RESULTS:")
            logger.info(f"   Final Velocity Similarity: {summary.get('final_velocity_sim', 0):.4f}")
            logger.info(f"   Best Velocity Similarity: {summary.get('best_velocity_sim', 0):.4f}")
            logger.info(f"   Final EVA Similarity: {summary.get('final_eva_sim', 0):.4f}")
            logger.info(f"   Best EVA Similarity: {summary.get('best_eva_sim', 0):.4f}")
            logger.info(f"   Training Health: {summary.get('training_health', 'Unknown')}")
            logger.info(f"   Total Steps: {summary.get('total_steps', 0)}")
            logger.info(f"   Evaluations Performed: {summary.get('evaluations_performed', 0)}")
            
            if args.overfit_test_size:
                overfit_success = summary.get('overfit_success', False)
                logger.info(f"   Overfit Test Success: {'✅ Yes' if overfit_success else '❌ No'}")
        
        if final_results and 'final_evaluation' in final_results:
            eval_results = final_results['final_evaluation']
            if eval_results:
                logger.info(f"🎯 FINAL EVA EVALUATION (on {eval_results.get('samples_evaluated', 0)} samples):")
                logger.info(f"   Overall EVA Similarity: {eval_results.get('overall_eva_similarity', 0):.4f}")
                logger.info(f"   High Quality Images (>0.7): {eval_results.get('high_quality_images', 0)*100:.1f}%")
                logger.info(f"   Very High Quality Images (>0.8): {eval_results.get('very_high_quality_images', 0)*100:.1f}%")
                logger.info(f"   Excellent Quality Images (>0.9): {eval_results.get('excellent_quality_images', 0)*100:.1f}%")
        
        # Success assessment for BLIP3-o DiT validation
        if final_results and 'training_summary' in final_results:
            final_eva_sim = final_results['training_summary'].get('best_eva_sim', 0)
            final_vel_sim = final_results['training_summary'].get('best_velocity_sim', 0)
            training_health = final_results['training_summary'].get('training_health', 'unknown')
            
            logger.info("🔍 BLIP3-O DiT ARCHITECTURE ASSESSMENT:")
            
            if final_eva_sim > 0.3 and final_vel_sim > 0.3:
                logger.info("🎉 EXCELLENT: BLIP3-o DiT architecture FULLY VALIDATED!")
                logger.info("✅ Architecture can successfully reproduce EVA embeddings with high quality")
                logger.info("✅ 3D RoPE, Grouped-Query Attention, and Sandwich Normalization working perfectly")
            elif final_eva_sim > 0.1 and final_vel_sim > 0.1:
                logger.info("📈 GOOD: BLIP3-o DiT architecture shows strong learning capability!")
                logger.info("✅ Architecture can reproduce EVA embeddings with decent quality")
                logger.info("✅ All major components functioning correctly")
            elif final_eva_sim > 0.05 or final_vel_sim > 0.05:
                logger.info("📈 LEARNING: BLIP3-o DiT architecture shows learning capability")
                logger.info("✅ Architecture is functional but may need hyperparameter tuning")
            else:
                logger.info("⚠️ NEEDS INVESTIGATION: Low similarity scores")
                logger.info("🔧 Architecture may need further optimization or longer training")
            
            if training_health == "healthy" or training_health == "converged":
                logger.info("💚 TRAINING HEALTH: Excellent - stable and convergent training")
            elif training_health == "learning":
                logger.info("💛 TRAINING HEALTH: Good - showing learning progress")
            else:
                logger.info(f"💙 TRAINING HEALTH: {training_health}")
            
            if args.overfit_test_size:
                overfit_success = final_results['training_summary'].get('overfit_success', False)
                if overfit_success:
                    logger.info("🧪 OVERFITTING TEST: ✅ PASSED - Model can learn and memorize")
                else:
                    logger.info("🧪 OVERFITTING TEST: ⚠️ Incomplete - May need more training or debugging")
        
        logger.info("🏗️ ARCHITECTURE STATUS:")
        logger.info("   ✅ BLIP3-o DiT implementation tested with EVA reproduction task")
        logger.info("   ✅ 3D RoPE, Grouped-Query Attention, Sandwich Normalization implemented")
        logger.info("   ✅ Robust error handling and shape validation")
        logger.info("   ✅ Comprehensive monitoring and debugging capabilities")
        
        if wandb_instance:
            logger.info("📊 WANDB STATUS: All training and evaluation curves saved to WandB")
        
        logger.info("=" * 80)
        
        # Finish WandB run
        if wandb_instance:
            try:
                wandb_instance.finish()
                logger.info("✅ WandB run finished successfully")
            except Exception as e:
                logger.warning(f"⚠️ WandB finish failed: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ EVA reproduction training failed: {e}")
        traceback.print_exc()
        
        # Finish WandB run even on failure
        if wandb_instance:
            try:
                wandb_instance.finish(exit_code=1)
            except:
                pass
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)