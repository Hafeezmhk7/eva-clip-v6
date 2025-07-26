#!/usr/bin/env python3
"""
EVA-CLIP Reproduction Training Script
train_eva_reproduction.py

This script tests if we can reproduce EVA-CLIP embeddings from noisy EVA-CLIP embeddings,
using CLIP embeddings as conditioning. This validates our DiT architecture.

KEY CHANGES:
- Target: EVA embeddings [B, N, 4096] (to reproduce)
- Conditioning: CLIP embeddings [B, N, 1024]
- Evaluation measures EVA similarity instead of CLIP similarity
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
    parser = argparse.ArgumentParser(description="EVA-CLIP Reproduction Test with DiT Architecture")
    
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
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
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
    parser.add_argument("--eval_every_n_steps", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_num_samples", type=int, default=1000,
                       help="Number of samples for evaluation")
    parser.add_argument("--eval_inference_steps", type=int, default=50,
                       help="Number of inference steps for evaluation")
    
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
            args.wandb_run_name = f"eva_repro_{args.model_size}_{args.training_mode}_{timestamp}"
        
        # Add automatic tags
        auto_tags = [
            "eva_reproduction",
            "dit_validation",
            args.model_size,
            args.training_mode,
            f"{args.max_training_shards}shards",
            f"bs{args.batch_size}",
            f"lr{args.learning_rate}",
        ]
        all_tags = list(set(auto_tags + args.wandb_tags))
        
        # WandB configuration
        config = {
            # Test configuration
            "test_type": "eva_reproduction",
            "test_purpose": "Validate DiT architecture by reproducing EVA from noisy EVA",
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
            
            # Implementation details
            "l2_normalization_enabled": True,
            "flow_matching_type": "rectified",
            "prediction_type": "velocity",
            "normalize_targets": True,
            
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
    """Create EVA reproduction DiT model"""
    from src.modules.models.blip3o_eva_dit import create_eva_reproduction_model, BLIP3oEVADiTConfig
    
    logger.info(f"Creating {args.model_size} model for EVA reproduction ({args.training_mode} mode)...")
    
    # Model configurations
    size_configs = {
        "tiny": {"hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 6, "intermediate_size": 1536},
        "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
        "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16, "intermediate_size": 4096},
    }
    
    config_params = size_configs[args.model_size].copy()
    config_params.update({
        "num_tokens": 257 if args.training_mode == "cls_patch" else 256,
        "training_mode": args.training_mode,
        "use_gradient_checkpointing": args.gradient_checkpointing,
        "clip_embedding_size": 1024,  # CLIP conditioning
        "eva_embedding_size": 4096,   # EVA input/output
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
    
    logger.info(f"✅ EVA Reproduction Model created: {model.get_num_parameters():,} parameters")
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
        })
        # Watch model for gradients and parameters
        wandb_instance.watch(model, log="all", log_freq=args.logging_steps * 5)
        logger.info("✅ EVA model registered with WandB for gradient tracking")
    
    return model

def create_eva_loss_function(args, logger):
    """Create EVA reproduction flow matching loss"""
    from src.modules.losses.blip3o_eva_loss import create_eva_reproduction_loss
    
    logger.info("Creating EVA reproduction flow matching loss...")
    
    loss_fn = create_eva_reproduction_loss(
        prediction_type="velocity",
        normalize_targets=True,  # Ensure targets are normalized
        flow_type="rectified",
    )
    
    logger.info(f"✅ EVA Reproduction Loss created with proper normalization")
    logger.info(f"   Target: EVA embeddings [B, N, 4096]")
    logger.info(f"   Conditioning: CLIP embeddings [B, N, 1024]")
    return loss_fn

def create_eva_dataloaders(args, logger):
    """Create data loaders for EVA reproduction with proper normalization"""
    from src.modules.datasets.blip3o_eva_dataset import create_eva_reproduction_dataloaders
    
    logger.info("Creating EVA reproduction dataloaders...")
    
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
    )
    
    logger.info(f"✅ EVA reproduction dataloaders created:")
    logger.info(f"   Train batches: {len(train_dataloader)}")
    logger.info(f"   Eval dataloader: {'Available' if eval_dataloader else 'None'}")
    logger.info(f"   TARGET: EVA embeddings [B, N, 4096] (to reproduce)")
    logger.info(f"   CONDITIONING: CLIP embeddings [B, N, 1024]")
    logger.info(f"   L2 Normalization: ENABLED")
    
    return train_dataloader, eval_dataloader

def create_eva_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, logger, wandb_instance=None):
    """Create EVA reproduction trainer with evaluation and WandB"""
    from src.modules.trainers.blip3o_eva_trainer import BLIP3oEVATrainer, create_eva_training_args
    
    logger.info("Creating EVA reproduction trainer with evaluation and WandB...")
    
    # Create training arguments
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
    
    # Create trainer with evaluation capabilities and WandB
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
        # WandB integration
        wandb_instance=wandb_instance,
        use_wandb=args.use_wandb,
    )
    
    # Override dataloader
    trainer.get_train_dataloader = lambda: train_dataloader
    
    logger.info("✅ EVA Reproduction Trainer created with evaluation and WandB integration")
    logger.info(f"   Evaluation every {args.eval_every_n_steps} steps")
    logger.info(f"   Evaluation samples: {args.eval_num_samples}")
    logger.info(f"   WandB tracking: {'Enabled' if wandb_instance else 'Disabled'}")
    return trainer

def save_eva_training_info(args, final_results, output_dir, logger, wandb_instance=None):
    """Save comprehensive EVA reproduction training information"""
    training_info = {
        'training_completed': True,
        'timestamp': datetime.now().isoformat(),
        'test_type': 'eva_reproduction',
        'test_purpose': 'Validate DiT architecture by reproducing EVA from noisy EVA using CLIP conditioning',
        
        # Test configuration
        'target_embeddings': 'EVA-CLIP [B, N, 4096]',
        'conditioning_embeddings': 'CLIP [B, N, 1024]',
        'expected_behavior': 'Model should learn to reproduce clean EVA embeddings from noisy EVA embeddings',
        
        # Training configuration
        'training_mode': args.training_mode,
        'model_size': args.model_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        
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
            'l2_normalization_enabled': True,
            'target_norms_fixed': True,
            'clean_flow_matching': True,
            'proper_evaluation': True,
            'eva_reproduction_test': True,
            'dit_architecture_validation': True,
            'wandb_integration': args.use_wandb,
        },
        
        # Normalization configuration
        'normalization_config': {
            'normalize_embeddings': True,
            'normalize_targets': True,
            'expected_eva_target_norm': 1.0,
            'expected_clip_conditioning_norm': 1.0,
            'expected_prediction_norm': 1.0,
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
                "final/test_type": "eva_reproduction",
            })
        
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
    """Main EVA reproduction training function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 Starting EVA-CLIP Reproduction Test with DiT Architecture")
    logger.info("=" * 70)
    logger.info("TEST PURPOSE:")
    logger.info("  ✅ Validate DiT architecture by reproducing EVA embeddings")
    logger.info("  ✅ Input: Noisy EVA embeddings [B, N, 4096]")
    logger.info("  ✅ Conditioning: CLIP embeddings [B, N, 1024]")
    logger.info("  ✅ Target: Clean EVA embeddings [B, N, 4096]")
    logger.info("  ✅ Evaluation: EVA cosine similarity")
    logger.info("=" * 70)
    logger.info(f"Training mode: {args.training_mode}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Embeddings: {args.chunked_embeddings_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Evaluation every {args.eval_every_n_steps} steps")
    logger.info(f"WandB tracking: {'Enabled' if args.use_wandb else 'Disabled'}")
    logger.info("=" * 70)
    
    # Initialize WandB early
    wandb_instance = setup_wandb(args, logger)
    
    try:
        # Setup
        device = setup_device(logger)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create components
        logger.info("🏗️ Creating EVA reproduction model components...")
        model = create_eva_model(args, logger, wandb_instance)
        loss_fn = create_eva_loss_function(args, logger)
        train_dataloader, eval_dataloader = create_eva_dataloaders(args, logger)
        
        # Move model to device
        model = model.to(device)
        
        # Create trainer with evaluation and WandB
        trainer = create_eva_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, logger, wandb_instance)
        
        # Start training
        logger.info("🚀 Starting EVA reproduction training to validate DiT architecture...")
        logger.info("📊 Expected behavior:")
        logger.info("  • EVA target norms should be ~1.0 (not ~32.0)")
        logger.info("  • CLIP conditioning norms should be ~1.0")
        logger.info("  • Prediction norms should be ~1.0")
        logger.info("  • Velocity similarity should increase from ~0.01 to >0.1")
        logger.info("  • EVA similarity should increase from ~0.01 to >0.1")
        logger.info("  • If this works, DiT architecture is validated!")
        logger.info("  • Evaluation every 100 steps to track progress")
        if wandb_instance:
            logger.info(f"  • All metrics tracked in WandB: {wandb_instance.run.url}")
        logger.info("")
        
        start_time = datetime.now()
        
        # Train model
        trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get final comprehensive evaluation
        logger.info("🔍 Running final comprehensive EVA reproduction evaluation...")
        final_results = trainer.get_final_evaluation()
        
        # Save model
        trainer.save_model()
        
        # Save training info
        save_eva_training_info(args, final_results, args.output_dir, logger, wandb_instance)
        
        # Final summary
        logger.info("=" * 70)
        logger.info("✅ EVA REPRODUCTION TEST COMPLETED!")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Model saved to: {args.output_dir}")
        
        if wandb_instance:
            logger.info(f"📊 WandB Run: {wandb_instance.run.url}")
            logger.info(f"📊 All training curves and metrics available in WandB")
        
        if final_results and 'training_summary' in final_results:
            summary = final_results['training_summary']
            logger.info(f"📊 FINAL EVA REPRODUCTION RESULTS:")
            logger.info(f"   Final Velocity Similarity: {summary.get('final_velocity_sim', 0):.4f}")
            logger.info(f"   Best Velocity Similarity: {summary.get('best_velocity_sim', 0):.4f}")
            logger.info(f"   Final EVA Similarity: {summary.get('final_eva_sim', 0):.4f}")
            logger.info(f"   Best EVA Similarity: {summary.get('best_eva_sim', 0):.4f}")
            logger.info(f"   Training Health: {summary.get('training_health', 'Unknown')}")
            logger.info(f"   Evaluations Performed: {summary.get('evaluations_performed', 0)}")
        
        if final_results and 'final_evaluation' in final_results:
            eval_results = final_results['final_evaluation']
            if eval_results:
                logger.info(f"🎯 FINAL EVA EVALUATION (on {eval_results.get('samples_evaluated', 0)} samples):")
                logger.info(f"   Overall EVA Similarity: {eval_results.get('overall_eva_similarity', 0):.4f}")
                logger.info(f"   High Quality Images (>0.7): {eval_results.get('high_quality_images', 0)*100:.1f}%")
                logger.info(f"   Very High Quality Images (>0.8): {eval_results.get('very_high_quality_images', 0)*100:.1f}%")
                logger.info(f"   Excellent Quality Images (>0.9): {eval_results.get('excellent_quality_images', 0)*100:.1f}%")
        
        # Success assessment for DiT validation
        if final_results and 'training_summary' in final_results:
            final_eva_sim = final_results['training_summary'].get('best_eva_sim', 0)
            if final_eva_sim > 0.3:
                logger.info("🎉 SUCCESS: DiT architecture VALIDATED! Excellent EVA reproduction!")
                logger.info("✅ DiT can successfully reproduce EVA embeddings from noisy EVA + CLIP conditioning")
            elif final_eva_sim > 0.1:
                logger.info("📈 PROGRESS: DiT architecture shows good learning capability!")
                logger.info("✅ DiT can reproduce EVA embeddings with decent quality")
            elif final_eva_sim > 0.05:
                logger.info("📈 LEARNING: DiT architecture shows some learning capability")
                logger.info("⚠️ May need hyperparameter tuning for better performance")
            else:
                logger.info("⚠️ NEEDS WORK: Low EVA similarity, DiT may need architecture adjustments")
        
        logger.info("🔧 ARCHITECTURE STATUS: DiT implementation tested with EVA reproduction task")
        if wandb_instance:
            logger.info("📊 WANDB STATUS: All training and evaluation curves saved to WandB")
        logger.info("=" * 70)
        
        # Finish WandB run
        if wandb_instance:
            wandb_instance.finish()
            logger.info("✅ WandB run finished")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ EVA reproduction training failed: {e}")
        traceback.print_exc()
        
        # Finish WandB run even on failure
        if wandb_instance:
            wandb_instance.finish(exit_code=1)
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)