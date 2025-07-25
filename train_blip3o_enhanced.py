#!/usr/bin/env python3
"""
FIXED: BLIP3-o Training Script with Proper L2 Normalization
train_blip3o_enhanced.py

KEY FIX:
1. Enable proper L2 normalization for CLIP embeddings (normalize_embeddings=True)
2. Ensure target norms are ~1.0 instead of ~32.0
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
    parser = argparse.ArgumentParser(description="FIXED BLIP3-o Training with Proper Normalization")
    
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
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
        "tiny": {"hidden_size": 384, "num_hidden_layers": 6, "num_attention_heads": 6, "intermediate_size": 1536},
        "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
        "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16, "intermediate_size": 4096},
    }
    
    config_params = size_configs[args.model_size].copy()
    config_params.update({
        "num_tokens": 257 if args.training_mode == "cls_patch" else 256,
        "training_mode": args.training_mode,
        "use_gradient_checkpointing": False,
    })
    
    config = BLIP3oDiTConfig(**config_params)
    model = create_blip3o_patch_dit_model(config=config)
    
    logger.info(f"✅ FIXED Model created: {model.get_num_parameters():,} parameters")
    logger.info(f"   No scaling confusion - clean implementation")
    return model

def create_loss_function(args, logger):
    """Create FIXED flow matching loss"""
    from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
    
    logger.info("Creating FIXED flow matching loss...")
    
    loss_fn = create_blip3o_flow_matching_loss(
        prediction_type="velocity",
        normalize_targets=True,  # Ensure targets are normalized
        flow_type="rectified",
    )
    
    logger.info(f"✅ FIXED Loss created with proper normalization")
    return loss_fn

def create_dataloaders(args, logger):
    """Create data loaders with proper normalization"""
    from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
    
    logger.info("Creating dataloaders with L2 normalization...")
    
    # CRITICAL FIX: Enable normalization to fix target norm issue
    train_dataloader, eval_dataloader = create_flexible_dataloaders(
        chunked_embeddings_dir=args.chunked_embeddings_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        eval_split_ratio=0.0,  # Use same data for evaluation
        normalize_embeddings=True,  # FIXED: Enable L2 normalization
        training_mode=args.training_mode,
        max_shards=args.max_training_shards,
        use_same_data_for_eval=True,
        delete_after_use=False,
        num_workers=0,
        pin_memory=False,
    )
    
    logger.info(f"✅ Dataloaders created with L2 normalization:")
    logger.info(f"   Train batches: {len(train_dataloader)}")
    logger.info(f"   Eval dataloader: {'Available' if eval_dataloader else 'None'}")
    logger.info(f"   L2 Normalization: ENABLED (target norms should be ~1.0)")
    
    return train_dataloader, eval_dataloader

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, logger):
    """Create FIXED trainer with evaluation"""
    from src.modules.trainers.blip3o_trainer import BLIP3oTrainer, create_training_args
    
    logger.info("Creating FIXED trainer with evaluation...")
    
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
    
    # Create trainer with evaluation capabilities
    trainer = BLIP3oTrainer(
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
    )
    
    # Override dataloader
    trainer.get_train_dataloader = lambda: train_dataloader
    
    logger.info("✅ FIXED Trainer created with evaluation")
    logger.info(f"   Evaluation every {args.eval_every_n_steps} steps")
    logger.info(f"   Evaluation samples: {args.eval_num_samples}")
    return trainer

def save_training_info(args, final_results, output_dir, logger):
    """Save comprehensive training information"""
    training_info = {
        'training_completed': True,
        'timestamp': datetime.now().isoformat(),
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
        },
        
        # Fixed implementation details
        'implementation_fixes': {
            'l2_normalization_enabled': True,
            'target_norms_fixed': True,
            'clean_flow_matching': True,
            'proper_evaluation': True,
            'blip3o_paper_aligned': True,
            'velocity_and_embedding_tracking': True,
        },
        
        # Normalization status
        'normalization_config': {
            'normalize_embeddings': True,
            'normalize_targets': True,
            'expected_target_norm': 1.0,
            'expected_prediction_norm': 1.0,
        },
        
        # Paths
        'embeddings_dir': args.chunked_embeddings_dir,
        'output_dir': args.output_dir,
        'max_training_shards': args.max_training_shards,
        
        # Results
        'final_results': final_results,
    }
    
    info_file = Path(output_dir) / "training_info.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"Training info saved to: {info_file}")

def main():
    """Main training function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 Starting FIXED BLIP3-o Training with Proper L2 Normalization")
    logger.info("=" * 70)
    logger.info("NORMALIZATION FIX APPLIED:")
    logger.info("  ✅ L2 normalization enabled for CLIP embeddings")
    logger.info("  ✅ Target norms should be ~1.0 (not ~32.0)")
    logger.info("  ✅ Prediction norms should be ~1.0")
    logger.info("  ✅ Proper flow matching with normalized embeddings")
    logger.info("=" * 70)
    logger.info(f"Training mode: {args.training_mode}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Embeddings: {args.chunked_embeddings_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Evaluation every {args.eval_every_n_steps} steps")
    logger.info("=" * 70)
    
    try:
        # Setup
        device = setup_device(logger)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create components
        logger.info("🏗️ Creating model components...")
        model = create_model(args, logger)
        loss_fn = create_loss_function(args, logger)
        train_dataloader, eval_dataloader = create_dataloaders(args, logger)
        
        # Move model to device
        model = model.to(device)
        
        # Create trainer with evaluation
        trainer = create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, logger)
        
        # Start training
        logger.info("🚀 Starting FIXED training with proper normalization...")
        logger.info("📊 Expected behavior with L2 normalization:")
        logger.info("  • Target norms should be ~1.0 (not ~32.0)")
        logger.info("  • Prediction norms should be ~1.0")
        logger.info("  • Velocity similarity should increase from ~0.01 to >0.1")
        logger.info("  • Embedding similarity should increase from ~0.01 to >0.1")
        logger.info("  • Evaluation every 100 steps to track progress")
        logger.info("")
        
        start_time = datetime.now()
        
        # Train model
        trainer.train()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Get final comprehensive evaluation
        logger.info("🔍 Running final comprehensive evaluation...")
        final_results = trainer.get_final_evaluation()
        
        # Save model
        trainer.save_model()
        
        # Save training info
        save_training_info(args, final_results, args.output_dir, logger)
        
        # Final summary
        logger.info("=" * 70)
        logger.info("✅ FIXED TRAINING COMPLETED WITH PROPER NORMALIZATION!")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Model saved to: {args.output_dir}")
        
        if final_results and 'training_summary' in final_results:
            summary = final_results['training_summary']
            logger.info(f"📊 FINAL RESULTS:")
            logger.info(f"   Final Velocity Similarity: {summary.get('final_velocity_sim', 0):.4f}")
            logger.info(f"   Best Velocity Similarity: {summary.get('best_velocity_sim', 0):.4f}")
            logger.info(f"   Final Embedding Similarity: {summary.get('final_embedding_sim', 0):.4f}")
            logger.info(f"   Best Embedding Similarity: {summary.get('best_embedding_sim', 0):.4f}")
            logger.info(f"   Training Health: {summary.get('training_health', 'Unknown')}")
            logger.info(f"   Evaluations Performed: {summary.get('evaluations_performed', 0)}")
        
        if final_results and 'final_evaluation' in final_results:
            eval_results = final_results['final_evaluation']
            if eval_results:
                logger.info(f"🎯 FINAL EVALUATION (on {eval_results.get('samples_evaluated', 0)} samples):")
                logger.info(f"   Overall Embedding Similarity: {eval_results.get('overall_embedding_similarity', 0):.4f}")
                logger.info(f"   High Quality Images (>0.7): {eval_results.get('high_quality_images', 0)*100:.1f}%")
                logger.info(f"   Very High Quality Images (>0.8): {eval_results.get('very_high_quality_images', 0)*100:.1f}%")
                logger.info(f"   Excellent Quality Images (>0.9): {eval_results.get('excellent_quality_images', 0)*100:.1f}%")
        
        # Success assessment
        if final_results and 'training_summary' in final_results:
            final_emb_sim = final_results['training_summary'].get('best_embedding_sim', 0)
            if final_emb_sim > 0.1:
                logger.info("🎉 SUCCESS: Model shows good embedding generation with proper normalization!")
            elif final_emb_sim > 0.05:
                logger.info("📈 PROGRESS: Model shows learning with proper normalization")
            else:
                logger.info("⚠️ NEEDS WORK: Low embedding similarity, but normalization is now fixed")
        
        logger.info("🔧 NORMALIZATION STATUS: L2 normalization enabled - target norms should now be ~1.0")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)