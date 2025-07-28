#!/usr/bin/env python3
"""
CLIP Reproduction Training Script - train_dit.py (UPDATED)
Main training script for reproducing CLIP embeddings from EVA embeddings

UPDATES:
1. ✅ 3D Rotary Position Embedding for spatial understanding
2. ✅ Sandwich Normalization (RMSNorm before and after attention/MLP)
3. ✅ WandB integration with comprehensive metrics logging
4. ✅ Handles IterableDataset length issues for WandB
5. ✅ BLIP3-o paper alignment

Key features:
1. Minimal normalization (only for evaluation similarity)
2. Raw embedding space training
3. Comprehensive monitoring and debugging
4. WandB experiment tracking
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
            logging.FileHandler('clip_reproduction_training.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CLIP Reproduction from EVA Embeddings with BLIP3-o DiT")
    
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
    
    # NEW: BLIP3-o architecture features
    parser.add_argument("--use_3d_rope", action="store_true", default=True,
                       help="Use 3D Rotary Position Embedding")
    parser.add_argument("--use_sandwich_norm", action="store_true", default=True,
                       help="Use sandwich normalization (RMSNorm before and after)")
    parser.add_argument("--no_3d_rope", action="store_true",
                       help="Disable 3D RoPE (use standard RoPE)")
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
    
    # NEW: WandB configuration
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Enable WandB logging")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="blip3o-clip-reproduction",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None,
                       help="WandB tags for the run")
    
    return parser.parse_args()

def setup_device_and_model(args, logger):
    """Setup device and create model with BLIP3-o features"""
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
    
    # Process architecture arguments
    use_3d_rope = args.use_3d_rope and not args.no_3d_rope
    use_sandwich_norm = args.use_sandwich_norm and not args.no_sandwich_norm
    
    logger.info("🏗️ BLIP3-o Architecture Configuration:")
    logger.info(f"  3D Rotary Position Embedding: {'✅ Enabled' if use_3d_rope else '❌ Disabled'}")
    logger.info(f"  Sandwich Normalization: {'✅ Enabled' if use_sandwich_norm else '❌ Disabled'}")
    
    # Import and create model - try multiple import paths
    model = None
    try:
        # Try the updated file first
        from blip3o_dit import create_clip_reproduction_model, BLIP3oCLIPDiTConfig
        logger.info("✅ Imported UPDATED model from blip3o_dit.py")
        model = create_clip_reproduction_model(
            model_size=args.model_size,
            training_mode=args.training_mode,
            use_3d_rope=use_3d_rope,
            use_sandwich_norm=use_sandwich_norm,
        )
    except ImportError as e:
        logger.warning(f"Failed to import from blip3o_dit.py: {e}")
        try:
            # Try using the modules init
            from src.modules import create_clip_reproduction_model, BLIP3oCLIPDiTConfig
            logger.info("✅ Imported model from src.modules")
            model = create_clip_reproduction_model(
                model_size=args.model_size,
                training_mode=args.training_mode,
                use_3d_rope=use_3d_rope,
                use_sandwich_norm=use_sandwich_norm,
            )
        except ImportError as e2:
            logger.error(f"❌ Could not import model from src.modules: {e2}")
            try:
                # Direct import from the actual file
                from src.modules.models.blip3o_dit import create_clip_reproduction_model, BLIP3oCLIPDiTConfig
                logger.info("✅ Imported model directly from src.modules.models.blip3o_dit")
                model = create_clip_reproduction_model(
                    model_size=args.model_size,
                    training_mode=args.training_mode,
                    use_3d_rope=use_3d_rope,
                    use_sandwich_norm=use_sandwich_norm,
                )
            except ImportError as e3:
                logger.error(f"❌ Could not import model directly: {e3}")
                raise ImportError("Could not import model from any path")
    
    if model is None:
        raise RuntimeError("Failed to create model")
    
    logger.info(f"Creating {args.model_size} model for {args.training_mode} mode...")
    
    model = model.to(device)
    
    logger.info(f"Model created with {model.get_num_parameters():,} parameters")
    logger.info(f"Model moved to {device}")
    
    # Print architecture validation
    if hasattr(model, 'config'):
        config = model.config
        logger.info("🔍 Architecture Validation:")
        logger.info(f"  3D RoPE: {'✅' if getattr(config, 'use_3d_rope', False) else '❌'}")
        logger.info(f"  Sandwich Norm: {'✅' if getattr(config, 'use_sandwich_norm', False) else '❌'}")
        logger.info(f"  Grid Size: {getattr(config, 'grid_size', 16)}x{getattr(config, 'grid_size', 16)}")
        logger.info(f"  Grouped-Query Attention: {getattr(config, 'num_attention_heads', 12)}/{getattr(config, 'num_key_value_heads', 4)}")
    
    return device, model

def create_loss_function(args, logger):
    """Create loss function"""
    loss_fn = None
    try:
        from blip3o_fm_loss import create_clip_reproduction_loss
        logger.info("✅ Imported loss from blip3o_fm_loss.py")
        loss_fn = create_clip_reproduction_loss(
            prediction_type="velocity",
            flow_type="rectified",
            loss_weight=1.0,
            debug_mode=args.debug_mode
        )
    except ImportError as e:
        logger.warning(f"Failed to import from blip3o_fm_loss.py: {e}")
        try:
            from src.modules import create_clip_reproduction_loss
            logger.info("✅ Imported loss from src.modules")
            loss_fn = create_clip_reproduction_loss(
                prediction_type="velocity",
                flow_type="rectified",
                loss_weight=1.0,
                debug_mode=args.debug_mode
            )
        except ImportError as e2:
            logger.error(f"❌ Could not import loss from src.modules: {e2}")
            try:
                from src.modules.losses.blip3o_fm_loss import create_clip_reproduction_loss
                logger.info("✅ Imported loss directly from src.modules.losses.blip3o_fm_loss")
                loss_fn = create_clip_reproduction_loss(
                    prediction_type="velocity",
                    flow_type="rectified",
                    loss_weight=1.0,
                    debug_mode=args.debug_mode
                )
            except ImportError as e3:
                logger.error(f"❌ Could not import loss directly: {e3}")
                raise ImportError("Could not import loss from any path")
    
    if loss_fn is None:
        raise RuntimeError("Failed to create loss function")
    
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
    train_dataloader, eval_dataloader = None, None
    try:
        from blip3o_datasets import create_clip_reproduction_dataloaders
        logger.info("✅ Imported dataset from blip3o_datasets.py")
        train_dataloader, eval_dataloader = create_clip_reproduction_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            training_mode=args.training_mode,
            max_shards=args.max_shards,
            normalize_embeddings=False,  # Disable normalization
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    except ImportError as e:
        logger.warning(f"Failed to import from blip3o_datasets.py: {e}")
        try:
            from src.modules import create_clip_reproduction_dataloaders
            logger.info("✅ Imported dataset from src.modules")
            train_dataloader, eval_dataloader = create_clip_reproduction_dataloaders(
                chunked_embeddings_dir=args.chunked_embeddings_dir,
                batch_size=args.batch_size,
                training_mode=args.training_mode,
                max_shards=args.max_shards,
                normalize_embeddings=False,  # Disable normalization
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available()
            )
        except ImportError as e2:
            logger.error(f"❌ Could not import dataset from src.modules: {e2}")
            try:
                from src.modules.datasets.blip3o_dataset import create_clip_reproduction_dataloaders
                logger.info("✅ Imported dataset directly from src.modules.datasets.blip3o_dataset")
                train_dataloader, eval_dataloader = create_clip_reproduction_dataloaders(
                    chunked_embeddings_dir=args.chunked_embeddings_dir,
                    batch_size=args.batch_size,
                    training_mode=args.training_mode,
                    max_shards=args.max_shards,
                    normalize_embeddings=False,  # Disable normalization
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available()
                )
            except ImportError as e3:
                logger.error(f"❌ Could not import dataset directly: {e3}")
                raise ImportError("Could not import dataset from any path")
    
    if train_dataloader is None:
        raise RuntimeError("Failed to create dataloaders")
    
    logger.info(f"Dataloaders created")
    
    # Safely get lengths
    train_length = get_dataloader_length_safe(train_dataloader)
    if train_length != "unknown":
        logger.info(f"  Training batches: {train_length:,}")
    else:
        logger.info(f"  Training batches: Estimated from IterableDataset")
    
    logger.info(f"  Evaluation available: {eval_dataloader is not None}")
    logger.info(f"  🚫 Normalization disabled in dataloaders")
    
    return train_dataloader, eval_dataloader

def create_trainer(model, loss_fn, train_dataloader, eval_dataloader, args, device, logger):
    """Create trainer with WandB integration"""
    # Process WandB arguments
    use_wandb = args.use_wandb
    
    # Create run name if not provided
    wandb_run_name = args.wandb_run_name
    if wandb_run_name is None and use_wandb:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch_features = []
        if getattr(model.config, 'use_3d_rope', False):
            arch_features.append("3drope")
        if getattr(model.config, 'use_sandwich_norm', False):
            arch_features.append("sandwich")
        arch_str = "_".join(arch_features) if arch_features else "standard"
        wandb_run_name = f"blip3o_{args.model_size}_{args.training_mode}_{arch_str}_{timestamp}"
    
    # WandB configuration
    wandb_config = {
        "model_size": args.model_size,
        "training_mode": args.training_mode,
        "use_3d_rope": getattr(model.config, 'use_3d_rope', False),
        "use_sandwich_norm": getattr(model.config, 'use_sandwich_norm', False),
        "batch_size": args.batch_size,
        "max_shards": args.max_shards,
        "experiment_version": "v3_with_blip3o_features",
    }
    
    # Add tags
    wandb_tags = ["blip3o", "clip_reproduction", "eva_conditioning"]
    if getattr(model.config, 'use_3d_rope', False):
        wandb_tags.append("3d_rope")
    if getattr(model.config, 'use_sandwich_norm', False):
        wandb_tags.append("sandwich_norm")
    if args.overfit_test_size:
        wandb_tags.append("overfit_test")
    if args.wandb_tags:
        wandb_tags.extend(args.wandb_tags)
    
    trainer = None
    try:
        from blip3o_trainer import create_clip_trainer
        logger.info("✅ Imported UPDATED trainer from blip3o_trainer.py")
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
            use_wandb=use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_config=wandb_config,
        )
    except ImportError as e:
        logger.warning(f"Failed to import from blip3o_trainer.py: {e}")
        try:
            from src.modules import create_clip_trainer
            logger.info("✅ Imported trainer from src.modules")
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
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=wandb_run_name,
                wandb_config=wandb_config,
            )
        except ImportError as e2:
            logger.error(f"❌ Could not import trainer from src.modules: {e2}")
            try:
                from src.modules.trainers.blip3o_trainer import create_clip_trainer
                logger.info("✅ Imported trainer directly from src.modules.trainers.blip3o_trainer")
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
                    use_wandb=use_wandb,
                    wandb_project=args.wandb_project,
                    wandb_run_name=wandb_run_name,
                    wandb_config=wandb_config,
                )
            except ImportError as e3:
                logger.error(f"❌ Could not import trainer directly: {e3}")
                raise ImportError("Could not import trainer from any path")
    
    if trainer is None:
        raise RuntimeError("Failed to create trainer")
    
    logger.info("Trainer created with WandB integration")
    logger.info(f"  📊 WandB enabled: {use_wandb}")
    if use_wandb:
        logger.info(f"  📊 WandB project: {args.wandb_project}")
        logger.info(f"  📊 WandB run name: {wandb_run_name}")
        logger.info(f"  📊 WandB tags: {wandb_tags}")
    
    return trainer

def load_config_if_available(args, logger):
    """Load configuration if available"""
    model_config = None
    
    # Try multiple import paths for config
    try:
        from src.modules import get_blip3o_clip_config, print_config_summary, FlowMatchingConfig, TrainingConfig, EvaluationConfig
        logger.info("✅ Imported config from src.modules")
        
        # Create all config objects with proper arguments
        model_config = get_blip3o_clip_config(
            args.model_size, 
            args.training_mode,
            use_3d_rope=args.use_3d_rope and not args.no_3d_rope,
            use_sandwich_norm=args.use_sandwich_norm and not args.no_sandwich_norm,
        )
        
        # Create flow config
        flow_config = FlowMatchingConfig(
            prediction_type="velocity",
            normalize_targets=True,
            flow_type="rectified",
            loss_scale=1.0,
        )
        
        # Create training config  
        training_config = TrainingConfig(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            debug_mode=args.debug_mode,
            overfit_test_size=args.overfit_test_size,
            eval_every_n_steps=args.eval_every_n_steps,
            eval_num_samples=args.eval_num_samples,
        )
        
        # Create evaluation config
        eval_config = EvaluationConfig(
            eval_every_n_steps=args.eval_every_n_steps,
            eval_num_samples=args.eval_num_samples,
            eval_inference_steps=50,
        )
        
        # Print config summary with all real objects
        print_config_summary(model_config, flow_config, training_config, eval_config)
        
        return model_config
        
    except ImportError as e:
        logger.warning(f"Failed to import config from src.modules: {e}")
        try:
            # Try direct import from the actual file
            from src.modules.config.blip3o_config import get_blip3o_clip_config, print_config_summary, FlowMatchingConfig, TrainingConfig, EvaluationConfig
            logger.info("✅ Imported config directly from src.modules.config.blip3o_config")
            
            # Create all config objects
            model_config = get_blip3o_clip_config(
                args.model_size, 
                args.training_mode,
                use_3d_rope=args.use_3d_rope and not args.no_3d_rope,
                use_sandwich_norm=args.use_sandwich_norm and not args.no_sandwich_norm,
            )
            
            flow_config = FlowMatchingConfig(
                prediction_type="velocity",
                normalize_targets=True,
                flow_type="rectified",
                loss_scale=1.0,
            )
            
            training_config = TrainingConfig(
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps,
                debug_mode=args.debug_mode,
                overfit_test_size=args.overfit_test_size,
                eval_every_n_steps=args.eval_every_n_steps,
                eval_num_samples=args.eval_num_samples,
            )
            
            eval_config = EvaluationConfig(
                eval_every_n_steps=args.eval_every_n_steps,
                eval_num_samples=args.eval_num_samples,
                eval_inference_steps=50,
            )
            
            print_config_summary(model_config, flow_config, training_config, eval_config)
            
            return model_config
            
        except ImportError as e2:
            logger.warning(f"⚠️ Could not import config from any path: {e2}")
            logger.warning("⚠️ Continuing without config summary.")
            return None
            
    except Exception as e:
        logger.error(f"Error creating config objects: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        # Return the model_config even if printing fails
        return model_config

def check_modules_availability(logger):
    """Check which modules are available"""
    logger.info("🔍 Checking module availability...")
    
    # Try to use the comprehensive modules init
    try:
        from src.modules import check_environment, print_environment_status
        logger.info("✅ Using comprehensive modules system")
        
        # Print environment status
        print_environment_status()
        
        # Check environment
        env_status = check_environment()
        if not env_status['all_available']:
            logger.warning(f"⚠️ Some modules not available: {env_status['missing_components']}")
        
        return env_status
    except ImportError:
        logger.info("📁 Using individual file imports")
        return {'all_available': True, 'missing_components': []}

def main():
    """Main training function with BLIP3-o features"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🚀 Starting CLIP Reproduction with BLIP3-o DiT")
    logger.info("=" * 80)
    logger.info("EXPERIMENT DETAILS:")
    logger.info("  📋 Task: Reproduce clean CLIP embeddings from EVA embeddings")
    logger.info("  🧠 Model: BLIP3-o DiT with advanced architecture features")
    logger.info("  🎯 Target: CLIP embeddings [B, N, 1024]")
    logger.info("  🎮 Conditioning: EVA embeddings [B, N, 4096]")
    logger.info("  🌊 Method: Rectified Flow Matching")
    logger.info("  🚫 Normalization: MINIMAL (only for evaluation similarity)")
    logger.info("=" * 80)
    logger.info("🏗️ BLIP3-o ARCHITECTURE FEATURES:")
    use_3d_rope = args.use_3d_rope and not args.no_3d_rope
    use_sandwich_norm = args.use_sandwich_norm and not args.no_sandwich_norm
    logger.info(f"  🌐 3D Rotary Position Embedding: {'✅ ENABLED' if use_3d_rope else '❌ DISABLED'}")
    logger.info(f"  🥪 Sandwich Normalization: {'✅ ENABLED' if use_sandwich_norm else '❌ DISABLED'}")
    logger.info(f"  🔍 Grouped-Query Attention: ✅ ENABLED")
    logger.info(f"  📊 WandB Logging: {'✅ ENABLED' if args.use_wandb and not args.no_wandb else '❌ DISABLED'}")
    logger.info("=" * 80)
    logger.info("📄 Updated file structure:")
    logger.info("  • Model: blip3o_dit.py (✅ 3D RoPE + Sandwich Norm)")
    logger.info("  • Loss: blip3o_fm_loss.py")
    logger.info("  • Dataset: blip3o_datasets.py")
    logger.info("  • Trainer: blip3o_trainer.py (✅ WandB Integration)")
    logger.info("  • Config: blip3o_config.py")
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
    if args.use_wandb and not args.no_wandb:
        logger.info(f"  📊 WandB project: {args.wandb_project}")
        if args.wandb_run_name:
            logger.info(f"  📊 WandB run: {args.wandb_run_name}")
    logger.info("=" * 80)
    
    try:
        # Check module availability
        env_status = check_modules_availability(logger)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config if available
        model_config = load_config_if_available(args, logger)
        
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
            'model_config': model_config.to_dict() if model_config and hasattr(model_config, 'to_dict') else {},
            'model_params': model.get_num_parameters() if hasattr(model, 'get_num_parameters') else 'unknown',
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'clip_reproduction_blip3o',
            'normalization_approach': 'minimal',
            'architecture_features': {
                '3d_rope': use_3d_rope,
                'sandwich_normalization': use_sandwich_norm,
                'grouped_query_attention': True,
                'minimal_normalization': True,
            },
            'file_names': {
                'model': 'blip3o_dit.py',
                'loss': 'blip3o_fm_loss.py',
                'dataset': 'blip3o_datasets.py',
                'trainer': 'blip3o_trainer.py',
                'config': 'blip3o_config.py',
                'training_script': 'train_dit.py',
            },
            'environment_status': env_status,
            'wandb_config': {
                'enabled': args.use_wandb and not args.no_wandb,
                'project': args.wandb_project,
                'run_name': args.wandb_run_name,
            },
            'updates_applied': [
                '3d_rotary_position_embedding',
                'sandwich_normalization_rms',
                'blip3o_paper_alignment',
                'wandb_integration',
                'iterable_dataset_wandb_fix',
                'comprehensive_metrics_logging',
                'minimal_normalization_approach',
                'raw_embedding_space_training',
                'gradient_flow_improvements',
                'proper_initialization',
                'numerical_stability',
                'overfitting_test_capability',
            ]
        }
        
        config_path = output_dir / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
        
        # Start training
        logger.info("\n🚀 Starting BLIP3-o training with advanced features...")
        logger.info("Expected behavior:")
        logger.info("  • Loss should decrease steadily")
        logger.info("  • Velocity similarity should increase")
        logger.info("  • CLIP similarity should improve during evaluation")
        logger.info("  • Gradients should be non-zero and stable")
        logger.info("  • Raw embeddings will be learned (no forced normalization)")
        logger.info("  • 3D RoPE should provide spatial understanding")
        logger.info("  • Sandwich normalization should improve gradient flow")
        logger.info("  • WandB should log comprehensive metrics")
        
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
        logger.info("🎉 BLIP3-o TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📊 RESULTS SUMMARY:")
        logger.info(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"  Total steps: {summary.get('total_steps', 0)}")
        logger.info(f"  Best loss: {summary.get('best_loss', float('inf')):.6f}")
        logger.info(f"  Best CLIP similarity: {summary.get('best_eval_similarity', 0):.4f}")
        logger.info(f"  🚫 Used minimal normalization approach")
        logger.info(f"  🌐 3D RoPE: {'✅' if use_3d_rope else '❌'}")
        logger.info(f"  🥪 Sandwich Norm: {'✅' if use_sandwich_norm else '❌'}")
        logger.info(f"  📊 WandB: {'✅' if summary.get('wandb_enabled', False) else '❌'}")
        
        # Evaluation results
        final_eval = summary.get('final_eval', {})
        if final_eval:
            logger.info(f"📊 FINAL EVALUATION:")
            logger.info(f"  CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
            logger.info(f"  High quality (>0.7): {final_eval.get('eval_high_quality', 0)*100:.1f}%")
            logger.info(f"  Very high quality (>0.8): {final_eval.get('eval_very_high_quality', 0)*100:.1f}%")
            logger.info(f"  Excellent quality (>0.9): {final_eval.get('eval_excellent_quality', 0)*100:.1f}%")
            logger.info(f"  Samples evaluated: {final_eval.get('eval_samples', 0)}")
        
        # Overfitting test results
        if args.overfit_test_size:
            overfit_success = summary.get('overfit_success', False)
            logger.info(f"🧪 OVERFITTING TEST: {'✅ PASSED' if overfit_success else '❌ FAILED'}")
            if overfit_success:
                logger.info("   ✅ Model can learn and memorize - BLIP3-o architecture is working!")
            else:
                logger.info("   ⚠️  Model struggles to overfit - check architecture/loss")
        
        # Architecture assessment
        best_sim = summary.get('best_eval_similarity', 0)
        logger.info(f"🏗️  BLIP3-o ARCHITECTURE ASSESSMENT:")
        if best_sim > 0.7:
            logger.info("   🎉 EXCELLENT: BLIP3-o DiT with 3D RoPE + Sandwich Norm working perfectly!")
        elif best_sim > 0.4:
            logger.info("   ✅ GOOD: BLIP3-o DiT architecture shows strong capability!")
        elif best_sim > 0.1:
            logger.info("   📈 FAIR: BLIP3-o DiT architecture is functional!")
        else:
            logger.info("   ⚠️  NEEDS WORK: Architecture may need tuning!")
        
        # WandB information
        if summary.get('wandb_enabled', False):
            logger.info(f"📊 WandB Dashboard: Check your {args.wandb_project} project for detailed metrics")
        
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