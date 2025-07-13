#!/usr/bin/env python3
"""
Main training script for BLIP3-o DiT with flow matching - UPDATED FOR CHUNKED TRAINING.
Trains a diffusion transformer to generate CLIP embeddings from EVA-CLIP conditioning.

This script implements the exact BLIP3-o training methodology as described in the paper.
UPDATED: Now supports both single-file and chunked embedding approaches.
UPDATED: Added support for scaling to 100k+ samples with chunked dataset loading.
"""

import os
import sys
import argparse
import logging
import torch
import wandb
from pathlib import Path
import json
from datetime import datetime
import traceback

# TEMPORARY FIX: Patch for transformers compatibility
def patch_trainer_for_compatibility():
    """Fix compute_loss method signature for newer transformers versions"""
    from src.modules.trainers.blip3o_trainer import BLIP3oTrainer
    
    # Store original method
    original_compute_loss = BLIP3oTrainer.compute_loss
    
    # Create new method that accepts the extra parameter
    def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Just ignore num_items_in_batch and call original method
        return original_compute_loss(self, model, inputs, return_outputs)
    
    # Replace the method
    BLIP3oTrainer.compute_loss = patched_compute_loss
    print("✅ Applied transformers compatibility patch")

# Apply the patch immediately
patch_trainer_for_compatibility()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.modules.config.blip3o_config import (
    BLIP3oDiTConfig, 
    FlowMatchingConfig, 
    TrainingConfig,
    get_default_blip3o_config,
    get_default_flow_matching_config,
    get_default_training_config
)
from src.modules.models.blip3o_dit import BLIP3oDiTModel, create_blip3o_dit_model
from src.modules.losses.flow_matching_loss import create_blip3o_flow_matching_loss
from src.modules.datasets.blip3o_dataset import create_blip3o_dataloaders
from src.modules.trainers.blip3o_trainer import BLIP3oTrainer, create_blip3o_training_args

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for BLIP3-o training."""
    parser = argparse.ArgumentParser(
        description="Train BLIP3-o DiT with flow matching (supports both single-file and chunked datasets)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments - now support both single file and chunked approaches
    embeddings_group = parser.add_mutually_exclusive_group(required=True)
    embeddings_group.add_argument(
        "--embeddings_path", type=str,
        help="Path to the BLIP3-o embeddings pickle file (single file approach)"
    )
    embeddings_group.add_argument(
        "--chunked_embeddings_dir", type=str,
        help="Path to directory containing chunked embedding files (chunked approach)"
    )
    
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for model checkpoints and logs"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_dim", type=int, default=1792,
                           help="Model hidden dimension")
    model_group.add_argument("--num_layers", type=int, default=24,
                           help="Number of transformer layers")
    model_group.add_argument("--num_heads", type=int, default=28,
                           help="Number of attention heads")
    model_group.add_argument("--num_kv_heads", type=int, default=None,
                           help="Number of KV heads (default: same as num_heads)")
    model_group.add_argument("--multiple_of", type=int, default=256,
                           help="FFN dimension multiple")
    model_group.add_argument("--ffn_dim_multiplier", type=float, default=None,
                           help="FFN dimension multiplier")
    model_group.add_argument("--gradient_checkpointing", action="store_true",
                           help="Enable gradient checkpointing for memory efficiency")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--num_epochs", type=int, default=10,
                           help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=32,
                           help="Training batch size per device")
    train_group.add_argument("--eval_batch_size", type=int, default=64,
                           help="Evaluation batch size per device")
    train_group.add_argument("--learning_rate", type=float, default=1e-4,
                           help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                           help="Weight decay for regularization")
    train_group.add_argument("--warmup_steps", type=int, default=1000,
                           help="Number of warmup steps")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Gradient accumulation steps")
    
    # Flow matching configuration
    flow_group = parser.add_argument_group("Flow Matching Configuration")
    flow_group.add_argument("--sigma_min", type=float, default=1e-4,
                          help="Minimum noise sigma for flow matching")
    flow_group.add_argument("--sigma_max", type=float, default=1.0,
                          help="Maximum noise sigma for flow matching")
    flow_group.add_argument("--prediction_type", type=str, default="v_prediction",
                          choices=["v_prediction", "epsilon"],
                          help="Flow matching prediction type")
    flow_group.add_argument("--schedule_type", type=str, default="linear",
                          choices=["linear", "cosine"],
                          help="Noise schedule type")
    flow_group.add_argument("--regularization_weight", type=float, default=0.0,
                          help="Regularization weight")
    
    # Data configuration - UPDATED for chunked support
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--subset_size", type=int, default=None,
                          help="Use subset of data for debugging (None for all data) - only for single-file")
    data_group.add_argument("--eval_split", type=float, default=0.1,
                          help="Fraction of data to use for evaluation")
    data_group.add_argument("--normalize_embeddings", action="store_true",
                          help="Normalize embeddings to unit norm")
    data_group.add_argument("--num_workers", type=int, default=4,
                          help="Number of dataloader workers (ignored for chunked datasets)")
    data_group.add_argument("--delete_after_use", action="store_true",
                          help="Delete embedding chunks after processing (chunked only)")
    
    # Logging and saving
    log_group = parser.add_argument_group("Logging and Saving")
    log_group.add_argument("--logging_steps", type=int, default=100,
                         help="Log metrics every N steps")
    log_group.add_argument("--save_steps", type=int, default=1000,
                         help="Save checkpoint every N steps")
    log_group.add_argument("--eval_steps", type=int, default=1000,
                         help="Evaluate model every N steps")
    log_group.add_argument("--wandb_project", type=str, default="blip3o-dit",
                         help="Weights & Biases project name")
    log_group.add_argument("--wandb_run_name", type=str, default=None,
                         help="Weights & Biases run name")
    log_group.add_argument("--no_wandb", action="store_true",
                         help="Disable Weights & Biases logging")
    
    # Hardware configuration
    hw_group = parser.add_argument_group("Hardware Configuration")
    hw_group.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (fp16)")
    hw_group.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 mixed precision training")
    hw_group.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    hw_group.add_argument("--compile_model", action="store_true",
                        help="Use torch.compile for model optimization")
    
    # Resume training
    resume_group = parser.add_argument_group("Resume Training")
    resume_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                            help="Path to checkpoint to resume training from")
    
    # Debug mode
    debug_group = parser.add_argument_group("Debug Configuration")
    debug_group.add_argument("--debug", action="store_true",
                           help="Enable debug mode with reduced data and epochs")
    debug_group.add_argument("--dry_run", action="store_true",
                           help="Run through setup without training")
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup and validate computation device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using specified device: {device}")
    
    return device


def setup_wandb(args, use_chunked: bool):
    """Initialize Weights & Biases logging."""
    if args.no_wandb:
        logger.info("Weights & Biases logging disabled")
        return
    
    # Create run name if not provided
    run_name = args.wandb_run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        approach = "chunked" if use_chunked else "single"
        run_name = f"blip3o-dit-{approach}-{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        tags=["blip3o", "dit", "flow-matching", "clip-generation"] + (["chunked"] if use_chunked else ["single-file"]),
        notes=f"BLIP3-o DiT training with flow matching for CLIP embedding generation ({'chunked' if use_chunked else 'single-file'} approach)",
    )
    
    logger.info(f"Initialized Weights & Biases: {wandb.run.url}")


def create_model_config(args) -> BLIP3oDiTConfig:
    """Create model configuration from arguments - FIXED dimensions."""
    return BLIP3oDiTConfig(
        input_size=16,                          # UPDATED: 16x16 = 256 tokens (was 8 for 64 tokens)
        patch_size=1,                           # Pre-tokenized (fixed)
        in_channels=1024,                       # CLIP dimension (FIXED: matches your embeddings)
        dim=args.model_dim,                     # Hidden dimension
        eva_embedding_size=4096,                # EVA-CLIP dimension (FIXED: matches your embeddings)
        n_layers=args.num_layers,               # Number of layers
        n_heads=args.num_heads,                 # Attention heads
        n_kv_heads=args.num_kv_heads or args.num_heads,  # KV heads
        multiple_of=args.multiple_of,           # FFN multiple
        ffn_dim_multiplier=args.ffn_dim_multiplier,  # FFN multiplier
        norm_eps=1e-5,                          # Layer norm epsilon
        qk_norm=True,                           # Query-key normalization
        learn_sigma=False,                      # Flow matching (fixed)
        _gradient_checkpointing=args.gradient_checkpointing,
    )


def create_flow_matching_config(args) -> FlowMatchingConfig:
    """Create flow matching configuration from arguments - FIXED dimensions."""
    return FlowMatchingConfig(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        prediction_type=args.prediction_type,
        clip_dim=1024,                          # FIXED: CLIP dimension matches your embeddings
        eva_dim=4096,                           # FIXED: EVA-CLIP dimension matches your embeddings
        regularization_weight=args.regularization_weight,
        schedule_type=args.schedule_type,
    )


def save_configs(args, output_dir: Path, use_chunked: bool):
    """Save all configurations to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training arguments
    training_config = {
        **vars(args),
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'approach': 'chunked' if use_chunked else 'single-file',
        'fixed_dimensions': {
            'clip_dim': 1024,  # Document the fixed dimensions
            'eva_dim': 4096,
            'tokens': 256,     # UPDATED: 256 tokens
        }
    }
    
    with open(output_dir / "training_args.json", 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Save model configuration
    model_config = create_model_config(args)
    with open(output_dir / "model_config.json", 'w') as f:
        json.dump(model_config.to_dict(), f, indent=2)
    
    # Save flow matching configuration
    flow_config = create_flow_matching_config(args)
    flow_config_dict = {
        'sigma_min': flow_config.sigma_min,
        'sigma_max': flow_config.sigma_max,
        'prediction_type': flow_config.prediction_type,
        'clip_dim': flow_config.clip_dim,
        'eva_dim': flow_config.eva_dim,
        'regularization_weight': flow_config.regularization_weight,
        'schedule_type': flow_config.schedule_type,
    }
    
    with open(output_dir / "flow_matching_config.json", 'w') as f:
        json.dump(flow_config_dict, f, indent=2)
    
    logger.info(f"Configurations saved to {output_dir}")


def load_manifest(chunked_dir: Path) -> dict:
    """Load and validate chunked embeddings manifest."""
    manifest_path = chunked_dir / "embeddings_manifest.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Validate manifest
    required_keys = ['total_shards', 'total_samples', 'format_version']
    for key in required_keys:
        if key not in manifest:
            raise ValueError(f"Invalid manifest: missing key '{key}'")
    
    # Check format version
    format_version = manifest.get('format_version', '')
    if '256' not in format_version:
        logger.warning(f"Manifest format version '{format_version}' may not be 256-token compatible")
    
    return manifest


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup debug mode
    if args.debug:
        logger.info("Debug mode enabled")
        args.subset_size = 1000
        args.num_epochs = 2
        args.logging_steps = 10
        args.save_steps = 100
        args.eval_steps = 100
        args.no_wandb = True
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("BLIP3-o DiT Training with Flow Matching (Single-File + Chunked Support)")
    logger.info("=" * 80)
    
    try:
        # Determine approach (single-file vs chunked)
        if args.embeddings_path:
            embeddings_path = Path(args.embeddings_path)
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            logger.info(f"Using SINGLE-FILE approach: {embeddings_path}")
            use_chunked = False
        else:
            embeddings_path = Path(args.chunked_embeddings_dir)
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Chunked embeddings directory not found: {embeddings_path}")
            
            # Load and validate manifest
            manifest = load_manifest(embeddings_path)
            logger.info(f"Using CHUNKED approach: {embeddings_path}")
            logger.info(f"  Total shards: {manifest['total_shards']}")
            logger.info(f"  Total samples: {manifest['total_samples']:,}")
            logger.info(f"  Format: {manifest['format_version']}")
            use_chunked = True
        
        # Save configurations
        save_configs(args, output_dir, use_chunked)
        
        # Setup device
        device = setup_device(args.device)
        
        # Setup wandb
        if not args.no_wandb:
            setup_wandb(args, use_chunked)
        
        # Create model configuration and model
        logger.info("Creating model...")
        model_config = create_model_config(args)
        model = create_blip3o_dit_model(config=model_config)
        model.to(device)
        
        # Log model info
        total_params = model.get_num_parameters(trainable_only=False)
        trainable_params = model.get_num_parameters(trainable_only=True)
        memory_footprint = model.get_memory_footprint()
        
        logger.info(f"Model created successfully:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Memory footprint: {memory_footprint}")
        logger.info(f"  Model dimensions: CLIP={model_config.in_channels}, EVA={model_config.eva_embedding_size}, Hidden={model_config.dim}")
        logger.info(f"  Tokens: {model_config.input_size}x{model_config.input_size} = {model_config.input_size**2}")
        
        # Compile model if requested
        if args.compile_model:
            try:
                model = torch.compile(model)
                logger.info("Model compiled for optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Create flow matching loss
        logger.info("Creating flow matching loss...")
        flow_config = create_flow_matching_config(args)
        flow_matching_loss = create_blip3o_flow_matching_loss(config=flow_config)
        
        # Create datasets and dataloaders - UPDATED for chunked support
        logger.info("Creating datasets...")
        
        if use_chunked:
            # Use chunked dataset approach
            from src.modules.datasets.blip3o_dataset import create_chunked_dataloaders
            
            logger.info("Loading chunked datasets...")
            train_dataloader, eval_dataloader = create_chunked_dataloaders(
                chunked_embeddings_dir=embeddings_path,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                eval_split_ratio=args.eval_split,
                normalize_embeddings=args.normalize_embeddings,
                delete_after_use=args.delete_after_use,
                num_workers=0,  # IterableDataset requires num_workers=0
            )
            
            # For chunked datasets, we can't get exact length, so use manifest
            total_samples = manifest['total_samples']
            train_samples = int(total_samples * (1 - args.eval_split))
            eval_samples = total_samples - train_samples
            
            logger.info(f"Chunked datasets created:")
            logger.info(f"  Total samples: {total_samples:,}")
            logger.info(f"  Training samples: {train_samples:,}")
            logger.info(f"  Evaluation samples: {eval_samples:,}")
            logger.info(f"  Total shards: {manifest['total_shards']}")
            logger.info(f"  Sequential loading: One shard at a time")
            logger.info(f"  Auto cleanup: {args.delete_after_use}")
            
            # Create dummy datasets for trainer compatibility
            class DummyDataset:
                def __init__(self, length):
                    self.length = length
                def __len__(self):
                    return self.length
            
            train_dataset = DummyDataset(train_samples)
            eval_dataset = DummyDataset(eval_samples) if eval_dataloader else None
            
        else:
            # Use traditional single-file approach
            logger.info("Loading single-file dataset...")
            train_dataloader, eval_dataloader = create_blip3o_dataloaders(
                embeddings_path=embeddings_path,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                subset_size=args.subset_size,
                normalize_embeddings=args.normalize_embeddings,
                eval_split_ratio=args.eval_split,
                expected_eva_dim=4096,     # Specify expected dimensions
                expected_clip_dim=1024,    # Specify expected dimensions
                expected_tokens=256,       # UPDATED: 256 tokens
            )
            
            train_dataset = train_dataloader.dataset
            eval_dataset = eval_dataloader.dataset if eval_dataloader else None
            
            logger.info(f"Single-file datasets created:")
            logger.info(f"  Training samples: {len(train_dataset)}")
            if eval_dataset:
                logger.info(f"  Evaluation samples: {len(eval_dataset)}")
        
        # Log dataloader info
        try:
            batches_per_epoch = len(train_dataloader)
            logger.info(f"  Training batches per epoch: {batches_per_epoch}")
        except:
            logger.info(f"  Training batches per epoch: Unknown (streaming/chunked)")
        
        # Create training arguments
        training_args = create_blip3o_training_args(
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps if eval_dataset else 0,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16 and not args.bf16,
            bf16=args.bf16,
            dataloader_num_workers=0 if use_chunked else args.num_workers,  # Chunked requires 0 workers
            # Fix for newer transformers compatibility
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Only keep the latest checkpoint to save disk space
        training_args.save_total_limit = 1
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = BLIP3oTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Check for dry run
        if args.dry_run:
            logger.info("Dry run completed successfully - exiting without training")
            return 0
        
        # Print training summary
        try:
            total_steps = len(train_dataloader) * args.num_epochs
            steps_info = f"Total steps: {total_steps}"
        except:
            steps_info = "Total steps: Unknown (streaming dataset)"
        
        logger.info("Training Summary:")
        logger.info(f"  Approach: {'Chunked' if use_chunked else 'Single-file'}")
        logger.info(f"  Epochs: {args.num_epochs}")
        try:
            logger.info(f"  Steps per epoch: {len(train_dataloader)}")
        except:
            logger.info(f"  Steps per epoch: Unknown (streaming)")
        logger.info(f"  {steps_info}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Weight decay: {args.weight_decay}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  Mixed precision: {'fp16' if args.fp16 else 'bf16' if args.bf16 else 'fp32'}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  3D RoPE: Enabled with proper spatial-temporal encoding")
        
        if use_chunked:
            logger.info(f"  Chunked features:")
            logger.info(f"    Sequential loading: One shard at a time")
            logger.info(f"    Memory efficient: Constant memory usage")
            logger.info(f"    Auto cleanup: {args.delete_after_use}")
            logger.info(f"    Scalable: Can handle 100k+ samples")
        
        # Start training
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        # Resume from checkpoint if specified
        resume_from_checkpoint = args.resume_from_checkpoint
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        
        # Train the model
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        logger.info("Training completed! Saving final model...")
        trainer.save_model()
        
        # Run final evaluation
        if eval_dataset:
            logger.info("Running final evaluation...")
            final_metrics = trainer.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # Generate sample outputs for verification (only for single-file approach)
        if not use_chunked:
            logger.info("Generating sample outputs...")
            try:
                # Get eval dataloader for sampling
                eval_dl = eval_dataloader if eval_dataloader else train_dataloader
                sample_batch = next(iter(eval_dl))
                eva_conditioning = sample_batch['eva_embeddings'][:4].to(device)
                
                model.eval()
                with torch.no_grad():
                    generated_samples = model.generate(
                        encoder_hidden_states=eva_conditioning,
                        num_inference_steps=50,
                    )
                
                # Save sample generation
                sample_output = {
                    'eva_conditioning': eva_conditioning.cpu(),
                    'generated_clip': generated_samples.cpu(),
                    'target_clip': sample_batch['clip_embeddings'][:4],
                    'captions': sample_batch['captions'][:4],
                }
                
                torch.save(sample_output, output_dir / 'sample_generation.pt')
                logger.info(f"Sample generation saved to {output_dir / 'sample_generation.pt'}")
                
            except Exception as e:
                logger.warning(f"Failed to generate samples: {e}")
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"🎉 Your BLIP3-o model with {'chunked' if use_chunked else 'single-file'} training is ready!")
        if use_chunked:
            logger.info("🚀 Chunked approach successfully handled large-scale training!")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'trainer' in locals():
            logger.info("Saving checkpoint...")
            trainer.save_model(output_dir / "interrupted_checkpoint")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return 1
        
    finally:
        # Clean up wandb
        if not args.no_wandb and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)