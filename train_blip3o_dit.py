#!/usr/bin/env python3
"""
FIXED Main training script for BLIP3-o DiT with flow matching - CHUNKED TRAINING.
Trains a diffusion transformer to generate CLIP embeddings from EVA-CLIP conditioning.

This script implements the exact BLIP3-o training methodology with chunked dataset support.
FIXED: All import issues, dataset handling, and trainer compatibility.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def patch_trainer_for_compatibility():
    """Fix compute_loss method signature for newer transformers versions"""
    try:
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
    except Exception as e:
        print(f"⚠️  Failed to apply compatibility patch: {e}")

# Apply the patch immediately
patch_trainer_for_compatibility()

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
from src.modules.datasets.blip3o_dataset import (
    create_chunked_dataloaders, 
    create_chunked_dataloader,
    BLIP3oEmbeddingDataset,
    chunked_collate_fn
)
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
        description="Train BLIP3-o DiT with flow matching using chunked embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments - chunked embeddings directory
    parser.add_argument(
        "--chunked_embeddings_dir", type=str, required=True,
        help="Path to directory containing chunked embedding files"
    )
    
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for model checkpoints and logs"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_dim", type=int, default=512,
                           help="Model hidden dimension (default: 512 for 3D RoPE compatibility)")
    model_group.add_argument("--num_layers", type=int, default=24,
                           help="Number of transformer layers")
    model_group.add_argument("--num_heads", type=int, default=8,
                           help="Number of attention heads (default: 8 for head_dim=64)")
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
    train_group.add_argument("--num_epochs", type=int, default=5,
                           help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=64,
                           help="Training batch size per device")
    train_group.add_argument("--eval_batch_size", type=int, default=128,
                           help="Evaluation batch size per device")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                           help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                           help="Weight decay for regularization")
    train_group.add_argument("--warmup_steps", type=int, default=20,
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
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--eval_split", type=float, default=0.1,
                          help="Fraction of data to use for evaluation")
    data_group.add_argument("--normalize_embeddings", action="store_true",
                          help="Normalize embeddings to unit norm")
    data_group.add_argument("--delete_after_use", action="store_true",
                          help="Delete embedding chunks after processing")
    
    # Logging and saving
    log_group = parser.add_argument_group("Logging and Saving")
    log_group.add_argument("--logging_steps", type=int, default=10,
                         help="Log metrics every N steps")
    log_group.add_argument("--save_steps", type=int, default=200,
                         help="Save checkpoint every N steps")
    log_group.add_argument("--eval_steps", type=int, default=50,
                         help="Evaluate model every N steps")
    log_group.add_argument("--wandb_project", type=str, default="blip3o-dit-256-tokens",
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
                           help="Enable debug mode with reduced epochs")
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


def setup_wandb(args):
    """Initialize Weights & Biases logging."""
    if args.no_wandb:
        logger.info("Weights & Biases logging disabled")
        return
    
    # Create run name if not provided
    run_name = args.wandb_run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"256-tokens-{timestamp}"
    
    # Initialize wandb
    try:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=["blip3o", "dit", "flow-matching", "clip-generation", "chunked", "256-tokens"],
            notes="BLIP3-o DiT training with flow matching for CLIP embedding generation (256 tokens, chunked approach)",
        )
        
        logger.info(f"Initialized Weights & Biases: {wandb.run.url}")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        args.no_wandb = True


def create_model_config(args) -> BLIP3oDiTConfig:
    """Create model configuration from arguments - FIXED dimensions."""
    return BLIP3oDiTConfig(
        input_size=16,                          # 16x16 = 256 tokens
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


def save_configs(args, output_dir: Path):
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
        'approach': 'chunked',
        'fixed_dimensions': {
            'clip_dim': 1024,  # Document the fixed dimensions
            'eva_dim': 4096,
            'tokens': 256,     # 256 tokens
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


def get_temp_directory():
    """Get temp directory for saving checkpoints."""
    if "TMPDIR" in os.environ:
        return Path(os.environ["TMPDIR"])
    elif "SCRATCH_SHARED" in os.environ:
        return Path(os.environ["SCRATCH_SHARED"])
    else:
        return Path("./temp")


def create_temp_output_dir(base_output_dir: str) -> Path:
    """Create output directory in temp storage."""
    temp_dir = get_temp_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create temp output directory
    temp_output_dir = temp_dir / f"blip3o_training_{timestamp}"
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create the requested output directory as a symlink if possible
    base_output_path = Path(base_output_dir)
    try:
        base_output_path.mkdir(parents=True, exist_ok=True)
        # Create a symlink from base to temp (if possible)
        if not (base_output_path / "temp_link").exists():
            (base_output_path / "temp_link").symlink_to(temp_output_dir)
        
        # Create info file
        with open(base_output_path / "temp_location.txt", 'w') as f:
            f.write(f"Actual training directory: {temp_output_dir}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Use this path to access model checkpoints and logs.\n")
        
        logger.info(f"Created symlink from {base_output_path} to {temp_output_dir}")
    except Exception as e:
        logger.warning(f"Could not create symlink: {e}")
    
    return temp_output_dir


class DummyDataset:
    """Dummy dataset for trainer compatibility."""
    def __init__(self, length):
        self.length = length
    
    def __len__(self):
        return self.length


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup debug mode
    if args.debug:
        logger.info("Debug mode enabled")
        args.num_epochs = 2
        args.logging_steps = 5
        args.save_steps = 50
        args.eval_steps = 25
        args.no_wandb = True
    
    # Setup output directory in temp storage
    temp_output_dir = create_temp_output_dir(args.output_dir)
    logger.info(f"Using temp output directory: {temp_output_dir}")
    
    # Setup logging to file
    log_file = temp_output_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("BLIP3-o DiT Training with Flow Matching (256 TOKENS, CHUNKED)")
    logger.info("=" * 80)
    
    try:
        # Validate chunked embeddings directory
        embeddings_path = Path(args.chunked_embeddings_dir)
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Chunked embeddings directory not found: {embeddings_path}")
        
        # Load and validate manifest
        manifest = load_manifest(embeddings_path)
        logger.info(f"Using CHUNKED approach: {embeddings_path}")
        logger.info(f"  Total shards: {manifest['total_shards']}")
        logger.info(f"  Total samples: {manifest['total_samples']:,}")
        logger.info(f"  Format: {manifest['format_version']}")
        
        # Save configurations
        save_configs(args, temp_output_dir)
        
        # Setup device
        device = setup_device(args.device)
        
        # Setup wandb
        if not args.no_wandb:
            setup_wandb(args)
        
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
        
        # Create chunked datasets and dataloaders
        logger.info("Creating chunked datasets...")
        
        train_dataloader, eval_dataloader = create_chunked_dataloaders(
            chunked_embeddings_dir=embeddings_path,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_split_ratio=args.eval_split,
            normalize_embeddings=args.normalize_embeddings,
            delete_after_use=args.delete_after_use,
            num_workers=0,  # IterableDataset requires num_workers=0
        )
        
        # Calculate sample counts from manifest
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
        train_dataset = DummyDataset(train_samples)
        eval_dataset = DummyDataset(eval_samples) if eval_dataloader else None
        
        # Create training arguments
        training_args = create_blip3o_training_args(
            output_dir=str(temp_output_dir),
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
            dataloader_num_workers=0,  # Chunked requires 0 workers
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Only keep the latest checkpoint to save disk space
        training_args.save_total_limit = 2
        
        # Create custom trainer that uses our dataloaders
        logger.info("Creating trainer...")
        trainer = BLIP3oTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Override the get_train_dataloader method to use our chunked dataloader
        def get_train_dataloader():
            return train_dataloader
        
        def get_eval_dataloader(eval_dataset=None):
            return eval_dataloader
        
        trainer.get_train_dataloader = get_train_dataloader
        trainer.get_eval_dataloader = get_eval_dataloader
        
        # Check for dry run
        if args.dry_run:
            logger.info("Dry run completed successfully - exiting without training")
            return 0
        
        # Print training summary
        logger.info("Training Summary:")
        logger.info(f"  Approach: Chunked (256 tokens)")
        logger.info(f"  Epochs: {args.num_epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Weight decay: {args.weight_decay}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  Mixed precision: {'fp16' if args.fp16 else 'bf16' if args.bf16 else 'fp32'}")
        logger.info(f"  Temp output directory: {temp_output_dir}")
        logger.info(f"  3D RoPE: Enabled with proper spatial-temporal encoding")
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
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {temp_output_dir}")
        logger.info(f"Base output directory info: {args.output_dir}")
        logger.info("🎉 Your BLIP3-o model with chunked training is ready!")
        logger.info("🚀 Chunked approach successfully handled large-scale training!")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'trainer' in locals():
            logger.info("Saving checkpoint...")
            trainer.save_model(temp_output_dir / "interrupted_checkpoint")
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