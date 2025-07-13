#!/usr/bin/env python3
"""
UPDATED Main training script for BLIP3-o DiT with flow matching and Temp Manager.
Trains a diffusion transformer to generate CLIP embeddings from EVA-CLIP conditioning.

ENHANCED FEATURES:
- Structured temp directory management with SnelliusTempManager
- Persistent embeddings storage with 14-day retention
- Job-specific temp for checkpoints and cache
- Automatic model archival to home directory
- Smart disk usage monitoring
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

def setup_temp_manager():
    """Setup temp manager for structured directory management."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src" / "utils"))
        from temp_manager import setup_snellius_environment
        manager = setup_snellius_environment("blip3o_workspace")
        return manager
    except ImportError:
        print("⚠️  Temp manager not available, using fallback directories")
        return None

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
    """Parse command line arguments for BLIP3-o training with temp manager."""
    parser = argparse.ArgumentParser(
        description="Train BLIP3-o DiT with flow matching using structured temp management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data source (either chunked_embeddings_dir OR let temp manager find it)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--chunked_embeddings_dir", type=str, 
        help="Path to directory containing chunked embedding files"
    )
    data_group.add_argument(
        "--auto_find_embeddings", action="store_true",
        help="Automatically find embeddings using temp manager"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/blip3o-dit-temp",
        help="Base output directory name (will be created in temp)"
    )
    
    parser.add_argument(
        "--final_model_name", type=str, default=None,
        help="Name for final model in home directory (auto-generated if None)"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_dim", type=int, default=512,
                           help="Model hidden dimension")
    model_group.add_argument("--num_layers", type=int, default=24,
                           help="Number of transformer layers")
    model_group.add_argument("--num_heads", type=int, default=8,
                           help="Number of attention heads")
    model_group.add_argument("--num_kv_heads", type=int, default=None,
                           help="Number of KV heads")
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
    
    # Data configuration
    data_config_group = parser.add_argument_group("Data Configuration")
    data_config_group.add_argument("--eval_split", type=float, default=0.1,
                          help="Fraction of data to use for evaluation")
    data_config_group.add_argument("--normalize_embeddings", action="store_true",
                          help="Normalize embeddings to unit norm")
    data_config_group.add_argument("--delete_after_use", action="store_true",
                          help="Delete embedding chunks after processing")
    
    # Logging and saving
    log_group = parser.add_argument_group("Logging and Saving")
    log_group.add_argument("--logging_steps", type=int, default=10,
                         help="Log metrics every N steps")
    log_group.add_argument("--save_steps", type=int, default=200,
                         help="Save checkpoint every N steps")
    log_group.add_argument("--eval_steps", type=int, default=50,
                         help="Evaluate model every N steps")
    log_group.add_argument("--wandb_project", type=str, default="blip3o-dit-256-tokens-temp",
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
    
    # Debug mode
    debug_group = parser.add_argument_group("Debug Configuration")
    debug_group.add_argument("--debug", action="store_true",
                           help="Enable debug mode with reduced epochs")
    debug_group.add_argument("--dry_run", action="store_true",
                           help="Run through setup without training")
    debug_group.add_argument("--show_temp_info", action="store_true",
                           help="Show temp directory information and exit")
    
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


def setup_wandb(args, temp_manager):
    """Initialize Weights & Biases logging."""
    if args.no_wandb:
        logger.info("Weights & Biases logging disabled")
        return
    
    # Create run name if not provided
    run_name = args.wandb_run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"256-tokens-temp-{job_id}-{timestamp}"
    
    # Prepare config with temp info
    config = vars(args).copy()
    if temp_manager:
        config.update({
            'temp_workspace': str(temp_manager.persistent_workspace),
            'job_temp': str(temp_manager.job_temp),
            'storage_type': 'structured_temp_management',
            'retention_policy': '14_days_scratch_shared'
        })
    
    # Initialize wandb
    try:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            tags=["blip3o", "dit", "flow-matching", "clip-generation", "chunked", "256-tokens", "temp-managed"],
            notes="BLIP3-o DiT training with structured temp management (256 tokens, chunked approach)",
        )
        
        logger.info(f"Initialized Weights & Biases: {wandb.run.url}")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        args.no_wandb = True


def find_embeddings_directory(args, temp_manager):
    """Find embeddings directory using temp manager or specified path."""
    
    if args.chunked_embeddings_dir:
        # Use specified directory
        embeddings_path = Path(args.chunked_embeddings_dir)
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Specified embeddings directory not found: {embeddings_path}")
        
        logger.info(f"Using specified embeddings directory: {embeddings_path}")
        return embeddings_path
    
    elif args.auto_find_embeddings and temp_manager:
        # Auto-find using temp manager
        embeddings_dir = temp_manager.get_embeddings_dir()
        
        # Look for any subdirectory with embeddings
        candidates = []
        for subdir in embeddings_dir.iterdir():
            if subdir.is_dir():
                manifest_file = subdir / "embeddings_manifest.json"
                if manifest_file.exists():
                    candidates.append(subdir)
        
        if not candidates:
            raise FileNotFoundError(
                f"No embeddings found in temp workspace: {embeddings_dir}\n"
                "Please run embedding extraction first:\n"
                "  python src/modules/extract_embeddings_g.py"
            )
        
        # Use the most recent one
        embeddings_path = max(candidates, key=lambda p: p.stat().st_mtime)
        logger.info(f"Auto-found embeddings directory: {embeddings_path}")
        return embeddings_path
    
    else:
        # Try to find in common locations
        search_locations = []
        
        if temp_manager:
            search_locations.append(temp_manager.get_embeddings_dir())
        
        # Add environment-based locations
        if "BLIP3O_EMBEDDINGS" in os.environ:
            search_locations.append(Path(os.environ["BLIP3O_EMBEDDINGS"]))
        
        if "TMPDIR" in os.environ:
            search_locations.append(Path(os.environ["TMPDIR"]) / "chunked_embeddings")
        
        # Search each location
        for location in search_locations:
            if location.exists():
                # Look for manifest files
                manifest_files = list(location.glob("**/embeddings_manifest.json"))
                if manifest_files:
                    embeddings_path = manifest_files[0].parent
                    logger.info(f"Found embeddings directory: {embeddings_path}")
                    return embeddings_path
        
        raise FileNotFoundError(
            "No embeddings directory found!\n"
            "Please either:\n"
            "1. Specify --chunked_embeddings_dir /path/to/embeddings\n"
            "2. Use --auto_find_embeddings with temp manager\n"
            "3. Run embedding extraction first: python src/modules/extract_embeddings_g.py"
        )


def create_model_config(args) -> BLIP3oDiTConfig:
    """Create model configuration from arguments."""
    return BLIP3oDiTConfig(
        input_size=16,                          # 16x16 = 256 tokens
        patch_size=1,                           # Pre-tokenized
        in_channels=1024,                       # CLIP dimension
        dim=args.model_dim,                     # Hidden dimension
        eva_embedding_size=4096,                # EVA-CLIP dimension
        n_layers=args.num_layers,               # Number of layers
        n_heads=args.num_heads,                 # Attention heads
        n_kv_heads=args.num_kv_heads or args.num_heads,  # KV heads
        norm_eps=1e-5,                          # Layer norm epsilon
        qk_norm=True,                           # Query-key normalization
        learn_sigma=False,                      # Flow matching
        _gradient_checkpointing=args.gradient_checkpointing,
    )


def create_flow_matching_config(args) -> FlowMatchingConfig:
    """Create flow matching configuration from arguments."""
    return FlowMatchingConfig(
        sigma_min=1e-4,
        sigma_max=1.0,
        prediction_type="v_prediction",
        clip_dim=1024,                          # CLIP dimension
        eva_dim=4096,                           # EVA-CLIP dimension
        regularization_weight=0.0,
        schedule_type="linear",
    )


def setup_training_directories(args, temp_manager):
    """Setup training directories using temp manager."""
    
    if temp_manager:
        # Use temp manager for structured storage
        
        # Create temp checkpoint directory for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        training_name = f"blip3o_256_tokens_{job_id}_{timestamp}"
        
        temp_checkpoint_dir = temp_manager.get_temp_checkpoints_dir() / training_name
        temp_checkpoint_dir.mkdir(exist_ok=True)
        
        # Create persistent checkpoint directory 
        persistent_checkpoint_dir = temp_manager.create_checkpoint_subdirectory(training_name)
        
        # Setup final model directory in home
        final_model_name = args.final_model_name or f"blip3o_256_tokens_{timestamp}"
        final_model_dir = Path.home() / "models" / final_model_name
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training directories (temp managed):")
        logger.info(f"  Temp checkpoints: {temp_checkpoint_dir}")
        logger.info(f"  Persistent checkpoints: {persistent_checkpoint_dir}")
        logger.info(f"  Final model (home): {final_model_dir}")
        
        return temp_checkpoint_dir, persistent_checkpoint_dir, final_model_dir
    
    else:
        # Fallback to basic temp directories
        if "TMPDIR" in os.environ:
            base_temp = Path(os.environ["TMPDIR"])
        else:
            base_temp = Path("./temp")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_checkpoint_dir = base_temp / f"blip3o_training_{timestamp}"
        temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # No persistent directory in fallback mode
        persistent_checkpoint_dir = None
        
        # Final model in current directory
        final_model_dir = Path(args.output_dir)
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Training directories (fallback):")
        logger.info(f"  Temp checkpoints: {temp_checkpoint_dir}")
        logger.info(f"  Final model: {final_model_dir}")
        
        return temp_checkpoint_dir, persistent_checkpoint_dir, final_model_dir


def save_configs(args, output_dir: Path, temp_manager=None):
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
        'approach': 'chunked_with_temp_manager',
        'fixed_dimensions': {
            'clip_dim': 1024,
            'eva_dim': 4096,
            'tokens': 256,
        }
    }
    
    if temp_manager:
        training_config['temp_management'] = {
            'workspace': str(temp_manager.persistent_workspace),
            'job_temp': str(temp_manager.job_temp),
            'retention_policy': '14_days_scratch_shared',
            'storage_structured': True,
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
    
    return manifest


class DummyDataset:
    """Dummy dataset for trainer compatibility."""
    def __init__(self, length):
        self.length = length
    
    def __len__(self):
        return self.length


def create_final_model_package(model, temp_checkpoint_dir, final_model_dir, args, temp_manager=None):
    """Create final model package in home directory."""
    
    logger.info(f"Creating final model package in: {final_model_dir}")
    
    # Copy model files
    for file_pattern in ["*.bin", "*.safetensors", "*.json", "*.txt"]:
        for file_path in temp_checkpoint_dir.glob(file_pattern):
            target_path = final_model_dir / file_path.name
            if file_path.is_file():
                import shutil
                shutil.copy2(file_path, target_path)
    
    # Create model loading script
    loading_script = final_model_dir / "load_model.py"
    with open(loading_script, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""Load BLIP3-o model trained with temp manager (256 tokens)"""
import sys
import torch
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent.parent  # Adjust as needed
sys.path.insert(0, str(project_root / "src"))

from src.modules.models.blip3o_dit import BLIP3oDiTModel
from src.modules.config.blip3o_config import BLIP3oDiTConfig
import json

def load_model():
    model_dir = Path(__file__).parent
    
    # Load config
    config_file = model_dir / "model_config.json"
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    config = BLIP3oDiTConfig(**config_dict)
    
    # Create model
    model = BLIP3oDiTModel(config)
    
    # Load weights
    model_file = model_dir / "pytorch_model.bin"
    if model_file.exists():
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded from {{model_file}}")
    else:
        print("⚠️  No weights found, using random initialization")
    
    print(f"✅ BLIP3-o model loaded successfully!")
    print(f"   Parameters: {{model.get_num_parameters():,}}")
    print(f"   Tokens: 256 (16x16 grid)")
    print(f"   Training: Chunked approach with temp manager")
    
    return model

if __name__ == "__main__":
    model = load_model()
''')
    
    # Create info file
    info_file = final_model_dir / "model_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"""BLIP3-o DiT Model (256 Tokens, Temp Managed)
============================================

Training Information:
- Date: {datetime.now().isoformat()}
- Job ID: {os.environ.get('SLURM_JOB_ID', 'local')}
- Approach: Chunked training with structured temp management
- Tokens: 256 (16x16 grid, NO pooling)
- Dimensions: CLIP=1024, EVA=4096, Hidden={args.model_dim}

Model Configuration:
- Layers: {args.num_layers}
- Heads: {args.num_heads}
- Epochs: {args.num_epochs}
- Batch size: {args.batch_size}
- Learning rate: {args.learning_rate}

Temp Management:
- Storage: Structured temp directories
- Retention: 14 days (scratch-shared)
- Auto-archived: Yes (to home directory)

Usage:
python load_model.py
""")
    
    # Make loading script executable
    loading_script.chmod(0o755)
    
    # Calculate final size
    total_size = sum(f.stat().st_size for f in final_model_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    logger.info(f"✅ Final model package created:")
    logger.info(f"   Location: {final_model_dir}")
    logger.info(f"   Size: {size_mb:.1f} MB")
    logger.info(f"   Files: {len(list(final_model_dir.iterdir()))}")


def main():
    """Main training function with temp manager."""
    args = parse_arguments()
    
    # Setup temp manager
    temp_manager = setup_temp_manager()
    
    if args.show_temp_info:
        if temp_manager:
            temp_manager.print_status()
        else:
            print("Temp manager not available")
        return 0
    
    # Setup debug mode
    if args.debug:
        logger.info("Debug mode enabled")
        args.num_epochs = 2
        args.logging_steps = 5
        args.save_steps = 20
        args.eval_steps = 10
        args.no_wandb = True
    
    # Setup training directories
    temp_checkpoint_dir, persistent_checkpoint_dir, final_model_dir = setup_training_directories(args, temp_manager)
    
    # Setup logging to file
    log_file = temp_checkpoint_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("BLIP3-o DiT Training with Temp Manager (256 TOKENS, CHUNKED)")
    logger.info("=" * 80)
    
    try:
        # Find embeddings directory
        embeddings_path = find_embeddings_directory(args, temp_manager)
        
        # Load and validate manifest
        manifest = load_manifest(embeddings_path)
        logger.info(f"Using embeddings: {embeddings_path}")
        logger.info(f"  Total shards: {manifest['total_shards']}")
        logger.info(f"  Total samples: {manifest['total_samples']:,}")
        logger.info(f"  Format: {manifest['format_version']}")
        
        # Save configurations
        save_configs(args, temp_checkpoint_dir, temp_manager)
        
        # Setup device
        device = setup_device(args.device)
        
        # Setup wandb
        if not args.no_wandb:
            setup_wandb(args, temp_manager)
        
        # Create model
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
            num_workers=0,
        )
        
        # Calculate sample counts from manifest
        total_samples = manifest['total_samples']
        train_samples = int(total_samples * (1 - args.eval_split))
        eval_samples = total_samples - train_samples
        
        logger.info(f"Chunked datasets created:")
        logger.info(f"  Total samples: {total_samples:,}")
        logger.info(f"  Training samples: {train_samples:,}")
        logger.info(f"  Evaluation samples: {eval_samples:,}")
        
        # Create dummy datasets for trainer compatibility
        train_dataset = DummyDataset(train_samples)
        eval_dataset = DummyDataset(eval_samples) if eval_dataloader else None
        
        # Create training arguments
        training_args = create_blip3o_training_args(
            output_dir=str(temp_checkpoint_dir),
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
            dataloader_num_workers=0,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = BLIP3oTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Override dataloader methods
        trainer.get_train_dataloader = lambda: train_dataloader
        trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
        
        # Check for dry run
        if args.dry_run:
            logger.info("Dry run completed successfully - exiting without training")
            return 0
        
        # Print training summary
        logger.info("Training Summary:")
        logger.info(f"  Approach: Chunked (256 tokens) with temp manager")
        logger.info(f"  Epochs: {args.num_epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Temp checkpoints: {temp_checkpoint_dir}")
        logger.info(f"  Final model: {final_model_dir}")
        
        if temp_manager:
            logger.info(f"  Storage: Structured temp management")
            logger.info(f"  Retention: 14 days (scratch-shared)")
            logger.info(f"  Auto-archive: Yes (to home directory)")
        
        # Start training
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        # Train the model
        trainer.train()
        
        # Save final model to temp
        logger.info("Training completed! Saving final model...")
        trainer.save_model()
        
        # Create final model package
        create_final_model_package(model, temp_checkpoint_dir, final_model_dir, args, temp_manager)
        
        # Copy to persistent checkpoint if available
        if persistent_checkpoint_dir and temp_manager:
            import shutil
            shutil.copytree(temp_checkpoint_dir, persistent_checkpoint_dir, dirs_exist_ok=True)
            logger.info(f"Model also saved to persistent storage: {persistent_checkpoint_dir}")
        
        # Run final evaluation
        if eval_dataset:
            logger.info("Running final evaluation...")
            final_metrics = trainer.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # Show final status
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {final_model_dir}")
        
        if temp_manager:
            logger.info("\n📊 Final Storage Status:")
            usage = temp_manager.get_disk_usage()
            for name, info in usage.items():
                if info.get('exists', False):
                    size_gb = info.get('total_size_gb', 0)
                    print(f"   {name}: {size_gb:.2f} GB")
        
        logger.info("🎉 Your BLIP3-o model with temp management is ready!")
        logger.info("🚀 Structured temp approach optimized storage usage!")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'trainer' in locals():
            logger.info("Saving checkpoint...")
            trainer.save_model(temp_checkpoint_dir / "interrupted_checkpoint")
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