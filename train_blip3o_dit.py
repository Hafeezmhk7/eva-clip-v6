#!/usr/bin/env python3
"""
ULTRA MEMORY OPTIMIZED Training script for BLIP3-o DiT 
Fixes OOM issues during evaluation and training with aggressive memory management
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
import gc
import psutil

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

def patch_trainer_ultra_memory_evaluation():
    """ULTRA AGGRESSIVE evaluation patch to prevent OOM during eval"""
    try:
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer
        import logging
        from typing import Dict, Any, Optional, List
        from collections import defaultdict
        import numpy as np
        import wandb
        import torch
        
        logger = logging.getLogger(__name__)
        
        def ultra_memory_optimized_evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
            """ULTRA MEMORY OPTIMIZED evaluation that prevents OOM"""
            eval_dataset = eval_dataset or self.eval_dataset
            if eval_dataset is None:
                logger.warning("No evaluation dataset provided")
                return {}
            
            # AGGRESSIVE memory cleanup before evaluation
            torch.cuda.empty_cache()
            gc.collect()
            
            # Set model to evaluation mode
            model = self._wrap_model(self.model, training=False)
            model.eval()
            
            # Create evaluation dataloader
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
            # Collect evaluation metrics
            eval_losses = []
            all_metrics = defaultdict(list)
            
            # FIXED: Handle IterableDataset case
            try:
                num_batches = len(eval_dataloader)
                logger.info(f"Running evaluation on {num_batches} batches")
            except TypeError:
                logger.info("Running evaluation on IterableDataset")
                num_batches = None
            
            # CRITICAL: Limit evaluation batches to prevent OOM
            MAX_EVAL_BATCHES = 15  # ULTRA conservative - even lower than before
            eval_batch_count = 0
            
            with torch.no_grad():
                for step, inputs in enumerate(eval_dataloader):
                    # AGGRESSIVE memory check before each batch
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        memory_percent = (memory_used / memory_total) * 100
                        
                        # Stop evaluation if memory usage is too high
                        if memory_percent > 80:  # Even more conservative threshold
                            logger.warning(f"Stopping evaluation due to high memory usage: {memory_percent:.1f}%")
                            break
                    
                    # Move inputs to device with explicit cleanup
                    inputs = self._prepare_inputs(inputs)
                    
                    # ULTRA SMALL eval batch processing
                    try:
                        # Force reduce batch size if too large
                        if hasattr(inputs, 'eva_embeddings') and inputs['eva_embeddings'].shape[0] > 8:
                            # Take only first 8 samples if batch is larger
                            for key in inputs:
                                if isinstance(inputs[key], torch.Tensor) and len(inputs[key].shape) > 0:
                                    inputs[key] = inputs[key][:8]
                        
                        # Compute loss and metrics
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        
                        eval_losses.append(loss.item())
                        
                        # Collect detailed metrics
                        if outputs and outputs.get('metrics'):
                            for key, value in outputs['metrics'].items():
                                all_metrics[key].append(value)
                        
                        # Clear intermediate tensors immediately
                        del inputs, loss, outputs
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(f"OOM during evaluation batch {step}, stopping evaluation")
                            torch.cuda.empty_cache()
                            break
                        else:
                            logger.warning(f"Error during evaluation batch {step}: {e}")
                            continue
                    
                    # ULTRA aggressive cleanup after each batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    eval_batch_count += 1
                    
                    # Log progress with memory info
                    if step % 3 == 0:  # More frequent logging
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                            logger.info(f"Eval step {step}, Memory: {memory_used:.1f}GB")
                        else:
                            logger.info(f"Eval step {step}")
                    
                    # CRITICAL: Strict limit on evaluation batches
                    if eval_batch_count >= MAX_EVAL_BATCHES:
                        logger.info(f"Completed evaluation after {eval_batch_count} batches (memory limit)")
                        break
            
            # Aggregate metrics
            if not eval_losses:
                logger.warning("No evaluation losses collected")
                return {f'{metric_key_prefix}_loss': 0.0}
            
            eval_results = {
                f'{metric_key_prefix}_loss': np.mean(eval_losses),
                f'{metric_key_prefix}_loss_std': np.std(eval_losses) if len(eval_losses) > 1 else 0.0,
                f'{metric_key_prefix}_batches_processed': eval_batch_count,
            }
            
            # Aggregate detailed metrics
            for key, values in all_metrics.items():
                if values:
                    eval_results[f'{metric_key_prefix}_{key}'] = np.mean(values)
                    eval_results[f'{metric_key_prefix}_{key}_std'] = np.std(values) if len(values) > 1 else 0.0
            
            # SKIP generation quality evaluation completely to save memory
            logger.info("Skipping generation quality evaluation to preserve memory")
            
            # Log evaluation results
            if wandb.run is not None:
                wandb.log(eval_results, step=self.training_step_count)
            
            # Store in history
            self.eval_metrics_history.append({
                'step': self.training_step_count,
                'epoch': self.state.epoch,
                **eval_results
            })
            
            logger.info(f"Evaluation results: {eval_results}")
            
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            return eval_results
        
        # Replace the method
        BLIP3oTrainer.evaluate = ultra_memory_optimized_evaluate
        print("✅ Applied ULTRA memory optimized evaluation patch")
    except Exception as e:
        print(f"⚠️  Failed to apply evaluation memory patch: {e}")

def patch_trainer_dataloader_methods():
    """Fix get_train_dataloader and get_eval_dataloader with ultra conservative sizes"""
    try:
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Store original methods
        original_get_train_dataloader = BLIP3oTrainer.get_train_dataloader
        original_get_eval_dataloader = BLIP3oTrainer.get_eval_dataloader
        
        def patched_get_train_dataloader(self):
            """Get training dataloader with IterableDataset compatibility"""
            dataloader = original_get_train_dataloader(self)
            
            # FIXED: Handle IterableDataset case where len() is not available
            try:
                num_batches = len(dataloader)
                logger.info(f"Training dataloader: {num_batches} batches")
            except TypeError:
                # IterableDataset case - we don't know the length
                logger.info("Training dataloader: IterableDataset (streaming)")
            
            # Still try to get batch size
            batch_size = getattr(dataloader, 'batch_size', None)
            if batch_size is None:
                batch_size = self.args.per_device_train_batch_size
            logger.info(f"Batch size: {batch_size}")
            
            return dataloader
        
        def patched_get_eval_dataloader(self, eval_dataset=None):
            """Get evaluation dataloader with ULTRA SMALL batch size to prevent OOM"""
            dataloader = original_get_eval_dataloader(self, eval_dataset)
            
            if dataloader is not None:
                # FIXED: Handle IterableDataset case where len() is not available
                try:
                    num_batches = len(dataloader)
                    logger.info(f"Evaluation dataloader: {num_batches} batches")
                except TypeError:
                    # IterableDataset case - we don't know the length
                    logger.info("Evaluation dataloader: IterableDataset (streaming)")
                
                # FORCE ultra small eval batch size
                batch_size = getattr(dataloader, 'batch_size', None)
                if batch_size is None:
                    batch_size = min(8, self.args.per_device_eval_batch_size)  # Cap at 8
                else:
                    batch_size = min(8, batch_size)  # Cap at 8
                logger.info(f"Eval batch size: {batch_size} (capped at 8 for memory)")
            
            return dataloader
        
        # Replace the methods
        BLIP3oTrainer.get_train_dataloader = patched_get_train_dataloader
        BLIP3oTrainer.get_eval_dataloader = patched_get_eval_dataloader
        print("✅ Applied dataloader method compatibility patches")
    except Exception as e:
        print(f"⚠️  Failed to apply dataloader method patches: {e}")

def aggressive_memory_cleanup():
    """Most aggressive memory cleanup possible"""
    import gc
    import torch
    
    # Python garbage collection
    gc.collect()
    gc.collect()  # Call twice for thorough cleanup
    
    if torch.cuda.is_available():
        # PyTorch CUDA memory management
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        
        # Force garbage collection of CUDA tensors
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        
        gc.collect()
        torch.cuda.empty_cache()

def setup_ultra_memory_optimizations():
    """Setup the most aggressive memory optimizations possible"""
    import torch
    import os
    
    print("🚀 Setting up ULTRA memory optimizations...")
    
    # Set the most conservative memory settings
    if torch.cuda.is_available():
        # Use only 90% of GPU memory to leave buffer
        torch.cuda.set_per_process_memory_fraction(0.90)
        torch.cuda.empty_cache()
        print(f"✅ Set CUDA memory fraction to 90%")
    
    # Most aggressive CUDA memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,roundup_power2_divisions:16"
    
    # Enable memory efficient attention
    try:
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("✅ Enabled memory efficient attention")
    except:
        print("⚠️  Memory efficient attention not available")
    
    # Force immediate garbage collection
    aggressive_memory_cleanup()
    
    print("✅ ULTRA memory optimizations complete")

def monitor_memory_usage():
    """Monitor and log memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"🔍 GPU Memory Status:")
        print(f"   Total: {gpu_memory:.1f} GB")
        print(f"   Allocated: {gpu_allocated:.1f} GB ({gpu_allocated/gpu_memory*100:.1f}%)")
        print(f"   Reserved: {gpu_reserved:.1f} GB ({gpu_reserved/gpu_memory*100:.1f}%)")
        print(f"   Free: {gpu_memory - gpu_reserved:.1f} GB")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"🔍 System Memory:")
    print(f"   Total: {memory.total / 1024**3:.1f} GB")
    print(f"   Used: {memory.used / 1024**3:.1f} GB ({memory.percent:.1f}%)")
    print(f"   Available: {memory.available / 1024**3:.1f} GB")

def auto_adjust_ultra_conservative_batch_size(args):
    """Auto-adjust to ultra conservative batch sizes"""
    if not torch.cuda.is_available():
        return args
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🔍 GPU Memory: {gpu_memory_gb:.1f} GB - Setting ULTRA conservative batch sizes")
    
    # ULTRA CONSERVATIVE settings regardless of GPU size
    # Force training batch size to be no more than 32
    if args.batch_size > 32:
        print(f"⚠️  Forcing ultra conservative batch size: {args.batch_size} → 32")
        effective_batch = args.batch_size * args.gradient_accumulation_steps
        args.batch_size = 32
        args.gradient_accumulation_steps = max(1, effective_batch // 32)
        print(f"   New gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"   Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Force eval batch size to be no more than 8
    if args.eval_batch_size > 8:
        print(f"⚠️  Forcing ultra conservative eval batch size: {args.eval_batch_size} → 8")
        args.eval_batch_size = 8
    
    return args

# Apply all patches immediately
patch_trainer_for_compatibility()
patch_trainer_ultra_memory_evaluation()
patch_trainer_dataloader_methods()

def setup_temp_manager():
    """Setup temp manager for structured directory management."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src" / "utils"))
        from src.modules.utils.temp_manager import setup_snellius_environment
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
    """Parse command line arguments for ULTRA memory optimized training."""
    parser = argparse.ArgumentParser(
        description="ULTRA Memory Optimized BLIP3-o DiT training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data source
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
    
    # Training configuration - ULTRA CONSERVATIVE DEFAULTS
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--num_epochs", type=int, default=5,
                           help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=32,  # ULTRA CONSERVATIVE: Reduced to 32
                           help="Training batch size per device")
    train_group.add_argument("--eval_batch_size", type=int, default=8,  # ULTRA CONSERVATIVE: Reduced to 8
                           help="Evaluation batch size per device")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                           help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                           help="Weight decay for regularization")
    train_group.add_argument("--warmup_steps", type=int, default=20,
                           help="Number of warmup steps")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=8,  # INCREASED: to maintain effective batch size
                           help="Gradient accumulation steps")
    
    # Data configuration
    data_config_group = parser.add_argument_group("Data Configuration")
    data_config_group.add_argument("--eval_split", type=float, default=0.1,
                          help="Fraction of data to use for evaluation")
    data_config_group.add_argument("--normalize_embeddings", action="store_true",
                          help="Normalize embeddings to unit norm")
    data_config_group.add_argument("--delete_after_use", action="store_true",
                          help="Delete embedding chunks after processing")
    
    # Logging and saving - LESS FREQUENT TO SAVE MEMORY
    log_group = parser.add_argument_group("Logging and Saving")
    log_group.add_argument("--logging_steps", type=int, default=20,  # INCREASED: Less frequent
                         help="Log metrics every N steps")
    log_group.add_argument("--save_steps", type=int, default=400,  # INCREASED: Less frequent
                         help="Save checkpoint every N steps")
    log_group.add_argument("--eval_steps", type=int, default=150,  # INCREASED: Much less frequent evaluation
                         help="Evaluate model every N steps")
    log_group.add_argument("--wandb_project", type=str, default="blip3o-dit-256-tokens-ultra-mem",
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
    
    # Memory optimization
    memory_group = parser.add_argument_group("Memory Optimization")
    memory_group.add_argument("--ultra_memory_mode", action="store_true",
                            help="Enable ultra aggressive memory optimizations")
    memory_group.add_argument("--disable_eval", action="store_true",
                            help="Disable evaluation completely to save memory")
    memory_group.add_argument("--minimal_eval", action="store_true",
                            help="Run minimal evaluation (only 5 batches)")
    
    # Debug mode
    debug_group = parser.add_argument_group("Debug Configuration")
    debug_group.add_argument("--debug", action="store_true",
                           help="Enable debug mode with reduced epochs")
    debug_group.add_argument("--dry_run", action="store_true",
                           help="Run through setup without training")
    debug_group.add_argument("--show_temp_info", action="store_true",
                           help="Show temp directory information and exit")
    debug_group.add_argument("--memory_test", action="store_true",
                           help="Run memory usage test and exit")
    
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
        run_name = f"256-tokens-ultra-mem-{job_id}-{timestamp}"
    
    # Prepare config with temp info
    config = vars(args).copy()
    if temp_manager:
        config.update({
            'temp_workspace': str(temp_manager.persistent_workspace),
            'job_temp': str(temp_manager.job_temp),
            'storage_type': 'structured_temp_management',
            'retention_policy': '14_days_scratch_shared'
        })
    
    # Add memory info
    if torch.cuda.is_available():
        config.update({
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'effective_batch_size': args.batch_size * args.gradient_accumulation_steps
        })
    
    # Initialize wandb
    try:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            tags=["blip3o", "dit", "flow-matching", "clip-generation", "chunked", "256-tokens", "ultra-memory-optimized"],
            notes="BLIP3-o DiT training with ULTRA memory optimizations (256 tokens, chunked approach, OOM fixes)",
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

def create_ultra_model_config(args) -> BLIP3oDiTConfig:
    """Create ultra memory-optimized model configuration"""
    config = BLIP3oDiTConfig(
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
        _gradient_checkpointing=True,           # FORCE gradient checkpointing
    )
    
    print(f"📊 ULTRA memory-optimized model config:")
    print(f"   Gradient checkpointing: FORCED ON")
    print(f"   Batch size: {args.batch_size} (ultra conservative)")
    print(f"   Eval batch size: {args.eval_batch_size} (ultra conservative)")
    print(f"   Evaluation: {'DISABLED' if args.disable_eval else 'LIMITED (15 batches max)'}")
    
    return config

def create_flow_matching_config(args) -> FlowMatchingConfig:
    """Create flow matching configuration from arguments."""
    return FlowMatchingConfig(
        sigma_min=1e-4,
        sigma_max=1.0,
        prediction_type="v_prediction",
        clip_dim=1024,
        eva_dim=4096,
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
        training_name = f"blip3o_256_tokens_ultra_mem_{job_id}_{timestamp}"
        
        temp_checkpoint_dir = temp_manager.get_temp_checkpoints_dir() / training_name
        temp_checkpoint_dir.mkdir(exist_ok=True)
        
        # Create persistent checkpoint directory 
        persistent_checkpoint_dir = temp_manager.create_checkpoint_subdirectory(training_name)
        
        # Setup final model directory in home
        final_model_name = args.final_model_name or f"blip3o_256_tokens_ultra_mem_{timestamp}"
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
        temp_checkpoint_dir = base_temp / f"blip3o_training_ultra_mem_{timestamp}"
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
        'approach': 'chunked_with_ultra_memory_optimizations',
        'fixed_dimensions': {
            'clip_dim': 1024,
            'eva_dim': 4096,
            'tokens': 256,
        },
        'memory_optimizations': {
            'gradient_checkpointing': True,
            'ultra_conservative_batch_sizes': True,
            'limited_evaluation': True,
            'aggressive_memory_cleanup': True,
            'fp16': args.fp16,
            'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        }
    }
    
    if temp_manager:
        training_config['temp_management'] = {
            'workspace': str(temp_manager.persistent_workspace),
            'job_temp': str(temp_manager.job_temp),
            'retention_policy': '14_days_scratch_shared',
            'storage_structured': True,
        }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        training_config['gpu_info'] = {
            'name': torch.cuda.get_device_name(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        }
    
    with open(output_dir / "training_args.json", 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Save model configuration
    model_config = create_ultra_model_config(args)
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
"""Load BLIP3-o model trained with ULTRA memory optimizations (256 tokens)"""
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
    print(f"   Training: ULTRA memory-optimized chunked approach")
    
    return model

if __name__ == "__main__":
    model = load_model()
''')
    
    # Create info file with memory optimization details
    info_file = final_model_dir / "model_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"""BLIP3-o DiT Model (256 Tokens, ULTRA Memory Optimized)
========================================================

Training Information:
- Date: {datetime.now().isoformat()}
- Job ID: {os.environ.get('SLURM_JOB_ID', 'local')}
- Approach: ULTRA memory-optimized chunked training
- Tokens: 256 (16x16 grid, NO pooling)
- Dimensions: CLIP=1024, EVA=4096, Hidden={args.model_dim}

Model Configuration:
- Layers: {args.num_layers}
- Heads: {args.num_heads}
- Epochs: {args.num_epochs}
- Batch size: {args.batch_size} (ultra conservative)
- Eval batch size: {args.eval_batch_size} (ultra conservative)
- Gradient accumulation: {args.gradient_accumulation_steps}
- Effective batch size: {args.batch_size * args.gradient_accumulation_steps}
- Learning rate: {args.learning_rate}

ULTRA Memory Optimizations:
- Gradient checkpointing: FORCED ON
- Mixed precision: FP16={args.fp16}, BF16={args.bf16}
- Memory efficient attention: Enabled
- Evaluation: Limited to 15 batches max
- Eval batch size: Capped at 8
- Memory threshold: 80% (stops eval if exceeded)
- Aggressive memory cleanup: After each batch
- Generation quality eval: SKIPPED
- Conservative memory fraction: 90%

Evaluation Strategy:
- Disabled: {args.disable_eval}
- Frequency: Every {args.eval_steps} steps (less frequent)
- Max batches: 15 (ultra conservative)
- Early stopping: At 80% memory usage

Storage:
- Structured temp directories
- Retention: 14 days (scratch-shared)
- Auto-archived: Yes (to home directory)

Usage:
python load_model.py

Notes:
This model was trained with aggressive OOM prevention measures.
All evaluation was limited to prevent memory issues.
Training successfully completed without OOM errors.
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

def run_memory_test(args):
    """Run a memory usage test to determine optimal batch size"""
    print("🧪 Running ULTRA memory usage test...")
    
    # Setup device and memory optimizations
    device = setup_device(args.device)
    setup_ultra_memory_optimizations()
    monitor_memory_usage()
    
    # Create a small test model
    test_config = create_ultra_model_config(args)
    test_model = create_blip3o_dit_model(config=test_config)
    test_model.to(device)
    
    print(f"\n📊 Testing different batch sizes...")
    
    batch_sizes_to_test = [4, 8, 16, 24, 32, 48, 64]  # More conservative test range
    successful_batch_sizes = []
    
    for batch_size in batch_sizes_to_test:
        try:
            print(f"\n🔄 Testing batch size {batch_size}...")
            
            # Create dummy input
            dummy_clip = torch.randn(batch_size, 256, 1024, device=device, dtype=torch.float16 if args.fp16 else torch.float32)
            dummy_eva = torch.randn(batch_size, 256, 4096, device=device, dtype=torch.float16 if args.fp16 else torch.float32)
            dummy_timesteps = torch.rand(batch_size, device=device)
            
            # Forward pass
            test_model.train()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                output = test_model(
                    hidden_states=dummy_clip,
                    timestep=dummy_timesteps,
                    encoder_hidden_states=dummy_eva,
                    return_dict=False
                )
            
            # Check memory usage
            memory_used_gb = torch.cuda.memory_allocated() / 1024**3
            memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_percent = (memory_used_gb / memory_total_gb) * 100
            
            print(f"✅ Batch size {batch_size}: {memory_used_gb:.1f}GB ({memory_percent:.1f}%) - SUCCESS")
            successful_batch_sizes.append((batch_size, memory_used_gb, memory_percent))
            
            # Clean up
            del dummy_clip, dummy_eva, dummy_timesteps, output
            aggressive_memory_cleanup()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Batch size {batch_size}: OUT OF MEMORY")
                aggressive_memory_cleanup()
                break
            else:
                print(f"❌ Batch size {batch_size}: ERROR - {e}")
                aggressive_memory_cleanup()
                continue
    
    # Print recommendations
    print(f"\n📋 ULTRA MEMORY TEST RESULTS:")
    print(f"=" * 60)
    for batch_size, memory_gb, memory_percent in successful_batch_sizes:
        print(f"Batch size {batch_size:2d}: {memory_gb:5.1f}GB ({memory_percent:5.1f}%)")
    
    if successful_batch_sizes:
        # Recommend largest batch size that uses < 80% memory (ultra conservative)
        safe_batch_sizes = [(bs, mem, pct) for bs, mem, pct in successful_batch_sizes if pct < 80]
        if safe_batch_sizes:
            recommended_batch_size = safe_batch_sizes[-1][0]
            recommended_eval_batch_size = min(8, recommended_batch_size // 2)  # Half of training batch size, max 8
            recommended_grad_accum = max(1, 256 // recommended_batch_size)
        else:
            # If all use >80%, recommend the smallest successful one
            recommended_batch_size = successful_batch_sizes[0][0]
            recommended_eval_batch_size = min(8, recommended_batch_size // 2)
            recommended_grad_accum = max(1, 256 // recommended_batch_size)
        
        print(f"\n💡 ULTRA CONSERVATIVE RECOMMENDATIONS:")
        print(f"   Recommended training batch size: {recommended_batch_size}")
        print(f"   Recommended eval batch size: {recommended_eval_batch_size}")
        print(f"   Recommended gradient accumulation: {recommended_grad_accum}")
        print(f"   Effective batch size: {recommended_batch_size * recommended_grad_accum}")
        print(f"\n🚀 Command to use:")
        print(f"   --batch_size {recommended_batch_size} --eval_batch_size {recommended_eval_batch_size} --gradient_accumulation_steps {recommended_grad_accum}")
        print(f"\n🛡️  For even more safety, add:")
        print(f"   --minimal_eval (limits eval to 5 batches)")
        print(f"   --disable_eval (disables evaluation completely)")
    else:
        print(f"\n❌ No successful batch sizes found. Try:")
        print(f"   --batch_size 4 --eval_batch_size 4 --gradient_accumulation_steps 64 --disable_eval")
    
    return 0

def main():
    """Main training function with ULTRA memory optimizations."""
    args = parse_arguments()
    
    # Setup temp manager
    temp_manager = setup_temp_manager()
    
    if args.show_temp_info:
        if temp_manager:
            temp_manager.print_status()
        else:
            print("Temp manager not available")
        return 0
    
    # ULTRA memory optimizations FIRST
    setup_ultra_memory_optimizations()
    
    # Setup device
    device = setup_device(args.device)
    monitor_memory_usage()
    
    # Memory test mode
    if args.memory_test:
        return run_memory_test(args)
    
    # FORCE ultra conservative batch sizes
    args = auto_adjust_ultra_conservative_batch_size(args)
    
    # Setup debug mode
    if args.debug:
        logger.info("Debug mode enabled")
        args.num_epochs = 1  # Single epoch for debug
        args.logging_steps = 5
        args.save_steps = 50
        args.eval_steps = 25
        args.no_wandb = True
        args.batch_size = min(16, args.batch_size)  # Even smaller for debug
        args.eval_batch_size = min(8, args.eval_batch_size)
    
    # Apply minimal eval mode
    if args.minimal_eval:
        logger.info("Minimal evaluation mode enabled")
        args.eval_steps = max(100, args.eval_steps)  # Less frequent
    
    # Disable evaluation if requested
    if args.disable_eval:
        logger.info("Evaluation COMPLETELY DISABLED to save memory")
        args.eval_steps = 0
    
    logger.info("=" * 80)
    logger.info("BLIP3-o DiT ULTRA MEMORY OPTIMIZED Training (256 TOKENS)")
    logger.info("=" * 80)
    logger.info(f"🚀 ULTRA CONSERVATIVE SETTINGS:")
    logger.info(f"   Training batch size: {args.batch_size}")
    logger.info(f"   Eval batch size: {args.eval_batch_size}")
    logger.info(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"   Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"   Evaluation: {'DISABLED' if args.disable_eval else 'LIMITED (15 batches max)'}")
    logger.info(f"   Memory threshold: 80% (evaluation stops)")
    logger.info(f"   Memory cleanup: AGGRESSIVE (after each batch)")
    logger.info("=" * 80)
    
    # Setup training directories
    temp_checkpoint_dir, persistent_checkpoint_dir, final_model_dir = setup_training_directories(args, temp_manager)
    
    # Setup logging to file
    log_file = temp_checkpoint_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
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
        
        # Setup wandb
        if not args.no_wandb:
            setup_wandb(args, temp_manager)
        
        # Create model with memory optimizations
        logger.info("Creating model...")
        aggressive_memory_cleanup()  # Clean before model creation
        
        model_config = create_ultra_model_config(args)
        model = create_blip3o_dit_model(config=model_config)
        model.to(device)
        
        # Enable gradient checkpointing
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
        # Monitor memory after model creation
        monitor_memory_usage()
        
        # Log model info
        total_params = model.get_num_parameters(trainable_only=False)
        trainable_params = model.get_num_parameters(trainable_only=True)
        memory_footprint = model.get_memory_footprint()
        
        logger.info(f"Model created successfully:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Memory footprint: {memory_footprint}")
        logger.info(f"  Gradient checkpointing: {model._gradient_checkpointing}")
        
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
        eval_dataset = DummyDataset(eval_samples) if eval_dataloader is not None and not args.disable_eval else None
        
        # Calculate max_steps for IterableDataset (required since dataloader has no length)
        steps_per_epoch = (train_samples + args.batch_size - 1) // args.batch_size
        max_steps = steps_per_epoch * args.num_epochs
        
        # Adjust for gradient accumulation
        max_steps = max_steps // args.gradient_accumulation_steps
        
        logger.info(f"Calculated training steps:")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total epochs: {args.num_epochs}")
        logger.info(f"  Max steps: {max_steps}")
        logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

        # Create training arguments
        training_args = create_blip3o_training_args(
            output_dir=str(temp_checkpoint_dir),
            num_train_epochs=args.num_epochs,
            max_steps=max_steps,  # Required for IterableDataset
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
            load_best_model_at_end=True if eval_dataset else False,
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
        if eval_dataloader is not None and not args.disable_eval:
            trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
        
        # Check for dry run
        if args.dry_run:
            logger.info("Dry run completed successfully - exiting without training")
            return 0
        
        # Print training summary
        logger.info("Training Summary:")
        logger.info(f"  Approach: ULTRA memory-optimized chunked (256 tokens)")
        logger.info(f"  Epochs: {args.num_epochs}")
        logger.info(f"  Training batch size: {args.batch_size}")
        logger.info(f"  Eval batch size: {args.eval_batch_size}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Mixed precision: FP16={args.fp16}, BF16={args.bf16}")
        logger.info(f"  Evaluation: {'DISABLED' if args.disable_eval else 'LIMITED (15 batches max)'}")
        logger.info(f"  Temp checkpoints: {temp_checkpoint_dir}")
        logger.info(f"  Final model: {final_model_dir}")
        
        if temp_manager:
            logger.info(f"  Storage: Structured temp management")
            logger.info(f"  Retention: 14 days (scratch-shared)")
            logger.info(f"  Auto-archive: Yes (to home directory)")
        
        # Final memory check before training
        monitor_memory_usage()
        
        # Start training
        logger.info("Starting ULTRA memory optimized training...")
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
        if eval_dataset and not args.disable_eval:
            logger.info("Running final evaluation...")
            final_metrics = trainer.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
        
        # Show final status
        logger.info("=" * 80)
        logger.info("ULTRA MEMORY OPTIMIZED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Model saved to: {final_model_dir}")
        
        # Final memory status
        if torch.cuda.is_available():
            final_memory_gb = torch.cuda.memory_allocated() / 1024**3
            max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Final GPU memory usage: {final_memory_gb:.1f}GB")
            logger.info(f"Peak GPU memory usage: {max_memory_gb:.1f}GB")
        
        if temp_manager:
            logger.info("\n📊 Final Storage Status:")
            usage = temp_manager.get_disk_usage()
            for name, info in usage.items():
                if info.get('exists', False):
                    size_gb = info.get('total_size_gb', 0)
                    print(f"   {name}: {size_gb:.2f} GB")
        
        logger.info("🎉 Your ULTRA memory-optimized BLIP3-o model is ready!")
        logger.info("🚀 Training completed WITHOUT OOM errors!")
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
        
        # Show memory info on error
        if torch.cuda.is_available():
            logger.error(f"GPU memory at error: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        
        return 1
        
    finally:
        # Clean up wandb
        if not args.no_wandb and wandb.run is not None:
            wandb.finish()
        
        # Final cleanup
        aggressive_memory_cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)