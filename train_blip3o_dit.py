#!/usr/bin/env python3
"""
FIXED Training script for BLIP3-o DiT with shard management and memory issues resolved
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

def patch_chunked_dataset_shard_management():
    """CRITICAL FIX: Patch the chunked dataset to prevent premature shard deletion"""
    try:
        from src.modules.datasets.blip3o_dataset import BLIP3oEmbeddingDataset
        
        # Store original method
        original_load_next_shard = BLIP3oEmbeddingDataset._load_next_shard
        
        def fixed_load_next_shard(self):
            """FIXED: Don't delete shards until we're sure we won't need them again"""
            # Clean up current shard data but DON'T delete the file yet
            if self.current_shard_data is not None:
                # Only clear memory, don't delete files
                del self.current_shard_data
                gc.collect()
            
            # Check if we have more shards
            if self.current_shard_idx >= len(self.shard_files):
                logger.info("No more shards to process")
                self.current_shard_data = None
                return False
            
            # Load current shard
            shard_path = self.shard_files[self.current_shard_idx]
            
            # Check if shard file exists
            if not shard_path.exists():
                logger.error(f"Shard file does not exist: {shard_path}")
                self.current_shard_idx += 1
                return self._load_next_shard()  # Try next shard
            
            try:
                self.current_shard_data = self._load_shard(shard_path)
                logger.info(f"Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: {shard_path}")
            except Exception as e:
                logger.error(f"Failed to load shard {shard_path}: {e}")
                self.current_shard_idx += 1
                return self._load_next_shard()  # Try next shard
            
            # Prepare samples
            self._prepare_current_shard_samples()
            
            logger.info(f"Loaded shard {self.current_shard_idx + 1}/{len(self.shard_files)}: "
                       f"{len(self.current_shard_samples)} samples")
            
            self.current_shard_idx += 1
            self.shards_processed += 1
            
            return True
        
        # Replace the method
        BLIP3oEmbeddingDataset._load_next_shard = fixed_load_next_shard
        print("✅ Applied CRITICAL shard management fix")
    except Exception as e:
        print(f"⚠️  Failed to apply shard management fix: {e}")

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
            
            # CRITICAL: Limit evaluation batches to prevent OOM
            MAX_EVAL_BATCHES = 10  # Even more conservative
            eval_batch_count = 0
            
            logger.info(f"Running LIMITED evaluation (max {MAX_EVAL_BATCHES} batches)")
            
            with torch.no_grad():
                for step, inputs in enumerate(eval_dataloader):
                    # AGGRESSIVE memory check before each batch
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        memory_percent = (memory_used / memory_total) * 100
                        
                        # Stop evaluation if memory usage is too high
                        if memory_percent > 75:  # Even more conservative threshold
                            logger.warning(f"Stopping evaluation due to high memory usage: {memory_percent:.1f}%")
                            break
                    
                    # Move inputs to device with explicit cleanup
                    inputs = self._prepare_inputs(inputs)
                    
                    try:
                        # Force tiny eval batch size
                        if isinstance(inputs, dict):
                            for key in inputs:
                                if isinstance(inputs[key], torch.Tensor) and len(inputs[key].shape) > 0:
                                    inputs[key] = inputs[key][:4]  # Only 4 samples max
                        
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

def create_safer_chunked_dataloaders(chunked_dir, batch_size, eval_batch_size, eval_split_ratio, normalize_embeddings):
    """Create dataloaders with SAFER settings to prevent shard deletion issues"""
    from src.modules.datasets.blip3o_dataset import create_chunked_dataloader
    
    # Create training dataloader with NO deletion
    train_dataloader = create_chunked_dataloader(
        chunked_embeddings_dir=chunked_dir,
        batch_size=batch_size,
        split="train",
        eval_split_ratio=eval_split_ratio,
        normalize_embeddings=normalize_embeddings,
        shuffle_shards=True,
        shuffle_within_shard=True,
        delete_after_use=False,  # CRITICAL: Don't delete shards during training
        num_workers=0,
        pin_memory=False,  # Disable to save memory
    )
    
    # Create evaluation dataloader with NO deletion
    eval_dataloader = create_chunked_dataloader(
        chunked_embeddings_dir=chunked_dir,
        batch_size=eval_batch_size,
        split="eval",
        eval_split_ratio=eval_split_ratio,
        normalize_embeddings=normalize_embeddings,
        shuffle_shards=False,
        shuffle_within_shard=False,
        delete_after_use=False,  # CRITICAL: Don't delete shards during eval
        num_workers=0,
        pin_memory=False,  # Disable to save memory
    )
    
    return train_dataloader, eval_dataloader

def aggressive_memory_cleanup():
    """Most aggressive memory cleanup possible"""
    import gc
    import torch
    
    # Python garbage collection
    for _ in range(3):  # Multiple passes
        gc.collect()
    
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

def setup_extreme_memory_optimizations():
    """Setup the most extreme memory optimizations possible"""
    import torch
    import os
    
    print("🚀 Setting up EXTREME memory optimizations...")
    
    # Set the most conservative memory settings
    if torch.cuda.is_available():
        # Use only 85% of GPU memory to leave larger buffer
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.cuda.empty_cache()
        print(f"✅ Set CUDA memory fraction to 85%")
    
    # Most aggressive CUDA memory allocation settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,roundup_power2_divisions:16"
    
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
    
    print("✅ EXTREME memory optimizations complete")

# Apply all patches immediately
patch_trainer_for_compatibility()
patch_chunked_dataset_shard_management()  # CRITICAL FIX
patch_trainer_ultra_memory_evaluation()

def parse_arguments():
    """Parse command line arguments for FIXED training."""
    parser = argparse.ArgumentParser(
        description="FIXED BLIP3-o DiT training",
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
        help="Base output directory name"
    )
    
    # Model configuration - EXTREMELY CONSERVATIVE DEFAULTS
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_dim", type=int, default=512,
                           help="Model hidden dimension")
    model_group.add_argument("--num_layers", type=int, default=12,  # REDUCED
                           help="Number of transformer layers")
    model_group.add_argument("--num_heads", type=int, default=8,
                           help="Number of attention heads")
    
    # Training configuration - EXTREMELY CONSERVATIVE DEFAULTS
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--num_epochs", type=int, default=3,  # REDUCED
                           help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=16,  # REDUCED
                           help="Training batch size per device")
    train_group.add_argument("--eval_batch_size", type=int, default=4,  # REDUCED
                           help="Evaluation batch size per device")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                           help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                           help="Weight decay")
    train_group.add_argument("--warmup_steps", type=int, default=20,
                           help="Number of warmup steps")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=16,  # INCREASED
                           help="Gradient accumulation steps")
    
    # Memory optimization
    memory_group = parser.add_argument_group("Memory Optimization")
    memory_group.add_argument("--disable_eval", action="store_true",
                            help="Disable evaluation completely to save memory")
    memory_group.add_argument("--minimal_eval", action="store_true",
                            help="Run minimal evaluation (only 5 batches)")
    
    # Hardware configuration
    hw_group = parser.add_argument_group("Hardware Configuration")
    hw_group.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training (fp16)")
    
    return parser.parse_args()

def main():
    """Main training function with CRITICAL fixes."""
    args = parse_arguments()
    
    # EXTREME memory optimizations FIRST
    setup_extreme_memory_optimizations()
    
    print("🚀 Starting FIXED BLIP3-o DiT Training (256 TOKENS)")
    print("=" * 80)
    print("CRITICAL FIXES APPLIED:")
    print("  ✅ Shard deletion disabled (prevents file not found errors)")
    print("  ✅ Memory optimizations increased")
    print("  ✅ Evaluation batches limited to 10 max")
    print("  ✅ Batch sizes reduced")
    print("  ✅ Model layers reduced")
    print("=" * 80)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Setup temp manager
    try:
        from src.modules.utils.temp_manager import setup_snellius_environment
        temp_manager = setup_snellius_environment("blip3o_workspace")
        print("✅ Temp manager initialized")
    except ImportError:
        print("⚠️  Temp manager not available, using fallback")
        temp_manager = None
    
    # Find embeddings directory
    if args.auto_find_embeddings and temp_manager:
        embeddings_dir = temp_manager.get_embeddings_dir()
        candidates = [d for d in embeddings_dir.iterdir() if d.is_dir() and (d / "embeddings_manifest.json").exists()]
        if not candidates:
            raise FileNotFoundError(f"No embeddings found in {embeddings_dir}")
        embeddings_path = candidates[0]
        print(f"Auto-found embeddings: {embeddings_path}")
    elif args.chunked_embeddings_dir:
        embeddings_path = Path(args.chunked_embeddings_dir)
        print(f"Using specified embeddings: {embeddings_path}")
    else:
        raise ValueError("Must specify --chunked_embeddings_dir or --auto_find_embeddings")
    
    # Load manifest
    manifest_path = embeddings_path / "embeddings_manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"Embeddings info:")
    print(f"  Total shards: {manifest['total_shards']}")
    print(f"  Total samples: {manifest['total_samples']:,}")
    print(f"  Format: {manifest['format_version']}")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"fixed_training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    try:
        # Import modules
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.blip3o_dit import create_blip3o_dit_model
        from src.modules.losses.flow_matching_loss import create_blip3o_flow_matching_loss
        from src.modules.trainers.blip3o_trainer import BLIP3oTrainer, create_blip3o_training_args
        
        # Create model config with REDUCED complexity
        model_config = BLIP3oDiTConfig(
            input_size=16,                      # 16x16 = 256 tokens
            patch_size=1,
            in_channels=1024,                   # CLIP dimension
            dim=args.model_dim,                 # Hidden dimension
            eva_embedding_size=4096,            # EVA-CLIP dimension
            n_layers=args.num_layers,           # REDUCED layers
            n_heads=args.num_heads,
            norm_eps=1e-5,
            qk_norm=True,
            learn_sigma=False,
            _gradient_checkpointing=True,       # Force gradient checkpointing
        )
        
        print(f"Creating model with {args.num_layers} layers...")
        model = create_blip3o_dit_model(config=model_config)
        model.to(device)
        
        # Force gradient checkpointing
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
        print(f"Model parameters: {model.get_num_parameters():,}")
        
        # Create flow matching loss
        flow_matching_loss = create_blip3o_flow_matching_loss()
        
        # Create SAFER dataloaders (no shard deletion)
        print("Creating SAFER dataloaders (no shard deletion)...")
        train_dataloader, eval_dataloader = create_safer_chunked_dataloaders(
            chunked_dir=embeddings_path,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_split_ratio=0.1,
            normalize_embeddings=True,
        )
        
        # Create dummy datasets for trainer
        class DummyDataset:
            def __init__(self, length):
                self.length = length
            def __len__(self):
                return self.length
        
        train_dataset = DummyDataset(manifest['total_samples'])
        eval_dataset = DummyDataset(manifest['total_samples'] // 10) if not args.disable_eval else None
        
        # Calculate training steps
        steps_per_epoch = (manifest['total_samples'] + args.batch_size - 1) // args.batch_size
        max_steps = (steps_per_epoch * args.num_epochs) // args.gradient_accumulation_steps
        
        print(f"Training steps: {max_steps} (effective batch size: {args.batch_size * args.gradient_accumulation_steps})")
        
        # Create training arguments
        training_args = create_blip3o_training_args(
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            logging_steps=10,
            save_steps=max(50, max_steps // 5),
            eval_steps=max(25, max_steps // 10) if not args.disable_eval else 0,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            load_best_model_at_end=False,  # Disable to prevent issues
        )
        
        # Create trainer
        print("Creating trainer...")
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
        
        print("Starting training...")
        aggressive_memory_cleanup()
        
        # Start training
        trainer.train()
        
        # Save final model
        print("Saving final model...")
        trainer.save_model()
        
        print("✅ Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        aggressive_memory_cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)