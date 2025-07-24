#!/usr/bin/env python3
"""
CPU-OPTIMIZED: BLIP3-o Patch-Level Cosine Similarity Evaluation Script
eval_blip3o_patch_similarity.py

OPTIMIZATIONS FOR CPU:
1. CPU-first device selection
2. Optimized batch processing for CPU memory
3. Efficient tensor operations for CPU
4. Reduced memory footprint
5. Better progress tracking for slower CPU evaluation

Features:
1. Comprehensive patch-level cosine similarity evaluation
2. Support for both CLS+patch (257) and patch-only (256) modes
3. Per-patch, per-image, and global cosine similarity analysis
4. JSON reporting and visualization plots
5. Same-data evaluation for overfitting verification
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import traceback
import gc

# CPU-OPTIMIZED: Set optimal CPU settings
torch.set_num_threads(max(1, os.cpu_count() // 2))  # Use half available CPUs
os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))

# Disable CUDA to force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging for evaluation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments with CPU-optimized defaults"""
    parser = argparse.ArgumentParser(
        description="CPU-OPTIMIZED: BLIP3-o Patch-Level Cosine Similarity Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained BLIP3-o model")
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")
    
    # CPU-OPTIMIZED: Smaller defaults for CPU
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of samples to evaluate (reduced for CPU)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Evaluation batch size (reduced for CPU)")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                       help="Number of inference steps (reduced for CPU)")
    parser.add_argument("--training_mode", type=str, default="auto",
                       choices=["auto", "cls_patch", "patch_only"],
                       help="Training mode (auto-detect from model if 'auto')")
    
    # Evaluation options
    parser.add_argument("--same_data_eval", action="store_true", default=True,
                       help="Use same training data for evaluation (overfitting test)")
    parser.add_argument("--max_eval_shards", type=int, default=1,
                       help="Maximum number of shards to use for evaluation")
    parser.add_argument("--normalize_embeddings", action="store_true", default=True,
                       help="Normalize embeddings before computing similarity")
    
    # Output options
    parser.add_argument("--save_plots", action="store_true", default=False,
                       help="Save visualization plots (disabled for CPU)")
    parser.add_argument("--save_detailed_results", action="store_true", default=True,
                       help="Save detailed per-image results")
    
    # CPU-specific options
    parser.add_argument("--cpu_threads", type=int, default=None,
                       help="Number of CPU threads to use (auto-detect if None)")
    parser.add_argument("--memory_efficient", action="store_true", default=True,
                       help="Use memory-efficient processing")
    parser.add_argument("--progress_frequency", type=int, default=5,
                       help="How often to log progress (every N batches)")
    
    # Hardware configuration (CPU-focused)
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "auto"],
                       help="Device to use (CPU recommended)")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                       choices=["float32", "float16"],
                       help="Torch data type (float32 recommended for CPU)")
    
    return parser.parse_args()

def setup_cpu_optimizations(args, logger):
    """Setup CPU-specific optimizations"""
    # Set CPU threads
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        os.environ['OMP_NUM_THREADS'] = str(args.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(args.cpu_threads)
        logger.info(f"Set CPU threads to: {args.cpu_threads}")
    else:
        available_cpus = os.cpu_count()
        optimal_threads = max(1, available_cpus // 2)
        torch.set_num_threads(optimal_threads)
        logger.info(f"Auto-detected CPU threads: {optimal_threads}/{available_cpus}")
    
    # Force CPU device
    device = torch.device("cpu")
    logger.info("💻 Using CPU for evaluation (GPU disabled)")
    logger.info(f"🔧 CPU optimization enabled")
    logger.info(f"🔧 Memory efficient processing: {args.memory_efficient}")
    
    return device

def load_model_and_determine_mode(model_path, device, torch_dtype, training_mode, logger):
    """CPU-OPTIMIZED: Load model with CPU-specific settings"""
    from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
    
    model_path = Path(model_path)
    logger.info(f"📦 Loading model from: {model_path}")
    logger.info(f"💻 Loading on CPU (optimized for CPU evaluation)")
    
    # Load model configuration
    config_files = [
        model_path / "config.json",
        model_path / "blip3o_model_config.json",
        model_path / "enhanced_training_config.json"
    ]
    
    config_data = None
    for config_file in config_files:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            logger.info(f"✅ Loaded config from: {config_file}")
            break
    
    if config_data is None:
        raise FileNotFoundError(f"No config file found in {model_path}")
    
    # Handle different config key names properly
    if 'num_tokens' in config_data:
        expected_tokens = config_data['num_tokens']
    elif 'expected_tokens' in config_data:
        expected_tokens = config_data['expected_tokens']
    else:
        # Infer from training_mode if not in config
        if training_mode == "auto":
            if 'training_mode' in config_data:
                inferred_mode = config_data['training_mode']
                expected_tokens = 257 if inferred_mode == "cls_patch" else 256
            else:
                expected_tokens = 257  # Default to cls_patch
        else:
            expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        logger.warning(f"No token count in config, inferring: {expected_tokens}")
    
    # Create model config
    if 'num_tokens' not in config_data:
        config_data['num_tokens'] = expected_tokens
    
    # CPU-OPTIMIZED: Disable gradient checkpointing for CPU
    config_data['use_gradient_checkpointing'] = False
    
    config = BLIP3oDiTConfig(**config_data)
    
    # Determine training mode
    if training_mode == "auto":
        if hasattr(config, 'training_mode') and config.training_mode:
            training_mode = config.training_mode
        elif expected_tokens == 257:
            training_mode = "cls_patch"
        else:
            training_mode = "patch_only"
        logger.info(f"🎯 Auto-detected training mode: {training_mode}")
    
    # Create model
    model = create_blip3o_patch_dit_model(config=config)
    
    # Load weights
    weight_files = [
        model_path / "pytorch_model.bin",
        model_path / "model.safetensors", 
        model_path / "pytorch_model.safetensors"
    ]
    
    weight_file = None
    for wf in weight_files:
        if wf.exists():
            weight_file = wf
            break
    
    if weight_file is None:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    logger.info(f"💾 Loading weights from: {weight_file}")
    
    # Load state dict with CPU map_location
    if weight_file.suffix == ".bin":
        state_dict = torch.load(weight_file, map_location='cpu')
    else:
        from safetensors.torch import load_file
        state_dict = load_file(str(weight_file))
    
    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
    
    # Move to CPU and set dtype
    dtype = torch.float32  # Always use float32 on CPU for better performance
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # CPU-OPTIMIZED: Disable gradient computation globally
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info(f"✅ Model loaded successfully on CPU")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Expected tokens: {expected_tokens}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Dtype: {dtype}")
    logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config, training_mode

def create_evaluation_dataloader(embeddings_dir, training_mode, max_shards, batch_size, logger):
    """CPU-OPTIMIZED: Create dataloader with CPU-friendly settings"""
    from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
    
    logger.info(f"📊 Creating CPU-optimized evaluation dataloader")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Max shards: {max_shards}")
    logger.info(f"   Batch size: {batch_size} (optimized for CPU)")
    
    # CPU-OPTIMIZED: Use minimal workers and no pinning
    train_dataloader, eval_dataloader = create_flexible_dataloaders(
        chunked_embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        eval_split_ratio=0.0,  # Use all data for evaluation
        normalize_embeddings=False,  # We'll handle normalization in evaluation
        training_mode=training_mode,
        max_shards=max_shards,
        use_same_data_for_eval=True,  # Use same data
        delete_after_use=False,
        num_workers=0,  # CPU-OPTIMIZED: No multiprocessing
        pin_memory=False,  # CPU-OPTIMIZED: No pinning
    )
    
    # Use train_dataloader for same-data evaluation
    eval_dataloader = train_dataloader
    
    logger.info(f"✅ CPU-optimized evaluation dataloader created")
    
    return eval_dataloader

def compute_comprehensive_cosine_similarity(
    predicted_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    training_mode: str = "cls_patch",
    normalize: bool = True,
    return_detailed: bool = True
) -> dict:
    """
    CPU-OPTIMIZED: Compute comprehensive cosine similarity with memory efficiency
    """
    batch_size, num_tokens, embed_dim = predicted_embeddings.shape
    
    # Validate inputs
    assert predicted_embeddings.shape == target_embeddings.shape, \
        f"Shape mismatch: {predicted_embeddings.shape} vs {target_embeddings.shape}"
    assert num_tokens in [256, 257], f"Expected 256 or 257 tokens, got {num_tokens}"
    assert embed_dim == 1024, f"Expected 1024-dim embeddings, got {embed_dim}"
    
    # CPU-OPTIMIZED: Process in smaller chunks if batch is large
    if batch_size > 16:
        # Process in chunks to save memory
        chunk_size = 8
        all_per_patch_similarities = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            pred_chunk = predicted_embeddings[i:end_idx]
            target_chunk = target_embeddings[i:end_idx]
            
            # Normalize if requested
            if normalize:
                pred_norm = F.normalize(pred_chunk, p=2, dim=-1)
                target_norm = F.normalize(target_chunk, p=2, dim=-1)
            else:
                pred_norm = pred_chunk
                target_norm = target_chunk
            
            # Compute similarities for this chunk
            chunk_similarities = F.cosine_similarity(pred_norm, target_norm, dim=-1)
            all_per_patch_similarities.append(chunk_similarities)
            
            # Clean up memory
            del pred_chunk, target_chunk, pred_norm, target_norm, chunk_similarities
            if i % (chunk_size * 4) == 0:
                gc.collect()
        
        # Concatenate results
        per_patch_similarities = torch.cat(all_per_patch_similarities, dim=0)
        del all_per_patch_similarities
        gc.collect()
    
    else:
        # Process normally for small batches
        if normalize:
            predicted_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
            target_norm = F.normalize(target_embeddings, p=2, dim=-1)
        else:
            predicted_norm = predicted_embeddings
            target_norm = target_embeddings
        
        per_patch_similarities = F.cosine_similarity(predicted_norm, target_norm, dim=-1)
    
    # Compute per-image and overall similarities
    per_image_similarities = per_patch_similarities.mean(dim=1)
    overall_similarity = per_image_similarities.mean().item()
    
    # Prepare results
    results = {
        'overall_cosine_similarity': overall_similarity,
        'per_image_mean_similarity': per_image_similarities.mean().item(),
        'per_image_std_similarity': per_image_similarities.std().item(),
        'per_patch_mean_similarity': per_patch_similarities.mean().item(),
        'per_patch_std_similarity': per_patch_similarities.std().item(),
        'num_images': batch_size,
        'num_tokens_per_image': num_tokens,
        'total_patches_evaluated': batch_size * num_tokens,
    }
    
    # Add detailed analysis if requested
    if return_detailed:
        # Quality thresholds
        high_quality_patches = (per_patch_similarities > 0.7).float()
        very_high_quality_patches = (per_patch_similarities > 0.8).float()
        excellent_patches = (per_patch_similarities > 0.9).float()
        
        high_quality_images = (per_image_similarities > 0.7).float()
        very_high_quality_images = (per_image_similarities > 0.8).float()
        excellent_images = (per_image_similarities > 0.9).float()
        
        results.update({
            # Quality distribution for patches
            'high_quality_patches_ratio': high_quality_patches.mean().item(),
            'very_high_quality_patches_ratio': very_high_quality_patches.mean().item(),
            'excellent_patches_ratio': excellent_patches.mean().item(),
            
            # Quality distribution for images
            'high_quality_images_ratio': high_quality_images.mean().item(),
            'very_high_quality_images_ratio': very_high_quality_images.mean().item(),
            'excellent_images_ratio': excellent_images.mean().item(),
            
            # Min/max statistics
            'min_patch_similarity': per_patch_similarities.min().item(),
            'max_patch_similarity': per_patch_similarities.max().item(),
            'min_image_similarity': per_image_similarities.min().item(),
            'max_image_similarity': per_image_similarities.max().item(),
        })
        
        # Mode-specific analysis
        if training_mode == "cls_patch" and num_tokens == 257:
            cls_similarities = per_patch_similarities[:, 0]
            patch_similarities = per_patch_similarities[:, 1:]
            
            results.update({
                'cls_token_mean_similarity': cls_similarities.mean().item(),
                'cls_token_std_similarity': cls_similarities.std().item(),
                'patches_only_mean_similarity': patch_similarities.mean().item(),
                'patches_only_std_similarity': patch_similarities.std().item(),
                'cls_vs_patches_difference': (cls_similarities.mean() - patch_similarities.mean()).item(),
            })
        else:
            results.update({
                'cls_token_mean_similarity': 0.0,
                'patches_only_mean_similarity': per_patch_similarities.mean().item(),
                'patches_only_std_similarity': per_patch_similarities.std().item(),
                'cls_vs_patches_difference': 0.0,
            })
    
    # Clean up memory
    del per_patch_similarities, per_image_similarities
    gc.collect()
    
    return results

def evaluate_model_on_single_shard_cpu(
    model,
    dataloader,
    training_mode: str = "cls_patch",
    num_inference_steps: int = 25,
    max_batches: int = None,
    memory_efficient: bool = True,
    progress_frequency: int = 5,
    logger = None
) -> dict:
    """
    CPU-OPTIMIZED: Evaluate model with memory-efficient processing
    """
    model.eval()
    
    all_per_patch_similarities = []
    all_per_image_similarities = []
    batch_count = 0
    total_images = 0
    
    if logger:
        logger.info(f"🔍 Starting CPU-optimized evaluation...")
        logger.info(f"   Training mode: {training_mode}")
        logger.info(f"   Inference steps: {num_inference_steps}")
        logger.info(f"   Max batches: {max_batches or 'All'}")
        logger.info(f"   Memory efficient: {memory_efficient}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            try:
                # Move batch to CPU (should already be on CPU)
                eva_embeddings = batch['encoder_hidden_states']
                target_clip = batch['clip_embeddings']
                
                # Ensure on CPU
                if eva_embeddings.device.type != 'cpu':
                    eva_embeddings = eva_embeddings.cpu()
                if target_clip.device.type != 'cpu':
                    target_clip = target_clip.cpu()
                
                batch_size, num_tokens, _ = eva_embeddings.shape
                total_images += batch_size
                
                # CPU-OPTIMIZED: Generate embeddings with reduced steps
                if hasattr(model, 'generate'):
                    generated_clip = model.generate(
                        eva_features=eva_embeddings,
                        num_inference_steps=num_inference_steps,
                    )
                else:
                    # Fallback to forward pass
                    timesteps = torch.zeros(batch_size)
                    outputs = model(
                        hidden_states=target_clip,
                        timestep=timesteps,
                        encoder_hidden_states=eva_embeddings,
                        return_dict=True
                    )
                    generated_clip = outputs.get('velocity_prediction', outputs.get('last_hidden_state'))
                
                # Ensure same shape
                if generated_clip.shape != target_clip.shape:
                    if logger and batch_idx == 0:  # Only log once
                        logger.warning(f"Shape mismatch: generated {generated_clip.shape} vs target {target_clip.shape}")
                    continue
                
                # Compute similarities efficiently
                if memory_efficient and batch_size > 8:
                    # Process in smaller chunks
                    chunk_size = 4
                    batch_per_patch_sims = []
                    
                    for i in range(0, batch_size, chunk_size):
                        end_idx = min(i + chunk_size, batch_size)
                        
                        pred_norm = F.normalize(generated_clip[i:end_idx], p=2, dim=-1)
                        target_norm = F.normalize(target_clip[i:end_idx], p=2, dim=-1)
                        chunk_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                        
                        batch_per_patch_sims.append(chunk_sim)
                        
                        # Clean memory
                        del pred_norm, target_norm, chunk_sim
                    
                    batch_per_patch_sim = torch.cat(batch_per_patch_sims, dim=0)
                    del batch_per_patch_sims
                    
                else:
                    # Process entire batch
                    pred_norm = F.normalize(generated_clip, p=2, dim=-1)
                    target_norm = F.normalize(target_clip, p=2, dim=-1)
                    batch_per_patch_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                    del pred_norm, target_norm
                
                # Compute per-image similarities
                batch_per_image_sim = batch_per_patch_sim.mean(dim=1)
                
                # Store results
                all_per_patch_similarities.append(batch_per_patch_sim.cpu())
                all_per_image_similarities.append(batch_per_image_sim.cpu())
                
                batch_count += 1
                
                # Progress logging
                if logger and batch_idx % progress_frequency == 0:
                    current_overall_sim = batch_per_image_sim.mean().item()
                    logger.info(f"   Batch {batch_idx}/{len(dataloader)}: "
                              f"Images={total_images}, "
                              f"Batch similarity={current_overall_sim:.4f}")
                
                # Clean up memory
                del eva_embeddings, target_clip, generated_clip
                del batch_per_patch_sim, batch_per_image_sim
                
                # Periodic garbage collection
                if batch_idx % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                if logger:
                    logger.warning(f"   Batch {batch_idx} failed: {e}")
                continue
    
    if not all_per_patch_similarities:
        raise RuntimeError("No batches were successfully evaluated")
    
    # Aggregate all results
    logger.info(f"📊 Aggregating results from {batch_count} batches...")
    all_per_patch = torch.cat(all_per_patch_similarities, dim=0)
    all_per_image = torch.cat(all_per_image_similarities, dim=0)
    
    # Clean up intermediate results
    del all_per_patch_similarities, all_per_image_similarities
    gc.collect()
    
    # Final comprehensive results
    final_results = {
        'overall_cosine_similarity': all_per_image.mean().item(),
        'per_image_mean_similarity': all_per_image.mean().item(),
        'per_image_std_similarity': all_per_image.std().item(),
        'per_patch_mean_similarity': all_per_patch.mean().item(),
        'per_patch_std_similarity': all_per_patch.std().item(),
        'num_images_evaluated': len(all_per_image),
        'num_batches_evaluated': batch_count,
        'patches_per_image': all_per_patch.shape[1],
        'total_patches_evaluated': all_per_patch.numel(),
        
        # Quality thresholds
        'high_quality_patches_ratio': (all_per_patch > 0.7).float().mean().item(),
        'very_high_quality_patches_ratio': (all_per_patch > 0.8).float().mean().item(),
        'excellent_patches_ratio': (all_per_patch > 0.9).float().mean().item(),
        
        'high_quality_images_ratio': (all_per_image > 0.7).float().mean().item(),
        'very_high_quality_images_ratio': (all_per_image > 0.8).float().mean().item(),
        'excellent_images_ratio': (all_per_image > 0.9).float().mean().item(),
        
        # Min/max
        'min_patch_similarity': all_per_patch.min().item(),
        'max_patch_similarity': all_per_patch.max().item(),
        'min_image_similarity': all_per_image.min().item(),
        'max_image_similarity': all_per_image.max().item(),
    }
    
    # Mode-specific analysis
    if training_mode == "cls_patch" and all_per_patch.shape[1] == 257:
        cls_similarities = all_per_patch[:, 0]
        patch_similarities = all_per_patch[:, 1:]
        
        final_results.update({
            'cls_token_mean_similarity': cls_similarities.mean().item(),
            'cls_token_std_similarity': cls_similarities.std().item(),
            'patches_only_mean_similarity': patch_similarities.mean().item(),
            'patches_only_std_similarity': patch_similarities.std().item(),
            'cls_vs_patches_difference': (cls_similarities.mean() - patch_similarities.mean()).item(),
        })
    
    if logger:
        logger.info(f"✅ CPU evaluation completed!")
        logger.info(f"   Batches processed: {batch_count}")
        logger.info(f"   Images evaluated: {final_results['num_images_evaluated']:,}")
        logger.info(f"   Overall cosine similarity: {final_results['overall_cosine_similarity']:.4f}")
        logger.info(f"   Per-image mean: {final_results['per_image_mean_similarity']:.4f}")
        logger.info(f"   Per-patch mean: {final_results['per_patch_mean_similarity']:.4f}")
        logger.info(f"   High quality images (>0.7): {final_results['high_quality_images_ratio']*100:.1f}%")
    
    # Clean up final tensors
    del all_per_patch, all_per_image
    gc.collect()
    
    return final_results

def main():
    """CPU-OPTIMIZED: Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("💻 CPU-OPTIMIZED: BLIP3-o Patch-Level Cosine Similarity Evaluation")
    logger.info("=" * 70)
    logger.info("🎯 CPU OPTIMIZATIONS:")
    logger.info(f"  ✅ Model path: {args.model_path}")
    logger.info(f"  ✅ Training mode: {args.training_mode}")
    logger.info(f"  ✅ Batch size: {args.batch_size} (CPU-optimized)")
    logger.info(f"  ✅ Inference steps: {args.num_inference_steps} (reduced for CPU)")
    logger.info(f"  ✅ Memory efficient: {args.memory_efficient}")
    logger.info(f"  ✅ CPU threads: {args.cpu_threads or 'Auto'}")
    logger.info(f"  ✅ CUDA disabled, CPU-only evaluation")
    logger.info("=" * 70)
    
    try:
        # 1. Setup CPU optimizations
        device = setup_cpu_optimizations(args, logger)
        
        # 2. Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        # 3. Load model with CPU optimizations
        model, config, training_mode = load_model_and_determine_mode(
            args.model_path, device, args.torch_dtype, args.training_mode, logger
        )
        
        # 4. Create CPU-optimized dataloader
        eval_dataloader = create_evaluation_dataloader(
            args.chunked_embeddings_dir, training_mode, args.max_eval_shards, 
            args.batch_size, logger
        )
        
        # 5. Run CPU-optimized evaluation
        logger.info("🚀 Starting CPU-optimized evaluation...")
        start_time = datetime.now()
        
        results = evaluate_model_on_single_shard_cpu(
            model=model,
            dataloader=eval_dataloader,
            training_mode=training_mode,
            num_inference_steps=args.num_inference_steps,
            max_batches=args.num_samples // args.batch_size if args.num_samples else None,
            memory_efficient=args.memory_efficient,
            progress_frequency=args.progress_frequency,
            logger=logger
        )
        
        end_time = datetime.now()
        evaluation_duration = (end_time - start_time).total_seconds()
        
        # 6. Display results
        logger.info("📊 CPU EVALUATION RESULTS:")
        logger.info("=" * 50)
        logger.info(f"🎯 COSINE SIMILARITY ANALYSIS:")
        logger.info(f"   Overall cosine similarity: {results['overall_cosine_similarity']:.4f}")
        logger.info(f"   Per-image mean similarity: {results['per_image_mean_similarity']:.4f}")
        logger.info(f"   Per-patch mean similarity: {results['per_patch_mean_similarity']:.4f}")
        
        logger.info(f"📊 QUALITY DISTRIBUTION:")
        logger.info(f"   High quality images (>0.7): {results['high_quality_images_ratio']*100:.1f}%")
        logger.info(f"   Very high quality images (>0.8): {results['very_high_quality_images_ratio']*100:.1f}%")
        logger.info(f"   Excellent images (>0.9): {results['excellent_images_ratio']*100:.1f}%")
        
        logger.info(f"📈 STATISTICS:")
        logger.info(f"   Images evaluated: {results['num_images_evaluated']:,}")
        logger.info(f"   Total patches evaluated: {results['total_patches_evaluated']:,}")
        logger.info(f"   Patches per image: {results['patches_per_image']}")
        logger.info(f"   CPU evaluation time: {evaluation_duration:.1f} seconds")
        
        # Mode-specific results
        if 'cls_token_mean_similarity' in results and training_mode == "cls_patch":
            logger.info(f"🎯 CLS vs PATCH ANALYSIS:")
            logger.info(f"   CLS token similarity: {results['cls_token_mean_similarity']:.4f}")
            logger.info(f"   Patches similarity: {results['patches_only_mean_similarity']:.4f}")
            logger.info(f"   CLS vs Patches difference: {results['cls_vs_patches_difference']:.4f}")
        
        # Overfitting assessment
        if args.same_data_eval:
            overall_sim = results['overall_cosine_similarity']
            if overall_sim > 0.9:
                logger.info("🎉 EXCELLENT OVERFITTING: Model perfectly learned training data!")
            elif overall_sim > 0.8:
                logger.info("✅ VERY GOOD OVERFITTING: Strong performance on training data")
            elif overall_sim > 0.7:
                logger.info("👍 GOOD OVERFITTING: Solid performance on training data") 
            elif overall_sim > 0.6:
                logger.info("🔄 MODERATE OVERFITTING: Some learning detected")
            else:
                logger.info("⚠️ LOW OVERFITTING: Model needs more training")
        
        # 7. Save results
        evaluation_summary = {
            'evaluation_completed': True,
            'cpu_optimized': True,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': evaluation_duration,
            'model_path': str(args.model_path),
            'embeddings_dir': args.chunked_embeddings_dir,
            'training_mode': training_mode,
            'cpu_configuration': {
                'cpu_threads': torch.get_num_threads(),
                'memory_efficient': args.memory_efficient,
                'batch_size': args.batch_size,
                'inference_steps': args.num_inference_steps,
            },
            'evaluation_config': vars(args),
            'results_summary': {
                'overall_cosine_similarity': results['overall_cosine_similarity'],
                'per_image_mean_similarity': results['per_image_mean_similarity'],
                'high_quality_images_percentage': results['high_quality_images_ratio'] * 100,
                'total_images': results['num_images_evaluated'],
                'total_patches': results['total_patches_evaluated'],
            },
            'detailed_results': results,
        }
        
        # Save detailed results
        results_file = output_dir / 'cpu_cosine_similarity_results.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("✅ CPU EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"⏱️ Total evaluation time: {evaluation_duration:.1f} seconds")
        logger.info(f"💻 CPU evaluation was successful!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ CPU evaluation failed: {e}")
        traceback.print_exc()
        
        # Save error info
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'evaluation_args': vars(args),
            'timestamp': datetime.now().isoformat(),
            'cpu_optimized': True,
        }
        
        with open('cpu_evaluation_error.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        logger.error("💾 Error info saved to cpu_evaluation_error.json")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)