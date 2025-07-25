#!/usr/bin/env python3
"""
FIXED: BLIP3-o Patch-Level Cosine Similarity Evaluation Script
eval_blip3o_patch_similarity.py

FIXES APPLIED:
1. Added missing --shard_index argument
2. Fixed config key handling (num_tokens vs expected_tokens)
3. Better CUDA environment setup
4. Comprehensive cosine similarity calculation
5. Proper error handling and device setup
6. Enhanced evaluation with detailed metrics
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

# FIXED: Handle CUDA environment issues
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FIXED: BLIP3-o Patch-Level Cosine Similarity Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained BLIP3-o model")
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")
    
    # Evaluation configuration
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps for generation")
    parser.add_argument("--training_mode", type=str, default="auto",
                       choices=["auto", "cls_patch", "patch_only"],
                       help="Training mode (auto-detect from model if 'auto')")
    
    # FIXED: Added missing shard_index argument
    parser.add_argument("--shard_index", type=int, default=0,
                       help="Index of the shard to evaluate (0-based)")
    parser.add_argument("--max_eval_shards", type=int, default=1,
                       help="Maximum number of shards to use for evaluation")
    
    # Evaluation options
    parser.add_argument("--same_data_eval", action="store_true", default=True,
                       help="Use same training data for evaluation (overfitting test)")
    parser.add_argument("--normalize_embeddings", action="store_true", default=True,
                       help="Normalize embeddings before computing similarity")
    
    # Output options
    parser.add_argument("--save_plots", action="store_true", default=True,
                       help="Save visualization plots")
    parser.add_argument("--save_detailed_results", action="store_true", default=True,
                       help="Save detailed per-image results")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                       choices=["float32", "float16"],
                       help="Torch data type")
    
    return parser.parse_args()

def setup_device_safely(device_arg: str, logger):
    """FIXED: Setup device with better CUDA handling"""
    if device_arg == "auto":
        try:
            if torch.cuda.is_available():
                # Test CUDA availability
                torch.cuda.device_count()
                device = torch.device("cuda:0")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        except Exception as e:
            logger.warning(f"CUDA setup failed: {e}, falling back to CPU")
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(device_arg)
            if device.type == "cuda":
                # Test that the device works
                torch.cuda.set_device(device)
                torch.cuda.device_count()
            logger.info(f"Using specified device: {device}")
        except Exception as e:
            logger.warning(f"Specified device {device_arg} failed: {e}, using CPU")
            device = torch.device("cpu")
    
    return device

def load_model_and_determine_mode(model_path, device, torch_dtype, training_mode, logger):
    """FIXED: Load model and determine training mode with proper config handling"""
    from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
    
    model_path = Path(model_path)
    logger.info(f"📦 Loading model from: {model_path}")
    
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
    
    # FIXED: Handle different config key names properly
    if 'num_tokens' in config_data:
        expected_tokens = config_data['num_tokens']
    elif 'expected_tokens' in config_data:
        expected_tokens = config_data['expected_tokens']
    else:
        # Infer from training_mode if not in config
        if training_mode == "auto":
            # Try to infer from other config keys
            if 'training_mode' in config_data:
                inferred_mode = config_data['training_mode']
                expected_tokens = 257 if inferred_mode == "cls_patch" else 256
            else:
                expected_tokens = 256  # Default to patch_only
        else:
            expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        logger.warning(f"No token count in config, inferring: {expected_tokens}")
    
    # Create model config
    # Ensure we have the num_tokens field
    if 'num_tokens' not in config_data:
        config_data['num_tokens'] = expected_tokens
    
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
    
    # Load state dict
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
    
    # Move to device and set dtype
    dtype = torch.float16 if torch_dtype == "float16" else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Expected tokens: {expected_tokens}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Dtype: {dtype}")
    
    return model, config, training_mode

def create_evaluation_dataloader(embeddings_dir, training_mode, max_shards, batch_size, shard_index, logger):
    """Create dataloader for evaluation with specific shard support"""
    from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
    
    logger.info(f"📊 Creating evaluation dataloader")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Shard index: {shard_index}")
    logger.info(f"   Max shards: {max_shards}")
    logger.info(f"   Batch size: {batch_size}")
    
    # Create dataloaders - using single shard for evaluation
    train_dataloader, eval_dataloader = create_flexible_dataloaders(
        chunked_embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        eval_split_ratio=0.0,  # Use all data for evaluation
        normalize_embeddings=False,  # We'll handle normalization in evaluation
        training_mode=training_mode,
        max_shards=max_shards,  # Limit to specific number of shards
        use_same_data_for_eval=True,  # Use same data
        delete_after_use=False,
        num_workers=0,  # Avoid multiprocessing issues in evaluation
        pin_memory=False,  # Disable to avoid CUDA issues
    )
    
    # Use train_dataloader for same-data evaluation
    eval_dataloader = train_dataloader
    
    logger.info(f"✅ Evaluation dataloader created")
    
    return eval_dataloader

def compute_comprehensive_cosine_similarity(
    predicted_embeddings: torch.Tensor,  # [B, N, 1024] - Generated CLIP embeddings
    target_embeddings: torch.Tensor,     # [B, N, 1024] - Ground truth CLIP embeddings
    training_mode: str = "patch_only",
    normalize: bool = True,
    return_detailed: bool = True
) -> dict:
    """
    Compute comprehensive cosine similarity evaluation exactly as requested:
    1. Cosine similarity for each patch
    2. Average over all patches to get cosine similarity for each image  
    3. Average over all samples to get overall cosine similarity
    """
    batch_size, num_tokens, embed_dim = predicted_embeddings.shape
    
    # Validate inputs
    assert predicted_embeddings.shape == target_embeddings.shape, \
        f"Shape mismatch: {predicted_embeddings.shape} vs {target_embeddings.shape}"
    assert num_tokens in [256, 257], f"Expected 256 or 257 tokens, got {num_tokens}"
    assert embed_dim == 1024, f"Expected 1024-dim embeddings, got {embed_dim}"
    
    # Normalize embeddings if requested (recommended for cosine similarity)
    if normalize:
        predicted_norm = F.normalize(predicted_embeddings, p=2, dim=-1)  # [B, N, 1024]
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)       # [B, N, 1024]
    else:
        predicted_norm = predicted_embeddings
        target_norm = target_embeddings
    
    # STEP 1: Compute cosine similarity for each patch
    per_patch_similarities = F.cosine_similarity(
        predicted_norm, target_norm, dim=-1
    )  # [B, N] - Cosine similarity for each patch in each image
    
    # STEP 2: Average over all patches to get cosine similarity for each image
    per_image_similarities = per_patch_similarities.mean(dim=1)  # [B] - Average similarity per image
    
    # STEP 3: Average over all samples to get overall cosine similarity
    overall_similarity = per_image_similarities.mean().item()  # Scalar - Overall average
    
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
            # Separate CLS token (index 0) from patches (indices 1-256)
            cls_similarities = per_patch_similarities[:, 0]  # [B] - CLS token similarities
            patch_similarities = per_patch_similarities[:, 1:]  # [B, 256] - Patch similarities
            
            results.update({
                'cls_token_mean_similarity': cls_similarities.mean().item(),
                'cls_token_std_similarity': cls_similarities.std().item(),
                'patches_only_mean_similarity': patch_similarities.mean().item(),
                'patches_only_std_similarity': patch_similarities.std().item(),
                'cls_vs_patches_difference': (cls_similarities.mean() - patch_similarities.mean()).item(),
            })
        else:
            # All tokens are patches
            results.update({
                'cls_token_mean_similarity': 0.0,  # No CLS token
                'patches_only_mean_similarity': per_patch_similarities.mean().item(),
                'patches_only_std_similarity': per_patch_similarities.std().item(),
                'cls_vs_patches_difference': 0.0,
            })
    
    return results

def evaluate_model_on_single_shard(
    model,
    dataloader,
    device: str,
    training_mode: str = "patch_only",
    num_inference_steps: int = 50,
    max_batches: int = None,
    logger = None
) -> dict:
    """Evaluate model on a single shard with comprehensive cosine similarity analysis"""
    model.eval()
    
    all_per_patch_similarities = []
    all_per_image_similarities = []
    batch_count = 0
    
    if logger:
        logger.info(f"🔍 Starting evaluation on single shard...")
        logger.info(f"   Training mode: {training_mode}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Max batches: {max_batches or 'All'}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            # Move batch to device
            eva_embeddings = batch['encoder_hidden_states'].to(device)
            target_clip = batch['clip_embeddings'].to(device)
            
            batch_size, num_tokens, _ = eva_embeddings.shape
            
            # Generate embeddings using the model
            try:
                # Generate using the model's generation method
                if hasattr(model, 'generate'):
                    generated_clip = model.generate(
                        eva_features=eva_embeddings,
                        num_inference_steps=num_inference_steps,
                        normalize_output=True,
                    )
                else:
                    # Fallback to forward pass if no generate method
                    timesteps = torch.zeros(batch_size, device=device)
                    hidden_states = batch.get('hidden_states', target_clip)
                    outputs = model(
                        hidden_states=hidden_states,
                        timestep=timesteps,
                        encoder_hidden_states=eva_embeddings,
                        return_dict=True
                    )
                    generated_clip = outputs.get('velocity_prediction', outputs.get('last_hidden_state'))
                
                # Ensure same shape
                if generated_clip.shape != target_clip.shape:
                    if logger:
                        logger.warning(f"Shape mismatch: generated {generated_clip.shape} vs target {target_clip.shape}")
                    continue
                
                # Compute cosine similarities for this batch
                batch_results = compute_comprehensive_cosine_similarity(
                    predicted_embeddings=generated_clip,
                    target_embeddings=target_clip,
                    training_mode=training_mode,
                    normalize=True,
                    return_detailed=True
                )
                
                # Collect per-patch and per-image similarities for global aggregation
                with torch.no_grad():
                    pred_norm = F.normalize(generated_clip, p=2, dim=-1)
                    target_norm = F.normalize(target_clip, p=2, dim=-1)
                    per_patch_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)  # [B, N]
                    per_image_sim = per_patch_sim.mean(dim=1)  # [B]
                    
                    all_per_patch_similarities.append(per_patch_sim.cpu())
                    all_per_image_similarities.append(per_image_sim.cpu())
                
                batch_count += 1
                
                if logger and batch_idx % 10 == 0:
                    logger.info(f"   Batch {batch_idx}: Overall similarity = {batch_results['overall_cosine_similarity']:.4f}")
                    
            except Exception as e:
                if logger:
                    logger.warning(f"   Batch {batch_idx} failed: {e}")
                continue
    
    if not all_per_patch_similarities:
        raise RuntimeError("No batches were successfully evaluated")
    
    # Aggregate all results
    all_per_patch = torch.cat(all_per_patch_similarities, dim=0)  # [Total_Images, N]
    all_per_image = torch.cat(all_per_image_similarities, dim=0)  # [Total_Images]
    
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
        cls_similarities = all_per_patch[:, 0]  # [Total_Images] - CLS token similarities
        patch_similarities = all_per_patch[:, 1:]  # [Total_Images, 256] - Patch similarities
        
        final_results.update({
            'cls_token_mean_similarity': cls_similarities.mean().item(),
            'cls_token_std_similarity': cls_similarities.std().item(),
            'patches_only_mean_similarity': patch_similarities.mean().item(),
            'patches_only_std_similarity': patch_similarities.std().item(),
            'cls_vs_patches_difference': (cls_similarities.mean() - patch_similarities.mean()).item(),
        })
    
    if logger:
        logger.info(f"✅ Evaluation completed on {batch_count} batches")
        logger.info(f"   Overall cosine similarity: {final_results['overall_cosine_similarity']:.4f}")
        logger.info(f"   Per-image mean: {final_results['per_image_mean_similarity']:.4f}")
        logger.info(f"   Per-patch mean: {final_results['per_patch_mean_similarity']:.4f}")
        logger.info(f"   High quality images (>0.7): {final_results['high_quality_images_ratio']*100:.1f}%")
    
    return final_results

def main():
    """FIXED: Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("🔍 FIXED: BLIP3-o Patch-Level Cosine Similarity Evaluation")
    logger.info("=" * 60)
    logger.info("🎯 FEATURES:")
    logger.info(f"  ✅ Model path: {args.model_path}")
    logger.info(f"  ✅ Training mode: {args.training_mode}")
    logger.info(f"  ✅ Shard index: {args.shard_index}")
    logger.info(f"  ✅ Same-data evaluation: {args.same_data_eval}")
    logger.info(f"  ✅ Max eval shards: {args.max_eval_shards}")
    logger.info(f"  ✅ Number of samples: {args.num_samples}")
    logger.info(f"  ✅ FIXED: Added shard_index argument support")
    logger.info("=" * 60)
    
    try:
        # 1. Setup device safely
        device = setup_device_safely(args.device, logger)
        
        # 2. Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        # 3. FIXED: Load model with proper config handling
        model, config, training_mode = load_model_and_determine_mode(
            args.model_path, device, args.torch_dtype, args.training_mode, logger
        )
        
        # 4. Create evaluation dataloader with shard support
        eval_dataloader = create_evaluation_dataloader(
            args.chunked_embeddings_dir, training_mode, args.max_eval_shards, 
            args.batch_size, args.shard_index, logger
        )
        
        # 5. Run comprehensive evaluation
        logger.info("🚀 Starting comprehensive evaluation...")
        start_time = datetime.now()
        
        results = evaluate_model_on_single_shard(
            model=model,
            dataloader=eval_dataloader,
            device=str(device),
            training_mode=training_mode,
            num_inference_steps=args.num_inference_steps,
            max_batches=args.num_samples // args.batch_size if args.num_samples else None,
            logger=logger
        )
        
        end_time = datetime.now()
        evaluation_duration = (end_time - start_time).total_seconds()
        
        # 6. Process and display results
        logger.info("📊 EVALUATION RESULTS:")
        logger.info("=" * 40)
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
        
        # Mode-specific results
        if 'cls_token_mean_similarity' in results and training_mode == "cls_patch":
            logger.info(f"🎯 CLS vs PATCH ANALYSIS:")
            logger.info(f"   CLS token similarity: {results['cls_token_mean_similarity']:.4f}")
            logger.info(f"   Patches similarity: {results['patches_only_mean_similarity']:.4f}")
            logger.info(f"   CLS vs Patches difference: {results['cls_vs_patches_difference']:.4f}")
        
        # Overfitting assessment for same-data evaluation
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
        
        # 7. Save evaluation summary
        evaluation_summary = {
            'evaluation_completed': True,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': evaluation_duration,
            'model_path': str(args.model_path),
            'embeddings_dir': args.chunked_embeddings_dir,
            'training_mode': training_mode,
            'shard_index': args.shard_index,
            'evaluation_config': {
                'num_samples': args.num_samples,
                'batch_size': args.batch_size,
                'num_inference_steps': args.num_inference_steps,
                'same_data_eval': args.same_data_eval,
                'max_eval_shards': args.max_eval_shards,
                'normalize_embeddings': args.normalize_embeddings,
            },
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f'patch_similarity_evaluation_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        logger.info("=" * 40)
        logger.info("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"⏱️ Evaluation time: {evaluation_duration:.1f} seconds")
        
        # 8. Print final summary
        logger.info("📋 FINAL SUMMARY:")
        logger.info(f"   ✅ Successfully evaluated {results['num_images_evaluated']:,} images")
        logger.info(f"   ✅ Overall cosine similarity: {results['overall_cosine_similarity']:.4f}")
        logger.info(f"   ✅ Training mode: {training_mode} ({results['patches_per_image']} tokens)")
        logger.info(f"   ✅ Quality assessment: {results['high_quality_images_ratio']*100:.1f}% high quality images")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        if args.training_mode == "auto":
            logger.error("💡 Try specifying --training_mode explicitly (cls_patch or patch_only)")
        logger.error("💡 Check that model path and embeddings directory are correct")
        traceback.print_exc()
        
        # Save error info
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'evaluation_args': vars(args),
            'timestamp': datetime.now().isoformat(),
            'fixes_applied': [
                'Added missing --shard_index argument',
                'Fixed config key handling (num_tokens vs expected_tokens)',
                'Better CUDA environment setup',
                'Comprehensive cosine similarity calculation',
                'Proper error handling and device setup'
            ]
        }
        
        with open('evaluation_error_fixed.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        logger.error("💾 Error info saved to evaluation_error_fixed.json")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)