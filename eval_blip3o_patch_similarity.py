#!/usr/bin/env python3
"""
FIXED: BLIP3-o Evaluation Script with Corrected Generation Method
eval_blip3o_patch_similarity.py
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
from typing import Dict, Any, Optional
import glob

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
    parser = argparse.ArgumentParser(
        description="FIXED BLIP3-o Patch-wise Cosine Similarity Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory or checkpoint")
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    
    # Evaluation parameters
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--training_mode", type=str, default="auto",
                       choices=["auto", "cls_patch", "patch_only"],
                       help="Training mode (auto-detect if 'auto')")
    
    # FIXED: Critical generation parameters
    parser.add_argument("--velocity_scale", type=float, default=0.1,
                       help="CRITICAL: Velocity scale (must match training)")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                       help="Guidance scale for generation")
    
    # Options
    parser.add_argument("--same_data_eval", action="store_true", default=True,
                       help="Use same training data for evaluation")
    parser.add_argument("--normalize_embeddings", action="store_true", default=True,
                       help="Normalize embeddings")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"],
                       help="Torch dtype for evaluation")
    
    # Testing options
    parser.add_argument("--test_multiple_steps", action="store_true", default=False,
                       help="Test with multiple inference steps")
    parser.add_argument("--save_detailed_results", action="store_true", default=True,
                       help="Save detailed evaluation results")
    
    return parser.parse_args()

def setup_device(device_arg: str, logger):
    """Setup device"""
    if device_arg == "auto":
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        except Exception as e:
            logger.warning(f"CUDA setup failed: {e}, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using device: {device}")
    
    return device

def get_torch_dtype(dtype_str: str):
    """Convert string to torch dtype"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)

def find_model_files(model_path):
    """Find model files in various checkpoint structures"""
    model_path = Path(model_path)
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔍 Searching for model files in: {model_path}")
    
    # Try direct path first
    if model_path.is_file():
        logger.info(f"Model path is a file: {model_path}")
        return model_path.parent, model_path.name, None, None
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Look for config files
    config_files = ["config.json", "blip3o_model_config.json", "training_info.json"]
    
    config_file = None
    for cf in config_files:
        if (model_path / cf).exists():
            config_file = cf
            logger.info(f"✅ Found config file: {cf}")
            break
    
    if config_file is None:
        json_files = list(model_path.glob("**/*.json"))
        config_files = [f for f in json_files if any(name in f.name.lower() for name in ['config', 'training'])]
        if config_files:
            config_file = config_files[0].name
            model_path = config_files[0].parent
            logger.info(f"✅ Found config file in subdirectory: {config_file}")
        else:
            raise FileNotFoundError(f"No config file found in {model_path} or subdirectories")
    
    # Look for model weight files
    weight_files = ["model.safetensors", "pytorch_model.bin", "pytorch_model.safetensors"]
    
    weight_file = None
    for wf in weight_files:
        if (model_path / wf).exists():
            weight_file = wf
            logger.info(f"✅ Found weight file: {wf}")
            break
    
    if weight_file is None:
        model_files = []
        for pattern in ["**/*.safetensors", "**/*.bin"]:
            model_files.extend(list(model_path.glob(pattern)))
        
        model_files = [f for f in model_files if any(name in f.name.lower() for name in ['model', 'pytorch'])]
        if model_files:
            weight_file = model_files[0].name
            model_path = model_files[0].parent
            logger.info(f"✅ Found weight file in subdirectory: {weight_file}")
        else:
            raise FileNotFoundError(f"No model weight file found in {model_path} or subdirectories")
    
    return model_path, config_file, weight_file, None

def load_model_and_config(model_path, device, training_mode, torch_dtype, velocity_scale, logger):
    """Load model with FIXED generation method"""
    from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
    
    # Find model files
    model_dir, config_file, weight_file, _ = find_model_files(model_path)
    
    logger.info(f"📦 Loading FIXED model from directory: {model_dir}")
    logger.info(f"📋 Config file: {config_file}")
    logger.info(f"💾 Weight file: {weight_file}")
    logger.info(f"🔧 Velocity scale: {velocity_scale}")
    
    # Load configuration
    config_path = model_dir / config_file
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    logger.info(f"✅ Loaded config from: {config_path}")
    
    # Determine training mode
    if training_mode == "auto":
        if 'training_mode' in config_data:
            training_mode = config_data['training_mode']
        elif 'num_tokens' in config_data:
            training_mode = "cls_patch" if config_data['num_tokens'] == 257 else "patch_only"
        else:
            training_mode = "patch_only"  # Default
        logger.info(f"🎯 Auto-detected training mode: {training_mode}")
    
    # Create config with proper defaults
    expected_tokens = 257 if training_mode == "cls_patch" else 256
    
    # Set default values for missing config parameters
    config_defaults = {
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'eva_embedding_size': 4096,
        'clip_embedding_size': 1024,
        'num_tokens': expected_tokens,
        'max_position_embeddings': max(expected_tokens, 257),
        'dropout_prob': 0.1,
        'training_mode': training_mode,
        'use_gradient_checkpointing': False,
        'output_scale': 0.1,
    }
    
    # Merge with loaded config
    for key, default_value in config_defaults.items():
        if key not in config_data:
            config_data[key] = default_value
            logger.info(f"🔧 Using default value for {key}: {default_value}")
    
    config = BLIP3oDiTConfig(**config_data)
    
    # Create model
    model = create_blip3o_patch_dit_model(config=config)
    
    # Load weights
    weight_path = model_dir / weight_file
    logger.info(f"💾 Loading weights from: {weight_path}")
    
    # Load state dict
    if weight_path.suffix == ".bin":
        state_dict = torch.load(weight_path, map_location='cpu')
    else:
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(weight_path))
        except ImportError:
            logger.error("safetensors not available, please install with: pip install safetensors")
            raise
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)} keys")
        if len(missing_keys) < 10:
            logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
        if len(unexpected_keys) < 10:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    # Move to device
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    
    logger.info(f"✅ FIXED Model loaded successfully")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Expected tokens: {expected_tokens}")
    logger.info(f"   Dtype: {torch_dtype}")
    logger.info(f"   Parameters: {model.get_num_parameters():,}")
    logger.info(f"   Generation will use velocity_scale: {velocity_scale}")
    
    return model, config, training_mode

def create_evaluation_dataloader(embeddings_dir, training_mode, batch_size, logger):
    """Create evaluation dataloader"""
    try:
        from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
        
        logger.info(f"📊 Creating evaluation dataloader")
        logger.info(f"   Embeddings dir: {embeddings_dir}")
        logger.info(f"   Training mode: {training_mode}")
        logger.info(f"   Batch size: {batch_size}")
        
        # Use same data for evaluation (overfitting test)
        train_dataloader, _ = create_flexible_dataloaders(
            chunked_embeddings_dir=embeddings_dir,
            batch_size=batch_size,
            eval_batch_size=batch_size,
            eval_split_ratio=0.0,
            normalize_embeddings=False,
            training_mode=training_mode,
            max_shards=1,  # Single shard for evaluation
            use_same_data_for_eval=True,
            delete_after_use=False,
            num_workers=0,
            pin_memory=False,
        )
        
        logger.info(f"✅ Evaluation dataloader created")
        return train_dataloader
        
    except ImportError as e:
        logger.error(f"Failed to import dataset module: {e}")
        logger.error("Please ensure the src.modules.datasets.blip3o_dataset module is available")
        raise
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        raise

def compute_patch_wise_cosine_similarity(
    predicted_embeddings: torch.Tensor,  # [B, N, 1024]
    target_embeddings: torch.Tensor,     # [B, N, 1024]
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute patch-wise cosine similarity (matches training methodology)
    """
    batch_size, num_tokens, embed_dim = predicted_embeddings.shape
    
    # Validate inputs
    assert predicted_embeddings.shape == target_embeddings.shape, \
        f"Shape mismatch: {predicted_embeddings.shape} vs {target_embeddings.shape}"
    assert num_tokens in [256, 257], f"Expected 256 or 257 tokens, got {num_tokens}"
    assert embed_dim == 1024, f"Expected 1024-dim embeddings, got {embed_dim}"
    
    # Normalize if requested
    if normalize:
        pred_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)
    else:
        pred_norm = predicted_embeddings
        target_norm = target_embeddings
    
    # Per-patch cosine similarities [B, N]
    per_patch_similarities = F.cosine_similarity(pred_norm, target_norm, dim=-1)
    
    # Per-image average similarities [B]
    per_image_similarities = per_patch_similarities.mean(dim=1)
    
    # Overall similarity (scalar)
    overall_similarity = per_image_similarities.mean().item()
    
    results = {
        'overall_cosine_similarity': overall_similarity,
        'per_image_mean_similarity': per_image_similarities.mean().item(),
        'per_image_std_similarity': per_image_similarities.std().item(),
        'per_patch_mean_similarity': per_patch_similarities.mean().item(),
        'per_patch_std_similarity': per_patch_similarities.std().item(),
        
        # Quality metrics
        'high_quality_patches_ratio': (per_patch_similarities > 0.7).float().mean().item(),
        'very_high_quality_patches_ratio': (per_patch_similarities > 0.8).float().mean().item(),
        'high_quality_images_ratio': (per_image_similarities > 0.7).float().mean().item(),
        'very_high_quality_images_ratio': (per_image_similarities > 0.8).float().mean().item(),
        
        # Statistics
        'min_patch_similarity': per_patch_similarities.min().item(),
        'max_patch_similarity': per_patch_similarities.max().item(),
        'min_image_similarity': per_image_similarities.min().item(),
        'max_image_similarity': per_image_similarities.max().item(),
        
        # Dataset info
        'num_images': batch_size,
        'num_tokens_per_image': num_tokens,
        'total_patches_evaluated': batch_size * num_tokens,
    }
    
    return results

def evaluate_model(
    model,
    dataloader,
    device: str,
    training_mode: str = "patch_only",
    num_inference_steps: int = 50,
    max_batches: int = None,
    normalize_embeddings: bool = True,
    velocity_scale: float = 0.1,  # CRITICAL: Must match training
    guidance_scale: float = 1.0,
    test_multiple_steps: bool = False,
    logger = None
) -> dict[str, float]:
    """FIXED: Evaluate model with corrected generation"""
    model.eval()
    
    all_per_patch_similarities = []
    all_per_image_similarities = []
    batch_count = 0
    total_generation_time = 0.0
    
    if logger:
        logger.info(f"🔍 Starting FIXED evaluation...")
        logger.info(f"   Training mode: {training_mode}")
        logger.info(f"   Max batches: {max_batches or 'All'}")
        logger.info(f"   Inference steps: {num_inference_steps}")
        logger.info(f"   Normalize embeddings: {normalize_embeddings}")
        logger.info(f"   FIXED Velocity scale: {velocity_scale}")
        logger.info(f"   Guidance scale: {guidance_scale}")
    
    # Test multiple inference steps if requested
    if test_multiple_steps and logger:
        logger.info("🧪 Testing multiple inference steps...")
        test_batch = next(iter(dataloader))
        eva_embeddings = test_batch['encoder_hidden_states'].to(device)
        target_clip = test_batch['clip_embeddings'].to(device)
        
        for test_steps in [10, 25, 50, 100]:
            start_time = datetime.now()
            generated_clip = model.generate(
                eva_features=eva_embeddings,
                num_inference_steps=test_steps,
                normalize_output=normalize_embeddings,
                guidance_scale=guidance_scale,
                velocity_scale=velocity_scale,  # CRITICAL FIX
            )
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Compute similarity
            batch_results = compute_patch_wise_cosine_similarity(
                predicted_embeddings=generated_clip,
                target_embeddings=target_clip,
                normalize=False,  # Already normalized in generation
            )
            
            logger.info(f"   Steps {test_steps:3d}: Similarity = {batch_results['overall_cosine_similarity']:.4f}, Time = {generation_time:.2f}s")
    
    # Main evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            try:
                # Move to device
                eva_embeddings = batch['encoder_hidden_states'].to(device)
                target_clip = batch['clip_embeddings'].to(device)
                
                # Record generation time
                start_time = datetime.now()
                
                # FIXED: Generate embeddings with correct parameters
                generated_clip = model.generate(
                    eva_features=eva_embeddings,
                    num_inference_steps=num_inference_steps,
                    normalize_output=normalize_embeddings,
                    guidance_scale=guidance_scale,
                    velocity_scale=velocity_scale,  # CRITICAL FIX
                )
                
                generation_time = (datetime.now() - start_time).total_seconds()
                total_generation_time += generation_time
                
                # Ensure same shape
                if generated_clip.shape != target_clip.shape:
                    logger.warning(f"Shape mismatch in batch {batch_idx}: {generated_clip.shape} vs {target_clip.shape}")
                    continue
                
                # Compute patch-wise similarities
                batch_results = compute_patch_wise_cosine_similarity(
                    predicted_embeddings=generated_clip,
                    target_embeddings=target_clip,
                    normalize=False,  # Already normalized in generation if requested
                )
                
                # Collect similarities for global statistics
                with torch.no_grad():
                    # Use same normalization as similarity computation
                    if normalize_embeddings:
                        pred_norm = F.normalize(generated_clip, p=2, dim=-1)
                        target_norm = F.normalize(target_clip, p=2, dim=-1)
                    else:
                        pred_norm = generated_clip
                        target_norm = target_clip
                        
                    per_patch_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                    per_image_sim = per_patch_sim.mean(dim=1)
                    
                    all_per_patch_similarities.append(per_patch_sim.cpu())
                    all_per_image_similarities.append(per_image_sim.cpu())
                
                batch_count += 1
                
                if logger and batch_idx % 5 == 0:
                    logger.info(f"   Batch {batch_idx}: Overall similarity = {batch_results['overall_cosine_similarity']:.4f}, "
                              f"Generation time = {generation_time:.2f}s")
                    
            except Exception as e:
                if logger:
                    logger.warning(f"   Batch {batch_idx} failed: {e}")
                continue
    
    if not all_per_patch_similarities:
        raise RuntimeError("No batches were successfully evaluated")
    
    # Aggregate results
    all_per_patch = torch.cat(all_per_patch_similarities, dim=0)
    all_per_image = torch.cat(all_per_image_similarities, dim=0)
    
    # Final results
    final_results = {
        'overall_cosine_similarity': all_per_image.mean().item(),
        'per_image_mean_similarity': all_per_image.mean().item(),
        'per_image_std_similarity': all_per_image.std().item(),
        'per_patch_mean_similarity': all_per_patch.mean().item(),
        'per_patch_std_similarity': all_per_patch.std().item(),
        
        # Dataset statistics
        'num_images_evaluated': len(all_per_image),
        'num_batches_evaluated': batch_count,
        'patches_per_image': all_per_patch.shape[1],
        'total_patches_evaluated': all_per_patch.numel(),
        
        # Performance metrics
        'avg_generation_time_per_batch': total_generation_time / batch_count if batch_count > 0 else 0.0,
        'total_generation_time': total_generation_time,
        
        # Quality distribution
        'high_quality_patches_ratio': (all_per_patch > 0.7).float().mean().item(),
        'very_high_quality_patches_ratio': (all_per_patch > 0.8).float().mean().item(),
        'excellent_quality_patches_ratio': (all_per_patch > 0.9).float().mean().item(),
        'high_quality_images_ratio': (all_per_image > 0.7).float().mean().item(),
        'very_high_quality_images_ratio': (all_per_image > 0.8).float().mean().item(),
        'excellent_quality_images_ratio': (all_per_image > 0.9).float().mean().item(),
        
        # Min/max
        'min_patch_similarity': all_per_patch.min().item(),
        'max_patch_similarity': all_per_patch.max().item(),
        'min_image_similarity': all_per_image.min().item(),
        'max_image_similarity': all_per_image.max().item(),
        
        # Percentiles
        'patch_similarity_p25': torch.quantile(all_per_patch, 0.25).item(),
        'patch_similarity_p50': torch.quantile(all_per_patch, 0.50).item(),
        'patch_similarity_p75': torch.quantile(all_per_patch, 0.75).item(),
        'image_similarity_p25': torch.quantile(all_per_image, 0.25).item(),
        'image_similarity_p50': torch.quantile(all_per_image, 0.50).item(),
        'image_similarity_p75': torch.quantile(all_per_image, 0.75).item(),
        
        # FIXED: Generation parameters used
        'velocity_scale_used': velocity_scale,
        'guidance_scale_used': guidance_scale,
        'num_inference_steps_used': num_inference_steps,
        'normalize_embeddings_used': normalize_embeddings,
    }
    
    if logger:
        logger.info(f"✅ FIXED Evaluation completed on {batch_count} batches")
        logger.info(f"   Overall cosine similarity: {final_results['overall_cosine_similarity']:.4f}")
        logger.info(f"   High quality images (>0.7): {final_results['high_quality_images_ratio']*100:.1f}%")
        logger.info(f"   Average generation time: {final_results['avg_generation_time_per_batch']:.2f}s per batch")
        logger.info(f"   Velocity scale used: {velocity_scale}")
    
    return final_results

def main():
    """Main evaluation function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🔍 FIXED BLIP3-o Patch-wise Cosine Similarity Evaluation")
    logger.info("=" * 70)
    logger.info("🎯 EVALUATION METHODOLOGY:")
    logger.info("  1. Load FIXED model with corrected generation")
    logger.info("  2. Generate embeddings using FIXED velocity scaling")
    logger.info("  3. Compute cosine similarity for each patch")
    logger.info("  4. Average over all patches to get image similarity")
    logger.info("  5. Average over all images to get overall similarity")
    logger.info("=" * 70)
    
    try:
        # Setup device and dtype
        device = setup_device(args.device, logger)
        torch_dtype = get_torch_dtype(args.torch_dtype)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model with FIXED generation
        model, config, training_mode = load_model_and_config(
            args.model_path, device, args.training_mode, torch_dtype, args.velocity_scale, logger
        )
        
        # Create dataloader
        eval_dataloader = create_evaluation_dataloader(
            args.chunked_embeddings_dir, training_mode, args.batch_size, logger
        )
        
        # Run evaluation
        logger.info("🚀 Starting FIXED patch-wise cosine similarity evaluation...")
        start_time = datetime.now()
        
        results = evaluate_model(
            model=model,
            dataloader=eval_dataloader,
            device=str(device),
            training_mode=training_mode,
            num_inference_steps=args.num_inference_steps,
            max_batches=args.num_samples // args.batch_size if args.num_samples else None,
            normalize_embeddings=args.normalize_embeddings,
            velocity_scale=args.velocity_scale,  # CRITICAL FIX
            guidance_scale=args.guidance_scale,
            test_multiple_steps=args.test_multiple_steps,
            logger=logger
        )
        
        end_time = datetime.now()
        evaluation_duration = (end_time - start_time).total_seconds()
        
        # Display results
        logger.info("📊 FIXED PATCH-WISE COSINE SIMILARITY RESULTS:")
        logger.info("=" * 60)
        logger.info(f"🎯 OVERALL COSINE SIMILARITY: {results['overall_cosine_similarity']:.4f}")
        logger.info(f"📊 Per-image mean: {results['per_image_mean_similarity']:.4f} ± {results['per_image_std_similarity']:.4f}")
        logger.info(f"📊 Per-patch mean: {results['per_patch_mean_similarity']:.4f} ± {results['per_patch_std_similarity']:.4f}")
        logger.info(f"📈 High quality images (>0.7): {results['high_quality_images_ratio']*100:.1f}%")
        logger.info(f"📈 Very high quality images (>0.8): {results['very_high_quality_images_ratio']*100:.1f}%")
        logger.info(f"📈 Excellent quality images (>0.9): {results['excellent_quality_images_ratio']*100:.1f}%")
        logger.info(f"📈 Images evaluated: {results['num_images_evaluated']:,}")
        logger.info(f"⏱️ Average generation time: {results['avg_generation_time_per_batch']:.2f}s per batch")
        logger.info(f"🔧 Velocity scale used: {results['velocity_scale_used']}")
        
        # Assessment
        overall_sim = results['overall_cosine_similarity']
        if overall_sim > 0.9:
            logger.info("🎉 OUTSTANDING: Exceptional alignment!")
        elif overall_sim > 0.8:
            logger.info("🎉 EXCELLENT: Outstanding alignment!")
        elif overall_sim > 0.6:
            logger.info("✅ VERY GOOD: Strong performance")
        elif overall_sim > 0.4:
            logger.info("🔄 GOOD: Solid learning")
        elif overall_sim > 0.2:
            logger.info("📈 IMPROVING: Some learning detected")
        elif overall_sim > 0.05:
            logger.info("🔧 FIXED ISSUES: Improvement from fixes")
        else:
            logger.info("⚠️ STILL NEEDS WORK: Check fixes applied correctly")
        
        # Save results
        evaluation_summary = {
            'evaluation_completed': True,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': evaluation_duration,
            'model_path': str(args.model_path),
            'training_mode': training_mode,
            'torch_dtype': str(torch_dtype),
            'fixes_applied': [
                "FIXED velocity scaling in generation",
                "FIXED timestep schedule",
                "FIXED normalization consistency",
                "FIXED generation parameter passing"
            ],
            'generation_parameters': {
                'num_inference_steps': args.num_inference_steps,
                'velocity_scale': args.velocity_scale,
                'guidance_scale': args.guidance_scale,
                'normalize_embeddings': args.normalize_embeddings,
            },
            'evaluation_methodology': {
                'step_1': 'Load model with FIXED generation method',
                'step_2': 'Generate embeddings using corrected velocity scaling',
                'step_3': 'Compute cosine similarity for each patch',
                'step_4': 'Average over all patches to get image similarity',
                'step_5': 'Average over all images to get overall similarity',
                'fixes_critical': 'velocity_scale parameter must match training value',
            },
            'results_summary': {
                'overall_cosine_similarity': results['overall_cosine_similarity'],
                'per_image_mean_similarity': results['per_image_mean_similarity'],
                'per_image_std_similarity': results['per_image_std_similarity'],
                'per_patch_mean_similarity': results['per_patch_mean_similarity'],
                'per_patch_std_similarity': results['per_patch_std_similarity'],
                'high_quality_images_percentage': results['high_quality_images_ratio'] * 100,
                'very_high_quality_images_percentage': results['very_high_quality_images_ratio'] * 100,
                'excellent_quality_images_percentage': results['excellent_quality_images_ratio'] * 100,
                'total_images': results['num_images_evaluated'],
                'total_patches': results['total_patches_evaluated'],
                'avg_generation_time_per_batch': results['avg_generation_time_per_batch'],
            },
            'detailed_results': results,
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f'fixed_evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("✅ FIXED EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"⏱️ Evaluation time: {evaluation_duration:.1f} seconds")
        
        # Final assessment
        if overall_sim > 0.1:
            logger.info("🎉 SUCCESS: Fixes improved evaluation results significantly!")
        elif overall_sim > 0.05:
            logger.info("📈 PROGRESS: Some improvement, may need further tuning")
        else:
            logger.info("⚠️ ISSUE: Results still low, check model training or other fixes needed")
        
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)