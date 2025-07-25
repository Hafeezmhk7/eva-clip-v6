#!/usr/bin/env python3
"""
FIXED: BLIP3-o Evaluation Script - Patch-wise Cosine Similarity
eval_blip3o_patch_similarity.py

Evaluates trained DiT model by computing:
1. Per-patch cosine similarity
2. Per-image average cosine similarity
3. Global average cosine similarity
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
        description="BLIP3-o Patch-wise Cosine Similarity Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
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
    
    # Options
    parser.add_argument("--same_data_eval", action="store_true", default=True,
                       help="Use same training data for evaluation")
    parser.add_argument("--normalize_embeddings", action="store_true", default=True,
                       help="Normalize embeddings")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
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

def load_model_and_config(model_path, device, training_mode, logger):
    """Load model and determine training mode"""
    from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
    
    model_path = Path(model_path)
    logger.info(f"📦 Loading model from: {model_path}")
    
    # Load configuration
    config_files = [
        model_path / "config.json",
        model_path / "blip3o_model_config.json",
        model_path / "training_info.json"
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
    
    # Determine training mode
    if training_mode == "auto":
        if 'training_mode' in config_data:
            training_mode = config_data['training_mode']
        elif 'num_tokens' in config_data:
            training_mode = "cls_patch" if config_data['num_tokens'] == 257 else "patch_only"
        else:
            training_mode = "patch_only"  # Default
        logger.info(f"🎯 Auto-detected training mode: {training_mode}")
    
    # Create config
    expected_tokens = 257 if training_mode == "cls_patch" else 256
    if 'num_tokens' not in config_data:
        config_data['num_tokens'] = expected_tokens
    
    config = BLIP3oDiTConfig(**config_data)
    
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
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
    
    # Move to device
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Expected tokens: {expected_tokens}")
    
    return model, config, training_mode

def create_evaluation_dataloader(embeddings_dir, training_mode, batch_size, logger):
    """Create evaluation dataloader"""
    from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
    
    logger.info(f"📊 Creating evaluation dataloader")
    
    # Use same data for evaluation (overfitting test)
    train_dataloader, _ = create_flexible_dataloaders(
        chunked_embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        eval_split_ratio=0.0,
        normalize_embeddings=False,
        training_mode=training_mode,
        max_shards=1,  # Single shard
        use_same_data_for_eval=True,
        delete_after_use=False,
        num_workers=0,
        pin_memory=False,
    )
    
    logger.info(f"✅ Evaluation dataloader created")
    return train_dataloader

def compute_patch_wise_cosine_similarity(
    predicted_embeddings: torch.Tensor,  # [B, N, 1024]
    target_embeddings: torch.Tensor,     # [B, N, 1024]
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute patch-wise cosine similarity:
    1. Cosine similarity for each patch
    2. Average over all patches to get image similarity  
    3. Average over all images to get overall similarity
    """
    batch_size, num_tokens, embed_dim = predicted_embeddings.shape
    
    # Validate inputs
    assert predicted_embeddings.shape == target_embeddings.shape
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
    logger = None
) -> Dict[str, float]:
    """Evaluate model with patch-wise similarity"""
    model.eval()
    
    all_per_patch_similarities = []
    all_per_image_similarities = []
    batch_count = 0
    
    if logger:
        logger.info(f"🔍 Starting evaluation...")
        logger.info(f"   Training mode: {training_mode}")
        logger.info(f"   Max batches: {max_batches or 'All'}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            # Move to device
            eva_embeddings = batch['encoder_hidden_states'].to(device)
            target_clip = batch['clip_embeddings'].to(device)
            
            try:
                # Generate embeddings
                generated_clip = model.generate(
                    eva_features=eva_embeddings,
                    num_inference_steps=num_inference_steps,
                    normalize_output=normalize_embeddings,
                    guidance_scale=1.0,
                )
                
                # Ensure same shape
                if generated_clip.shape != target_clip.shape:
                    logger.warning(f"Shape mismatch: {generated_clip.shape} vs {target_clip.shape}")
                    continue
                
                # Compute patch-wise similarities
                batch_results = compute_patch_wise_cosine_similarity(
                    predicted_embeddings=generated_clip,
                    target_embeddings=target_clip,
                    normalize=normalize_embeddings,
                )
                
                # Collect similarities
                with torch.no_grad():
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
                    logger.info(f"   Batch {batch_idx}: Overall similarity = {batch_results['overall_cosine_similarity']:.4f}")
                    
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
        
        # Quality distribution
        'high_quality_patches_ratio': (all_per_patch > 0.7).float().mean().item(),
        'very_high_quality_patches_ratio': (all_per_patch > 0.8).float().mean().item(),
        'high_quality_images_ratio': (all_per_image > 0.7).float().mean().item(),
        'very_high_quality_images_ratio': (all_per_image > 0.8).float().mean().item(),
        
        # Min/max
        'min_patch_similarity': all_per_patch.min().item(),
        'max_patch_similarity': all_per_patch.max().item(),
        'min_image_similarity': all_per_image.min().item(),
        'max_image_similarity': all_per_image.max().item(),
    }
    
    if logger:
        logger.info(f"✅ Evaluation completed on {batch_count} batches")
        logger.info(f"   Overall cosine similarity: {final_results['overall_cosine_similarity']:.4f}")
        logger.info(f"   High quality images (>0.7): {final_results['high_quality_images_ratio']*100:.1f}%")
    
    return final_results

def main():
    """Main evaluation function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🔍 BLIP3-o Patch-wise Cosine Similarity Evaluation")
    logger.info("=" * 60)
    logger.info("🎯 EVALUATION METHODOLOGY:")
    logger.info("  1. Compute cosine similarity for each patch")
    logger.info("  2. Average over all patches to get image similarity")
    logger.info("  3. Average over all images to get overall similarity")
    logger.info("=" * 60)
    
    try:
        # Setup device
        device = setup_device(args.device, logger)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model, config, training_mode = load_model_and_config(
            args.model_path, device, args.training_mode, logger
        )
        
        # Create dataloader
        eval_dataloader = create_evaluation_dataloader(
            args.chunked_embeddings_dir, training_mode, args.batch_size, logger
        )
        
        # Run evaluation
        logger.info("🚀 Starting patch-wise cosine similarity evaluation...")
        start_time = datetime.now()
        
        results = evaluate_model(
            model=model,
            dataloader=eval_dataloader,
            device=str(device),
            training_mode=training_mode,
            num_inference_steps=args.num_inference_steps,
            max_batches=args.num_samples // args.batch_size if args.num_samples else None,
            normalize_embeddings=args.normalize_embeddings,
            logger=logger
        )
        
        end_time = datetime.now()
        evaluation_duration = (end_time - start_time).total_seconds()
        
        # Display results
        logger.info("📊 PATCH-WISE COSINE SIMILARITY RESULTS:")
        logger.info("=" * 50)
        logger.info(f"🎯 OVERALL COSINE SIMILARITY: {results['overall_cosine_similarity']:.4f}")
        logger.info(f"📊 Per-image mean: {results['per_image_mean_similarity']:.4f}")
        logger.info(f"📊 Per-patch mean: {results['per_patch_mean_similarity']:.4f}")
        logger.info(f"📈 High quality images (>0.7): {results['high_quality_images_ratio']*100:.1f}%")
        logger.info(f"📈 Images evaluated: {results['num_images_evaluated']:,}")
        
        # Assessment
        overall_sim = results['overall_cosine_similarity']
        if overall_sim > 0.8:
            logger.info("🎉 EXCELLENT: Outstanding alignment!")
        elif overall_sim > 0.6:
            logger.info("✅ VERY GOOD: Strong performance")
        elif overall_sim > 0.4:
            logger.info("🔄 GOOD: Solid learning")
        elif overall_sim > 0.2:
            logger.info("📈 IMPROVING: Some learning detected")
        else:
            logger.info("⚠️ NEEDS IMPROVEMENT: Low similarity")
        
        # Save results
        evaluation_summary = {
            'evaluation_completed': True,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': evaluation_duration,
            'model_path': str(args.model_path),
            'training_mode': training_mode,
            'evaluation_methodology': {
                'step_1': 'Compute cosine similarity for each patch',
                'step_2': 'Average over all patches to get image similarity',
                'step_3': 'Average over all images to get overall similarity',
                'normalize_embeddings': args.normalize_embeddings,
            },
            'results_summary': {
                'overall_cosine_similarity': results['overall_cosine_similarity'],
                'per_image_mean_similarity': results['per_image_mean_similarity'],
                'per_patch_mean_similarity': results['per_patch_mean_similarity'],
                'high_quality_images_percentage': results['high_quality_images_ratio'] * 100,
                'total_images': results['num_images_evaluated'],
                'total_patches': results['total_patches_evaluated'],
            },
            'detailed_results': results,
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f'evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"⏱️ Evaluation time: {evaluation_duration:.1f} seconds")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)