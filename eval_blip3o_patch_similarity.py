#!/usr/bin/env python3
"""
EVA-CLIP Reproduction Evaluation Script
eval_eva_reproduction.py

Comprehensive evaluation of trained EVA reproduction model.
"""

import os
import sys
import argparse
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Evaluate EVA-CLIP Reproduction Model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")
    
    # Evaluation parameters
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    
    # Analysis options
    parser.add_argument("--create_plots", action="store_true", default=True,
                       help="Create visualization plots")
    parser.add_argument("--save_samples", action="store_true", default=False,
                       help="Save sample comparisons")
    parser.add_argument("--detailed_analysis", action="store_true", default=True,
                       help="Perform detailed similarity analysis")
    
    return parser.parse_args()

def load_model(model_path, device, logger):
    """Load trained EVA reproduction model"""
    from src.modules.models.blip3o_eva_dit import BLIP3oEVADiTModel, BLIP3oEVADiTConfig
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load config
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = BLIP3oEVADiTConfig(**config_dict)
    else:
        logger.warning("No config found, using default")
        config = BLIP3oEVADiTConfig()
    
    # Load model
    model = BLIP3oEVADiTModel(config)
    
    # Load weights
    model_file = Path(model_path) / "pytorch_model.bin"
    if not model_file.exists():
        model_file = Path(model_path) / "model.safetensors"
    
    if model_file.exists():
        if model_file.suffix == ".bin":
            state_dict = torch.load(model_file, map_location=device)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
        
        model.load_state_dict(state_dict)
        logger.info(f"✅ Model loaded from: {model_file}")
    else:
        logger.error(f"❌ No model weights found in: {model_path}")
        raise FileNotFoundError(f"Model weights not found")
    
    model = model.to(device)
    model.eval()
    
    return model

def create_dataloader(chunked_embeddings_dir, batch_size, training_mode, logger):
    """Create evaluation dataloader"""
    from src.modules.datasets.blip3o_eva_dataset import create_eva_reproduction_dataloaders
    
    logger.info("Creating evaluation dataloader...")
    
    _, eval_dataloader = create_eva_reproduction_dataloaders(
        chunked_embeddings_dir=chunked_embeddings_dir,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        normalize_embeddings=True,
        training_mode=training_mode,
        max_shards=1,
        use_same_data_for_eval=True,
        num_workers=0,
    )
    
    logger.info(f"✅ Evaluation dataloader created")
    return eval_dataloader

def evaluate_model(model, dataloader, num_samples, inference_steps, device, logger):
    """Comprehensive model evaluation"""
    logger.info(f"Starting comprehensive evaluation on {num_samples} samples...")
    
    all_similarities = []
    all_per_image_sims = []
    all_per_patch_sims = []
    
    # Detailed metrics
    eva_norms = []
    clip_norms = []
    generated_norms = []
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if samples_processed >= num_samples:
                break
            
            # Move to device
            clip_features = batch['encoder_hidden_states'].to(device)  # CLIP conditioning
            target_eva = batch['eva_embeddings'].to(device)            # EVA targets
            
            batch_size = clip_features.shape[0]
            
            # Generate EVA embeddings
            generated_eva = model.generate(
                clip_features=clip_features,
                num_inference_steps=inference_steps,
                normalize_output=True
            )
            
            # Normalize targets
            target_eva_norm = torch.nn.functional.normalize(target_eva, p=2, dim=-1)
            
            # Compute similarities
            per_patch_sim = torch.nn.functional.cosine_similarity(
                generated_eva, target_eva_norm, dim=-1
            )
            per_image_sim = per_patch_sim.mean(dim=1)
            
            # Store results
            all_similarities.append(per_patch_sim.cpu())
            all_per_image_sims.append(per_image_sim.cpu())
            all_per_patch_sims.extend(per_patch_sim.cpu().numpy())
            
            # Collect norms
            eva_norms.extend(torch.norm(target_eva_norm, dim=-1).cpu().numpy().flatten())
            clip_norms.extend(torch.norm(clip_features, dim=-1).cpu().numpy().flatten())
            generated_norms.extend(torch.norm(generated_eva, dim=-1).cpu().numpy().flatten())
            
            samples_processed += batch_size
    
    # Aggregate results
    all_patch_sims = torch.cat(all_similarities, dim=0)
    all_image_sims = torch.cat(all_per_image_sims, dim=0)
    
    # Compute comprehensive metrics
    results = {
        'overall_eva_similarity': all_image_sims.mean().item(),
        'per_image_mean': all_image_sims.mean().item(),
        'per_image_std': all_image_sims.std().item(),
        'per_image_median': all_image_sims.median().item(),
        'per_patch_mean': all_patch_sims.mean().item(),
        'per_patch_std': all_patch_sims.std().item(),
        'per_patch_median': all_patch_sims.median().item(),
        
        # Quality thresholds
        'high_quality_images': (all_image_sims > 0.7).float().mean().item(),
        'very_high_quality_images': (all_image_sims > 0.8).float().mean().item(),
        'excellent_quality_images': (all_image_sims > 0.9).float().mean().item(),
        
        'high_quality_patches': (all_patch_sims > 0.7).float().mean().item(),
        'very_high_quality_patches': (all_patch_sims > 0.8).float().mean().item(),
        'excellent_quality_patches': (all_patch_sims > 0.9).float().mean().item(),
        
        # Distribution statistics
        'similarity_percentiles': {
            '5th': np.percentile(all_image_sims.numpy(), 5),
            '25th': np.percentile(all_image_sims.numpy(), 25),
            '75th': np.percentile(all_image_sims.numpy(), 75),
            '95th': np.percentile(all_image_sims.numpy(), 95),
        },
        
        # Norm statistics
        'eva_target_norm_mean': np.mean(eva_norms),
        'eva_target_norm_std': np.std(eva_norms),
        'clip_conditioning_norm_mean': np.mean(clip_norms),
        'clip_conditioning_norm_std': np.std(clip_norms),
        'generated_norm_mean': np.mean(generated_norms),
        'generated_norm_std': np.std(generated_norms),
        
        # Sample counts
        'samples_evaluated': samples_processed,
        'inference_steps': inference_steps,
    }
    
    # Store raw data for plotting
    results['raw_data'] = {
        'per_image_similarities': all_image_sims.numpy(),
        'per_patch_similarities': np.array(all_per_patch_sims),
        'eva_norms': np.array(eva_norms),
        'clip_norms': np.array(clip_norms),
        'generated_norms': np.array(generated_norms),
    }
    
    logger.info("✅ Evaluation completed")
    return results

def create_plots(results, output_dir, logger):
    """Create comprehensive evaluation plots"""
    logger.info("Creating evaluation plots...")
    
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Similarity Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Per-image similarity histogram
    axes[0, 0].hist(results['raw_data']['per_image_similarities'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(results['per_image_mean'], color='red', linestyle='--', label=f'Mean: {results["per_image_mean"]:.3f}')
    axes[0, 0].axvline(results['per_image_median'], color='orange', linestyle='--', label=f'Median: {results["per_image_median"]:.3f}')
    axes[0, 0].set_xlabel('EVA Cosine Similarity (Per Image)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('EVA Similarity Distribution (Per Image)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Per-patch similarity histogram
    axes[0, 1].hist(results['raw_data']['per_patch_similarities'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(results['per_patch_mean'], color='red', linestyle='--', label=f'Mean: {results["per_patch_mean"]:.3f}')
    axes[0, 1].axvline(results['per_patch_median'], color='orange', linestyle='--', label=f'Median: {results["per_patch_median"]:.3f}')
    axes[0, 1].set_xlabel('EVA Cosine Similarity (Per Patch)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('EVA Similarity Distribution (Per Patch)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quality thresholds
    quality_data = {
        'High (>0.7)': results['high_quality_images'] * 100,
        'Very High (>0.8)': results['very_high_quality_images'] * 100,
        'Excellent (>0.9)': results['excellent_quality_images'] * 100,
    }
    
    bars = axes[1, 0].bar(quality_data.keys(), quality_data.values(), alpha=0.7)
    axes[1, 0].set_ylabel('Percentage of Images (%)')
    axes[1, 0].set_title('EVA Quality Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom')
    
    # Norm comparison
    norm_data = {
        'EVA Targets': results['eva_target_norm_mean'],
        'CLIP Conditioning': results['clip_conditioning_norm_mean'],
        'Generated EVA': results['generated_norm_mean'],
    }
    
    bars = axes[1, 1].bar(norm_data.keys(), norm_data.values(), alpha=0.7)
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', label='Expected (1.0)')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].set_title('Embedding Norms Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "eva_reproduction_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed similarity analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Scatter plot of per-image similarities
    similarities = results['raw_data']['per_image_similarities']
    indices = np.arange(len(similarities))
    
    colors = ['red' if sim < 0.5 else 'orange' if sim < 0.7 else 'green' for sim in similarities]
    ax.scatter(indices, similarities, c=colors, alpha=0.6, s=10)
    
    ax.axhline(y=0.7, color='orange', linestyle='--', label='High Quality Threshold')
    ax.axhline(y=0.8, color='green', linestyle='--', label='Very High Quality Threshold')
    ax.axhline(y=results['per_image_mean'], color='blue', linestyle='-', label=f'Mean: {results["per_image_mean"]:.3f}')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('EVA Cosine Similarity')
    ax.set_title('Per-Sample EVA Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "per_sample_similarity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Plots saved to: {plots_dir}")

def save_results(results, output_dir, logger):
    """Save evaluation results"""
    logger.info("Saving evaluation results...")
    
    # Remove raw data for JSON serialization
    results_to_save = results.copy()
    raw_data = results_to_save.pop('raw_data')
    
    # Save JSON results
    results_file = Path(output_dir) / "eva_reproduction_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save raw data as numpy
    raw_data_file = Path(output_dir) / "eva_reproduction_raw_data.npz"
    np.savez(raw_data_file, **raw_data)
    
    logger.info(f"✅ Results saved to: {results_file}")
    logger.info(f"✅ Raw data saved to: {raw_data_file}")

def print_summary(results, logger):
    """Print evaluation summary"""
    logger.info("=" * 70)
    logger.info("📊 EVA REPRODUCTION EVALUATION SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"🎯 Overall EVA Similarity: {results['overall_eva_similarity']:.4f}")
    logger.info(f"📊 Per-Image Statistics:")
    logger.info(f"   Mean: {results['per_image_mean']:.4f}")
    logger.info(f"   Std:  {results['per_image_std']:.4f}")
    logger.info(f"   Median: {results['per_image_median']:.4f}")
    
    logger.info(f"📊 Quality Distribution:")
    logger.info(f"   High Quality (>0.7):    {results['high_quality_images']*100:.1f}%")
    logger.info(f"   Very High Quality (>0.8): {results['very_high_quality_images']*100:.1f}%")
    logger.info(f"   Excellent Quality (>0.9): {results['excellent_quality_images']*100:.1f}%")
    
    logger.info(f"📏 Normalization Status:")
    logger.info(f"   EVA Targets:     {results['eva_target_norm_mean']:.3f} ± {results['eva_target_norm_std']:.3f}")
    logger.info(f"   CLIP Conditioning: {results['clip_conditioning_norm_mean']:.3f} ± {results['clip_conditioning_norm_std']:.3f}")
    logger.info(f"   Generated EVA:   {results['generated_norm_mean']:.3f} ± {results['generated_norm_std']:.3f}")
    
    logger.info(f"📈 Evaluation Details:")
    logger.info(f"   Samples: {results['samples_evaluated']:,}")
    logger.info(f"   Inference Steps: {results['inference_steps']}")
    
    # Assessment
    overall_sim = results['overall_eva_similarity']
    if overall_sim > 0.8:
        logger.info("🎉 EXCELLENT: Outstanding EVA reproduction! DiT architecture works perfectly!")
    elif overall_sim > 0.6:
        logger.info("✅ GOOD: Strong EVA reproduction! DiT architecture is working well!")
    elif overall_sim > 0.4:
        logger.info("📈 FAIR: Decent EVA reproduction! DiT shows learning capability!")
    elif overall_sim > 0.2:
        logger.info("⚠️ POOR: Low EVA reproduction! DiT needs improvement!")
    else:
        logger.info("❌ FAILED: Very low EVA reproduction! Check DiT implementation!")
    
    logger.info("=" * 70)

def main():
    """Main evaluation function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🔍 Starting EVA-CLIP Reproduction Evaluation")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.chunked_embeddings_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info("=" * 50)
    
    try:
        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model = load_model(args.model_path, device, logger)
        
        # Create dataloader
        dataloader = create_dataloader(
            args.chunked_embeddings_dir, 
            args.batch_size, 
            args.training_mode, 
            logger
        )
        
        # Evaluate model
        results = evaluate_model(
            model, dataloader, args.num_samples, 
            args.inference_steps, device, logger
        )
        
        # Create plots
        if args.create_plots:
            create_plots(results, args.output_dir, logger)
        
        # Save results
        save_results(results, args.output_dir, logger)
        
        # Print summary
        print_summary(results, logger)
        
        logger.info("✅ EVA reproduction evaluation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)