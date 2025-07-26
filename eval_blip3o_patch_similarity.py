#!/usr/bin/env python3
"""
EVA-CLIP Reproduction Evaluation Script
Comprehensive evaluation of trained models
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
from tqdm import tqdm
import torch.nn.functional as F

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
                       help="Path to trained model checkpoint")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                       help="Path to embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")
    
    # Evaluation parameters
    parser.add_argument("--num_samples", type=int, default=2000,
                       help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--training_mode", type=str, default="patch_only",
                       choices=["patch_only", "cls_patch"],
                       help="Training mode")
    
    # Analysis options
    parser.add_argument("--save_visualizations", action="store_true", default=True,
                       help="Save visualization plots")
    parser.add_argument("--detailed_analysis", action="store_true", default=True,
                       help="Perform detailed analysis")
    
    return parser.parse_args()

def load_model(model_path, device, logger):
    """Load trained model"""
    logger.info(f"Loading model from: {model_path}")
    
    try:
        from fixed_model import create_eva_reproduction_model
    except ImportError:
        logger.error("Could not import model. Make sure fixed_model.py is available.")
        raise
    
    model_path = Path(model_path)
    
    # Load checkpoint
    if model_path.is_file() and model_path.suffix == '.pt':
        # Direct checkpoint file
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model (you may need to adjust config based on checkpoint)
        model = create_eva_reproduction_model(model_size="base")  # Adjust as needed
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
        
    elif model_path.is_dir():
        # Model directory - look for latest checkpoint
        checkpoints = list(model_path.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {model_path}")
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        
        model = create_eva_reproduction_model(model_size="base")  # Adjust as needed
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded latest checkpoint: {latest_checkpoint}")
        
    else:
        raise ValueError(f"Invalid model path: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded with {model.get_num_parameters():,} parameters")
    return model

def create_evaluation_dataloader(embeddings_dir, batch_size, training_mode, logger):
    """Create evaluation dataloader"""
    try:
        from fixed_dataset import create_eva_reproduction_dataloaders
    except ImportError:
        logger.error("Could not import dataset. Make sure fixed_dataset.py is available.")
        raise
    
    logger.info("Creating evaluation dataloader...")
    
    _, eval_dataloader = create_eva_reproduction_dataloaders(
        chunked_embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        training_mode=training_mode,
        max_shards=None,  # Use all available shards for evaluation
        normalize_embeddings=True,
        num_workers=0
    )
    
    logger.info("Evaluation dataloader created")
    return eval_dataloader

def evaluate_model(model, dataloader, num_samples, inference_steps, device, logger):
    """Comprehensive model evaluation"""
    logger.info(f"Starting evaluation on {num_samples} samples...")
    
    all_similarities = []
    all_per_image_sims = []
    generated_norms = []
    target_norms = []
    
    samples_processed = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            if samples_processed >= num_samples:
                break
            
            try:
                # Move to device
                clip_features = batch['encoder_hidden_states'].to(device)
                target_eva = batch['eva_embeddings'].to(device)
                
                batch_size = clip_features.shape[0]
                
                # Generate EVA embeddings
                generated_eva = model.generate(
                    clip_features=clip_features,
                    num_inference_steps=inference_steps,
                    normalize_output=True
                )
                
                # Normalize targets for fair comparison
                target_eva_norm = F.normalize(target_eva, p=2, dim=-1)
                
                # Compute similarities
                per_patch_sim = F.cosine_similarity(generated_eva, target_eva_norm, dim=-1)
                per_image_sim = per_patch_sim.mean(dim=1)
                
                # Store results
                all_similarities.append(per_patch_sim.cpu())
                all_per_image_sims.append(per_image_sim.cpu())
                
                # Collect norms
                generated_norms.extend(torch.norm(generated_eva, dim=-1).cpu().numpy().flatten())
                target_norms.extend(torch.norm(target_eva_norm, dim=-1).cpu().numpy().flatten())
                
                samples_processed += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'samples': samples_processed,
                    'avg_sim': per_image_sim.mean().item()
                })
                
            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
                continue
    
    if not all_similarities:
        raise RuntimeError("No successful evaluations")
    
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
        'generated_norm_mean': np.mean(generated_norms),
        'generated_norm_std': np.std(generated_norms),
        'target_norm_mean': np.mean(target_norms),
        'target_norm_std': np.std(target_norms),
        
        # Sample counts
        'samples_evaluated': samples_processed,
        'inference_steps': inference_steps,
        
        # Raw data for plotting
        'raw_data': {
            'per_image_similarities': all_image_sims.numpy(),
            'per_patch_similarities': all_patch_sims.numpy(),
            'generated_norms': np.array(generated_norms),
            'target_norms': np.array(target_norms),
        }
    }
    
    logger.info("Evaluation completed successfully")
    return results

def create_visualizations(results, output_dir, logger):
    """Create comprehensive visualizations"""
    logger.info("Creating visualizations...")
    
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # 1. Main evaluation plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Per-image similarity histogram
    per_image_sims = results['raw_data']['per_image_similarities']
    axes[0, 0].hist(per_image_sims, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].axvline(results['per_image_mean'], color='red', linestyle='--', 
                      label=f'Mean: {results["per_image_mean"]:.3f}')
    axes[0, 0].axvline(results['per_image_median'], color='orange', linestyle='--', 
                      label=f'Median: {results["per_image_median"]:.3f}')
    axes[0, 0].set_xlabel('EVA Cosine Similarity (Per Image)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('EVA Similarity Distribution (Per Image)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Per-patch similarity histogram
    per_patch_sims = results['raw_data']['per_patch_similarities']
    axes[0, 1].hist(per_patch_sims, bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 1].axvline(results['per_patch_mean'], color='red', linestyle='--', 
                      label=f'Mean: {results["per_patch_mean"]:.3f}')
    axes[0, 1].axvline(results['per_patch_median'], color='orange', linestyle='--', 
                      label=f'Median: {results["per_patch_median"]:.3f}')
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
    
    bars = axes[1, 0].bar(quality_data.keys(), quality_data.values(), 
                         alpha=0.7, color=['orange', 'green', 'gold'])
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
        'Generated': results['generated_norm_mean'],
        'Target': results['target_norm_mean'],
    }
    
    bars = axes[1, 1].bar(norm_data.keys(), norm_data.values(), 
                         alpha=0.7, color=['purple', 'cyan'])
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
    
    # 2. Detailed similarity scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    indices = np.arange(len(per_image_sims))
    colors = ['red' if sim < 0.5 else 'orange' if sim < 0.7 else 'green' for sim in per_image_sims]
    
    ax.scatter(indices, per_image_sims, c=colors, alpha=0.6, s=10)
    ax.axhline(y=0.7, color='orange', linestyle='--', label='High Quality Threshold')
    ax.axhline(y=0.8, color='green', linestyle='--', label='Very High Quality Threshold')
    ax.axhline(y=results['per_image_mean'], color='blue', linestyle='-', 
              label=f'Mean: {results["per_image_mean"]:.3f}')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('EVA Cosine Similarity')
    ax.set_title('Per-Sample EVA Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "per_sample_similarity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to: {plots_dir}")

def save_results(results, output_dir, logger):
    """Save evaluation results"""
    logger.info("Saving evaluation results...")
    
    # Remove raw data for JSON serialization
    results_to_save = results.copy()
    raw_data = results_to_save.pop('raw_data')
    
    # Save JSON results
    results_file = Path(output_dir) / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save raw data as numpy
    raw_data_file = Path(output_dir) / "evaluation_raw_data.npz"
    np.savez(raw_data_file, **raw_data)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Raw data saved to: {raw_data_file}")

def print_evaluation_summary(results, logger):
    """Print comprehensive evaluation summary"""
    logger.info("=" * 80)
    logger.info("📊 EVA REPRODUCTION EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"🎯 Overall EVA Similarity: {results['overall_eva_similarity']:.4f}")
    
    logger.info(f"📊 Per-Image Statistics:")
    logger.info(f"   Mean: {results['per_image_mean']:.4f}")
    logger.info(f"   Std:  {results['per_image_std']:.4f}")
    logger.info(f"   Median: {results['per_image_median']:.4f}")
    
    logger.info(f"📊 Quality Distribution (Images):")
    logger.info(f"   High Quality (>0.7):    {results['high_quality_images']*100:.1f}%")
    logger.info(f"   Very High Quality (>0.8): {results['very_high_quality_images']*100:.1f}%")
    logger.info(f"   Excellent Quality (>0.9): {results['excellent_quality_images']*100:.1f}%")
    
    logger.info(f"📊 Quality Distribution (Patches):")
    logger.info(f"   High Quality (>0.7):    {results['high_quality_patches']*100:.1f}%")
    logger.info(f"   Very High Quality (>0.8): {results['very_high_quality_patches']*100:.1f}%")
    logger.info(f"   Excellent Quality (>0.9): {results['excellent_quality_patches']*100:.1f}%")
    
    logger.info(f"📏 Normalization Status:")
    logger.info(f"   Generated Norms: {results['generated_norm_mean']:.3f} ± {results['generated_norm_std']:.3f}")
    logger.info(f"   Target Norms:    {results['target_norm_mean']:.3f} ± {results['target_norm_std']:.3f}")
    
    logger.info(f"📈 Evaluation Details:")
    logger.info(f"   Samples: {results['samples_evaluated']:,}")
    logger.info(f"   Inference Steps: {results['inference_steps']}")
    
    # Assessment
    overall_sim = results['overall_eva_similarity']
    if overall_sim > 0.8:
        logger.info("🎉 EXCELLENT: Outstanding EVA reproduction! Model works perfectly!")
    elif overall_sim > 0.6:
        logger.info("✅ GOOD: Strong EVA reproduction! Model is working well!")
    elif overall_sim > 0.4:
        logger.info("📈 FAIR: Decent EVA reproduction! Model shows capability!")
    elif overall_sim > 0.2:
        logger.info("⚠️ POOR: Low EVA reproduction! Model needs improvement!")
    else:
        logger.info("❌ FAILED: Very low EVA reproduction! Check implementation!")
    
    logger.info("=" * 80)

def main():
    """Main evaluation function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🔍 Starting EVA-CLIP Reproduction Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.embeddings_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info(f"Inference steps: {args.inference_steps}")
    logger.info("=" * 60)
    
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
        dataloader = create_evaluation_dataloader(
            args.embeddings_dir, 
            args.batch_size, 
            args.training_mode, 
            logger
        )
        
        # Evaluate model
        results = evaluate_model(
            model, dataloader, args.num_samples, 
            args.inference_steps, device, logger
        )
        
        # Create visualizations
        if args.save_visualizations:
            create_visualizations(results, args.output_dir, logger)
        
        # Save results
        save_results(results, args.output_dir, logger)
        
        # Print summary
        print_evaluation_summary(results, logger)
        
        # Save evaluation metadata
        eval_metadata = {
            'evaluation_completed': True,
            'timestamp': datetime.now().isoformat(),
            'args': vars(args),
            'device': str(device),
            'model_parameters': model.get_num_parameters(),
        }
        
        metadata_file = output_dir / 'evaluation_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(eval_metadata, f, indent=2)
        
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