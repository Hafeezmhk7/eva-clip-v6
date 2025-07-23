"""
Detailed BLIP3-o Evaluator with Comprehensive Cosine Similarity Analysis and Plotting
src/modules/evaluation/blip3o_detailed_evaluator.py

Features:
1. Per-patch cosine similarity for each image
2. Per-image average cosine similarity
3. Global average cosine similarity
4. Detailed JSON reporting
5. Comprehensive plots and visualizations
6. Support for both CLS+patch and patch-only modes
7. Same-data evaluation (overfitting test)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from tqdm import tqdm
from pathlib import Path
import json
import time
from datetime import datetime
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


class BLIP3oDetailedEvaluator:
    """
    Comprehensive BLIP3-o Evaluator with detailed cosine similarity analysis
    """
    
    def __init__(
        self,
        model,
        device: str = "auto",
        training_mode: str = "cls_patch",
        num_inference_steps: int = 50,
        normalize_embeddings: bool = True,
    ):
        self.model = model
        self.device = self._setup_device(device)
        self.training_mode = training_mode
        self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        self.num_inference_steps = num_inference_steps
        self.normalize_embeddings = normalize_embeddings
        
        # Move model to device and set eval mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✅ Detailed BLIP3-o Evaluator initialized")
        logger.info(f"🎯 Training mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"📊 Inference steps: {self.num_inference_steps}")
        logger.info(f"🔧 Normalize embeddings: {self.normalize_embeddings}")
    
    def _setup_device(self, device_arg: str) -> torch.device:
        """Setup computation device"""
        if device_arg == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(device_arg)
        return device
    
    def evaluate_detailed_cosine_similarity(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        save_dir: Optional[str] = None,
        save_plots: bool = True,
        plot_distribution: bool = True,
        plot_per_image: bool = True,
        plot_heatmap: bool = True,
        same_data_eval: bool = False,
    ) -> Dict[str, Any]:
        """
        Comprehensive cosine similarity evaluation with detailed analysis
        
        Args:
            dataloader: DataLoader with EVA and CLIP embeddings
            max_batches: Maximum number of batches to evaluate (None for all)
            save_dir: Directory to save results and plots
            save_plots: Whether to save plots
            plot_distribution: Whether to plot similarity distributions
            plot_per_image: Whether to plot per-image analysis
            plot_heatmap: Whether to plot similarity heatmaps
            same_data_eval: Whether this is same-data evaluation (overfitting test)
            
        Returns:
            Comprehensive evaluation results
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔍 Starting detailed cosine similarity evaluation")
        logger.info(f"📊 Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"🔄 Same data eval: {same_data_eval}")
        logger.info(f"🎯 Max batches: {max_batches or 'All'}")
        
        # Storage for all results
        all_patch_similarities = []      # List of [num_patches] arrays for each image
        all_image_averages = []          # List of scalar averages for each image
        all_generated_embeddings = []    # All generated embeddings
        all_target_embeddings = []       # All target embeddings
        batch_info = []                  # Metadata for each batch
        
        total_patches = 0
        total_images = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if max_batches and batch_idx >= max_batches:
                    break
                
                try:
                    # Extract inputs
                    eva_embeddings = batch['encoder_hidden_states'].to(self.device)  # [B, N, 4096]
                    clip_targets = batch['clip_embeddings'].to(self.device)          # [B, N, 1024]
                    
                    batch_size, seq_len, eva_dim = eva_embeddings.shape
                    _, clip_seq_len, clip_dim = clip_targets.shape
                    
                    # Validate shapes
                    if seq_len != self.expected_tokens or clip_seq_len != self.expected_tokens:
                        logger.warning(f"Token count mismatch: expected {self.expected_tokens}, "
                                     f"got EVA={seq_len}, CLIP={clip_seq_len}")
                        continue
                    
                    # Generate embeddings
                    generated = self.model.generate(
                        eva_features=eva_embeddings,
                        num_inference_steps=self.num_inference_steps,
                        return_intermediate=False
                    )  # [B, N, 1024]
                    
                    # Normalize if requested
                    if self.normalize_embeddings:
                        generated = F.normalize(generated, p=2, dim=-1)
                        clip_targets = F.normalize(clip_targets, p=2, dim=-1)
                    
                    # Compute per-patch cosine similarities
                    # Reshape to [B*N, 1024] for batch cosine similarity
                    generated_flat = generated.view(-1, clip_dim)      # [B*N, 1024]
                    targets_flat = clip_targets.view(-1, clip_dim)     # [B*N, 1024]
                    
                    # Per-patch similarities
                    patch_similarities = F.cosine_similarity(
                        generated_flat, targets_flat, dim=-1
                    )  # [B*N]
                    
                    # Reshape back to [B, N]
                    patch_similarities = patch_similarities.view(batch_size, seq_len)
                    
                    # Per-image averages
                    image_averages = patch_similarities.mean(dim=1)  # [B]
                    
                    # Store results
                    for i in range(batch_size):
                        all_patch_similarities.append(patch_similarities[i].cpu().numpy())
                        all_image_averages.append(image_averages[i].cpu().item())
                    
                    all_generated_embeddings.append(generated.cpu())
                    all_target_embeddings.append(clip_targets.cpu())
                    
                    # Batch metadata
                    batch_info.append({
                        'batch_idx': batch_idx,
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'mean_patch_similarity': patch_similarities.mean().item(),
                        'mean_image_average': image_averages.mean().item(),
                        'std_image_average': image_averages.std().item(),
                        'min_patch_similarity': patch_similarities.min().item(),
                        'max_patch_similarity': patch_similarities.max().item(),
                    })
                    
                    total_patches += batch_size * seq_len
                    total_images += batch_size
                    
                    # Progress logging
                    if batch_idx % 10 == 0:
                        current_avg = np.mean(all_image_averages)
                        logger.info(f"Batch {batch_idx}: Current avg similarity = {current_avg:.4f}")
                
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Compute comprehensive statistics
        logger.info("📊 Computing comprehensive statistics...")
        
        # Convert to numpy arrays
        all_patch_similarities_flat = np.concatenate(all_patch_similarities)
        all_image_averages = np.array(all_image_averages)
        
        # Compute detailed statistics
        results = self._compute_detailed_statistics(
            all_patch_similarities,
            all_patch_similarities_flat,
            all_image_averages,
            batch_info,
            total_images,
            total_patches,
            same_data_eval
        )
        
        # Save results
        if save_dir:
            self._save_detailed_results(results, save_path, same_data_eval)
        
        # Generate plots
        if save_plots and save_dir:
            self._generate_comprehensive_plots(
                results,
                all_patch_similarities,
                all_image_averages,
                save_path,
                plot_distribution,
                plot_per_image,
                plot_heatmap,
                same_data_eval
            )
        
        return results
    
    def _compute_detailed_statistics(
        self,
        all_patch_similarities: List[np.ndarray],
        all_patch_similarities_flat: np.ndarray,
        all_image_averages: np.ndarray,
        batch_info: List[Dict],
        total_images: int,
        total_patches: int,
        same_data_eval: bool
    ) -> Dict[str, Any]:
        """Compute comprehensive statistics"""
        
        # Global statistics
        global_mean = float(np.mean(all_patch_similarities_flat))
        global_std = float(np.std(all_patch_similarities_flat))
        global_median = float(np.median(all_patch_similarities_flat))
        global_min = float(np.min(all_patch_similarities_flat))
        global_max = float(np.max(all_patch_similarities_flat))
        
        # Per-image statistics
        image_mean = float(np.mean(all_image_averages))
        image_std = float(np.std(all_image_averages))
        image_median = float(np.median(all_image_averages))
        image_min = float(np.min(all_image_averages))
        image_max = float(np.max(all_image_averages))
        
        # Quality thresholds
        high_quality_patches = float(np.mean(all_patch_similarities_flat > 0.7))
        very_high_quality_patches = float(np.mean(all_patch_similarities_flat > 0.8))
        excellent_quality_patches = float(np.mean(all_patch_similarities_flat > 0.9))
        
        high_quality_images = float(np.mean(all_image_averages > 0.7))
        very_high_quality_images = float(np.mean(all_image_averages > 0.8))
        excellent_quality_images = float(np.mean(all_image_averages > 0.9))
        
        # Distribution analysis
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        patch_percentiles = {f'p{p}': float(np.percentile(all_patch_similarities_flat, p)) 
                           for p in percentiles}
        image_percentiles = {f'p{p}': float(np.percentile(all_image_averages, p)) 
                           for p in percentiles}
        
        # Mode-specific analysis
        mode_analysis = {}
        if self.training_mode == "cls_patch" and len(all_patch_similarities) > 0:
            # CLS+Patch mode: analyze CLS vs patches separately
            cls_similarities = [patches[0] for patches in all_patch_similarities]  # First token (CLS)
            patch_similarities = [patches[1:] for patches in all_patch_similarities]  # Rest (patches)
            
            mode_analysis = {
                'cls_token_stats': {
                    'mean': float(np.mean(cls_similarities)),
                    'std': float(np.std(cls_similarities)),
                    'min': float(np.min(cls_similarities)),
                    'max': float(np.max(cls_similarities)),
                },
                'patch_only_stats': {
                    'mean': float(np.mean(np.concatenate(patch_similarities))),
                    'std': float(np.std(np.concatenate(patch_similarities))),
                    'min': float(np.min(np.concatenate(patch_similarities))),
                    'max': float(np.max(np.concatenate(patch_similarities))),
                },
                'cls_vs_patch_correlation': float(np.corrcoef(
                    cls_similarities,
                    [np.mean(patches) for patches in patch_similarities]
                )[0, 1])
            }
        
        # Training quality assessment
        quality_assessment = self._assess_training_quality(
            global_mean, image_mean, high_quality_images, same_data_eval
        )
        
        # Comprehensive results
        results = {
            # Metadata
            'evaluation_timestamp': datetime.now().isoformat(),
            'training_mode': self.training_mode,
            'expected_tokens': self.expected_tokens,
            'same_data_evaluation': same_data_eval,
            'total_images': total_images,
            'total_patches': total_patches,
            'num_inference_steps': self.num_inference_steps,
            
            # Global patch-level statistics
            'global_patch_statistics': {
                'mean_cosine_similarity': global_mean,
                'std_cosine_similarity': global_std,
                'median_cosine_similarity': global_median,
                'min_cosine_similarity': global_min,
                'max_cosine_similarity': global_max,
                'percentiles': patch_percentiles,
            },
            
            # Per-image statistics
            'per_image_statistics': {
                'mean_cosine_similarity': image_mean,
                'std_cosine_similarity': image_std,
                'median_cosine_similarity': image_median,
                'min_cosine_similarity': image_min,
                'max_cosine_similarity': image_max,
                'percentiles': image_percentiles,
            },
            
            # Quality metrics
            'quality_metrics': {
                'high_quality_patches_ratio': high_quality_patches,
                'very_high_quality_patches_ratio': very_high_quality_patches,
                'excellent_quality_patches_ratio': excellent_quality_patches,
                'high_quality_images_ratio': high_quality_images,
                'very_high_quality_images_ratio': very_high_quality_images,
                'excellent_quality_images_ratio': excellent_quality_images,
            },
            
            # Mode-specific analysis
            'mode_specific_analysis': mode_analysis,
            
            # Training quality assessment
            'quality_assessment': quality_assessment,
            
            # Batch information
            'batch_statistics': {
                'num_batches': len(batch_info),
                'batch_info': batch_info,
                'mean_batch_similarity': float(np.mean([b['mean_patch_similarity'] for b in batch_info])),
                'std_batch_similarity': float(np.std([b['mean_patch_similarity'] for b in batch_info])),
            },
            
            # Raw data for further analysis (first 1000 images only to save space)
            'sample_data': {
                'per_image_averages': all_image_averages[:1000].tolist(),
                'sample_patch_similarities': [patches.tolist() for patches in all_patch_similarities[:100]],
            }
        }
        
        return results
    
    def _assess_training_quality(
        self, 
        global_mean: float, 
        image_mean: float, 
        high_quality_ratio: float,
        same_data_eval: bool
    ) -> Dict[str, Any]:
        """Assess overall training quality"""
        
        # Quality thresholds
        if same_data_eval:
            # For same-data evaluation, we expect higher similarity (overfitting)
            thresholds = {
                'excellent': 0.85,
                'very_good': 0.75,
                'good': 0.65,
                'fair': 0.50,
                'poor': 0.35
            }
        else:
            # For general evaluation
            thresholds = {
                'excellent': 0.75,
                'very_good': 0.65,
                'good': 0.55,
                'fair': 0.45,
                'poor': 0.35
            }
        
        # Determine quality level
        if image_mean >= thresholds['excellent']:
            quality_level = 'excellent'
            quality_message = "🎉 EXCELLENT: Outstanding alignment!"
        elif image_mean >= thresholds['very_good']:
            quality_level = 'very_good'
            quality_message = "✅ VERY GOOD: Strong performance"
        elif image_mean >= thresholds['good']:
            quality_level = 'good'
            quality_message = "👍 GOOD: Solid performance"
        elif image_mean >= thresholds['fair']:
            quality_level = 'fair'
            quality_message = "🔄 FAIR: Reasonable progress"
        elif image_mean >= thresholds['poor']:
            quality_level = 'poor'
            quality_message = "⚠️ POOR: Needs improvement"
        else:
            quality_level = 'very_poor'
            quality_message = "❌ VERY POOR: Significant issues"
        
        # Additional assessments
        overfitting_indicator = None
        if same_data_eval:
            overfitting_indicator = image_mean > 0.8
            if overfitting_indicator:
                quality_message += " (Good overfitting detected!)"
        
        # Consistency assessment
        consistency_good = high_quality_ratio > 0.5
        
        return {
            'quality_level': quality_level,
            'quality_message': quality_message,
            'global_mean_similarity': global_mean,
            'image_mean_similarity': image_mean,
            'high_quality_ratio': high_quality_ratio,
            'consistency_good': consistency_good,
            'overfitting_indicator': overfitting_indicator,
            'same_data_evaluation': same_data_eval,
            'training_mode': self.training_mode,
            'recommendations': self._generate_recommendations(
                quality_level, image_mean, high_quality_ratio, same_data_eval
            )
        }
    
    def _generate_recommendations(
        self, 
        quality_level: str, 
        image_mean: float, 
        high_quality_ratio: float,
        same_data_eval: bool
    ) -> List[str]:
        """Generate training recommendations based on results"""
        recommendations = []
        
        if same_data_eval:
            if image_mean < 0.7:
                recommendations.append("Low same-data performance suggests training issues")
                recommendations.append("Check gradient flow and loss computation")
                recommendations.append("Verify data loading and model architecture")
            elif image_mean > 0.9:
                recommendations.append("Excellent overfitting - model can learn the data")
                recommendations.append("Ready for training on larger dataset")
            else:
                recommendations.append("Good same-data performance - continue training")
        else:
            if quality_level in ['poor', 'very_poor']:
                recommendations.append("Consider longer training or higher learning rate")
                recommendations.append("Check data quality and model architecture")
                recommendations.append("Verify loss function implementation")
            elif quality_level == 'fair':
                recommendations.append("Continue training - model is learning")
                recommendations.append("Consider adjusting hyperparameters")
            elif quality_level in ['good', 'very_good']:
                recommendations.append("Good progress - continue current strategy")
                recommendations.append("Consider evaluation on larger dataset")
            else:  # excellent
                recommendations.append("Excellent performance achieved!")
                recommendations.append("Model ready for downstream applications")
        
        # Mode-specific recommendations
        if self.training_mode == "cls_patch":
            recommendations.append("CLS+patch mode: Monitor CLS vs patch performance")
        else:
            recommendations.append("Patch-only mode: Consider trying CLS+patch for comparison")
        
        return recommendations
    
    def _save_detailed_results(self, results: Dict[str, Any], save_path: Path, same_data_eval: bool):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if same_data_eval:
            filename = f"same_data_evaluation_detailed_{self.training_mode}_{timestamp}.json"
        else:
            filename = f"detailed_evaluation_{self.training_mode}_{timestamp}.json"
        
        with open(save_path / filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save a summary file
        summary = {
            'evaluation_type': 'same_data' if same_data_eval else 'general',
            'training_mode': self.training_mode,
            'timestamp': timestamp,
            'key_metrics': {
                'global_mean_similarity': results['global_patch_statistics']['mean_cosine_similarity'],
                'image_mean_similarity': results['per_image_statistics']['mean_cosine_similarity'],
                'quality_level': results['quality_assessment']['quality_level'],
                'quality_message': results['quality_assessment']['quality_message'],
                'high_quality_images_ratio': results['quality_metrics']['high_quality_images_ratio'],
                'total_images': results['total_images'],
                'total_patches': results['total_patches'],
            },
            'recommendations': results['quality_assessment']['recommendations']
        }
        
        summary_filename = f"evaluation_summary_{self.training_mode}_{timestamp}.json"
        with open(save_path / summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Results saved: {filename}")
        logger.info(f"✅ Summary saved: {summary_filename}")
    
    def _generate_comprehensive_plots(
        self,
        results: Dict[str, Any],
        all_patch_similarities: List[np.ndarray],
        all_image_averages: np.ndarray,
        save_path: Path,
        plot_distribution: bool,
        plot_per_image: bool,
        plot_heatmap: bool,
        same_data_eval: bool
    ):
        """Generate comprehensive plots and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "same_data" if same_data_eval else "general"
        
        # Set up plot style
        plt.rcParams.update({'font.size': 10})
        
        # 1. Distribution plots
        if plot_distribution:
            self._plot_similarity_distributions(
                all_patch_similarities, all_image_averages, results,
                save_path / f"{prefix}_distributions_{self.training_mode}_{timestamp}.png"
            )
        
        # 2. Per-image analysis
        if plot_per_image:
            self._plot_per_image_analysis(
                all_image_averages, results,
                save_path / f"{prefix}_per_image_{self.training_mode}_{timestamp}.png"
            )
        
        # 3. Heatmap visualization
        if plot_heatmap and len(all_patch_similarities) > 0:
            self._plot_similarity_heatmap(
                all_patch_similarities[:50],  # First 50 images
                save_path / f"{prefix}_heatmap_{self.training_mode}_{timestamp}.png"
            )
        
        # 4. Mode-specific plots
        if self.training_mode == "cls_patch":
            self._plot_cls_vs_patch_analysis(
                all_patch_similarities, results,
                save_path / f"{prefix}_cls_vs_patch_{timestamp}.png"
            )
        
        # 5. Quality assessment plot
        self._plot_quality_assessment(
            results,
            save_path / f"{prefix}_quality_assessment_{self.training_mode}_{timestamp}.png"
        )
        
        logger.info(f"✅ All plots saved to {save_path}")
    
    def _plot_similarity_distributions(self, all_patch_similarities, all_image_averages, results, save_path):
        """Plot similarity distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Flatten patch similarities
        all_patches_flat = np.concatenate(all_patch_similarities)
        
        # 1. Patch-level distribution
        axes[0, 0].hist(all_patches_flat, bins=50, alpha=0.7, density=True, color='skyblue')
        axes[0, 0].axvline(np.mean(all_patches_flat), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_patches_flat):.3f}')
        axes[0, 0].axvline(np.median(all_patches_flat), color='green', linestyle='--', 
                          label=f'Median: {np.median(all_patches_flat):.3f}')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Patch-Level Cosine Similarity Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Image-level distribution
        axes[0, 1].hist(all_image_averages, bins=30, alpha=0.7, density=True, color='lightcoral')
        axes[0, 1].axvline(np.mean(all_image_averages), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_image_averages):.3f}')
        axes[0, 1].axvline(np.median(all_image_averages), color='green', linestyle='--', 
                          label=f'Median: {np.median(all_image_averages):.3f}')
        axes[0, 1].set_xlabel('Average Cosine Similarity')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Per-Image Average Similarity Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Box plot comparison
        box_data = [all_patches_flat, all_image_averages]
        axes[1, 0].boxplot(box_data, labels=['All Patches', 'Image Averages'])
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('Similarity Distribution Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Quality metrics
        quality_levels = ['High (>0.7)', 'Very High (>0.8)', 'Excellent (>0.9)']
        patch_ratios = [
            np.mean(all_patches_flat > 0.7),
            np.mean(all_patches_flat > 0.8),
            np.mean(all_patches_flat > 0.9)
        ]
        image_ratios = [
            np.mean(all_image_averages > 0.7),
            np.mean(all_image_averages > 0.8),
            np.mean(all_image_averages > 0.9)
        ]
        
        x = np.arange(len(quality_levels))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, patch_ratios, width, label='Patches', alpha=0.7)
        axes[1, 1].bar(x + width/2, image_ratios, width, label='Images', alpha=0.7)
        axes[1, 1].set_xlabel('Quality Level')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_title('Quality Distribution')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(quality_levels)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_image_analysis(self, all_image_averages, results, save_path):
        """Plot per-image analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Image similarity over sequence
        axes[0, 0].plot(all_image_averages[:200], marker='o', markersize=2, alpha=0.7)
        axes[0, 0].axhline(np.mean(all_image_averages), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_image_averages):.3f}')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Average Cosine Similarity')
        axes[0, 0].set_title('Per-Image Similarity (First 200 Images)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cumulative average
        cumulative_avg = np.cumsum(all_image_averages) / np.arange(1, len(all_image_averages) + 1)
        axes[0, 1].plot(cumulative_avg)
        axes[0, 1].set_xlabel('Number of Images')
        axes[0, 1].set_ylabel('Cumulative Average Similarity')
        axes[0, 1].set_title('Convergence of Average Similarity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Similarity vs image rank
        sorted_similarities = np.sort(all_image_averages)[::-1]
        axes[1, 0].plot(sorted_similarities)
        axes[1, 0].set_xlabel('Image Rank (Best to Worst)')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('Similarity by Image Rank')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistics table
        stats_text = f"""
        Total Images: {len(all_image_averages)}
        Mean: {np.mean(all_image_averages):.4f}
        Std: {np.std(all_image_averages):.4f}
        Min: {np.min(all_image_averages):.4f}
        Max: {np.max(all_image_averages):.4f}
        Median: {np.median(all_image_averages):.4f}
        
        High Quality (>0.7): {np.mean(all_image_averages > 0.7)*100:.1f}%
        Very High (>0.8): {np.mean(all_image_averages > 0.8)*100:.1f}%
        Excellent (>0.9): {np.mean(all_image_averages > 0.9)*100:.1f}%
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                        transform=axes[1, 1].transAxes, family='monospace')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_similarity_heatmap(self, patch_similarities_sample, save_path):
        """Plot heatmap of patch similarities"""
        # Create heatmap data
        num_images = min(len(patch_similarities_sample), 50)
        heatmap_data = np.array([sim for sim in patch_similarities_sample[:num_images]])
        
        if self.training_mode == "cls_patch":
            # 257 tokens: CLS + 16x16 patches
            heatmap_reshaped = np.zeros((num_images, 17, 16))  # Extra row for CLS
            for i in range(num_images):
                heatmap_reshaped[i, 0, 0] = heatmap_data[i, 0]  # CLS token
                patches_16x16 = heatmap_data[i, 1:].reshape(16, 16)
                heatmap_reshaped[i, 1:, :] = patches_16x16
        else:
            # 256 tokens: 16x16 patches only
            heatmap_reshaped = heatmap_data.reshape(num_images, 16, 16)
        
        # Plot average heatmap
        avg_heatmap = np.mean(heatmap_reshaped, axis=0)
        
        plt.figure(figsize=(12, 8))
        
        if self.training_mode == "cls_patch":
            # Special handling for CLS+patch
            gs = plt.GridSpec(2, 2, height_ratios=[1, 16], width_ratios=[1, 16])
            
            # CLS token
            ax_cls = plt.subplot(gs[0, 0])
            im_cls = ax_cls.imshow([[avg_heatmap[0, 0]]], cmap='viridis', aspect='equal')
            ax_cls.set_title('CLS Token', fontsize=10)
            ax_cls.set_xticks([])
            ax_cls.set_yticks([])
            
            # Patch grid
            ax_patches = plt.subplot(gs[1, :])
            im_patches = ax_patches.imshow(avg_heatmap[1:, :], cmap='viridis', aspect='equal')
            ax_patches.set_title('Patch Similarities (16x16 Grid)', fontsize=12)
            ax_patches.set_xlabel('Patch X')
            ax_patches.set_ylabel('Patch Y')
            
            # Colorbar
            plt.colorbar(im_patches, ax=[ax_cls, ax_patches], shrink=0.8)
        else:
            # Standard 16x16 heatmap
            plt.imshow(avg_heatmap, cmap='viridis', aspect='equal')
            plt.colorbar(shrink=0.8)
            plt.title('Average Patch Similarity Heatmap (16x16)')
            plt.xlabel('Patch X')
            plt.ylabel('Patch Y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cls_vs_patch_analysis(self, all_patch_similarities, results, save_path):
        """Plot CLS vs patch analysis for CLS+patch mode"""
        if self.training_mode != "cls_patch":
            return
        
        # Extract CLS and patch similarities
        cls_similarities = [patches[0] for patches in all_patch_similarities]
        patch_averages = [np.mean(patches[1:]) for patches in all_patch_similarities]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. CLS vs Patch scatter
        axes[0, 0].scatter(cls_similarities, patch_averages, alpha=0.5)
        axes[0, 0].set_xlabel('CLS Token Similarity')
        axes[0, 0].set_ylabel('Average Patch Similarity')
        axes[0, 0].set_title('CLS vs Patch Similarity Correlation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add correlation line
        correlation = np.corrcoef(cls_similarities, patch_averages)[0, 1]
        z = np.polyfit(cls_similarities, patch_averages, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(sorted(cls_similarities), p(sorted(cls_similarities)), "r--", alpha=0.8)
        axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 0].transAxes, verticalalignment='top')
        
        # 2. Distribution comparison
        axes[0, 1].hist(cls_similarities, alpha=0.5, label='CLS Token', bins=30, density=True)
        axes[0, 1].hist(patch_averages, alpha=0.5, label='Patch Average', bins=30, density=True)
        axes[0, 1].set_xlabel('Cosine Similarity')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('CLS vs Patch Similarity Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. CLS vs Patch over images
        indices = range(min(100, len(cls_similarities)))
        axes[1, 0].plot(indices, [cls_similarities[i] for i in indices], 
                       label='CLS Token', alpha=0.7)
        axes[1, 0].plot(indices, [patch_averages[i] for i in indices], 
                       label='Patch Average', alpha=0.7)
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('CLS vs Patch Similarity Over Images')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistics comparison
        stats_text = f"""
        CLS Token Statistics:
        Mean: {np.mean(cls_similarities):.4f}
        Std:  {np.std(cls_similarities):.4f}
        
        Patch Statistics:
        Mean: {np.mean(patch_averages):.4f}
        Std:  {np.std(patch_averages):.4f}
        
        Correlation: {correlation:.4f}
        
        Performance Comparison:
        CLS > Patches: {np.mean([c > p for c, p in zip(cls_similarities, patch_averages)])*100:.1f}%
        CLS < Patches: {np.mean([c < p for c, p in zip(cls_similarities, patch_averages)])*100:.1f}%
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                        transform=axes[1, 1].transAxes, family='monospace')
        axes[1, 1].set_title('CLS vs Patch Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_assessment(self, results, save_path):
        """Plot quality assessment visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        quality_metrics = results['quality_metrics']
        quality_assessment = results['quality_assessment']
        
        # 1. Quality level pie chart
        quality_levels = ['High (>0.7)', 'Very High (>0.8)', 'Excellent (>0.9)']
        quality_values = [
            quality_metrics['high_quality_images_ratio'] - quality_metrics['very_high_quality_images_ratio'],
            quality_metrics['very_high_quality_images_ratio'] - quality_metrics['excellent_quality_images_ratio'],
            quality_metrics['excellent_quality_images_ratio']
        ]
        quality_values.append(1 - quality_metrics['high_quality_images_ratio'])  # Below 0.7
        quality_levels.append('Below High (<0.7)')
        
        colors = ['lightgreen', 'green', 'darkgreen', 'lightcoral']
        axes[0, 0].pie(quality_values, labels=quality_levels, colors=colors, autopct='%1.1f%%')
        axes[0, 0].set_title('Image Quality Distribution')
        
        # 2. Quality metrics bar chart
        metrics = ['High Quality\nImages', 'Very High Quality\nImages', 'Excellent Quality\nImages',
                  'High Quality\nPatches', 'Very High Quality\nPatches', 'Excellent Quality\nPatches']
        values = [
            quality_metrics['high_quality_images_ratio'],
            quality_metrics['very_high_quality_images_ratio'],
            quality_metrics['excellent_quality_images_ratio'],
            quality_metrics['high_quality_patches_ratio'],
            quality_metrics['very_high_quality_patches_ratio'],
            quality_metrics['excellent_quality_patches_ratio']
        ]
        
        colors = ['lightblue'] * 3 + ['lightcoral'] * 3
        bars = axes[0, 1].bar(metrics, values, color=colors)
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].set_title('Quality Metrics Overview')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Overall assessment
        assessment_text = f"""
        Overall Assessment: {quality_assessment['quality_level'].upper()}
        
        {quality_assessment['quality_message']}
        
        Key Metrics:
        • Global Mean Similarity: {quality_assessment['global_mean_similarity']:.4f}
        • Image Mean Similarity: {quality_assessment['image_mean_similarity']:.4f}
        • High Quality Ratio: {quality_assessment['high_quality_ratio']:.3f}
        • Consistency: {'Good' if quality_assessment['consistency_good'] else 'Needs Improvement'}
        
        Training Mode: {results['training_mode']}
        Total Images: {results['total_images']:,}
        Total Patches: {results['total_patches']:,}
        """
        
        axes[1, 0].text(0.05, 0.95, assessment_text, fontsize=10, verticalalignment='top',
                       transform=axes[1, 0].transAxes, family='monospace')
        axes[1, 0].set_title('Overall Assessment')
        axes[1, 0].axis('off')
        
        # 4. Recommendations
        recommendations_text = "Recommendations:\n\n"
        for i, rec in enumerate(quality_assessment['recommendations'], 1):
            recommendations_text += f"{i}. {rec}\n"
        
        axes[1, 1].text(0.05, 0.95, recommendations_text, fontsize=10, verticalalignment='top',
                       transform=axes[1, 1].transAxes, family='monospace')
        axes[1, 1].set_title('Training Recommendations')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_detailed_evaluator(
    model,
    training_mode: str = "cls_patch",
    device: str = "auto",
    **kwargs
) -> BLIP3oDetailedEvaluator:
    """Factory function for creating detailed evaluator"""
    return BLIP3oDetailedEvaluator(
        model=model,
        training_mode=training_mode,
        device=device,
        **kwargs
    )