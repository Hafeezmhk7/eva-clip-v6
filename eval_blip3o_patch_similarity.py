#!/usr/bin/env python3
"""
BLIP3-o Patch-Level Cosine Similarity Evaluation Script
eval_blip3o_patch_similarity.py

Evaluates the cosine similarity between predicted DiT patches and ground truth CLIP patches:
1. 256 patch-level cosine similarities per image
2. Average cosine similarity per image
3. Detailed statistics and visualizations
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoModel, CLIPImageProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import json
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "src"))

def setup_paths():
    """Setup import paths with enhanced error handling"""
    try:
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.blip3o_patch_dit import BLIP3oPatchDiTModel, create_blip3o_patch_dit_model
        logger.info("✅ BLIP3-o modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("💡 Make sure src/modules are properly set up")
        return False

class BLIP3oPatchSimilarityEvaluator:
    """
    Evaluator for patch-level cosine similarity between DiT predictions and CLIP ground truth
    """
    
    def __init__(self, device="auto"):
        self.device = self._setup_device(device)
        self.clip_processor = None
        self.clip_model = None
        self.eva_processor = None
        self.eva_model = None
        self.blip3o_model = None
        self.model_info = {}
        
        logger.info("BLIP3-o patch similarity evaluator initialized")
    
    def _setup_device(self, device_arg):
        if device_arg == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logger.info("Using CPU")
        else:
            device = torch.device(device_arg)
        return device
    
    def load_models(self):
        """Load CLIP, EVA-CLIP, and BLIP3-o models"""
        logger.info("Loading foundation models...")
        
        # Load CLIP ViT-L/14
        logger.info("   Loading CLIP ViT-L/14...")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_model.eval()
            logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise
        
        # Load EVA-CLIP-8B
        logger.info("   Loading EVA-CLIP-8B...")
        try:
            self.eva_model = AutoModel.from_pretrained(
                "BAAI/EVA-CLIP-8B", 
                trust_remote_code=True
            ).to(self.device)
            self.eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.eva_model.eval()
            logger.info("✅ EVA-CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load EVA-CLIP model: {e}")
            raise
        
        logger.info("✅ All foundation models loaded successfully")
    
    def load_blip3o_model(self, model_path):
        """Load the trained BLIP3-o patch-level model"""
        logger.info(f"Loading BLIP3-o patch model from: {model_path}")
        
        model_path = Path(model_path)
        
        # Load config
        config_files = [
            model_path / "enhanced_training_config.json",
            model_path / "config.json",
            model_path / "blip3o_model_config.json"
        ]
        
        config_file = None
        config_dict = None
        
        for cf in config_files:
            if cf.exists():
                config_file = cf
                with open(config_file, 'r') as f:
                    full_config = json.load(f)
                
                if cf.name == "enhanced_training_config.json":
                    config_dict = full_config.get('model_config', {})
                    self.model_info = {
                        'enhanced': True,
                        'training_mode': full_config.get('training_strategy', {}).get('mode', 'unknown'),
                        'enhanced_features': full_config.get('enhanced_hyperparameters', {}),
                        'architecture': full_config.get('architecture', 'unknown'),
                    }
                    logger.info("✅ Enhanced training config detected")
                else:
                    config_dict = full_config
                    self.model_info = {'enhanced': False}
                
                break
        
        if not config_file:
            raise FileNotFoundError(f"No config file found in {model_path}")
        
        logger.info(f"Using config: {config_file.name}")
        
        # Create model config
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        
        try:
            config = BLIP3oDiTConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Direct config creation failed: {e}")
            # Try with common defaults
            config_dict.update({
                'hidden_size': config_dict.get('hidden_size', 768),
                'num_hidden_layers': config_dict.get('num_hidden_layers', 12),
                'num_attention_heads': config_dict.get('num_attention_heads', 12),
                'intermediate_size': config_dict.get('intermediate_size', 3072),
            })
            config = BLIP3oDiTConfig(**config_dict)
        
        # Create model
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model
        self.blip3o_model = create_blip3o_patch_dit_model(config)
        
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
        
        if not weight_file:
            raise FileNotFoundError(f"No weight file found in {model_path}")
        
        logger.info(f"Loading weights from: {weight_file}")
        
        if weight_file.suffix == ".bin":
            state_dict = torch.load(weight_file, map_location=self.device)
        else:
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(weight_file))
            except ImportError:
                logger.error("SafeTensors not installed, cannot load .safetensors file")
                raise
        
        # Clean up state dict if needed
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            logger.info("Removed DDP prefix from state dict")
        
        # Load state dict
        missing_keys, unexpected_keys = self.blip3o_model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        
        self.blip3o_model = self.blip3o_model.to(self.device)
        self.blip3o_model.eval()
        
        # Log model info
        param_count = sum(p.numel() for p in self.blip3o_model.parameters())
        logger.info(f"✅ BLIP3-o model loaded successfully")
        logger.info(f"   Parameters: {param_count:,}")
        logger.info(f"   Enhanced: {self.model_info.get('enhanced', False)}")
    
    def extract_clip_patch_embeddings(self, images):
        """Extract CLIP patch embeddings (256 patches, 1024-dim each)"""
        logger.info(f"Extracting CLIP patch embeddings for {len(images)} images...")
        
        patch_embeddings = []
        batch_size = 8
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                
                inputs = self.clip_processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get vision model outputs with hidden states
                vision_outputs = self.clip_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token)
                patches = vision_outputs.last_hidden_state[:, 1:, :]  # [B, 256, 1024]
                
                # Validate dimensions
                assert patches.shape[1] == 256, f"Expected 256 patches, got {patches.shape[1]}"
                assert patches.shape[2] == 1024, f"Expected 1024-dim patches, got {patches.shape[2]}"
                
                patch_embeddings.append(patches.cpu())
        
        result = torch.cat(patch_embeddings, dim=0)
        logger.info(f"CLIP patch embeddings shape: {result.shape}")
        return result
    
    def extract_eva_embeddings(self, images):
        """Extract EVA-CLIP embeddings for conditioning"""
        logger.info(f"Extracting EVA-CLIP embeddings for {len(images)} images...")
        
        eva_embeddings = []
        batch_size = 4
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                
                inputs = self.eva_processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                
                vision_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token)
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [B, 256, 4096]
                eva_embeddings.append(patch_embeddings.cpu())
        
        return torch.cat(eva_embeddings, dim=0)
    
    def generate_blip3o_patches(self, eva_embeddings, num_inference_steps=50):
        """Generate BLIP3-o patch embeddings"""
        logger.info(f"Generating BLIP3-o patches for {eva_embeddings.shape[0]} images...")
        
        eva_embeddings = eva_embeddings.to(self.device)
        generated_patches = []
        batch_size = 2
        
        with torch.no_grad():
            for i in range(0, eva_embeddings.shape[0], batch_size):
                end_idx = min(i + batch_size, eva_embeddings.shape[0])
                batch_eva = eva_embeddings[i:end_idx]
                
                try:
                    # Generate CLIP patch embeddings
                    generated = self.blip3o_model.generate(
                        eva_features=batch_eva,
                        num_inference_steps=num_inference_steps,
                    )  # [B, 256, 1024]
                    
                    generated_patches.append(generated.cpu())
                    
                except Exception as e:
                    logger.error(f"Error generating patches for batch {i//batch_size}: {e}")
                    # Add zero patches for failed batch
                    batch_size_actual = batch_eva.shape[0]
                    zero_patches = torch.zeros(batch_size_actual, 256, 1024)
                    generated_patches.append(zero_patches)
        
        result = torch.cat(generated_patches, dim=0)
        logger.info(f"Generated patches shape: {result.shape}")
        return result
    
    def compute_patch_cosine_similarity(self, predicted_patches, target_patches, normalize_embeddings=True, 
                                       similarity_metrics=['cosine']):
        """
        Compute patch-level similarities with multiple metrics and normalization options
        
        Args:
            predicted_patches: [N, 256, 1024] - Generated patches
            target_patches: [N, 256, 1024] - Ground truth CLIP patches
            normalize_embeddings: Whether to normalize embeddings to unit norm
            similarity_metrics: List of metrics to compute ['cosine', 'dot_product', 'l2_distance']
            
        Returns:
            Dict with similarity metrics and comprehensive analysis
        """
        logger.info(f"Computing patch-level similarities...")
        logger.info(f"Predicted patches shape: {predicted_patches.shape}")
        logger.info(f"Target patches shape: {target_patches.shape}")
        logger.info(f"Metrics: {similarity_metrics}")
        logger.info(f"Normalization: {'ON' if normalize_embeddings else 'OFF'}")
        
        # Analyze embedding norms before normalization
        pred_norms = torch.norm(predicted_patches, p=2, dim=-1)  # [N, 256]
        target_norms = torch.norm(target_patches, p=2, dim=-1)   # [N, 256]
        
        logger.info(f"Predicted embeddings - norm stats: mean={pred_norms.mean():.4f}, std={pred_norms.std():.4f}")
        logger.info(f"Target embeddings - norm stats: mean={target_norms.mean():.4f}, std={target_norms.std():.4f}")
        
        results = {}
        
        for metric in similarity_metrics:
            logger.info(f"Computing {metric} similarity...")
            
            if metric == 'cosine':
                # Cosine similarity - always normalize for this metric
                pred_norm = F.normalize(predicted_patches, p=2, dim=-1)  # [N, 256, 1024]
                target_norm = F.normalize(target_patches, p=2, dim=-1)   # [N, 256, 1024]
                
                # Element-wise cosine similarity for each patch
                patch_similarities = torch.sum(pred_norm * target_norm, dim=-1)  # [N, 256]
                
            elif metric == 'dot_product':
                # Dot product similarity - use normalization option
                if normalize_embeddings:
                    pred_use = F.normalize(predicted_patches, p=2, dim=-1)
                    target_use = F.normalize(target_patches, p=2, dim=-1)
                    logger.info("Using normalized embeddings for dot product")
                else:
                    pred_use = predicted_patches
                    target_use = target_patches
                    logger.info("Using raw embeddings for dot product")
                
                patch_similarities = torch.sum(pred_use * target_use, dim=-1)  # [N, 256]
                
            elif metric == 'l2_distance':
                # L2 distance - use normalization option (convert to similarity: negative distance)
                if normalize_embeddings:
                    pred_use = F.normalize(predicted_patches, p=2, dim=-1)
                    target_use = F.normalize(target_patches, p=2, dim=-1)
                    logger.info("Using normalized embeddings for L2 distance")
                else:
                    pred_use = predicted_patches
                    target_use = target_patches
                    logger.info("Using raw embeddings for L2 distance")
                
                # L2 distance (negative for similarity interpretation)
                patch_similarities = -torch.norm(pred_use - target_use, p=2, dim=-1)  # [N, 256]
                
            else:
                logger.warning(f"Unknown metric: {metric}, skipping...")
                continue
            
            # Per-image average similarity
            per_image_avg_similarity = torch.mean(patch_similarities, dim=1)  # [N]
            
            # Overall statistics
            all_patch_similarities = patch_similarities.flatten()
            
            metric_results = {
                # Raw similarities
                'patch_similarities': patch_similarities.numpy(),  # [N, 256]
                'per_image_avg_similarity': per_image_avg_similarity.numpy(),  # [N]
                
                # Overall statistics
                'overall_mean_similarity': float(torch.mean(all_patch_similarities)),
                'overall_std_similarity': float(torch.std(all_patch_similarities)),
                'overall_min_similarity': float(torch.min(all_patch_similarities)),
                'overall_max_similarity': float(torch.max(all_patch_similarities)),
                'overall_median_similarity': float(torch.median(all_patch_similarities)),
                
                # Per-image statistics
                'per_image_mean': float(torch.mean(per_image_avg_similarity)),
                'per_image_std': float(torch.std(per_image_avg_similarity)),
                'per_image_min': float(torch.min(per_image_avg_similarity)),
                'per_image_max': float(torch.max(per_image_avg_similarity)),
                'per_image_median': float(torch.median(per_image_avg_similarity)),
                
                # Dataset info
                'num_images': predicted_patches.shape[0],
                'num_patches_per_image': 256,
                'total_patches': predicted_patches.shape[0] * 256,
                'metric': metric,
                'normalized': normalize_embeddings if metric != 'cosine' else True,
            }
            
            # Quality metrics (adjusted for each similarity type)
            if metric == 'l2_distance':
                # For L2 distance, more negative (smaller distance) is better
                metric_results.update({
                    'high_similarity_patches_ratio': float(torch.sum(all_patch_similarities > -0.5) / len(all_patch_similarities)),
                    'good_similarity_patches_ratio': float(torch.sum(all_patch_similarities > -1.0) / len(all_patch_similarities)),
                    'poor_similarity_patches_ratio': float(torch.sum(all_patch_similarities < -2.0) / len(all_patch_similarities)),
                    'excellent_images_ratio': float(torch.sum(per_image_avg_similarity > -0.5) / len(per_image_avg_similarity)),
                    'good_images_ratio': float(torch.sum(per_image_avg_similarity > -1.0) / len(per_image_avg_similarity)),
                    'poor_images_ratio': float(torch.sum(per_image_avg_similarity < -2.0) / len(per_image_avg_similarity)),
                })
            else:
                # For cosine and dot product, higher is better
                metric_results.update({
                    'high_similarity_patches_ratio': float(torch.sum(all_patch_similarities > 0.8) / len(all_patch_similarities)),
                    'good_similarity_patches_ratio': float(torch.sum(all_patch_similarities > 0.6) / len(all_patch_similarities)),
                    'poor_similarity_patches_ratio': float(torch.sum(all_patch_similarities < 0.3) / len(all_patch_similarities)),
                    'excellent_images_ratio': float(torch.sum(per_image_avg_similarity > 0.8) / len(per_image_avg_similarity)),
                    'good_images_ratio': float(torch.sum(per_image_avg_similarity > 0.6) / len(per_image_avg_similarity)),
                    'poor_images_ratio': float(torch.sum(per_image_avg_similarity < 0.4) / len(per_image_avg_similarity)),
                })
            
            results[metric] = metric_results
        
        # Add embedding norm analysis
        results['embedding_analysis'] = {
            'predicted_norms': {
                'mean': float(pred_norms.mean()),
                'std': float(pred_norms.std()),
                'min': float(pred_norms.min()),
                'max': float(pred_norms.max()),
                'median': float(pred_norms.median()),
            },
            'target_norms': {
                'mean': float(target_norms.mean()),
                'std': float(target_norms.std()),
                'min': float(target_norms.min()),
                'max': float(target_norms.max()),
                'median': float(target_norms.median()),
            },
            'norm_ratio': float(pred_norms.mean() / target_norms.mean()),
            'norm_correlation': float(torch.corrcoef(torch.stack([pred_norms.flatten(), target_norms.flatten()]))[0, 1]),
        }
        
        return results
    
    def analyze_patch_patterns(self, patch_similarities):
        """Analyze spatial patterns in patch similarities"""
        logger.info("Analyzing spatial patch patterns...")
        
        # Reshape to spatial grid [N, 16, 16]
        spatial_similarities = patch_similarities.reshape(-1, 16, 16)
        
        # Compute spatial statistics
        patterns = {
            'center_vs_edge': self._analyze_center_vs_edge(spatial_similarities),
            'spatial_variance': self._analyze_spatial_variance(spatial_similarities),
            'hotspots': self._find_similarity_hotspots(spatial_similarities),
        }
        
        return patterns
    
    def _analyze_center_vs_edge(self, spatial_similarities):
        """Compare center vs edge patch similarities"""
        N, H, W = spatial_similarities.shape
        
        # Define center region (inner 8x8)
        center_start, center_end = 4, 12
        center_patches = spatial_similarities[:, center_start:center_end, center_start:center_end]
        
        # Define edge region (outer border)
        edge_mask = torch.ones(H, W, dtype=torch.bool)
        edge_mask[1:-1, 1:-1] = False
        edge_patches = spatial_similarities[:, edge_mask]
        
        return {
            'center_mean': float(torch.mean(center_patches)),
            'edge_mean': float(torch.mean(edge_patches)),
            'center_vs_edge_diff': float(torch.mean(center_patches) - torch.mean(edge_patches)),
        }
    
    def _analyze_spatial_variance(self, spatial_similarities):
        """Analyze spatial variance within images"""
        # Compute variance for each image
        per_image_variance = torch.var(spatial_similarities.view(spatial_similarities.shape[0], -1), dim=1)
        
        return {
            'mean_spatial_variance': float(torch.mean(per_image_variance)),
            'std_spatial_variance': float(torch.std(per_image_variance)),
        }
    
    def _find_similarity_hotspots(self, spatial_similarities):
        """Find common high/low similarity regions"""
        # Average across all images to find common patterns
        avg_spatial = torch.mean(spatial_similarities, dim=0)  # [16, 16]
        
        # Find hotspots (highest similarities)
        flat_avg = avg_spatial.flatten()
        top_indices = torch.topk(flat_avg, k=5).indices
        hotspot_coords = [(int(idx // 16), int(idx % 16)) for idx in top_indices]
        
        # Find coldspots (lowest similarities)
        bottom_indices = torch.topk(flat_avg, k=5, largest=False).indices
        coldspot_coords = [(int(idx // 16), int(idx % 16)) for idx in bottom_indices]
        
        return {
            'avg_spatial_pattern': avg_spatial.numpy().tolist(),
            'hotspots': hotspot_coords,
            'coldspots': coldspot_coords,
            'spatial_range': float(torch.max(avg_spatial) - torch.min(avg_spatial)),
        }
    
    def create_metric_visualizations(self, metric_results, output_dir, metric_name):
        """Create visualization plots for a specific metric"""
        logger.info(f"Creating {metric_name} visualization plots...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Per-image average similarity histogram
        plt.figure(figsize=(10, 6))
        plt.hist(metric_results['per_image_avg_similarity'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(metric_results['per_image_mean'], color='red', linestyle='--', 
                   label=f'Mean: {metric_results["per_image_mean"]:.3f}')
        plt.axvline(metric_results['per_image_median'], color='green', linestyle='--', 
                   label=f'Median: {metric_results["per_image_median"]:.3f}')
        plt.xlabel(f'Average {metric_name.title()} Similarity per Image')
        plt.ylabel('Number of Images')
        plt.title(f'Distribution of Per-Image Average {metric_name.title()} Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'per_image_{metric_name}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. All patch similarities histogram
        all_similarities = metric_results['patch_similarities'].flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(all_similarities, bins=100, alpha=0.7, edgecolor='black')
        plt.axvline(metric_results['overall_mean_similarity'], color='red', linestyle='--', 
                   label=f'Mean: {metric_results["overall_mean_similarity"]:.3f}')
        plt.axvline(metric_results['overall_median_similarity'], color='green', linestyle='--', 
                   label=f'Median: {metric_results["overall_median_similarity"]:.3f}')
        plt.xlabel(f'Patch-Level {metric_name.title()} Similarity')
        plt.ylabel('Number of Patches')
        plt.title(f'Distribution of All Patch-Level {metric_name.title()} Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'all_patches_{metric_name}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Quality distribution pie chart (metric-specific thresholds)
        per_image_avg = metric_results['per_image_avg_similarity']
        
        if metric_name == 'l2_distance':
            # For L2 distance, more negative (smaller distance) is better
            excellent = np.sum(per_image_avg > -0.5) / len(per_image_avg) * 100
            good = np.sum((per_image_avg > -1.0) & (per_image_avg <= -0.5)) / len(per_image_avg) * 100
            fair = np.sum((per_image_avg > -2.0) & (per_image_avg <= -1.0)) / len(per_image_avg) * 100
            poor = np.sum(per_image_avg <= -2.0) / len(per_image_avg) * 100
            quality_labels = ['Excellent (>-0.5)', 'Good (-1.0 to -0.5)', 'Fair (-2.0 to -1.0)', 'Poor (≤-2.0)']
        else:
            # For cosine and dot product, higher is better
            excellent = np.sum(per_image_avg > 0.8) / len(per_image_avg) * 100
            good = np.sum((per_image_avg > 0.6) & (per_image_avg <= 0.8)) / len(per_image_avg) * 100
            fair = np.sum((per_image_avg > 0.4) & (per_image_avg <= 0.6)) / len(per_image_avg) * 100
            poor = np.sum(per_image_avg <= 0.4) / len(per_image_avg) * 100
            quality_labels = ['Excellent (>0.8)', 'Good (0.6-0.8)', 'Fair (0.4-0.6)', 'Poor (≤0.4)']
        
        quality_values = [excellent, good, fair, poor]
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        
        plt.figure(figsize=(8, 8))
        plt.pie(quality_values, labels=quality_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Image Quality Distribution - {metric_name.title()} Similarity')
        plt.savefig(output_dir / f'{metric_name}_quality_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"{metric_name} visualizations saved to: {output_dir}")
    
    def evaluate_on_dataset(self, images, captions_per_image, num_inference_steps=50, 
                          output_dir=None, save_detailed=True, normalize_embeddings=True,
                          similarity_metrics=['cosine']):
        """
        Main evaluation function for patch-level similarity with multiple metrics
        
        Args:
            images: List of PIL images
            captions_per_image: List of caption lists (for metadata)
            num_inference_steps: Number of inference steps for generation
            output_dir: Directory to save results
            save_detailed: Whether to save detailed per-image results
            normalize_embeddings: Whether to normalize embeddings
            similarity_metrics: List of similarity metrics to compute
            
        Returns:
            Dictionary with evaluation results for all metrics
        """
        logger.info(f"Starting patch-level similarity evaluation on {len(images)} images")
        logger.info(f"Metrics: {similarity_metrics}")
        logger.info(f"Normalization: {normalize_embeddings}")
        
        # Extract EVA embeddings for conditioning
        logger.info("Step 1: Extracting EVA-CLIP embeddings...")
        eva_embeddings = self.extract_eva_embeddings(images)
        
        # Extract ground truth CLIP patches
        logger.info("Step 2: Extracting ground truth CLIP patches...")
        target_patches = self.extract_clip_patch_embeddings(images)
        
        # Generate BLIP3-o patches
        logger.info("Step 3: Generating BLIP3-o patches...")
        predicted_patches = self.generate_blip3o_patches(eva_embeddings, num_inference_steps)
        
        # Compute similarities with multiple metrics
        logger.info("Step 4: Computing patch-level similarities...")
        similarity_results = self.compute_patch_cosine_similarity(
            predicted_patches, target_patches, normalize_embeddings, similarity_metrics
        )
        
        # Analyze spatial patterns for the primary metric
        primary_metric = similarity_metrics[0]
        if primary_metric in similarity_results:
            logger.info("Step 5: Analyzing spatial patterns...")
            patterns = self.analyze_patch_patterns(similarity_results[primary_metric]['patch_similarities'])
            similarity_results['spatial_patterns'] = patterns
        
        # Add metadata
        similarity_results.update({
            'model_info': self.model_info,
            'evaluation_config': {
                'num_inference_steps': num_inference_steps,
                'num_images': len(images),
                'captions_provided': len([cap for caps in captions_per_image for cap in caps]),
                'normalize_embeddings': normalize_embeddings,
                'similarity_metrics': similarity_metrics,
            },
            'timestamp': time.time(),
        })
        
        # Create visualizations for each metric
        if output_dir:
            viz_dir = Path(output_dir) / 'visualizations'
            for metric in similarity_metrics:
                if metric in similarity_results:
                    metric_viz_dir = viz_dir / metric
                    self.create_metric_visualizations(similarity_results[metric], metric_viz_dir, metric)
        
        # Save detailed results
        if output_dir and save_detailed:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            with open(output_path / 'patch_similarity_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for metric, metric_data in similarity_results.items():
                    if isinstance(metric_data, dict) and 'patch_similarities' in metric_data:
                        json_metric_data = metric_data.copy()
                        json_metric_data['patch_similarities'] = metric_data['patch_similarities'].tolist()
                        json_metric_data['per_image_avg_similarity'] = metric_data['per_image_avg_similarity'].tolist()
                        json_results[metric] = json_metric_data
                    else:
                        json_results[metric] = metric_data
                        
                json.dump(json_results, f, indent=2)
            
            # Save detailed per-image results for each metric
            for metric in similarity_metrics:
                if metric in similarity_results:
                    metric_data = similarity_results[metric]
                    per_image_details = []
                    
                    for i, (patches_sim, avg_sim) in enumerate(zip(
                        metric_data['patch_similarities'], 
                        metric_data['per_image_avg_similarity']
                    )):
                        per_image_details.append({
                            'image_id': i,
                            'metric': metric,
                            'average_similarity': float(avg_sim),
                            'patch_similarities': patches_sim.tolist(),
                            'max_patch_similarity': float(np.max(patches_sim)),
                            'min_patch_similarity': float(np.min(patches_sim)),
                            'std_patch_similarity': float(np.std(patches_sim)),
                            'captions': captions_per_image[i] if i < len(captions_per_image) else [],
                        })
                    
                    with open(output_path / f'per_image_details_{metric}.json', 'w') as f:
                        json.dump(per_image_details, f, indent=2)
            
            logger.info(f"Detailed results saved to: {output_path}")
        
        return similarity_results

def load_coco_samples(coco_root, num_samples=1000):
    """Load COCO validation samples"""
    logger.info(f"Loading {num_samples} COCO samples...")
    
    coco_root = Path(coco_root)
    annotations_file = coco_root / "annotations" / "captions_val2017.json"
    images_dir = coco_root / "images" / "val2017"
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    images_info = {img['id']: img for img in coco_data['images']}
    image_captions = {}
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(ann['caption'])
    
    # Load samples
    images = []
    captions_per_image = []
    image_ids = []
    
    loaded_count = 0
    failed_count = 0
    
    for image_id, captions in image_captions.items():
        if loaded_count >= num_samples:
            break
        
        if image_id not in images_info:
            continue
        
        image_info = images_info[image_id]
        image_path = images_dir / image_info['file_name']
        
        if not image_path.exists():
            failed_count += 1
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            captions_per_image.append(captions[:5])  # Max 5 captions
            image_ids.append(image_id)
            loaded_count += 1
            
            if loaded_count % 100 == 0:
                logger.info(f"Loaded {loaded_count}/{num_samples} images...")
                
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            failed_count += 1
            continue
    
    logger.info(f"Successfully loaded {len(images)} images")
    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count} images")
    
    return images, captions_per_image, image_ids

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate BLIP3-o Patch-Level Cosine Similarity")
    parser.add_argument("--coco_root", type=str, required=True, help="COCO dataset root")
    parser.add_argument("--model_path", type=str, required=True, help="BLIP3-o model path")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--output_dir", type=str, default="./patch_similarity_results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting BLIP3-o Patch-Level Cosine Similarity Evaluation")
    logger.info("=" * 70)
    
    # Setup imports
    if not setup_paths():
        return 1
    
    # Initialize evaluator
    try:
        evaluator = BLIP3oPatchSimilarityEvaluator(device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return 1
    
    # Load models
    try:
        logger.info("Loading models...")
        evaluator.load_models()
        evaluator.load_blip3o_model(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return 1
    
    # Load COCO samples
    try:
        images, captions_per_image, image_ids = load_coco_samples(args.coco_root, args.num_samples)
    except Exception as e:
        logger.error(f"Failed to load COCO samples: {e}")
        return 1
    
    # Run evaluation
    try:
        results = evaluator.evaluate_on_dataset(
            images=images,
            captions_per_image=captions_per_image,
            num_inference_steps=args.num_inference_steps,
            output_dir=args.output_dir,
            save_detailed=True
        )
        
        # Print summary results
        print(f"\n{'='*70}")
        print("🎯 BLIP3-O PATCH-LEVEL COSINE SIMILARITY RESULTS")
        print(f"{'='*70}")
        
        print(f"📊 Dataset Information:")
        print(f"   Images evaluated: {results['num_images']:,}")
        print(f"   Total patches: {results['total_patches']:,}")
        print(f"   Patches per image: {results['num_patches_per_image']}")
        
        print(f"\n🎯 Overall Patch-Level Statistics:")
        print(f"   Mean similarity: {results['overall_mean_similarity']:.4f}")
        print(f"   Std similarity:  {results['overall_std_similarity']:.4f}")
        print(f"   Min similarity:  {results['overall_min_similarity']:.4f}")
        print(f"   Max similarity:  {results['overall_max_similarity']:.4f}")
        print(f"   Median similarity: {results['overall_median_similarity']:.4f}")
        
        print(f"\n🖼️  Per-Image Average Statistics:")
        print(f"   Mean per-image avg: {results['per_image_mean']:.4f}")
        print(f"   Std per-image avg:  {results['per_image_std']:.4f}")
        print(f"   Min per-image avg:  {results['per_image_min']:.4f}")
        print(f"   Max per-image avg:  {results['per_image_max']:.4f}")
        print(f"   Median per-image avg: {results['per_image_median']:.4f}")
        
        print(f"\n🏆 Quality Assessment:")
        print(f"   Excellent images (>0.8): {results['excellent_images_ratio']*100:.1f}%")
        print(f"   Good images (>0.6):     {results['good_images_ratio']*100:.1f}%")
        print(f"   Poor images (<0.4):     {results['poor_images_ratio']*100:.1f}%")
        
        print(f"\n📊 Patch Quality Distribution:")
        print(f"   High similarity patches (>0.8): {results['high_similarity_patches_ratio']*100:.1f}%")
        print(f"   Good similarity patches (>0.6): {results['good_similarity_patches_ratio']*100:.1f}%")
        print(f"   Poor similarity patches (<0.3): {results['poor_similarity_patches_ratio']*100:.1f}%")
        
        if 'patterns' in results:
            patterns = results['patterns']
            print(f"\n🗺️  Spatial Pattern Analysis:")
            print(f"   Center vs Edge similarity diff: {patterns['center_vs_edge']['center_vs_edge_diff']:+.4f}")
            print(f"   Center mean: {patterns['center_vs_edge']['center_mean']:.4f}")
            print(f"   Edge mean:   {patterns['center_vs_edge']['edge_mean']:.4f}")
            print(f"   Mean spatial variance: {patterns['spatial_variance']['mean_spatial_variance']:.4f}")
        
        print(f"\n💾 Results saved to: {args.output_dir}")
        print(f"   📈 Visualizations: {args.output_dir}/visualizations/")
        print(f"   📋 Main results: patch_similarity_results.json")
        print(f"   🖼️  Per-image details: per_image_details.json")
        
        # Performance assessment
        overall_quality = results['per_image_mean']
        if overall_quality > 0.8:
            print(f"\n🎉 EXCELLENT: Very high patch-level similarity!")
        elif overall_quality > 0.6:
            print(f"\n✅ GOOD: Strong patch-level similarity")
        elif overall_quality > 0.4:
            print(f"\n⚠️  FAIR: Moderate patch-level similarity")
        else:
            print(f"\n❌ POOR: Low patch-level similarity - model needs improvement")
        
        print(f"{'='*70}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)