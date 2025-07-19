#!/usr/bin/env python3
"""
Comprehensive Evaluation and Analysis Script for BLIP3-o DiT
Combines recall evaluation, alignment analysis, and performance metrics
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Comprehensive BLIP3-o Evaluation")
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained BLIP3-o model")
    parser.add_argument("--coco_root", type=str, required=True,
                       help="Path to COCO dataset root")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=5000,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10],
                       help="K values for recall@K evaluation")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./comprehensive_evaluation",
                       help="Output directory for results")
    parser.add_argument("--create_visualizations", action="store_true",
                       help="Create evaluation visualizations")
    
    # Analysis parameters
    parser.add_argument("--analyze_embeddings", action="store_true",
                       help="Analyze embedding quality and distribution")
    parser.add_argument("--compare_methods", action="store_true",
                       help="Compare multiple evaluation methods")
    
    return parser.parse_args()

class ComprehensiveEvaluator:
    """Comprehensive evaluator for BLIP3-o model"""
    
    def __init__(self, model, clip_model, clip_processor, eva_model, eva_processor, device):
        self.model = model
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.eva_model = eva_model
        self.eva_processor = eva_processor
        self.device = device
        
        # Set models to eval mode
        self.model.eval()
        self.clip_model.eval()
        self.eva_model.eval()
    
    def extract_blip3o_features(self, images: List, method: str = "global") -> torch.Tensor:
        """Extract features using BLIP3-o with different output methods"""
        with torch.no_grad():
            # Step 1: Extract EVA-CLIP features for conditioning
            eva_features = []
            for img in images:
                inputs = self.eva_processor(images=img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device).half()
                
                outputs = self.eva_model.vision_model(pixel_values=pixel_values)
                patch_embeddings = outputs.last_hidden_state[:, 1:, :]
                
                # Reshape to 256 tokens
                batch_size, num_patches, hidden_dim = patch_embeddings.shape
                grid_size = int(np.sqrt(num_patches))
                spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
                tokens = spatial_grid.reshape(batch_size, 256, hidden_dim)
                
                eva_features.append(tokens.squeeze().cpu().float())
                del outputs, patch_embeddings, spatial_grid, tokens, pixel_values
            
            eva_conditioning = torch.stack(eva_features).to(self.device)
            
            # Step 2: Generate using BLIP3-o
            if method == "global":
                # Generate global features for recall
                generated = self.model.generate(
                    encoder_hidden_states=eva_conditioning,
                    num_inference_steps=50,
                    return_global_only=True,  # Get [B, 768]
                )
                return generated
            
            elif method == "patch_averaged":
                # Generate patch features and average
                generated = self.model.generate(
                    encoder_hidden_states=eva_conditioning,
                    num_inference_steps=50,
                    return_global_only=False,  # Get [B, 256, 1024]
                )
                # Average pool and apply CLIP projection
                if generated.dim() == 3:  # [B, 256, 1024]
                    pooled = generated.mean(dim=1)  # [B, 1024]
                    if hasattr(self.model, 'frozen_clip_visual_proj') and self.model.frozen_clip_visual_proj:
                        projected = self.model.frozen_clip_visual_proj(pooled)  # [B, 768]
                        return F.normalize(projected, p=2, dim=-1)
                    else:
                        # Use CLIP model projection
                        projected = self.clip_model.visual_projection(pooled)
                        return F.normalize(projected, p=2, dim=-1)
                else:
                    return generated
            
            else:
                raise ValueError(f"Unknown method: {method}")
    
    def extract_clip_baseline_features(self, images: List, method: str = "cls") -> torch.Tensor:
        """Extract baseline CLIP features with different methods"""
        with torch.no_grad():
            clip_features = []
            for img in images:
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.clip_model.vision_model(**inputs)
                
                if method == "cls":
                    # Use CLS token (standard approach)
                    cls_token = outputs.last_hidden_state[:, 0, :]  # [1, 1024]
                    projected = self.clip_model.visual_projection(cls_token)  # [1, 768]
                    clip_features.append(projected.squeeze().cpu().float())
                
                elif method == "patch_averaged":
                    # Average patch embeddings
                    patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Remove CLS
                    global_features = patch_embeddings.mean(dim=1)  # [1, 1024]
                    projected = self.clip_model.visual_projection(global_features)  # [1, 768]
                    clip_features.append(projected.squeeze().cpu().float())
                
                del outputs
            
            return torch.stack(clip_features).to(self.device)
    
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """Extract text features using CLIP"""
        with torch.no_grad():
            text_features = []
            batch_size = 32
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.clip_processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.clip_model.text_model(**inputs)
                pooled_output = outputs.pooler_output  # [batch, 768]
                
                text_features.append(pooled_output.cpu().float())
                del outputs, pooled_output
            
            return torch.cat(text_features, dim=0).to(self.device)
    
    def compute_comprehensive_recall(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        caption_to_image: Dict[int, int],
        image_ids: List[int],
        k_values: List[int]
    ) -> Dict[str, Any]:
        """Compute comprehensive recall metrics with analysis"""
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(image_features, text_features.t())
        
        # Initialize results
        recall_results = {f'recall@{k}': 0.0 for k in k_values}
        detailed_results = []
        similarity_scores = []
        
        # For each image query
        for i, image_id in enumerate(image_ids):
            image_similarities = similarity_matrix[i]
            
            # Get ground truth caption indices for this image
            gt_caption_indices = [
                idx for idx, img_id in caption_to_image.items() 
                if img_id == image_id
            ]
            
            # Get top-K caption indices
            top_k_indices = torch.topk(image_similarities, max(k_values), dim=0)[1]
            
            # Check recall for each K
            query_results = {'image_id': image_id, 'gt_captions': len(gt_caption_indices)}
            
            for k in k_values:
                top_k = top_k_indices[:k]
                
                # Check if any ground truth caption is in top-k
                correct = any(
                    caption_idx.item() in [idx for idx in gt_caption_indices]
                    for caption_idx in top_k
                )
                
                query_results[f'recall@{k}'] = correct
                if correct:
                    recall_results[f'recall@{k}'] += 1.0
            
            # Store similarity statistics
            gt_similarities = [image_similarities[idx].item() for idx in gt_caption_indices]
            query_results['max_gt_similarity'] = max(gt_similarities) if gt_similarities else 0.0
            query_results['mean_gt_similarity'] = np.mean(gt_similarities) if gt_similarities else 0.0
            
            detailed_results.append(query_results)
            similarity_scores.extend(gt_similarities)
        
        # Normalize recall scores
        total_queries = len(image_ids)
        for k in k_values:
            recall_results[f'recall@{k}'] /= total_queries
            recall_results[f'recall@{k}'] *= 100.0
        
        # Add comprehensive analysis
        comprehensive_results = {
            'recall_scores': recall_results,
            'detailed_results': detailed_results,
            'similarity_analysis': {
                'mean_similarity': np.mean(similarity_scores) if similarity_scores else 0.0,
                'std_similarity': np.std(similarity_scores) if similarity_scores else 0.0,
                'min_similarity': np.min(similarity_scores) if similarity_scores else 0.0,
                'max_similarity': np.max(similarity_scores) if similarity_scores else 0.0,
                'similarity_distribution': np.histogram(similarity_scores, bins=20)[0].tolist() if similarity_scores else [],
            },
            'performance_analysis': {
                'queries_with_high_similarity': sum(1 for s in similarity_scores if s > 0.8) / len(similarity_scores) * 100 if similarity_scores else 0.0,
                'queries_with_medium_similarity': sum(1 for s in similarity_scores if 0.5 < s <= 0.8) / len(similarity_scores) * 100 if similarity_scores else 0.0,
                'queries_with_low_similarity': sum(1 for s in similarity_scores if s <= 0.5) / len(similarity_scores) * 100 if similarity_scores else 0.0,
            }
        }
        
        return comprehensive_results
    
    def evaluate_comprehensive(
        self,
        dataset,
        batch_size: int = 32,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation with multiple methods"""
        
        print(f"Extracting text features for {len(dataset.all_captions)} captions...")
        text_features = self.extract_text_features(dataset.all_captions)
        
        print(f"Extracting image features using multiple methods...")
        
        # Methods to evaluate
        methods = {
            'blip3o_global': ('blip3o', 'global'),
            'blip3o_patch_averaged': ('blip3o', 'patch_averaged'),
            'clip_cls': ('clip', 'cls'),
            'clip_patch_averaged': ('clip', 'patch_averaged'),
        }
        
        all_results = {}
        all_features = {}
        
        for method_name, (model_type, feature_type) in methods.items():
            print(f"Evaluating method: {method_name}")
            
            method_features = []
            
            for i in tqdm(range(0, len(dataset.image_ids), batch_size), desc=f"Processing {method_name}"):
                batch_image_ids = dataset.image_ids[i:i + batch_size]
                batch_images = []
                
                # Load images
                for image_id in batch_image_ids:
                    image_path = dataset.get_image_path(image_id)
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                    batch_images.append(image)
                
                # Extract features
                if model_type == 'blip3o':
                    batch_features = self.extract_blip3o_features(batch_images, feature_type)
                else:  # clip
                    batch_features = self.extract_clip_baseline_features(batch_images, feature_type)
                
                method_features.append(batch_features.cpu())
                
                del batch_images, batch_features
                torch.cuda.empty_cache()
            
            # Concatenate features
            method_features = torch.cat(method_features, dim=0).to(self.device)
            all_features[method_name] = method_features
            
            # Compute comprehensive recall
            method_results = self.compute_comprehensive_recall(
                method_features, text_features, dataset.caption_to_image, dataset.image_ids, k_values
            )
            
            all_results[method_name] = method_results
        
        # Compute comparisons
        comparison_results = self._compute_method_comparisons(all_results, k_values)
        
        return {
            'method_results': all_results,
            'comparisons': comparison_results,
            'evaluation_info': {
                'num_images': len(dataset.image_ids),
                'num_captions': len(dataset.all_captions),
                'methods_evaluated': list(methods.keys()),
                'k_values': k_values,
                'timestamp': datetime.now().isoformat(),
            }
        }
    
    def _compute_method_comparisons(self, all_results: Dict, k_values: List[int]) -> Dict:
        """Compute detailed comparisons between methods"""
        
        # Extract recall scores for easier comparison
        method_scores = {}
        for method_name, results in all_results.items():
            method_scores[method_name] = results['recall_scores']
        
        # Compute improvements relative to baselines
        baseline_methods = ['clip_cls', 'clip_patch_averaged']
        blip3o_methods = ['blip3o_global', 'blip3o_patch_averaged']
        
        improvements = {}
        for blip3o_method in blip3o_methods:
            for baseline_method in baseline_methods:
                comparison_key = f"{blip3o_method}_vs_{baseline_method}"
                improvements[comparison_key] = {}
                
                for k in k_values:
                    blip3o_score = method_scores[blip3o_method][f'recall@{k}']
                    baseline_score = method_scores[baseline_method][f'recall@{k}']
                    
                    abs_improvement = blip3o_score - baseline_score
                    rel_improvement = (abs_improvement / baseline_score * 100) if baseline_score > 0 else 0
                    
                    improvements[comparison_key][f'recall@{k}'] = {
                        'absolute': abs_improvement,
                        'relative': rel_improvement,
                        'blip3o_score': blip3o_score,
                        'baseline_score': baseline_score
                    }
        
        # Find best performing methods
        best_methods = {}
        for k in k_values:
            best_score = 0
            best_method = None
            for method_name, scores in method_scores.items():
                if scores[f'recall@{k}'] > best_score:
                    best_score = scores[f'recall@{k}']
                    best_method = method_name
            
            best_methods[f'recall@{k}'] = {
                'method': best_method,
                'score': best_score
            }
        
        return {
            'improvements': improvements,
            'best_methods': best_methods,
            'method_ranking': self._rank_methods(method_scores, k_values)
        }
    
    def _rank_methods(self, method_scores: Dict, k_values: List[int]) -> Dict:
        """Rank methods by performance"""
        
        # Calculate overall scores (weighted average)
        weights = {1: 0.5, 5: 0.3, 10: 0.2}  # R@1 most important
        
        overall_scores = {}
        for method_name, scores in method_scores.items():
            weighted_score = sum(
                scores[f'recall@{k}'] * weights.get(k, 1.0/len(k_values))
                for k in k_values
            )
            overall_scores[method_name] = weighted_score
        
        # Sort by overall score
        ranked_methods = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'overall_ranking': ranked_methods,
            'individual_rankings': {
                f'recall@{k}': sorted(
                    [(method, scores[f'recall@{k}']) for method, scores in method_scores.items()],
                    key=lambda x: x[1], reverse=True
                )
                for k in k_values
            }
        }

def create_evaluation_visualizations(results: Dict, output_dir: Path):
    """Create comprehensive evaluation visualizations"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Extract data for visualization
    methods = list(results['method_results'].keys())
    k_values = results['evaluation_info']['k_values']
    
    # 1. Recall scores comparison
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 6))
    if len(k_values) == 1:
        axes = [axes]
    
    for i, k in enumerate(k_values):
        scores = [
            results['method_results'][method]['recall_scores'][f'recall@{k}']
            for method in methods
        ]
        
        bars = axes[i].bar(range(len(methods)), scores)
        axes[i].set_title(f'Recall@{k} Comparison')
        axes[i].set_ylabel('Recall (%)')
        axes[i].set_xticks(range(len(methods)))
        axes[i].set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{score:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Similarity distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, method in enumerate(methods):
        similarity_data = results['method_results'][method]['similarity_analysis']
        
        # Create histogram
        bins = np.linspace(0, 1, 21)
        hist_data = similarity_data['similarity_distribution']
        if hist_data:
            axes[i].hist(bins[:-1], bins=bins, weights=hist_data, alpha=0.7, edgecolor='black')
        
        axes[i].set_title(f'{method} - Similarity Distribution')
        axes[i].set_xlabel('Similarity Score')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_sim = similarity_data['mean_similarity']
        std_sim = similarity_data['std_similarity']
        axes[i].axvline(mean_sim, color='red', linestyle='--', label=f'Mean: {mean_sim:.3f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Improvement heatmap
    improvements_data = results['comparisons']['improvements']
    
    # Create improvement matrix
    improvement_matrix = []
    labels = []
    
    for comparison_name, comparison_data in improvements_data.items():
        row = []
        for k in k_values:
            row.append(comparison_data[f'recall@{k}']['absolute'])
        improvement_matrix.append(row)
        labels.append(comparison_name.replace('_', ' ').title())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'R@{k}' for k in k_values])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(k_values)):
            text = ax.text(j, i, f'{improvement_matrix[i][j]:.1f}%',
                         ha="center", va="center", color="black", weight="bold")
    
    ax.set_title('Recall Improvements (Absolute %)')
    plt.colorbar(im, ax=ax, label='Improvement (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main comprehensive evaluation function"""
    logger = setup_logging()
    args = parse_arguments()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load models
        logger.info("Loading models for comprehensive evaluation...")
        
        from src.modules.models.blip3o_dit import load_blip3o_dit_model
        from transformers import CLIPModel, CLIPProcessor, AutoModel, CLIPImageProcessor
        
        # Load BLIP3-o model
        model = load_blip3o_dit_model(args.model_path, device=str(device))
        
        # Load CLIP models
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Load EVA-CLIP
        eva_model = AutoModel.from_pretrained("BAAI/EVA-CLIP-8B", trust_remote_code=True).to(device)
        eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        logger.info("✅ All models loaded successfully")
        
        # Load dataset
        logger.info("Loading COCO evaluation dataset...")
        from recall_evaluation import COCOEvaluationDataset
        dataset = COCOEvaluationDataset(args.coco_root, max_samples=args.max_samples)
        
        # Create comprehensive evaluator
        evaluator = ComprehensiveEvaluator(
            model=model,
            clip_model=clip_model,
            clip_processor=clip_processor,
            eva_model=eva_model,
            eva_processor=eva_processor,
            device=device
        )
        
        # Run comprehensive evaluation
        logger.info("Starting comprehensive evaluation...")
        results = evaluator.evaluate_comprehensive(
            dataset=dataset,
            batch_size=args.batch_size,
            k_values=args.k_values
        )
        
        # Print comprehensive results
        print("\n" + "="*80)
        print("COMPREHENSIVE BLIP3-o EVALUATION RESULTS")
        print("="*80)
        
        # Print method results
        for method_name, method_results in results['method_results'].items():
            print(f"\n📊 {method_name.upper()} Results:")
            for k in args.k_values:
                score = method_results['recall_scores'][f'recall@{k}']
                print(f"   Recall@{k}: {score:.1f}%")
            
            # Print similarity analysis
            sim_analysis = method_results['similarity_analysis']
            print(f"   Mean Similarity: {sim_analysis['mean_similarity']:.3f}")
            print(f"   High Similarity Queries: {method_results['performance_analysis']['queries_with_high_similarity']:.1f}%")
        
        # Print best methods
        print(f"\n🏆 BEST PERFORMING METHODS:")
        for k in args.k_values:
            best = results['comparisons']['best_methods'][f'recall@{k}']
            print(f"   Recall@{k}: {best['method']} ({best['score']:.1f}%)")
        
        # Print key improvements
        print(f"\n📈 KEY IMPROVEMENTS:")
        best_improvement = results['comparisons']['improvements']['blip3o_global_vs_clip_cls']
        for k in args.k_values:
            improvement = best_improvement[f'recall@{k}']
            print(f"   BLIP3-o vs CLIP Baseline R@{k}: {improvement['absolute']:+.1f}% ({improvement['relative']:+.1f}% relative)")
        
        # Save detailed results
        results_file = output_dir / f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Create visualizations
        if args.create_visualizations:
            logger.info("Creating evaluation visualizations...")
            create_evaluation_visualizations(results, output_dir)
            logger.info("✅ Visualizations saved")
        
        # Final assessment
        best_blip3o_improvement = max(
            results['comparisons']['improvements']['blip3o_global_vs_clip_cls'][f'recall@{k}']['absolute']
            for k in args.k_values
        )
        
        if best_blip3o_improvement > 20:
            print(f"\n🎉 OUTSTANDING! BLIP3-o shows major improvement (up to {best_blip3o_improvement:.1f}%)")
        elif best_blip3o_improvement > 10:
            print(f"\n✅ EXCELLENT! BLIP3-o shows significant improvement (up to {best_blip3o_improvement:.1f}%)")
        elif best_blip3o_improvement > 0:
            print(f"\n👍 GOOD! BLIP3-o shows improvement (up to {best_blip3o_improvement:.1f}%)")
        else:
            print(f"\n⚠️  BLIP3-o needs further optimization")
        
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)