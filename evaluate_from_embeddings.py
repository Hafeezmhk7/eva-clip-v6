#!/usr/bin/env python3
"""
Fast COCO Recall Evaluation from Pre-computed Embeddings

This script performs rapid recall evaluation using pre-extracted embeddings,
avoiding the need to recompute embeddings every time. Perfect for quick
experimentation and validation.

Features:
- Load pre-computed embeddings (pickle or npz format)
- Fast recall computation (no model inference needed)
- Support for different evaluation modes
- Detailed analysis and comparison
- Multiple output formats

Usage:
    python evaluate_from_embeddings.py --embeddings_file <path> [options]
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fast COCO Recall Evaluation from Pre-computed Embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--embeddings_file", type=str, required=True,
        help="Path to pre-computed embeddings file (.pkl or .npz)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--k_values", type=int, nargs='+', default=[1, 5, 10],
        help="K values for Recall@K computation"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results/fast_recall",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--save_detailed", action="store_true",
        help="Save detailed results and analysis"
    )
    parser.add_argument(
        "--compare_raw", action="store_true",
        help="Also compare raw embeddings (if available)"
    )
    parser.add_argument(
        "--analyze_similarities", action="store_true",
        help="Perform detailed similarity analysis"
    )
    parser.add_argument(
        "--validate_baseline", action="store_true",
        help="Validate CLIP baseline against expected ranges"
    )
    parser.add_argument(
        "--subset_size", type=int, default=None,
        help="Use only a subset of embeddings for faster testing"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_embeddings(embeddings_file: Union[str, Path], logger) -> Dict:
    """Load pre-computed embeddings from file."""
    embeddings_file = Path(embeddings_file)
    
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    logger.info(f"Loading embeddings from {embeddings_file}...")
    
    if embeddings_file.suffix == '.pkl':
        # Load pickle file
        with open(embeddings_file, 'rb') as f:
            embeddings_data = pickle.load(f)
    
    elif embeddings_file.suffix == '.npz':
        # Load NPZ file
        npz_data = np.load(embeddings_file)
        
        embeddings_data = {
            'clip_vision_embeddings': torch.from_numpy(npz_data['clip_vision_embeddings']),
            'generated_clip_embeddings': torch.from_numpy(npz_data['generated_clip_embeddings']),
            'text_embeddings': torch.from_numpy(npz_data['text_embeddings']),
            'image_ids': npz_data['image_ids'].tolist(),
            'captions': npz_data['captions'].tolist(),
            'image_paths': npz_data['image_paths'].tolist(),
            'metadata': {
                'num_samples': len(npz_data['image_ids']),
                'embedding_dim': npz_data['clip_vision_embeddings'].shape[1],
                'loaded_from': 'npz',
            }
        }
        
        # Load raw embeddings if available
        if 'clip_vision_raw' in npz_data:
            embeddings_data['clip_vision_raw'] = torch.from_numpy(npz_data['clip_vision_raw'])
            embeddings_data['generated_clip_raw'] = torch.from_numpy(npz_data['generated_clip_raw'])
    
    else:
        raise ValueError(f"Unsupported file format: {embeddings_file.suffix}")
    
    logger.info(f"✅ Loaded {embeddings_data['metadata']['num_samples']} samples")
    
    return embeddings_data


def validate_embeddings(embeddings_data: Dict, logger) -> Dict[str, bool]:
    """Validate loaded embeddings."""
    logger.info("Validating embeddings...")
    
    validation_results = {}
    
    # Check shapes
    clip_vision = embeddings_data['clip_vision_embeddings']
    generated = embeddings_data['generated_clip_embeddings']
    text = embeddings_data['text_embeddings']
    
    logger.info(f"Embedding shapes:")
    logger.info(f"  CLIP vision: {clip_vision.shape}")
    logger.info(f"  Generated:   {generated.shape}")
    logger.info(f"  Text:        {text.shape}")
    
    # Check normalization
    def check_normalization(emb_tensor, name):
        norms = torch.norm(emb_tensor, p=2, dim=-1)
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
        max_deviation = torch.abs(norms - 1.0).max().item()
        
        is_normalized = abs(mean_norm - 1.0) < 0.01
        
        logger.info(f"  {name}: mean norm = {mean_norm:.6f}, std = {std_norm:.6f}, max dev = {max_deviation:.6f}")
        
        return {
            'is_normalized': is_normalized,
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'max_deviation': max_deviation,
        }
    
    validation_results['clip_vision'] = check_normalization(clip_vision, "CLIP vision")
    validation_results['generated'] = check_normalization(generated, "Generated")
    validation_results['text'] = check_normalization(text, "Text")
    
    # Check consistency
    num_samples = len(embeddings_data['image_ids'])
    shapes_consistent = (
        clip_vision.shape[0] == num_samples and
        generated.shape[0] == num_samples and
        text.shape[0] == num_samples
    )
    
    validation_results['shapes_consistent'] = shapes_consistent
    validation_results['num_samples'] = num_samples
    
    if shapes_consistent:
        logger.info("✅ All embedding shapes are consistent")
    else:
        logger.warning("⚠️  Embedding shapes are inconsistent!")
    
    return validation_results


def compute_recall_fast(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: List[int],
    method_name: str = "Method"
) -> Dict[str, float]:
    """
    Fast recall computation for pre-computed embeddings.
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]
        k_values: K values for Recall@K
        method_name: Name for logging
        
    Returns:
        Dictionary of recall metrics
    """
    logger = logging.getLogger(__name__)
    
    # Ensure normalized
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Compute similarity matrix: [N, N] for single caption evaluation
    similarity_matrix = torch.mm(image_embeddings, text_embeddings.t())
    
    logger.info(f"{method_name} similarity matrix: {similarity_matrix.shape}")
    logger.info(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    logger.info(f"Similarity mean: {similarity_matrix.mean():.4f}, std: {similarity_matrix.std():.4f}")
    
    # For single caption evaluation, create simple 1-to-1 mapping
    num_samples = similarity_matrix.shape[0]
    image_to_text_mapping = [[i] for i in range(num_samples)]
    
    # Compute recall for each K
    recall_results = {}
    
    for k in k_values:
        correct_retrievals = 0
        total_queries = num_samples
        
        # Get top-k text indices for each image
        _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)  # [N, k]
        
        # Debug: Show first few retrievals for k=1
        if k == 1:
            logger.info(f"DEBUG: First 5 {method_name} retrievals:")
            for i in range(min(5, num_samples)):
                correct_texts = image_to_text_mapping[i]
                retrieved_text = top_k_indices[i, 0].item()
                similarity_score = similarity_matrix[i, retrieved_text].item()
                is_correct = retrieved_text in correct_texts
                logger.info(f"  Image {i}: retrieved text {retrieved_text}, similarity {similarity_score:.4f}, "
                           f"correct: {is_correct}, expected: {correct_texts}")
        
        # Check recall for each image
        for img_idx, correct_text_indices in enumerate(image_to_text_mapping):
            retrieved_indices = top_k_indices[img_idx].cpu().numpy()
            
            # Check if any retrieved text is correct for this image
            if any(ret_idx in correct_text_indices for ret_idx in retrieved_indices):
                correct_retrievals += 1
        
        recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0
        recall_results[f'recall@{k}'] = recall_at_k
        
        logger.info(f"{method_name} Recall@{k}: {correct_retrievals}/{total_queries} = {recall_at_k:.4f} ({recall_at_k*100:.2f}%)")
    
    # Additional metrics
    recall_results.update({
        'num_queries': total_queries,
        'num_gallery': num_samples,
        'embedding_dim': image_embeddings.shape[1],
        'similarity_mean': similarity_matrix.mean().item(),
        'similarity_std': similarity_matrix.std().item(),
        'similarity_min': similarity_matrix.min().item(),
        'similarity_max': similarity_matrix.max().item(),
    })
    
    return recall_results


def analyze_similarities(
    clip_embeddings: torch.Tensor,
    generated_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_ids: List,
    captions: List
) -> Dict[str, any]:
    """Perform detailed similarity analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Performing detailed similarity analysis...")
    
    # Normalize embeddings
    clip_norm = F.normalize(clip_embeddings, p=2, dim=-1)
    generated_norm = F.normalize(generated_embeddings, p=2, dim=-1)
    text_norm = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Compute all similarity matrices
    clip_text_sim = torch.mm(clip_norm, text_norm.t())
    generated_text_sim = torch.mm(generated_norm, text_norm.t())
    clip_generated_sim = torch.mm(clip_norm, generated_norm.t())
    
    # Extract diagonal (correct pairs)
    clip_text_correct = torch.diagonal(clip_text_sim)
    generated_text_correct = torch.diagonal(generated_text_sim)
    clip_generated_correct = torch.diagonal(clip_generated_sim)
    
    analysis = {
        'clip_text_similarities': {
            'correct_pairs_mean': clip_text_correct.mean().item(),
            'correct_pairs_std': clip_text_correct.std().item(),
            'correct_pairs_min': clip_text_correct.min().item(),
            'correct_pairs_max': clip_text_correct.max().item(),
            'all_pairs_mean': clip_text_sim.mean().item(),
            'all_pairs_std': clip_text_sim.std().item(),
        },
        'generated_text_similarities': {
            'correct_pairs_mean': generated_text_correct.mean().item(),
            'correct_pairs_std': generated_text_correct.std().item(),
            'correct_pairs_min': generated_text_correct.min().item(),
            'correct_pairs_max': generated_text_correct.max().item(),
            'all_pairs_mean': generated_text_sim.mean().item(),
            'all_pairs_std': generated_text_sim.std().item(),
        },
        'clip_generated_similarities': {
            'correct_pairs_mean': clip_generated_correct.mean().item(),
            'correct_pairs_std': clip_generated_correct.std().item(),
            'correct_pairs_min': clip_generated_correct.min().item(),
            'correct_pairs_max': clip_generated_correct.max().item(),
            'all_pairs_mean': clip_generated_sim.mean().item(),
            'all_pairs_std': clip_generated_sim.std().item(),
        },
        'improvement_analysis': {
            'mean_improvement': (generated_text_correct - clip_text_correct).mean().item(),
            'std_improvement': (generated_text_correct - clip_text_correct).std().item(),
            'positive_improvements': (generated_text_correct > clip_text_correct).sum().item(),
            'negative_improvements': (generated_text_correct < clip_text_correct).sum().item(),
            'correlation': torch.corrcoef(torch.stack([clip_text_correct, generated_text_correct]))[0, 1].item(),
        }
    }
    
    # Find best and worst improvements
    improvements = generated_text_correct - clip_text_correct
    
    # Top 5 improvements
    top_improvements_idx = torch.argsort(improvements, descending=True)[:5]
    # Bottom 5 improvements  
    bottom_improvements_idx = torch.argsort(improvements, descending=False)[:5]
    
    analysis['sample_analysis'] = {
        'top_improvements': [
            {
                'index': idx.item(),
                'image_id': image_ids[idx],
                'caption': captions[idx][:100] + "..." if len(captions[idx]) > 100 else captions[idx],
                'clip_similarity': clip_text_correct[idx].item(),
                'generated_similarity': generated_text_correct[idx].item(),
                'improvement': improvements[idx].item(),
            }
            for idx in top_improvements_idx
        ],
        'bottom_improvements': [
            {
                'index': idx.item(),
                'image_id': image_ids[idx],
                'caption': captions[idx][:100] + "..." if len(captions[idx]) > 100 else captions[idx],
                'clip_similarity': clip_text_correct[idx].item(),
                'generated_similarity': generated_text_correct[idx].item(),
                'improvement': improvements[idx].item(),
            }
            for idx in bottom_improvements_idx
        ]
    }
    
    return analysis


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n📊 {title}")
    print("=" * (len(title) + 4))
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if 'recall' in key.lower() or 'similarity' in key.lower():
                print(f"   {key:35s}: {value:.4f}")
            else:
                print(f"   {key:35s}: {value:.6f}")
        else:
            print(f"   {key:35s}: {value}")


def validate_clip_baseline(clip_recall_metrics: Dict[str, float]) -> Dict[str, bool]:
    """Validate CLIP baseline against expected community ranges."""
    expected_ranges = {
        'recall@1': (0.30, 0.37),   # 30-37%
        'recall@5': (0.55, 0.62),   # 55-62%
        'recall@10': (0.65, 0.72),  # 65-72%
    }
    
    validation = {}
    
    for metric, (min_val, max_val) in expected_ranges.items():
        if metric in clip_recall_metrics:
            actual_val = clip_recall_metrics[metric]
            is_valid = min_val <= actual_val <= max_val
            validation[metric] = {
                'is_valid': is_valid,
                'actual': actual_val,
                'expected_range': (min_val, max_val),
                'status': '✅' if is_valid else '⚠️',
            }
    
    return validation


def main():
    """Main evaluation function."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    print("⚡ Fast COCO Recall Evaluation from Pre-computed Embeddings")
    print("=" * 70)
    print(f"📁 Embeddings file: {args.embeddings_file}")
    print(f"📊 K values: {args.k_values}")
    print(f"📁 Results directory: {args.results_dir}")
    if args.subset_size:
        print(f"🔬 Using subset: {args.subset_size} samples")
    print("=" * 70)
    
    try:
        # Load embeddings
        embeddings_data = load_embeddings(args.embeddings_file, logger)
        
        # Validate embeddings
        validation_results = validate_embeddings(embeddings_data, logger)
        
        # Apply subset if requested
        if args.subset_size and args.subset_size < embeddings_data['metadata']['num_samples']:
            logger.info(f"Using subset of {args.subset_size} samples")
            for key in ['clip_vision_embeddings', 'generated_clip_embeddings', 'text_embeddings']:
                embeddings_data[key] = embeddings_data[key][:args.subset_size]
            embeddings_data['image_ids'] = embeddings_data['image_ids'][:args.subset_size]
            embeddings_data['captions'] = embeddings_data['captions'][:args.subset_size]
            embeddings_data['image_paths'] = embeddings_data['image_paths'][:args.subset_size]
        
        # Extract embeddings
        clip_vision_emb = embeddings_data['clip_vision_embeddings']
        generated_emb = embeddings_data['generated_clip_embeddings']
        text_emb = embeddings_data['text_embeddings']
        
        actual_samples = clip_vision_emb.shape[0]
        logger.info(f"Running evaluation on {actual_samples} samples")
        
        # Compute recall for both methods
        print(f"\n🚀 Computing recall metrics for {actual_samples} samples...")
        
        # Method A: CLIP vision → text retrieval
        clip_recall = compute_recall_fast(
            image_embeddings=clip_vision_emb,
            text_embeddings=text_emb,
            k_values=args.k_values,
            method_name="CLIP_Vision"
        )
        
        # Method B: Generated → text retrieval
        generated_recall = compute_recall_fast(
            image_embeddings=generated_emb,
            text_embeddings=text_emb,
            k_values=args.k_values,
            method_name="Generated"
        )
        
        # Combine results
        combined_metrics = {}
        
        for key, value in clip_recall.items():
            combined_metrics[f'clip_vision_{key}'] = value
        
        for key, value in generated_recall.items():
            combined_metrics[f'generated_{key}'] = value
        
        # Compute differences
        for k in args.k_values:
            recall_a = clip_recall[f'recall@{k}']
            recall_b = generated_recall[f'recall@{k}']
            combined_metrics[f'recall@{k}_difference'] = recall_b - recall_a
            combined_metrics[f'recall@{k}_relative_change'] = (recall_b - recall_a) / recall_a * 100 if recall_a > 0 else 0
        
        # Print results
        print("\n" + "=" * 70)
        print("📊 FAST RECALL EVALUATION RESULTS")
        print("=" * 70)
        
        print("\n🎯 Method (a): CLIP Vision → Text Retrieval")
        clip_metrics = {k.replace('clip_vision_', ''): v for k, v in combined_metrics.items() 
                       if k.startswith('clip_vision_') and not k.endswith(('_difference', '_relative_change'))}
        print_metrics(clip_metrics, "CLIP Vision → Text")
        
        print("\n🎯 Method (b): Generated → Text Retrieval")
        generated_metrics = {k.replace('generated_', ''): v for k, v in combined_metrics.items() 
                            if k.startswith('generated_') and not k.endswith(('_difference', '_relative_change'))}
        print_metrics(generated_metrics, "Generated → Text")
        
        print("\n📈 Comparison and Differences")
        comparison_metrics = {k: v for k, v in combined_metrics.items() 
                             if 'difference' in k or 'relative_change' in k}
        print_metrics(comparison_metrics, "Method Comparison")
        
        # Validate CLIP baseline if requested
        if args.validate_baseline:
            print("\n🔍 CLIP Baseline Validation")
            validation = validate_clip_baseline(clip_metrics)
            
            for metric, info in validation.items():
                print(f"   {metric:15s}: {info['actual']:.4f} {info['status']} "
                      f"(expected: {info['expected_range'][0]:.2f}-{info['expected_range'][1]:.2f})")
        
        # Detailed similarity analysis
        if args.analyze_similarities:
            print("\n🔬 Performing detailed similarity analysis...")
            similarity_analysis = analyze_similarities(
                clip_vision_emb, generated_emb, text_emb,
                embeddings_data['image_ids'], embeddings_data['captions']
            )
            
            print("\n📈 Similarity Analysis Summary:")
            print(f"   Mean improvement: {similarity_analysis['improvement_analysis']['mean_improvement']:+.4f}")
            print(f"   Positive improvements: {similarity_analysis['improvement_analysis']['positive_improvements']}/{actual_samples}")
            print(f"   Correlation: {similarity_analysis['improvement_analysis']['correlation']:.4f}")
        
        # Compare raw embeddings if available and requested
        if args.compare_raw and 'clip_vision_raw' in embeddings_data:
            print("\n🔧 Comparing raw embeddings (before projection)...")
            
            clip_raw = embeddings_data['clip_vision_raw']
            generated_raw = embeddings_data['generated_clip_raw']
            
            # Note: Can't compare raw to text directly since text is in 768-dim space
            # But we can compare the raw embeddings to each other
            clip_raw_norm = F.normalize(clip_raw, p=2, dim=-1)
            generated_raw_norm = F.normalize(generated_raw, p=2, dim=-1)
            
            raw_similarity = F.cosine_similarity(clip_raw_norm, generated_raw_norm)
            print(f"   Raw embedding similarity: {raw_similarity.mean():.4f} ± {raw_similarity.std():.4f}")
        
        # Save results if requested
        if args.save_detailed:
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save combined metrics
            results_file = results_dir / f"fast_recall_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(combined_metrics, f, indent=2)
            
            # Save detailed analysis if performed
            if args.analyze_similarities:
                analysis_file = results_dir / f"similarity_analysis_{timestamp}.json"
                with open(analysis_file, 'w') as f:
                    json.dump(similarity_analysis, f, indent=2)
            
            print(f"\n💾 Results saved to {results_dir}")
        
        # Final summary
        print("\n🎯 SUMMARY")
        print("-" * 40)
        
        for k in args.k_values:
            recall_a = combined_metrics[f'clip_vision_recall@{k}']
            recall_b = combined_metrics[f'generated_recall@{k}']
            difference = combined_metrics[f'recall@{k}_difference']
            
            print(f"Recall@{k:2d}: CLIP {recall_a:.3f} vs Generated {recall_b:.3f} "
                  f"({difference:+.3f})")
        
        # Performance info
        evaluation_time = datetime.datetime.now()
        print(f"\n⚡ Fast evaluation completed at {evaluation_time.strftime('%H:%M:%S')}")
        print(f"📊 Evaluated {actual_samples} samples instantly (no model inference needed)")
        
        print("\n✅ Fast evaluation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)