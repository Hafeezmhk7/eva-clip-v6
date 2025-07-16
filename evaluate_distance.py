#!/usr/bin/env python3
"""
BLIP3-o DiT Distance Evaluation Script (Task 3)
Evaluates various distance metrics between target CLIP embeddings and predicted embeddings.

This script evaluates the direct distance between:
- Target: CLIP ViT-L/14 vision embeddings (ground truth)
- Predicted: Generated CLIP embeddings (EVA-CLIP → BLIP3-o DiT)

Supports both raw embeddings and CLIP-aligned embeddings (with visual projection).

Usage:
    python evaluate_distance.py --blip3o_model_path <path> --coco_root <path> [options]
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.modules.evaluation.evaluator import BLIP3oEvaluator
from src.modules.evaluation.distance_metrics import print_distance_metrics


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
        description="BLIP3-o DiT Distance Evaluation (Task 3) - Direct Embedding Distance Metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--blip3o_model_path", type=str, required=True,
        help="Path to trained BLIP3-o DiT model directory"
    )
    parser.add_argument(
        "--coco_root", type=str, required=True,
        help="Path to MS-COCO dataset root directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--use_visual_projection", action="store_true",
        help="Use CLIP's visual projection for fair comparison (768-dim aligned space)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results/distance",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--save_detailed", action="store_true",
        help="Save detailed results (embeddings and per-sample distances)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_paths(args):
    """Validate input paths."""
    # Check BLIP3-o model path
    blip3o_path = Path(args.blip3o_model_path)
    if not blip3o_path.exists():
        raise FileNotFoundError(f"BLIP3-o model path not found: {blip3o_path}")
    
    # Check for required model files
    required_files = ["config.json", "pytorch_model.bin"]
    missing_files = []
    for file_name in required_files:
        if not (blip3o_path / file_name).exists():
            # Try alternative names
            alternatives = {
                "pytorch_model.bin": ["model.safetensors", "pytorch_model.safetensors"],
                "config.json": ["blip3o_model_config.json"]
            }
            
            found = False
            for alt in alternatives.get(file_name, []):
                if (blip3o_path / alt).exists():
                    found = True
                    break
            
            if not found:
                missing_files.append(file_name)
    
    if missing_files:
        print(f"⚠️  Warning: Some model files not found: {missing_files}")
        print("This might be okay if using alternative file names.")
    
    # Check COCO path
    coco_path = Path(args.coco_root)
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO dataset path not found: {coco_path}")
    
    # Check for COCO structure
    images_dir = coco_path / "images" / "val2017"
    annotations_file = coco_path / "annotations" / "captions_val2017.json"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO images directory not found: {images_dir}")
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"COCO annotations file not found: {annotations_file}")
    
    print(f"✅ Paths validated:")
    print(f"   BLIP3-o model: {blip3o_path}")
    print(f"   COCO dataset: {coco_path}")


def main():
    """Main evaluation function."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    print("📏 BLIP3-o DiT Distance Evaluation (Task 3)")
    print("=" * 80)
    print("🎯 Evaluates direct distance between target and predicted embeddings")
    print("")
    print("Target embeddings:    CLIP ViT-L/14 vision features")
    print("Predicted embeddings: EVA-CLIP → BLIP3-o DiT → generated features")
    print("")
    if args.use_visual_projection:
        print("🔧 Using CLIP visual projection (768-dim aligned space)")
        print("   Both embeddings projected to CLIP's aligned space for fair comparison")
    else:
        print("🔧 Using raw embeddings (1024-dim CLIP space)")
        print("   Direct comparison in CLIP's native feature space")
    print("=" * 80)
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Create results directory
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator
        print("🚀 Initializing evaluator...")
        evaluator = BLIP3oEvaluator(
            blip3o_model_path=args.blip3o_model_path,
            device=args.device,
        )
        
        # Run distance evaluation
        print("📏 Running distance evaluation...")
        print(f"   Max samples: {args.max_samples or 'All'}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Device: {evaluator.device}")
        print(f"   Visual projection: {args.use_visual_projection}")
        
        metrics = evaluator.evaluate_distance(
            coco_root=args.coco_root,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            use_visual_projection=args.use_visual_projection,
            save_results=args.save_detailed,
            results_dir=results_dir if args.save_detailed else None,
        )
        
        # Check for errors
        if 'error' in metrics:
            print(f"❌ Evaluation failed: {metrics['error']}")
            return 1
        
        # Print results
        print("\n" + "=" * 80)
        print("📊 DISTANCE EVALUATION RESULTS")
        print("=" * 80)
        
        # Print comprehensive distance metrics
        print_distance_metrics(metrics, "Target vs Predicted Embedding Distances")
        
        # Print key summary metrics
        print("\n🎯 KEY DISTANCE METRICS SUMMARY")
        print("-" * 60)
        
        key_metrics = [
            ('L2 Distance (mean)', 'l2_distance_mean'),
            ('L1 Distance (mean)', 'l1_distance_mean'),
            ('Cosine Distance (mean)', 'cosine_distance_mean'),
            ('Cosine Similarity (mean)', 'cosine_similarity_mean'),
            ('MSE Distance', 'mse_distance'),
            ('MAE Distance', 'mae_distance'),
            ('RMSE Distance', 'rmse_distance'),
        ]
        
        for name, key in key_metrics:
            value = metrics.get(key, 0)
            print(f"{name:30s}: {value:.6f}")
        
        # Interpretation
        print("\n🔍 INTERPRETATION")
        print("-" * 60)
        
        cosine_sim = metrics.get('cosine_similarity_mean', 0)
        l2_dist = metrics.get('l2_distance_mean', 0)
        
        print(f"Cosine similarity: {cosine_sim:.4f}")
        if cosine_sim > 0.9:
            print("   ✅ EXCELLENT: Predicted embeddings are very similar to targets")
        elif cosine_sim > 0.8:
            print("   ✅ GOOD: Predicted embeddings are quite similar to targets")
        elif cosine_sim > 0.7:
            print("   ⚠️  FAIR: Predicted embeddings are moderately similar to targets")
        elif cosine_sim > 0.5:
            print("   ⚠️  POOR: Predicted embeddings are somewhat different from targets")
        else:
            print("   ❌ VERY POOR: Predicted embeddings are very different from targets")
        
        print(f"\nL2 distance: {l2_dist:.6f}")
        embedding_dim = metrics.get('embedding_dimension', 768)
        normalized_l2 = l2_dist / embedding_dim**0.5  # Normalize by sqrt of dimension
        print(f"Normalized L2 (per √dim): {normalized_l2:.6f}")
        
        if normalized_l2 < 0.1:
            print("   ✅ EXCELLENT: Very low distance between embeddings")
        elif normalized_l2 < 0.2:
            print("   ✅ GOOD: Low distance between embeddings")
        elif normalized_l2 < 0.4:
            print("   ⚠️  FAIR: Moderate distance between embeddings")
        elif normalized_l2 < 0.6:
            print("   ⚠️  POOR: High distance between embeddings")
        else:
            print("   ❌ VERY POOR: Very high distance between embeddings")
        
        # Technical details
        print("\n🔬 TECHNICAL DETAILS")
        print("-" * 60)
        print(f"Embedding space: {metrics.get('embedding_space', 'unknown')}")
        print(f"Embedding dimension: {metrics.get('embedding_dimension', 'unknown')}")
        print(f"Number of images evaluated: {metrics.get('num_images_evaluated', 'unknown')}")
        print(f"Uses visual projection: {metrics.get('uses_visual_projection', 'unknown')}")
        
        if args.use_visual_projection:
            print("• Both embeddings projected to CLIP's 768-dim aligned space")
            print("• Fair comparison methodology ensures meaningful distances")
        else:
            print("• Raw 1024-dim CLIP embeddings used")
            print("• Direct comparison in CLIP's native feature space")
        
        # Comparison context
        print("\n📊 DISTANCE METRICS CONTEXT")
        print("-" * 60)
        print("• L2 Distance: Euclidean distance (sensitive to all dimensions)")
        print("• L1 Distance: Manhattan distance (less sensitive to outliers)")
        print("• Cosine Distance: 1 - cosine_similarity (direction-based)")
        print("• MSE/MAE/RMSE: Standard regression error metrics")
        print("• Lower distances = better prediction quality")
        print("• Higher cosine similarity = better directional alignment")
        
        # Save summary results
        summary_file = results_dir / "distance_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Summary results saved to: {summary_file}")
        
        if args.save_detailed:
            print(f"📁 Detailed results saved to: {results_dir}")
            print("   • distance_detailed_results.json: Per-sample distances and embeddings")
            print("   • distance_distribution_analysis.json: Distance distribution analysis")
        
        print("\n✅ Distance evaluation completed successfully!")
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