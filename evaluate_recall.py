#!/usr/bin/env python3
"""
BLIP3-o DiT Recall Evaluation Script (Task 2)

This script evaluates recall metrics (Recall@1, Recall@5, Recall@10) for image-to-text retrieval:
(a) Image → CLIP ViT-L/14 → retrieval against text captions
(b) Image → EVA-CLIP → BLIP3-o DiT → retrieval against text captions

Usage:
    python evaluate_recall.py --blip3o_model_path <path> --coco_root <path> [options]
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
from src.modules.evaluation.metrics import print_metrics, compare_metrics


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
        description="BLIP3-o DiT Recall Evaluation (Task 2)",
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
        "--k_values", type=int, nargs='+', default=[1, 5, 10],
        help="K values for Recall@K computation"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results/recall",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--save_detailed", action="store_true",
        help="Save detailed results (embeddings and similarities)"
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
    
    print("🎯 BLIP3-o DiT Recall Evaluation (Task 2)")
    print("=" * 50)
    print("This script evaluates recall metrics for image-to-text retrieval:")
    print("(a) Image → CLIP ViT-L/14 → retrieval against text captions")
    print("(b) Image → EVA-CLIP → BLIP3-o DiT → retrieval against text captions")
    print(f"Metrics: Recall@{args.k_values}")
    print("=" * 50)
    
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
        
        # Run recall evaluation
        print("🎯 Running recall evaluation...")
        print(f"   Max samples: {args.max_samples or 'All'}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   K values: {args.k_values}")
        print(f"   Device: {evaluator.device}")
        
        metrics = evaluator.evaluate_recall(
            coco_root=args.coco_root,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            k_values=args.k_values,
            save_results=args.save_detailed,
            results_dir=results_dir if args.save_detailed else None,
        )
        
        # Check for errors
        if 'error' in metrics:
            print(f"❌ Evaluation failed: {metrics['error']}")
            return 1
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 RECALL EVALUATION RESULTS")
        print("=" * 60)
        
        print("\n🎯 Method (a): Image -> CLIP Vision -> Text Retrieval")
        method_a_metrics = {
            k.replace('clip_vision_', ''): v 
            for k, v in metrics.items() 
            if k.startswith('clip_vision_') and not k.endswith(('_difference', '_relative_change'))
        }
        print_metrics(method_a_metrics, "Image -> CLIP Vision -> Text")
        
        print("\n🎯 Method (b): Image -> EVA-CLIP -> BLIP3-o -> Text Retrieval")
        method_b_metrics = {
            k.replace('generated_', ''): v 
            for k, v in metrics.items() 
            if k.startswith('generated_') and not k.endswith(('_difference', '_relative_change'))
        }
        print_metrics(method_b_metrics, "Image -> Generated CLIP -> Text")
        
        print("\n📈 Comparison and Differences")
        comparison_metrics = {
            k: v for k, v in metrics.items() 
            if 'difference' in k or 'relative_change' in k
        }
        print_metrics(comparison_metrics, "Method Comparison")
        
        # Detailed summary for each K value
        print("\n🎯 DETAILED RECALL COMPARISON")
        print("-" * 50)
        
        for k in args.k_values:
            recall_a = metrics.get(f'clip_vision_recall@{k}', 0)
            recall_b = metrics.get(f'generated_recall@{k}', 0)
            difference = metrics.get(f'recall@{k}_difference', 0)
            relative_change = metrics.get(f'recall@{k}_relative_change', 0)
            
            print(f"\nRecall@{k} (Image-to-Text Retrieval):")
            print(f"  Method (a) - Image→CLIP Vision→Text:    {recall_a:.4f} ({recall_a*100:.2f}%)")
            print(f"  Method (b) - Image→EVA→BLIP3o→Text:     {recall_b:.4f} ({recall_b*100:.2f}%)")
            print(f"  Difference (b - a):                     {difference:+.4f} ({difference*100:+.2f}%)")
            print(f"  Relative change:                        {relative_change:+.2f}%")
            
            if difference > 0:
                print(f"  ✅ EVA→BLIP3o shows BETTER Image-to-Text Recall@{k}")
            elif difference < 0:
                print(f"  ⚠️  EVA→BLIP3o shows LOWER Image-to-Text Recall@{k}")
            else:
                print(f"  ➖ EVA→BLIP3o shows SIMILAR Image-to-Text Recall@{k}")
        
        # Overall summary
        print("\n🎯 OVERALL SUMMARY")
        print("-" * 40)
        
        # Calculate average improvement
        total_improvement = 0
        count = 0
        
        for k in args.k_values:
            difference = metrics.get(f'recall@{k}_difference', 0)
            total_improvement += difference
            count += 1
        
        avg_improvement = total_improvement / count if count > 0 else 0
        
        print(f"Average image-to-text recall improvement: {avg_improvement:+.4f} ({avg_improvement*100:+.2f}%)")
        print(f"Number of images evaluated: {metrics.get('clip_vision_num_queries', 0)}")
        print(f"Text gallery size: {metrics.get('clip_vision_num_gallery', 0)}")
        
        if avg_improvement > 0.01:  # Threshold for significant improvement
            print("✅ EVA→BLIP3o embeddings show SIGNIFICANT improvement in image-to-text retrieval")
        elif avg_improvement < -0.01:
            print("⚠️  EVA→BLIP3o embeddings show SIGNIFICANT degradation in image-to-text retrieval")
        else:
            print("➖ EVA→BLIP3o embeddings show SIMILAR performance to CLIP vision in image-to-text retrieval")
        
        # Save summary results
        summary_file = results_dir / "recall_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Summary results saved to: {summary_file}")
        
        if args.save_detailed:
            print(f"📁 Detailed results saved to: {results_dir}")
        
        print("\n✅ Recall evaluation completed successfully!")
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