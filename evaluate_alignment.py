#!/usr/bin/env python3
"""
BLIP3-o DiT Alignment Evaluation Script (Task 1)
FIXED: Compatible with corrected evaluator parameter names

This script evaluates the alignment between text and vision embeddings using cosine similarity:
(a) CLIP text encoder + CLIP ViT-L/14 vision encoder
(b) CLIP text encoder + Generated CLIP embeddings (EVA-CLIP -> BLIP3-o DiT)

Usage:
    python evaluate_alignment.py --blip3o_model_path <path> --coco_root <path> [options]
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
        description="BLIP3-o DiT Alignment Evaluation (Task 1)",
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
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results/alignment",
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
    
    print("🔍 BLIP3-o DiT Alignment Evaluation (Task 1)")
    print("=" * 50)
    print("This script evaluates alignment using cosine similarity:")
    print("(a) CLIP text + CLIP vision embeddings")
    print("(b) CLIP text + Generated CLIP embeddings (EVA -> BLIP3-o DiT)")
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
        
        # Run alignment evaluation
        print("🔍 Running alignment evaluation...")
        print(f"   Max samples: {args.max_samples or 'All'}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Device: {evaluator.device}")
        
        metrics = evaluator.evaluate_alignment(
            coco_root=args.coco_root,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            save_results=args.save_detailed,
            results_dir=results_dir if args.save_detailed else None,
        )
        
        # Check for errors
        if 'error' in metrics:
            print(f"❌ Evaluation failed: {metrics['error']}")
            return 1
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 ALIGNMENT EVALUATION RESULTS")
        print("=" * 60)
        
        print("\n🎯 Method (a): CLIP Text + CLIP Vision")
        method_a_metrics = {
            k.replace('clip_text_clip_vision_', ''): v 
            for k, v in metrics.items() 
            if k.startswith('clip_text_clip_vision_')
        }
        print_metrics(method_a_metrics, "CLIP Text + CLIP Vision")
        
        print("\n🎯 Method (b): CLIP Text + Generated CLIP (EVA -> BLIP3-o)")
        method_b_metrics = {
            k.replace('clip_text_generated_', ''): v 
            for k, v in metrics.items() 
            if k.startswith('clip_text_generated_')
        }
        print_metrics(method_b_metrics, "CLIP Text + Generated CLIP")
        
        print("\n📈 Comparison and Differences")
        comparison_metrics = {
            k: v for k, v in metrics.items() 
            if 'difference' in k or 'correlation' in k
        }
        print_metrics(comparison_metrics, "Method Comparison")
        
        # Summary
        method_a_mean = metrics.get('clip_text_clip_vision_mean', 0)
        method_b_mean = metrics.get('clip_text_generated_mean', 0)
        difference = metrics.get('difference_mean', 0)
        correlation = metrics.get('correlation', 0)
        
        print("\n🎯 SUMMARY")
        print("-" * 40)
        print(f"Method (a) - CLIP Text + CLIP Vision:     {method_a_mean:.4f}")
        print(f"Method (b) - CLIP Text + Generated CLIP:  {method_b_mean:.4f}")
        print(f"Difference (b - a):                       {difference:+.4f}")
        print(f"Correlation between methods:              {correlation:.4f}")
        
        if difference > 0:
            print("✅ Generated CLIP embeddings show BETTER alignment with text")
        elif difference < 0:
            print("⚠️  Generated CLIP embeddings show LOWER alignment with text")
        else:
            print("➖ Generated CLIP embeddings show SIMILAR alignment with text")
        
        # Save summary results
        summary_file = results_dir / "alignment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Summary results saved to: {summary_file}")
        
        if args.save_detailed:
            print(f"📁 Detailed results saved to: {results_dir}")
        
        print("\n✅ Alignment evaluation completed successfully!")
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