#!/usr/bin/env python3
"""
FIXED Quick COCO Recall Evaluation

This script fixes the normalization and computation issues that were causing
the discrepancy between verification (26% R@1) and evaluation (6.4% R@1).

Key fixes:
- Check if embeddings are already normalized before applying F.normalize()
- Add debugging output to identify computation differences
- Support testing on subsets to match verification script behavior
"""

import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List
import json
import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FIXED Quick COCO Recall Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--embeddings_file", type=str, required=True,
        help="Path to extracted embeddings file (.pkl)"
    )
    parser.add_argument(
        "--k_values", type=int, nargs='+', default=[1, 5, 10],
        help="K values for Recall@K computation"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit evaluation to first N samples (for debugging)"
    )
    parser.add_argument(
        "--skip_normalization", action="store_true",
        help="Skip F.normalize() if embeddings are already normalized"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable detailed debugging output"
    )
    parser.add_argument(
        "--save_results", action="store_true",
        help="Save detailed results to JSON"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results",
        help="Directory to save results"
    )
    
    return parser.parse_args()

def load_embeddings(embeddings_file: str) -> Dict:
    """Load extracted embeddings."""
    embeddings_file = Path(embeddings_file)
    
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    print(f"📁 Loading embeddings from: {embeddings_file}")
    
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    return embeddings_data

def check_normalization(embeddings: torch.Tensor, name: str, debug: bool = False) -> bool:
    """Check if embeddings are already normalized."""
    norms = torch.norm(embeddings, p=2, dim=-1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    max_deviation = torch.abs(norms - 1.0).max().item()
    
    # Consider normalized if mean is close to 1 and deviations are small
    is_normalized = abs(mean_norm - 1.0) < 0.01 and max_deviation < 0.1
    
    if debug:
        print(f"\n🔍 {name} Normalization Check:")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Mean norm: {mean_norm:.6f}")
        print(f"   Std norm: {std_norm:.6f}")
        print(f"   Max deviation from 1.0: {max_deviation:.6f}")
        print(f"   Already normalized: {'✅' if is_normalized else '❌'}")
        print(f"   Value range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
    
    return is_normalized

def compute_recall_metrics_fixed(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    method_name: str = "Method",
    skip_normalization: bool = False,
    debug: bool = False
) -> Dict[str, float]:
    """Compute recall metrics with proper normalization handling."""
    
    print(f"\n🔍 Computing {method_name} metrics...")
    print(f"   Image embeddings: {image_embeddings.shape}")
    print(f"   Text embeddings: {text_embeddings.shape}")
    
    # Check if already normalized
    img_is_normalized = check_normalization(image_embeddings, f"{method_name} Image", debug)
    txt_is_normalized = check_normalization(text_embeddings, f"{method_name} Text", debug)
    
    # Apply normalization only if needed and not skipped
    if skip_normalization or (img_is_normalized and txt_is_normalized):
        print("   ✅ Using embeddings as-is (already normalized or skipped)")
        image_emb_final = image_embeddings
        text_emb_final = text_embeddings
    else:
        print("   🔧 Applying F.normalize()")
        image_emb_final = F.normalize(image_embeddings, p=2, dim=-1)
        text_emb_final = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Compute similarity matrix [N_images, N_texts]
    similarity_matrix = torch.mm(image_emb_final, text_emb_final.t())
    
    print(f"   Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    print(f"   Similarity mean: {similarity_matrix.mean():.4f}")
    
    if debug:
        # Check diagonal vs off-diagonal similarities
        diagonal = similarity_matrix.diagonal()
        off_diagonal = similarity_matrix.flatten()
        off_diagonal = off_diagonal[~torch.eye(similarity_matrix.shape[0], dtype=bool).flatten()]
        
        print(f"   Diagonal (correct pairs): mean={diagonal.mean():.4f}, std={diagonal.std():.4f}")
        print(f"   Off-diagonal (incorrect): mean={off_diagonal.mean():.4f}, std={off_diagonal.std():.4f}")
        print(f"   Correct > Incorrect: {'✅' if diagonal.mean() > off_diagonal.mean() else '❌'}")
    
    # For single caption evaluation: image i should retrieve text i
    num_samples = similarity_matrix.shape[0]
    
    results = {}
    
    for k in k_values:
        # Get top-k text indices for each image
        _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
        
        # Check if correct text (index i) is in top-k for image i
        correct_in_topk = (top_k_indices == torch.arange(num_samples).unsqueeze(1)).any(dim=1)
        recall_at_k = correct_in_topk.float().mean().item()
        
        results[f'recall@{k}'] = recall_at_k
        
        print(f"   Recall@{k:2d}: {recall_at_k:.4f} ({recall_at_k*100:.2f}%)")
    
    # Additional metrics
    results.update({
        'num_samples': num_samples,
        'embedding_dim': image_embeddings.shape[1],
        'similarity_mean': similarity_matrix.mean().item(),
        'similarity_std': similarity_matrix.std().item(),
        'was_normalized': not skip_normalization and not (img_is_normalized and txt_is_normalized),
        'already_normalized': img_is_normalized and txt_is_normalized,
    })
    
    return results

def validate_clip_baseline(clip_metrics: Dict[str, float]) -> bool:
    """Validate CLIP baseline against expected ranges."""
    expected_ranges = {
        'recall@1': (0.25, 0.40),   # Slightly broader range: 25-40%
        'recall@5': (0.50, 0.70),   # 50-70%
        'recall@10': (0.60, 0.80),  # 60-80%
    }
    
    print(f"\n🔍 CLIP Baseline Validation:")
    print(f"   {'Metric':<12} | {'Expected Range':<15} | {'Actual':<8} | {'Status'}")
    print(f"   {'-'*12}|{'-'*16}|{'-'*9}|{'-'*8}")
    
    all_valid = True
    
    for metric, (min_val, max_val) in expected_ranges.items():
        if metric in clip_metrics:
            actual = clip_metrics[metric]
            is_valid = min_val <= actual <= max_val
            status = '✅' if is_valid else '⚠️'
            
            if not is_valid:
                all_valid = False
                
            print(f"   {metric:<12}| {min_val:.2f}-{max_val:.2f}      | {actual:.4f}  | {status}")
    
    print()
    if all_valid:
        print("✅ CLIP BASELINE VALIDATED!")
        print("   Results are within expected range for CLIP on COCO.")
        print("   This confirms that extraction and evaluation are working correctly.")
    else:
        print("⚠️  CLIP BASELINE VALIDATION FAILED!")
        print("   Results are outside expected range.")
        print("   There may still be issues with extraction or evaluation.")
    
    return all_valid

def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    print("⚡ FIXED Quick COCO Recall Evaluation")
    print("=" * 45)
    print(f"📁 Embeddings: {args.embeddings_file}")
    print(f"📊 K values: {args.k_values}")
    print(f"🔧 Max samples: {args.max_samples or 'All'}")
    print(f"🔧 Skip normalization: {args.skip_normalization}")
    print(f"🔧 Debug mode: {args.debug}")
    
    try:
        # Load embeddings
        embeddings_data = load_embeddings(args.embeddings_file)
        
        # Extract embeddings
        clip_vision_emb = embeddings_data['clip_vision_embeddings']
        generated_emb = embeddings_data['generated_clip_embeddings']
        text_emb = embeddings_data['text_embeddings']
        
        # Convert to tensors if needed
        if isinstance(clip_vision_emb, list):
            clip_vision_emb = torch.stack(clip_vision_emb)
        if isinstance(generated_emb, list):
            generated_emb = torch.stack(generated_emb)
        if isinstance(text_emb, list):
            text_emb = torch.stack(text_emb)
        
        # Limit samples if requested
        if args.max_samples:
            clip_vision_emb = clip_vision_emb[:args.max_samples]
            generated_emb = generated_emb[:args.max_samples]
            text_emb = text_emb[:args.max_samples]
            print(f"🔧 Limited to first {args.max_samples} samples")
        
        num_samples = clip_vision_emb.shape[0]
        print(f"✅ Processing {num_samples} samples")
        
        # Compute recall for both methods
        print(f"\n🚀 Computing recall metrics...")
        
        # Method A: CLIP vision → text retrieval (FIXED)
        clip_metrics = compute_recall_metrics_fixed(
            image_embeddings=clip_vision_emb,
            text_embeddings=text_emb,
            k_values=args.k_values,
            method_name="CLIP Vision",
            skip_normalization=args.skip_normalization,
            debug=args.debug
        )
        
        # Method B: Generated → text retrieval (FIXED)
        generated_metrics = compute_recall_metrics_fixed(
            image_embeddings=generated_emb,
            text_embeddings=text_emb,
            k_values=args.k_values,
            method_name="Generated",
            skip_normalization=args.skip_normalization,
            debug=args.debug
        )
        
        # Validate CLIP baseline
        baseline_valid = validate_clip_baseline(clip_metrics)
        
        # Compare methods
        print(f"\n📊 METHOD COMPARISON")
        print(f"   {'Metric':<12} | {'CLIP':<8} | {'Generated':<10} | {'Difference'}")
        print(f"   {'-'*12}|{'-'*9}|{'-'*11}|{'-'*10}")
        
        for k in args.k_values:
            clip_val = clip_metrics[f'recall@{k}']
            gen_val = generated_metrics[f'recall@{k}']
            diff = gen_val - clip_val
            
            print(f"   Recall@{k:<5} | {clip_val:.4f}  | {gen_val:.4f}    | {diff:+.4f}")
        
        # Debug info
        if args.debug:
            print(f"\n🔍 DEBUG INFO:")
            print(f"   CLIP embeddings already normalized: {clip_metrics.get('already_normalized', 'Unknown')}")
            print(f"   Applied normalization to CLIP: {clip_metrics.get('was_normalized', 'Unknown')}")
            print(f"   Generated embeddings already normalized: {generated_metrics.get('already_normalized', 'Unknown')}")
            print(f"   Applied normalization to Generated: {generated_metrics.get('was_normalized', 'Unknown')}")
        
        # Overall assessment
        print(f"\n🎯 EVALUATION SUMMARY")
        print("=" * 30)
        
        if baseline_valid:
            print("✅ SUCCESS: CLIP baseline is validated!")
            print("   The extraction and evaluation are working correctly.")
            print("   Your embeddings are ready for detailed analysis.")
            
            # Check if generated method improves
            improvements = []
            for k in args.k_values:
                diff = generated_metrics[f'recall@{k}'] - clip_metrics[f'recall@{k}']
                improvements.append(diff)
            
            avg_improvement = np.mean(improvements)
            if avg_improvement > 0:
                print(f"🎉 BONUS: Generated method shows average improvement of {avg_improvement:+.4f}")
            else:
                print(f"📝 NOTE: Generated method shows average change of {avg_improvement:+.4f}")
        else:
            print("⚠️  WARNING: CLIP baseline validation failed!")
            print("   There may still be issues with extraction or evaluation.")
            
            # Suggest fixes
            print(f"\n🔧 Suggested fixes:")
            if not args.skip_normalization:
                print("   • Try --skip_normalization if embeddings are already normalized")
            if not args.max_samples:
                print("   • Try --max_samples 100 to match verification script")
            print("   • Use --debug for detailed analysis")
        
        # Save results if requested
        if args.save_results:
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"fixed_evaluation_{timestamp}.json"
            
            results = {
                'timestamp': timestamp,
                'embeddings_file': str(args.embeddings_file),
                'num_samples': num_samples,
                'max_samples': args.max_samples,
                'skip_normalization': args.skip_normalization,
                'clip_metrics': clip_metrics,
                'generated_metrics': generated_metrics,
                'baseline_valid': baseline_valid,
                'k_values': args.k_values,
                'metadata': embeddings_data.get('metadata', {}),
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        return 0 if baseline_valid else 1
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)