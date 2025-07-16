"""
Distance Metrics for BLIP3-o DiT Evaluation (Task 3)
Computes various distance metrics between target CLIP embeddings and predicted embeddings.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


def compute_l2_distance(
    target_embeddings: torch.Tensor,
    predicted_embeddings: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute L2 (Euclidean) distance between target and predicted embeddings.
    
    Args:
        target_embeddings: Target CLIP embeddings [N, D]
        predicted_embeddings: Predicted embeddings [N, D]
        reduction: How to reduce distances ("mean", "sum", "none")
        
    Returns:
        L2 distances
    """
    # Ensure same device
    if target_embeddings.device != predicted_embeddings.device:
        target_embeddings = target_embeddings.cpu()
        predicted_embeddings = predicted_embeddings.cpu()
    
    # Compute L2 distance
    distances = torch.norm(target_embeddings - predicted_embeddings, p=2, dim=-1)
    
    if reduction == "mean":
        return distances.mean()
    elif reduction == "sum":
        return distances.sum()
    else:
        return distances


def compute_l1_distance(
    target_embeddings: torch.Tensor,
    predicted_embeddings: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute L1 (Manhattan) distance between target and predicted embeddings.
    
    Args:
        target_embeddings: Target CLIP embeddings [N, D]
        predicted_embeddings: Predicted embeddings [N, D]
        reduction: How to reduce distances ("mean", "sum", "none")
        
    Returns:
        L1 distances
    """
    # Ensure same device
    if target_embeddings.device != predicted_embeddings.device:
        target_embeddings = target_embeddings.cpu()
        predicted_embeddings = predicted_embeddings.cpu()
    
    # Compute L1 distance
    distances = torch.norm(target_embeddings - predicted_embeddings, p=1, dim=-1)
    
    if reduction == "mean":
        return distances.mean()
    elif reduction == "sum":
        return distances.sum()
    else:
        return distances


def compute_cosine_distance(
    target_embeddings: torch.Tensor,
    predicted_embeddings: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute cosine distance (1 - cosine_similarity) between target and predicted embeddings.
    
    Args:
        target_embeddings: Target CLIP embeddings [N, D]
        predicted_embeddings: Predicted embeddings [N, D]
        reduction: How to reduce distances ("mean", "sum", "none")
        
    Returns:
        Cosine distances
    """
    # Ensure same device
    if target_embeddings.device != predicted_embeddings.device:
        target_embeddings = target_embeddings.cpu()
        predicted_embeddings = predicted_embeddings.cpu()
    
    # Normalize embeddings
    target_norm = F.normalize(target_embeddings, p=2, dim=-1)
    predicted_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
    
    # Compute cosine similarity
    cosine_sim = (target_norm * predicted_norm).sum(dim=-1)
    
    # Convert to cosine distance
    cosine_dist = 1.0 - cosine_sim
    
    if reduction == "mean":
        return cosine_dist.mean()
    elif reduction == "sum":
        return cosine_dist.sum()
    else:
        return cosine_dist


def compute_mse_distance(
    target_embeddings: torch.Tensor,
    predicted_embeddings: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Mean Squared Error between target and predicted embeddings.
    
    Args:
        target_embeddings: Target CLIP embeddings [N, D]
        predicted_embeddings: Predicted embeddings [N, D]
        reduction: How to reduce distances ("mean", "sum", "none")
        
    Returns:
        MSE distances
    """
    # Ensure same device
    if target_embeddings.device != predicted_embeddings.device:
        target_embeddings = target_embeddings.cpu()
        predicted_embeddings = predicted_embeddings.cpu()
    
    # Compute MSE
    mse = F.mse_loss(predicted_embeddings, target_embeddings, reduction=reduction)
    
    return mse


def compute_mae_distance(
    target_embeddings: torch.Tensor,
    predicted_embeddings: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Mean Absolute Error between target and predicted embeddings.
    
    Args:
        target_embeddings: Target CLIP embeddings [N, D]
        predicted_embeddings: Predicted embeddings [N, D]
        reduction: How to reduce distances ("mean", "sum", "none")
        
    Returns:
        MAE distances
    """
    # Ensure same device
    if target_embeddings.device != predicted_embeddings.device:
        target_embeddings = target_embeddings.cpu()
        predicted_embeddings = predicted_embeddings.cpu()
    
    # Compute MAE
    mae = F.l1_loss(predicted_embeddings, target_embeddings, reduction=reduction)
    
    return mae


def compute_comprehensive_distance_metrics(
    target_embeddings: torch.Tensor,      # [N, D] - Target CLIP embeddings
    predicted_embeddings: torch.Tensor,   # [N, D] - Predicted embeddings
) -> Dict[str, float]:
    """
    Compute comprehensive distance metrics between target and predicted embeddings.
    
    Args:
        target_embeddings: Target CLIP embeddings [N, D]
        predicted_embeddings: Predicted embeddings [N, D]
        
    Returns:
        Dictionary containing various distance metrics
    """
    with torch.no_grad():
        # Ensure same shape
        assert target_embeddings.shape == predicted_embeddings.shape, \
            f"Shape mismatch: target {target_embeddings.shape} vs predicted {predicted_embeddings.shape}"
        
        # Ensure same device
        if target_embeddings.device != predicted_embeddings.device:
            target_embeddings = target_embeddings.cpu()
            predicted_embeddings = predicted_embeddings.cpu()
        
        # Basic distance metrics
        l2_distances = compute_l2_distance(target_embeddings, predicted_embeddings, reduction="none")
        l1_distances = compute_l1_distance(target_embeddings, predicted_embeddings, reduction="none")
        cosine_distances = compute_cosine_distance(target_embeddings, predicted_embeddings, reduction="none")
        
        # Statistical metrics
        metrics = {
            # L2 (Euclidean) distance statistics
            'l2_distance_mean': l2_distances.mean().item(),
            'l2_distance_std': l2_distances.std().item(),
            'l2_distance_min': l2_distances.min().item(),
            'l2_distance_max': l2_distances.max().item(),
            'l2_distance_median': l2_distances.median().item(),
            
            # L1 (Manhattan) distance statistics
            'l1_distance_mean': l1_distances.mean().item(),
            'l1_distance_std': l1_distances.std().item(),
            'l1_distance_min': l1_distances.min().item(),
            'l1_distance_max': l1_distances.max().item(),
            'l1_distance_median': l1_distances.median().item(),
            
            # Cosine distance statistics
            'cosine_distance_mean': cosine_distances.mean().item(),
            'cosine_distance_std': cosine_distances.std().item(),
            'cosine_distance_min': cosine_distances.min().item(),
            'cosine_distance_max': cosine_distances.max().item(),
            'cosine_distance_median': cosine_distances.median().item(),
            
            # MSE and MAE
            'mse_distance': compute_mse_distance(target_embeddings, predicted_embeddings).item(),
            'mae_distance': compute_mae_distance(target_embeddings, predicted_embeddings).item(),
            'rmse_distance': torch.sqrt(compute_mse_distance(target_embeddings, predicted_embeddings)).item(),
        }
        
        # Additional analysis metrics
        # Embedding norm comparison
        target_norms = torch.norm(target_embeddings, p=2, dim=-1)
        predicted_norms = torch.norm(predicted_embeddings, p=2, dim=-1)
        
        metrics.update({
            'target_norm_mean': target_norms.mean().item(),
            'predicted_norm_mean': predicted_norms.mean().item(),
            'norm_difference_mean': (predicted_norms - target_norms).mean().item(),
            'norm_difference_abs_mean': torch.abs(predicted_norms - target_norms).mean().item(),
            'norm_ratio_mean': (predicted_norms / (target_norms + 1e-8)).mean().item(),
        })
        
        # Cosine similarity (complement of cosine distance)
        cosine_similarities = 1.0 - cosine_distances
        metrics.update({
            'cosine_similarity_mean': cosine_similarities.mean().item(),
            'cosine_similarity_std': cosine_similarities.std().item(),
            'cosine_similarity_min': cosine_similarities.min().item(),
            'cosine_similarity_max': cosine_similarities.max().item(),
        })
        
        # Relative error metrics
        # Avoid division by zero
        target_norms_safe = torch.clamp(target_norms, min=1e-8)
        relative_l2_error = l2_distances / target_norms_safe
        
        metrics.update({
            'relative_l2_error_mean': relative_l2_error.mean().item(),
            'relative_l2_error_std': relative_l2_error.std().item(),
            'relative_l2_error_median': relative_l2_error.median().item(),
        })
        
        # Token-wise analysis (if embeddings have multiple tokens)
        if len(target_embeddings.shape) == 3:  # [N, num_tokens, D]
            # Token-wise L2 distances
            token_l2_distances = torch.norm(target_embeddings - predicted_embeddings, p=2, dim=-1)  # [N, num_tokens]
            
            metrics.update({
                'token_l2_distance_mean': token_l2_distances.mean().item(),
                'token_l2_distance_std': token_l2_distances.std().item(),
                'token_l2_distance_min': token_l2_distances.min().item(),
                'token_l2_distance_max': token_l2_distances.max().item(),
                'worst_token_l2_distance_mean': token_l2_distances.max(dim=-1)[0].mean().item(),
                'best_token_l2_distance_mean': token_l2_distances.min(dim=-1)[0].mean().item(),
            })
        
        # Add sample count
        metrics['num_samples'] = len(target_embeddings)
        metrics['embedding_dimension'] = target_embeddings.shape[-1]
        
        return metrics


def compute_per_sample_distances(
    target_embeddings: torch.Tensor,      # [N, D] - Target CLIP embeddings
    predicted_embeddings: torch.Tensor,   # [N, D] - Predicted embeddings
) -> Dict[str, torch.Tensor]:
    """
    Compute per-sample distance metrics for detailed analysis.
    
    Args:
        target_embeddings: Target CLIP embeddings [N, D]
        predicted_embeddings: Predicted embeddings [N, D]
        
    Returns:
        Dictionary containing per-sample distance tensors
    """
    with torch.no_grad():
        # Ensure same device
        if target_embeddings.device != predicted_embeddings.device:
            target_embeddings = target_embeddings.cpu()
            predicted_embeddings = predicted_embeddings.cpu()
        
        per_sample_metrics = {
            'l2_distances': compute_l2_distance(target_embeddings, predicted_embeddings, reduction="none"),
            'l1_distances': compute_l1_distance(target_embeddings, predicted_embeddings, reduction="none"),
            'cosine_distances': compute_cosine_distance(target_embeddings, predicted_embeddings, reduction="none"),
            'cosine_similarities': 1.0 - compute_cosine_distance(target_embeddings, predicted_embeddings, reduction="none"),
        }
        
        # Add embedding norms
        per_sample_metrics.update({
            'target_norms': torch.norm(target_embeddings, p=2, dim=-1),
            'predicted_norms': torch.norm(predicted_embeddings, p=2, dim=-1),
        })
        
        return per_sample_metrics


def analyze_distance_distribution(
    target_embeddings: torch.Tensor,
    predicted_embeddings: torch.Tensor,
    num_bins: int = 50
) -> Dict[str, any]:
    """
    Analyze the distribution of distances between target and predicted embeddings.
    
    Args:
        target_embeddings: Target CLIP embeddings [N, D]
        predicted_embeddings: Predicted embeddings [N, D]
        num_bins: Number of bins for histogram analysis
        
    Returns:
        Dictionary containing distribution analysis
    """
    with torch.no_grad():
        # Get per-sample distances
        per_sample = compute_per_sample_distances(target_embeddings, predicted_embeddings)
        
        analysis = {}
        
        for metric_name, distances in per_sample.items():
            if 'distances' in metric_name or 'similarities' in metric_name:
                distances_np = distances.cpu().numpy()
                
                # Compute percentiles
                percentiles = [5, 10, 25, 50, 75, 90, 95]
                percentile_values = np.percentile(distances_np, percentiles)
                
                analysis[f'{metric_name}_percentiles'] = {
                    f'p{p}': v for p, v in zip(percentiles, percentile_values)
                }
                
                # Compute histogram
                hist, bin_edges = np.histogram(distances_np, bins=num_bins)
                analysis[f'{metric_name}_histogram'] = {
                    'counts': hist.tolist(),
                    'bin_edges': bin_edges.tolist(),
                }
        
        return analysis


def print_distance_metrics(metrics: Dict[str, float], title: str = "Distance Metrics"):
    """
    Pretty print distance metrics.
    
    Args:
        metrics: Dictionary of distance metrics
        title: Title for the metrics report
    """
    print(f"\n📏 {title}")
    print("=" * (len(title) + 4))
    
    # Group metrics by type
    metric_groups = {
        'L2 Distance': [k for k in metrics.keys() if 'l2_distance' in k],
        'L1 Distance': [k for k in metrics.keys() if 'l1_distance' in k],
        'Cosine Distance': [k for k in metrics.keys() if 'cosine_distance' in k],
        'Cosine Similarity': [k for k in metrics.keys() if 'cosine_similarity' in k],
        'MSE/MAE/RMSE': [k for k in metrics.keys() if any(x in k for x in ['mse', 'mae', 'rmse'])],
        'Norm Analysis': [k for k in metrics.keys() if 'norm' in k],
        'Relative Error': [k for k in metrics.keys() if 'relative' in k],
        'Token Analysis': [k for k in metrics.keys() if 'token' in k],
        'General Info': [k for k in metrics.keys() if k in ['num_samples', 'embedding_dimension']],
    }
    
    for group_name, metric_keys in metric_groups.items():
        if metric_keys:
            print(f"\n   {group_name}:")
            for key in metric_keys:
                value = metrics[key]
                if isinstance(value, float):
                    if 'distance' in key or 'error' in key:
                        print(f"      {key:35s}: {value:.6f}")
                    elif 'similarity' in key or 'ratio' in key:
                        print(f"      {key:35s}: {value:.4f}")
                    else:
                        print(f"      {key:35s}: {value:.4f}")
                else:
                    print(f"      {key:35s}: {value}")


if __name__ == "__main__":
    # Test the distance metrics functions
    print("🧪 Testing distance metrics...")
    
    # Create dummy data
    target = torch.randn(100, 768)  # Target embeddings
    predicted = target + torch.randn(100, 768) * 0.1  # Predicted with some noise
    
    # Test comprehensive metrics
    metrics = compute_comprehensive_distance_metrics(target, predicted)
    print_distance_metrics(metrics, "Test Distance Metrics")
    
    # Test per-sample metrics
    per_sample = compute_per_sample_distances(target, predicted)
    print(f"\n✅ Per-sample metrics computed for {len(per_sample)} metric types")
    
    # Test distribution analysis
    distribution = analyze_distance_distribution(target, predicted)
    print(f"✅ Distribution analysis completed")
    
    print("✅ All distance metric tests passed!")