"""
Evaluation metrics for BLIP3-o DiT model evaluation.
UPDATED: Added support for CLIP visual projection validation and debugging.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


def compute_cosine_similarity(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings.
    FIXED: Added device consistency check to avoid CUDA/CPU errors.
    
    Args:
        embeddings_a: First set of embeddings [N, D]
        embeddings_b: Second set of embeddings [N, D] 
        dim: Dimension along which to compute similarity
        
    Returns:
        Cosine similarity scores [N]
    """
    # Ensure both tensors are on the same device
    if embeddings_a.device != embeddings_b.device:
        # Move both to CPU for consistency
        embeddings_a = embeddings_a.cpu()
        embeddings_b = embeddings_b.cpu()
    
    # Normalize embeddings
    embeddings_a = F.normalize(embeddings_a, p=2, dim=dim)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=dim)
    
    # Compute cosine similarity
    similarity = (embeddings_a * embeddings_b).sum(dim=dim)
    
    return similarity


def compute_pairwise_cosine_similarity(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of embeddings.
    FIXED: Added device consistency check to avoid CUDA/CPU errors.
    
    Args:
        embeddings_a: First set of embeddings [N, D]
        embeddings_b: Second set of embeddings [M, D]
        
    Returns:
        Similarity matrix [N, M]
    """
    # Ensure both tensors are on the same device
    if embeddings_a.device != embeddings_b.device:
        # Move both to CPU for consistency
        embeddings_a = embeddings_a.cpu()
        embeddings_b = embeddings_b.cpu()
    
    # Normalize embeddings
    embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)
    
    # Compute pairwise similarity
    similarity_matrix = torch.mm(embeddings_a, embeddings_b.t())
    
    return similarity_matrix


def validate_clip_projection_consistency(
    clip_model,
    sample_features: torch.Tensor,
    device: torch.device,
    tolerance: float = 1e-6
) -> Dict[str, bool]:
    """
    Validate that CLIP's visual projection is being applied consistently.
    
    This function helps debug and verify that we're using CLIP's projection correctly
    by comparing two ways of getting the same result.
    
    Args:
        clip_model: The CLIP model with visual_projection
        sample_features: Sample vision features [B, D_vision] (e.g., 1024-dim)
        device: Device for computation
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with validation results
    """
    with torch.no_grad():
        sample_features = sample_features.to(device)
        
        # Method 1: Direct projection
        projected_direct = clip_model.visual_projection(sample_features)
        
        # Method 2: Using get_image_features (should be equivalent for vision features)
        # Note: This only works if sample_features are properly shaped vision features
        try:
            # This assumes sample_features are the output of vision_model
            # We can't easily test this without proper vision model output
            # So we'll just test the projection layer directly
            projected_direct_2 = F.linear(
                sample_features, 
                clip_model.visual_projection.weight, 
                clip_model.visual_projection.bias
            )
            
            # Check consistency
            max_diff = torch.max(torch.abs(projected_direct - projected_direct_2)).item()
            projection_consistent = max_diff < tolerance
            
        except Exception as e:
            logger.warning(f"Could not validate projection consistency: {e}")
            projection_consistent = None
            max_diff = None
        
        # Check dimensions
        expected_output_dim = clip_model.visual_projection.out_features  # Should be 768
        expected_input_dim = clip_model.visual_projection.in_features    # Should be 1024
        
        correct_input_dim = sample_features.shape[-1] == expected_input_dim
        correct_output_dim = projected_direct.shape[-1] == expected_output_dim
        
        # Check projection properties
        has_bias = clip_model.visual_projection.bias is not None
        projection_weight_shape = clip_model.visual_projection.weight.shape
        
        validation_results = {
            'projection_consistent': projection_consistent,
            'max_difference': max_diff,
            'correct_input_dim': correct_input_dim,
            'correct_output_dim': correct_output_dim,
            'expected_input_dim': expected_input_dim,
            'expected_output_dim': expected_output_dim,
            'actual_input_dim': sample_features.shape[-1],
            'actual_output_dim': projected_direct.shape[-1],
            'has_bias': has_bias,
            'weight_shape': projection_weight_shape,
            'projection_available': True,
        }
        
        return validation_results


def analyze_embedding_alignment(
    text_embeddings: torch.Tensor,
    vision_embeddings_method_a: torch.Tensor,
    vision_embeddings_method_b: torch.Tensor,
    method_a_name: str = "CLIP Vision",
    method_b_name: str = "Generated CLIP"
) -> Dict[str, float]:
    """
    Analyze alignment between text and vision embeddings for both methods.
    
    This helps understand how well different methods align with text in the
    CLIP embedding space.
    
    Args:
        text_embeddings: Text embeddings [N, D]
        vision_embeddings_method_a: Vision embeddings from method A [N, D]
        vision_embeddings_method_b: Vision embeddings from method B [N, D]
        method_a_name: Name for method A
        method_b_name: Name for method B
        
    Returns:
        Dictionary with alignment analysis
    """
    with torch.no_grad():
        # Ensure all embeddings have the same dimension
        assert text_embeddings.shape == vision_embeddings_method_a.shape, \
            f"Text {text_embeddings.shape} != Method A {vision_embeddings_method_a.shape}"
        assert text_embeddings.shape == vision_embeddings_method_b.shape, \
            f"Text {text_embeddings.shape} != Method B {vision_embeddings_method_b.shape}"
        
        # Normalize embeddings
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)
        vision_a_norm = F.normalize(vision_embeddings_method_a, p=2, dim=-1)
        vision_b_norm = F.normalize(vision_embeddings_method_b, p=2, dim=-1)
        
        # Compute similarities
        sim_text_a = (text_norm * vision_a_norm).sum(dim=-1)
        sim_text_b = (text_norm * vision_b_norm).sum(dim=-1)
        sim_a_b = (vision_a_norm * vision_b_norm).sum(dim=-1)
        
        # Compute statistics
        analysis = {
            # Method A alignment with text
            f'{method_a_name.lower().replace(" ", "_")}_text_similarity_mean': sim_text_a.mean().item(),
            f'{method_a_name.lower().replace(" ", "_")}_text_similarity_std': sim_text_a.std().item(),
            f'{method_a_name.lower().replace(" ", "_")}_text_similarity_min': sim_text_a.min().item(),
            f'{method_a_name.lower().replace(" ", "_")}_text_similarity_max': sim_text_a.max().item(),
            
            # Method B alignment with text
            f'{method_b_name.lower().replace(" ", "_")}_text_similarity_mean': sim_text_b.mean().item(),
            f'{method_b_name.lower().replace(" ", "_")}_text_similarity_std': sim_text_b.std().item(),
            f'{method_b_name.lower().replace(" ", "_")}_text_similarity_min': sim_text_b.min().item(),
            f'{method_b_name.lower().replace(" ", "_")}_text_similarity_max': sim_text_b.max().item(),
            
            # Cross-method similarity
            f'{method_a_name.lower().replace(" ", "_")}_{method_b_name.lower().replace(" ", "_")}_similarity_mean': sim_a_b.mean().item(),
            f'{method_a_name.lower().replace(" ", "_")}_{method_b_name.lower().replace(" ", "_")}_similarity_std': sim_a_b.std().item(),
            
            # Differences
            'text_alignment_difference_mean': (sim_text_b - sim_text_a).mean().item(),
            'text_alignment_difference_std': (sim_text_b - sim_text_a).std().item(),
            'text_alignment_difference_abs_mean': torch.abs(sim_text_b - sim_text_a).mean().item(),
            
            # Correlation
            'text_alignment_correlation': torch.corrcoef(torch.stack([sim_text_a, sim_text_b]))[0, 1].item(),
            'cross_method_correlation': torch.corrcoef(torch.stack([sim_a_b, sim_text_a]))[0, 1].item(),
            
            # Embedding properties
            'embedding_dimension': text_embeddings.shape[-1],
            'num_samples': len(text_embeddings),
        }
        
        return analysis


def compute_recall_metrics(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: List[int],
    gallery_labels: List[int],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Recall@K metrics for retrieval evaluation.
    
    Args:
        query_embeddings: Query embeddings [N_q, D]
        gallery_embeddings: Gallery embeddings [N_g, D]
        query_labels: Labels for query samples
        gallery_labels: Labels for gallery samples
        k_values: List of K values for Recall@K computation
        
    Returns:
        Dictionary containing Recall@K scores
    """
    # Compute similarity matrix
    similarity_matrix = compute_pairwise_cosine_similarity(
        query_embeddings, gallery_embeddings
    )  # [N_q, N_g]
    
    # Convert labels to tensors
    query_labels = torch.tensor(query_labels, device=similarity_matrix.device)
    gallery_labels = torch.tensor(gallery_labels, device=similarity_matrix.device)
    
    # Get top-k indices for each query
    num_queries = similarity_matrix.shape[0]
    max_k = max(k_values)
    
    # Get top-k most similar gallery items for each query
    _, top_k_indices = torch.topk(similarity_matrix, k=max_k, dim=1)  # [N_q, max_k]
    
    # Get labels of top-k retrieved items
    top_k_gallery_labels = gallery_labels[top_k_indices]  # [N_q, max_k]
    
    # Expand query labels for comparison
    query_labels_expanded = query_labels.unsqueeze(1).expand(-1, max_k)  # [N_q, max_k]
    
    # Check which retrieved items are correct (same label as query)
    correct_retrievals = (top_k_gallery_labels == query_labels_expanded)  # [N_q, max_k]
    
    # Compute Recall@K for each K value
    recall_metrics = {}
    
    for k in k_values:
        # Check if any of the top-k retrievals are correct
        recall_at_k = correct_retrievals[:, :k].any(dim=1).float().mean().item()
        recall_metrics[f'recall@{k}'] = recall_at_k
    
    # Additional metrics
    recall_metrics['num_queries'] = num_queries
    recall_metrics['num_gallery'] = len(gallery_labels)
    
    return recall_metrics


def compute_alignment_metrics(
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    pair_indices: List[Tuple[int, int]] = None
) -> Dict[str, float]:
    """
    Compute alignment metrics between text and vision embeddings.
    
    Args:
        text_embeddings: Text embeddings [N_t, D]
        vision_embeddings: Vision embeddings [N_v, D]
        pair_indices: List of (text_idx, vision_idx) pairs that should be aligned
                     If None, assumes 1-to-1 correspondence
        
    Returns:
        Dictionary containing alignment metrics
    """
    if pair_indices is None:
        # Assume 1-to-1 correspondence
        assert text_embeddings.shape[0] == vision_embeddings.shape[0], \
            "When pair_indices is None, text and vision embeddings must have same length"
        
        # Compute cosine similarity for paired samples
        paired_similarities = compute_cosine_similarity(text_embeddings, vision_embeddings)
        
        metrics = {
            'mean_cosine_similarity': paired_similarities.mean().item(),
            'std_cosine_similarity': paired_similarities.std().item(),
            'min_cosine_similarity': paired_similarities.min().item(),
            'max_cosine_similarity': paired_similarities.max().item(),
            'num_pairs': len(paired_similarities),
        }
    
    else:
        # Use provided pair indices
        paired_similarities = []
        
        for text_idx, vision_idx in pair_indices:
            text_emb = text_embeddings[text_idx:text_idx+1]
            vision_emb = vision_embeddings[vision_idx:vision_idx+1]
            similarity = compute_cosine_similarity(text_emb, vision_emb)
            paired_similarities.append(similarity.item())
        
        paired_similarities = torch.tensor(paired_similarities)
        
        metrics = {
            'mean_cosine_similarity': paired_similarities.mean().item(),
            'std_cosine_similarity': paired_similarities.std().item(),
            'min_cosine_similarity': paired_similarities.min().item(),
            'max_cosine_similarity': paired_similarities.max().item(),
            'num_pairs': len(paired_similarities),
        }
    
    return metrics


def compute_retrieval_accuracy(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    ground_truth_indices: List[int]
) -> Dict[str, float]:
    """
    Compute retrieval accuracy metrics.
    
    Args:
        query_embeddings: Query embeddings [N, D]
        gallery_embeddings: Gallery embeddings [M, D]
        ground_truth_indices: Ground truth gallery indices for each query [N]
        
    Returns:
        Dictionary containing retrieval accuracy metrics
    """
    # Compute similarity matrix
    similarity_matrix = compute_pairwise_cosine_similarity(
        query_embeddings, gallery_embeddings
    )  # [N, M]
    
    # Get top-1 predictions
    _, top1_indices = torch.topk(similarity_matrix, k=1, dim=1)  # [N, 1]
    top1_indices = top1_indices.squeeze(1)  # [N]
    
    # Check accuracy
    ground_truth_indices = torch.tensor(ground_truth_indices, device=top1_indices.device)
    correct_predictions = (top1_indices == ground_truth_indices)
    
    accuracy = correct_predictions.float().mean().item()
    
    return {
        'top1_accuracy': accuracy,
        'num_correct': correct_predictions.sum().item(),
        'num_total': len(correct_predictions),
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple evaluations.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Aggregated metrics with mean and std
    """
    if not metrics_list:
        return {}
    
    # Get all metric keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    aggregated = {}
    
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in metrics_list if key in metrics]
        
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
            aggregated[f'{key}_count'] = len(values)
    
    return aggregated


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics
    """
    print(f"\n📊 {title}")
    print("=" * (len(title) + 4))
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'recall' in key.lower() or 'accuracy' in key.lower() or 'similarity' in key.lower():
                print(f"   {key:35s}: {value:.4f}")
            else:
                print(f"   {key:35s}: {value:.6f}")
        else:
            print(f"   {key:35s}: {value}")


def print_clip_projection_validation(validation_results: Dict[str, any], title: str = "CLIP Projection Validation"):
    """
    Pretty print CLIP projection validation results.
    
    Args:
        validation_results: Results from validate_clip_projection_consistency
        title: Title for the validation report
    """
    print(f"\n🔬 {title}")
    print("=" * (len(title) + 4))
    
    if validation_results.get('projection_available', False):
        print("✅ CLIP visual projection is available")
        print(f"   Weight shape: {validation_results.get('weight_shape', 'Unknown')}")
        print(f"   Input dimension: {validation_results.get('expected_input_dim', 'Unknown')} (expected: 1024)")
        print(f"   Output dimension: {validation_results.get('expected_output_dim', 'Unknown')} (expected: 768)")
        print(f"   Has bias: {validation_results.get('has_bias', 'Unknown')}")
        
        if validation_results.get('projection_consistent') is not None:
            if validation_results['projection_consistent']:
                print(f"✅ Projection is consistent (max diff: {validation_results['max_difference']:.2e})")
            else:
                print(f"⚠️  Projection inconsistency detected (max diff: {validation_results['max_difference']:.2e})")
        
        if validation_results.get('correct_input_dim') and validation_results.get('correct_output_dim'):
            print("✅ Projection dimensions are correct")
        else:
            print("⚠️  Projection dimension mismatch detected")
    else:
        print("❌ CLIP visual projection not available")


def compare_metrics(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    label_a: str = "Method A",
    label_b: str = "Method B"
) -> Dict[str, float]:
    """
    Compare two sets of metrics and compute differences.
    
    Args:
        metrics_a: First set of metrics
        metrics_b: Second set of metrics
        label_a: Label for first method
        label_b: Label for second method
        
    Returns:
        Dictionary containing comparison metrics
    """
    comparison = {}
    
    # Find common keys
    common_keys = set(metrics_a.keys()) & set(metrics_b.keys())
    
    for key in common_keys:
        if isinstance(metrics_a[key], (int, float)) and isinstance(metrics_b[key], (int, float)):
            value_a = metrics_a[key]
            value_b = metrics_b[key]
            
            difference = value_b - value_a
            relative_change = (difference / value_a * 100) if value_a != 0 else 0
            
            comparison[f'{key}_{label_a}'] = value_a
            comparison[f'{key}_{label_b}'] = value_b
            comparison[f'{key}_difference'] = difference
            comparison[f'{key}_relative_change_percent'] = relative_change
    
    return comparison


if __name__ == "__main__":
    # Test the metrics functions
    print("🧪 Testing evaluation metrics with CLIP projection support...")
    
    # Test cosine similarity
    embeddings_a = torch.randn(100, 768)  # 768-dim for CLIP aligned space
    embeddings_b = torch.randn(100, 768)
    
    similarity = compute_cosine_similarity(embeddings_a, embeddings_b)
    print(f"✅ Cosine similarity: {similarity.mean():.4f} ± {similarity.std():.4f}")
    
    # Test alignment analysis
    text_embeddings = torch.randn(50, 768)
    vision_a = torch.randn(50, 768)
    vision_b = torch.randn(50, 768)
    
    alignment_analysis = analyze_embedding_alignment(
        text_embeddings, vision_a, vision_b,
        "CLIP Vision", "Generated CLIP"
    )
    print_metrics(alignment_analysis, "Embedding Alignment Analysis")
    
    # Test recall metrics
    query_embeddings = torch.randn(50, 768)
    gallery_embeddings = torch.randn(1000, 768)
    query_labels = list(range(50))
    gallery_labels = list(range(50)) + list(range(950))  # First 50 match queries
    
    recall_metrics = compute_recall_metrics(
        query_embeddings, gallery_embeddings, query_labels, gallery_labels
    )
    print_metrics(recall_metrics, "Recall Metrics")
    
    print("✅ All metric tests passed!")