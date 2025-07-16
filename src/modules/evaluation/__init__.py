"""
Evaluation module for BLIP3-o DiT model.

Contains:
- COCO dataset loading utilities
- Alignment evaluation metrics  
- Recall evaluation metrics
- Main evaluator class
"""

from .coco_dataset import COCOEvaluationDataset, create_coco_dataloader
from .metrics import compute_cosine_similarity, compute_recall_metrics
from .evaluator import BLIP3oEvaluator

__all__ = [
    "COCOEvaluationDataset",
    "create_coco_dataloader", 
    "compute_cosine_similarity",
    "compute_recall_metrics",
    "BLIP3oEvaluator",
]

from .distance_metrics import (
    compute_comprehensive_distance_metrics,
    compute_per_sample_distances,
    analyze_distance_distribution,
    print_distance_metrics
)

__all__ = [
    "compute_comprehensive_distance_metrics",
    "compute_per_sample_distances", 
    "analyze_distance_distribution",
    "print_distance_metrics",
 ]