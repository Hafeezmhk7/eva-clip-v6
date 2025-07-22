"""
Evaluation utilities for BLIP3-o DiT - Image-to-Text Recall

Contains:
- BLIP3oRecallEvaluator: Primary evaluation for image-to-text recall
- Evaluation utilities and metrics
- Model comparison utilities
"""

import logging

logger = logging.getLogger(__name__)

# Import recall evaluator from correct path
RECALL_EVALUATOR_AVAILABLE = False
try:
    from .evaluation.blip3o_recall_evaluator import (
        BLIP3oRecallEvaluator,
        create_recall_evaluator,
    )
    logger.debug("✅ BLIP3-o recall evaluator loaded")
    RECALL_EVALUATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ BLIP3-o recall evaluator not available: {e}")
    BLIP3oRecallEvaluator = None
    create_recall_evaluator = None

# Set defaults
BLIP3oEvaluator = BLIP3oRecallEvaluator
create_evaluator = create_recall_evaluator
DEFAULT_EVALUATOR = "recall"

if RECALL_EVALUATOR_AVAILABLE:
    logger.info("✅ Using BLIP3-o recall evaluator as default")
else:
    logger.warning("❌ No evaluators available")

__all__ = [
    "RECALL_EVALUATOR_AVAILABLE",
    "DEFAULT_EVALUATOR",
]

# Export recall evaluator if available
if RECALL_EVALUATOR_AVAILABLE:
    __all__.extend([
        "BLIP3oRecallEvaluator",
        "create_recall_evaluator",
        "BLIP3oEvaluator",
        "create_evaluator",
    ])

def get_evaluator_class(evaluator_type: str = "auto"):
    """
    Get the evaluator class
    
    Args:
        evaluator_type: "auto" or "recall"
        
    Returns:
        Evaluator class
    """
    if evaluator_type in ("auto", "recall"):
        if not RECALL_EVALUATOR_AVAILABLE:
            raise RuntimeError("Recall evaluator not available")
        return BLIP3oRecallEvaluator
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")

def get_evaluator_factory(evaluator_type: str = "auto"):
    """
    Get the evaluator factory function
    
    Args:
        evaluator_type: "auto" or "recall"
        
    Returns:
        Evaluator factory function
    """
    if evaluator_type in ("auto", "recall"):
        if not RECALL_EVALUATOR_AVAILABLE:
            raise RuntimeError("Recall evaluator not available")
        return create_recall_evaluator
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")

def print_evaluation_status():
    """Print status of available evaluation utilities"""
    print("📊 BLIP3-o Evaluation Status")
    print("=" * 35)
    print(f"Default evaluator: {DEFAULT_EVALUATOR}")
    print()
    print("Available evaluators:")
    
    if RECALL_EVALUATOR_AVAILABLE:
        print("  ✅ Recall Evaluator (Primary)")
        print("    - Image-to-text recall metrics")
        print("    - Recall@1, Recall@5, Recall@10")
        print("    - Baseline comparison")
        print("    - Quality metrics")
    else:
        print("  ❌ Recall Evaluator")
    
    print()
    print("Evaluation metrics:")
    print("  🎯 Primary: Image-to-text recall")
    print("  📈 Quality: Embedding similarity")
    print("  🔍 Comparison: Baseline vs BLIP3-o")
    
    print("=" * 35)

# Add utility functions to exports
__all__.extend([
    "get_evaluator_class",
    "get_evaluator_factory",
    "print_evaluation_status",
])

# Log evaluation module status
if RECALL_EVALUATOR_AVAILABLE:
    logger.info("BLIP3-o evaluation utilities loaded successfully")
else:
    logger.warning("BLIP3-o evaluation utilities not fully available")