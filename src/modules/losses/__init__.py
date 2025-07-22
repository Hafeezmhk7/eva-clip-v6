"""
Loss functions module for BLIP3-o DiT - Patch-Level Training (Paper-Aligned)

Contains:
- BLIP3oFlowMatchingLoss: Patch-level flow matching loss (primary)
- Enhanced contrastive loss for better image-text alignment
- Factory functions for creating loss instances
- Paper-aligned training objectives
"""

import logging

logger = logging.getLogger(__name__)

# Import patch-level flow matching loss (main loss function following BLIP3-o paper)
PATCH_FLOW_MATCHING_AVAILABLE = False
BLIP3oFlowMatchingLoss = None
create_blip3o_flow_matching_loss = None

try:
    from .blip3o_flow_matching_loss import (
        BLIP3oFlowMatchingLoss,
        create_blip3o_flow_matching_loss,
    )
    PATCH_FLOW_MATCHING_AVAILABLE = True
    logger.info("✅ BLIP3-o patch-level flow matching loss loaded successfully")
    
except ImportError as e:
    PATCH_FLOW_MATCHING_AVAILABLE = False
    logger.error(f"❌ Failed to load patch-level flow matching loss: {e}")
    raise ImportError(f"BLIP3-o patch-level flow matching loss is required but failed to load: {e}")

# Use patch-level flow matching as the main loss (paper-aligned)
create_blip3o_loss = create_blip3o_flow_matching_loss
DEFAULT_LOSS_TYPE = "patch_level"

logger.info("✅ Using BLIP3-o patch-level flow matching loss as primary loss")

# Build exports list
__all__ = [
    # Primary loss interface (paper-aligned)
    "BLIP3oFlowMatchingLoss",
    "create_blip3o_flow_matching_loss",
    "create_blip3o_loss",
    "DEFAULT_LOSS_TYPE",
    
    # Availability flag
    "PATCH_FLOW_MATCHING_AVAILABLE",
]

def get_loss_function(loss_type: str = "auto", enhanced: bool = True, **kwargs):
    """
    Get the appropriate loss function (always patch-level for paper alignment)
    
    Args:
        loss_type: Ignored, always returns patch-level flow matching loss
        enhanced: Whether to use enhanced version with contrastive loss
        **kwargs: Arguments to pass to loss function factory
        
    Returns:
        BLIP3oFlowMatchingLoss instance
    """
    if not PATCH_FLOW_MATCHING_AVAILABLE:
        raise RuntimeError("BLIP3-o patch-level flow matching loss not available")
    
    return create_blip3o_flow_matching_loss(enhanced=enhanced, **kwargs)

def create_loss(enhanced: bool = True, **kwargs):
    """
    Create a BLIP3-o loss function (always patch-level for paper alignment)
    
    Args:
        enhanced: Whether to use enhanced version with contrastive loss
        **kwargs: Loss configuration arguments
        
    Returns:
        BLIP3oFlowMatchingLoss instance
    """
    if not PATCH_FLOW_MATCHING_AVAILABLE:
        raise RuntimeError("BLIP3-o patch-level flow matching loss not available")
    
    return create_blip3o_flow_matching_loss(enhanced=enhanced, **kwargs)

def print_loss_status():
    """Print status of available loss functions"""
    print("📉 BLIP3-o Loss Functions Status")
    print("=" * 40)
    print(f"Loss type: {DEFAULT_LOSS_TYPE}")
    print()
    print("Available loss function (Paper-Aligned):")
    
    if PATCH_FLOW_MATCHING_AVAILABLE:
        print("  ✅ Patch-Level Flow Matching (Primary Loss)")
        print("    - Direct patch-level supervision")
        print("    - 256-token flow matching training")
        print("    - Rectified flow velocity prediction")
        print("    - Enhanced contrastive loss option")
        print("    - Patch-level alignment optimization")
        print("    - Global coherence loss")
        print("    - Optimized for image-text recall")
        print("    - Paper-aligned training objective")
    else:
        print("  ❌ Patch-Level Flow Matching (REQUIRED)")
    
    print()
    print("Loss components:")
    print("  🎯 Flow Matching: Velocity prediction on CLIP patches")
    print("  🔗 Contrastive: Patch-level alignment")
    print("  🌐 Global: Overall coherence")
    print("  📊 Metrics: Recall optimization")
    
    print()
    print("Training objective:")
    print("  📐 Input: Noisy CLIP patches [B, 256, 1024]")
    print("  🎯 Target: Clean CLIP patches [B, 256, 1024]")
    print("  🔄 Conditioning: EVA-CLIP patches [B, 256, 4096]")
    print("  📊 Goal: Maximize image-to-text recall")
    
    print("=" * 40)

# Add utility functions to exports
__all__.extend([
    "get_loss_function",
    "create_loss",
    "print_loss_status", 
])

# Backward compatibility aliases
create_blip3o_flow_matching = create_blip3o_flow_matching_loss  # Legacy alias

# Ensure the loss is available
if not PATCH_FLOW_MATCHING_AVAILABLE:
    logger.error("❌ BLIP3-o patch-level flow matching loss is required but not available!")
    raise ImportError("BLIP3-o patch-level flow matching loss is required for this project")

logger.info("BLIP3-o patch-level flow matching loss loaded successfully - Paper-aligned training")