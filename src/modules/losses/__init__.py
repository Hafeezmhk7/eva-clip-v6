"""
Loss functions module for BLIP3-o DiT - Global Training Only

Contains:
- GlobalFlowMatchingLoss: Global training loss (primary)
- EnhancedGlobalFlowMatchingLoss: Enhanced version with regularization
- Factory functions for creating loss instances
"""

import logging

logger = logging.getLogger(__name__)

# Import global flow matching loss (your main loss function)
GLOBAL_FLOW_MATCHING_AVAILABLE = False
GlobalFlowMatchingLoss = None
EnhancedGlobalFlowMatchingLoss = None
create_global_flow_matching_loss = None

try:
    from .global_flow_matching_loss import (
        GlobalFlowMatchingLoss,
        EnhancedGlobalFlowMatchingLoss,
        create_global_flow_matching_loss,
    )
    GLOBAL_FLOW_MATCHING_AVAILABLE = True
    logger.info("✅ Global flow matching loss loaded successfully")
    
except ImportError as e:
    GLOBAL_FLOW_MATCHING_AVAILABLE = False
    logger.error(f"❌ Failed to load global flow matching loss: {e}")
    raise ImportError(f"Global flow matching loss is required but failed to load: {e}")

# Use global flow matching as the main loss (no fallbacks)
BLIP3oFlowMatchingLoss = GlobalFlowMatchingLoss
create_blip3o_flow_matching_loss = create_global_flow_matching_loss
DEFAULT_LOSS_TYPE = "global"

logger.info("✅ Using Global flow matching loss as primary loss")

# Build exports list
__all__ = [
    # Primary loss interface
    "BLIP3oFlowMatchingLoss",
    "create_blip3o_flow_matching_loss",
    "DEFAULT_LOSS_TYPE",
    
    # Global loss specific
    "GlobalFlowMatchingLoss",
    "EnhancedGlobalFlowMatchingLoss", 
    "create_global_flow_matching_loss",
    "GLOBAL_FLOW_MATCHING_AVAILABLE",
]

def get_loss_function(loss_type: str = "auto", enhanced: bool = False, **kwargs):
    """
    Get the appropriate loss function (always global)
    
    Args:
        loss_type: Ignored, always returns global loss
        enhanced: Whether to use enhanced version with regularization
        **kwargs: Arguments to pass to loss function factory
        
    Returns:
        Global flow matching loss instance
    """
    if not GLOBAL_FLOW_MATCHING_AVAILABLE:
        raise RuntimeError("Global flow matching loss not available")
    
    return create_global_flow_matching_loss(enhanced=enhanced, **kwargs)

def create_loss(enhanced: bool = False, **kwargs):
    """
    Create a BLIP3-o loss function (always global)
    
    Args:
        enhanced: Whether to use enhanced version
        **kwargs: Loss configuration arguments
        
    Returns:
        Global flow matching loss instance
    """
    if not GLOBAL_FLOW_MATCHING_AVAILABLE:
        raise RuntimeError("Global flow matching loss not available")
    
    return create_global_flow_matching_loss(enhanced=enhanced, **kwargs)

def print_loss_status():
    """Print status of available loss functions"""
    print("📉 BLIP3-o Loss Functions Status")
    print("=" * 35)
    print(f"Loss type: {DEFAULT_LOSS_TYPE}")
    print()
    print("Available loss function:")
    
    if GLOBAL_FLOW_MATCHING_AVAILABLE:
        print("  ✅ Global Flow Matching (Primary Loss)")
        print("    - Direct global feature supervision")
        print("    - Contrastive loss option")
        print("    - Enhanced version with regularization")
        print("    - Optimized for recall performance")
    else:
        print("  ❌ Global Flow Matching (REQUIRED)")
    
    print("=" * 35)

# Add utility functions to exports
__all__.extend([
    "get_loss_function",
    "create_loss",
    "print_loss_status", 
])

# Backward compatibility aliases
create_blip3o_loss = get_loss_function  # Legacy alias

# Ensure the loss is available
if not GLOBAL_FLOW_MATCHING_AVAILABLE:
    logger.error("❌ Global flow matching loss is required but not available!")
    raise ImportError("Global flow matching loss is required for this project")

logger.info("BLIP3-o global loss function loaded successfully")