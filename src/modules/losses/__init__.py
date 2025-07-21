"""
Loss functions module for BLIP3-o DiT - FIXED for Dual Supervision
"""

from .flow_matching_loss import (
    FlowMatchingLoss,
    BLIP3oFlowMatchingLoss,
    create_blip3o_flow_matching_loss,
)

# Import dual supervision components with better error handling
DUAL_SUPERVISION_AVAILABLE = False
try:
    from .dual_supervision_flow_matching_loss import (
        DualSupervisionFlowMatchingLoss,
        create_dual_supervision_loss,
    )
    DUAL_SUPERVISION_AVAILABLE = True
    print("✅ Dual supervision loss loaded successfully")
    
except ImportError as e:
    DUAL_SUPERVISION_AVAILABLE = False
    print(f"⚠️ Dual supervision loss import failed: {e}")
    print("⚠️ Dual supervision loss not available")
    
except Exception as e:
    DUAL_SUPERVISION_AVAILABLE = False
    print(f"⚠️ Unexpected error loading dual supervision loss: {e}")
    print("⚠️ Dual supervision loss not available")

__all__ = [
    "FlowMatchingLoss",
    "BLIP3oFlowMatchingLoss",
    "create_blip3o_flow_matching_loss",
    "DUAL_SUPERVISION_AVAILABLE",
]

if DUAL_SUPERVISION_AVAILABLE:
    __all__.extend([
        "DualSupervisionFlowMatchingLoss",
        "create_dual_supervision_loss",
    ])