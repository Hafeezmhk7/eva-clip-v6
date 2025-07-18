"""
Loss functions module for BLIP3-o DiT - Updated for Dual Supervision
"""

from .flow_matching_loss import (
    FlowMatchingLoss,
    BLIP3oFlowMatchingLoss,
    create_blip3o_flow_matching_loss,
)

# Import dual supervision components
try:
    from .dual_supervision_flow_matching_loss import (
        DualSupervisionFlowMatchingLoss,
        create_dual_supervision_loss,
    )
    DUAL_SUPERVISION_AVAILABLE = True
except ImportError:
    DUAL_SUPERVISION_AVAILABLE = False
    print("⚠️ Dual supervision loss not available")

__all__ = [
    "FlowMatchingLoss",
    "BLIP3oFlowMatchingLoss", 
    "create_blip3o_flow_matching_loss",
]

if DUAL_SUPERVISION_AVAILABLE:
    __all__.extend([
        "DualSupervisionFlowMatchingLoss",
        "create_dual_supervision_loss",
    ])