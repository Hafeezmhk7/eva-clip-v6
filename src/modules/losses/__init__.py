"""
BLIP3-o Losses Module
src/modules/losses/__init__.py
"""

import logging

logger = logging.getLogger(__name__)

# Import availability flag
LOSS_AVAILABLE = False

try:
    from .blip3o_flow_matching_loss import (
        BLIP3oFlowMatchingLoss,
        create_blip3o_flow_matching_loss,
    )
    LOSS_AVAILABLE = True
    logger.info("✅ BLIP3-o flow matching loss loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import loss: {e}")
    BLIP3oFlowMatchingLoss = None
    create_blip3o_flow_matching_loss = None

# Main exports
__all__ = [
    "LOSS_AVAILABLE",
]

if LOSS_AVAILABLE:
    __all__.extend([
        "BLIP3oFlowMatchingLoss",
        "create_blip3o_flow_matching_loss",
    ])