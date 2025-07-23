"""
src/modules/losses/__init__.py
Loss modules initialization
"""

from .blip3o_flow_matching_loss import BLIP3oFlowMatchingLoss, create_blip3o_flow_matching_loss

# Log initialization
import logging
logger = logging.getLogger(__name__)
logger.info("BLIP3-o loss modules initialized")

__all__ = [
    "BLIP3oFlowMatchingLoss",
    "create_blip3o_flow_matching_loss",
]