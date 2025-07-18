"""
Configuration module for BLIP3-o DiT.

Contains configuration classes for:
- Model architecture (BLIP3oDiTConfig)
- Flow matching loss (FlowMatchingConfig)
- Training parameters (TrainingConfig)
"""

from .blip3o_config import (
    BLIP3oDiTConfig,
    FlowMatchingConfig,
    get_default_blip3o_config,
    get_default_flow_matching_config
)

__all__ = [
    "BLIP3oDiTConfig",
    "FlowMatchingConfig",
    "get_default_blip3o_config",
    "get_default_flow_matching_config"
]