"""
FIXED Configuration module for BLIP3-o DiT - Global Training
Place this file as: src/modules/config/__init__.py

Contains configuration classes for:
- Model architecture (BLIP3oDiTConfig)
- Flow matching loss (FlowMatchingConfig)
- Training parameters (TrainingConfig)
"""

from .blip3o_config import (
    BLIP3oDiTConfig,
    FlowMatchingConfig,
    TrainingConfig,
    get_default_blip3o_config,
    get_global_blip3o_config,
    get_multi_gpu_config,
    get_default_flow_matching_config,
    get_enhanced_flow_matching_config,
    get_default_training_config,
    validate_config_compatibility,
    print_config_summary,
)

__all__ = [
    "BLIP3oDiTConfig",
    "FlowMatchingConfig", 
    "TrainingConfig",
    "get_default_blip3o_config",
    "get_global_blip3o_config",
    "get_multi_gpu_config",
    "get_default_flow_matching_config",
    "get_enhanced_flow_matching_config",
    "get_default_training_config",
    "validate_config_compatibility",
    "print_config_summary",
]