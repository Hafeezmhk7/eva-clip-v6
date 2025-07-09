"""
BLIP3-o DiT Implementation
A PyTorch implementation of BLIP3-o Diffusion Transformer with Flow Matching.

This package provides:
- BLIP3-o DiT model architecture based on NextDiT
- Flow matching loss for training 
- EVA-CLIP to CLIP embedding generation
- Custom HuggingFace trainer integration
- Inference utilities

Usage:
    from src.modules.models.blip3o_dit import BLIP3oDiTModel
    from src.modules.config.blip3o_config import BLIP3oDiTConfig
    from src.modules.losses.flow_matching_loss import BLIP3oFlowMatchingLoss
"""

__version__ = "1.0.0"
__author__ = "BLIP3-o Implementation Team"
__description__ = "BLIP3-o Diffusion Transformer with Flow Matching"

# Core imports for convenience
from .modules.config.blip3o_config import (
    BLIP3oDiTConfig,
    FlowMatchingConfig,
    TrainingConfig,
    get_default_blip3o_config,
    get_default_flow_matching_config,
    get_default_training_config,
)

from .modules.models.blip3o_dit import (
    BLIP3oDiTModel,
    create_blip3o_dit_model,
)

from .modules.losses.flow_matching_loss import (
    BLIP3oFlowMatchingLoss,
    FlowMatchingLoss,
    create_blip3o_flow_matching_loss,
)

from .modules.datasets.blip3o_dataset import (
    BLIP3oEmbeddingDataset,
    create_blip3o_dataloader,
    create_blip3o_dataloaders,
)

from .modules.trainers.blip3o_trainer import (
    BLIP3oTrainer,
    create_blip3o_training_args,
)

from .modules.inference.blip3o_inference import (
    BLIP3oInference,
    load_blip3o_inference,
)

__all__ = [
    # Config
    "BLIP3oDiTConfig",
    "FlowMatchingConfig", 
    "TrainingConfig",
    "get_default_blip3o_config",
    "get_default_flow_matching_config",
    "get_default_training_config",
    
    # Models
    "BLIP3oDiTModel",
    "create_blip3o_dit_model",
    
    # Losses
    "BLIP3oFlowMatchingLoss",
    "FlowMatchingLoss",
    "create_blip3o_flow_matching_loss",
    
    # Datasets
    "BLIP3oEmbeddingDataset",
    "create_blip3o_dataloader",
    "create_blip3o_dataloaders",
    
    # Trainers
    "BLIP3oTrainer",
    "create_blip3o_training_args",
    
    # Inference
    "BLIP3oInference",
    "load_blip3o_inference",
]