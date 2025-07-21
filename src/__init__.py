"""
BLIP3-o DiT Implementation - FIXED
A PyTorch implementation of BLIP3-o Diffusion Transformer with Flow Matching.

This package provides:
- BLIP3-o DiT model architecture based on NextDiT
- Flow matching loss for training 
- EVA-CLIP to CLIP embedding generation
- Custom HuggingFace trainer integration
- Inference utilities
- FIXED: Dual supervision with global generation training
"""

__version__ = "1.0.0"
__author__ = "BLIP3-o Implementation Team"
__description__ = "BLIP3-o Diffusion Transformer with Flow Matching - FIXED"

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
    create_blip3o_dit_model as create_standard_model,
)

from .modules.losses.flow_matching_loss import (
    BLIP3oFlowMatchingLoss,
    FlowMatchingLoss,
    create_blip3o_flow_matching_loss,
)

from .modules.datasets.blip3o_dataset import (
    BLIP3oEmbeddingDataset,
    create_chunked_dataloader,
    create_chunked_dataloaders,
)

from .modules.trainers.blip3o_trainer import (
    BLIP3oTrainer as StandardTrainer,
    create_blip3o_training_args as create_standard_training_args,
)

from .modules.inference.blip3o_inference import (
    BLIP3oInference,
    load_blip3o_inference,
)

# Try to import dual supervision components
DUAL_SUPERVISION_AVAILABLE = False
try:
    from .modules.models.dual_supervision_blip3o_dit import (
        DualSupervisionBLIP3oDiTModel,
        create_blip3o_dit_model as create_dual_supervision_model,
    )
    
    from .modules.losses.dual_supervision_flow_matching_loss import (
        DualSupervisionFlowMatchingLoss,
        create_dual_supervision_loss,
    )
    
    from .modules.trainers.dual_supervision_blip3o_trainer import (
        DualSupervisionBLIP3oTrainer,
        create_blip3o_training_args as create_dual_supervision_training_args,
    )
    
    # Use dual supervision as default
    create_blip3o_dit_model = create_dual_supervision_model
    BLIP3oTrainer = DualSupervisionBLIP3oTrainer
    create_blip3o_training_args = create_dual_supervision_training_args
    
    DUAL_SUPERVISION_AVAILABLE = True
    print("✅ FIXED Dual supervision components loaded successfully")
    
except ImportError as e:
    # Use standard components as fallback
    create_blip3o_dit_model = create_standard_model
    BLIP3oTrainer = StandardTrainer
    create_blip3o_training_args = create_standard_training_args
    
    print(f"⚠️ Dual supervision import failed: {e}")
    print("⚠️ Using standard components as fallback")

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
    "create_chunked_dataloader",
    "create_chunked_dataloaders",
    
    # Trainers
    "BLIP3oTrainer",
    "create_blip3o_training_args",
    
    # Inference
    "BLIP3oInference",
    "load_blip3o_inference",
    
    # Flags
    "DUAL_SUPERVISION_AVAILABLE",
]

if DUAL_SUPERVISION_AVAILABLE:
    __all__.extend([
        "DualSupervisionBLIP3oDiTModel",
        "DualSupervisionFlowMatchingLoss",
        "DualSupervisionBLIP3oTrainer",
        "create_dual_supervision_loss",
    ])