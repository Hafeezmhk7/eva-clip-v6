"""
Model modules for BLIP3-o DiT - Patch-Level Training (Paper-Aligned)

Contains:
- BLIP3oPatchDiTModel: Patch-level DiT model (primary)
- Model creation and loading utilities for 256-token patch training
- Paper-aligned architecture with EVA-CLIP conditioning
"""

import logging

logger = logging.getLogger(__name__)

# Import patch-level model (primary model following BLIP3-o paper)
PATCH_MODEL_AVAILABLE = False
BLIP3oPatchDiTModel = None
create_blip3o_patch_dit_model = None

try:
    from .blip3o_patch_dit import (
        BLIP3oPatchDiTModel,
        create_blip3o_patch_dit_model,
        RotaryPositionalEmbedding3D,
        TimestepEmbedder,
        MultiHeadAttention,
        BLIP3oDiTBlock,
    )
    PATCH_MODEL_AVAILABLE = True
    logger.info("✅ BLIP3-o patch-level DiT model loaded successfully")
    
except ImportError as e:
    PATCH_MODEL_AVAILABLE = False
    logger.error(f"❌ Failed to load patch-level DiT model: {e}")
    raise ImportError(f"BLIP3-o patch-level DiT model is required but failed to load: {e}")

# Use patch-level model as the main model (paper-aligned)
BLIP3oDiTModel = BLIP3oPatchDiTModel
create_blip3o_dit_model = create_blip3o_patch_dit_model
DEFAULT_MODEL_TYPE = "patch_level"

logger.info("✅ Using BLIP3-o patch-level DiT model as primary model")

# Build exports list
__all__ = [
    # Primary model interface (paper-aligned)
    "BLIP3oDiTModel",
    "create_blip3o_dit_model", 
    "DEFAULT_MODEL_TYPE",
    
    # Patch-level model specific
    "BLIP3oPatchDiTModel",
    "create_blip3o_patch_dit_model",
    "PATCH_MODEL_AVAILABLE",
    
    # Model components
    "RotaryPositionalEmbedding3D",
    "TimestepEmbedder", 
    "MultiHeadAttention",
    "BLIP3oDiTBlock",
]

def get_model_class(model_type: str = "auto"):
    """
    Get the model class (always returns BLIP3oPatchDiTModel for paper alignment)
    
    Args:
        model_type: Ignored, always returns patch-level model
        
    Returns:
        BLIP3oPatchDiTModel class
    """
    if not PATCH_MODEL_AVAILABLE:
        raise RuntimeError("BLIP3-o patch-level DiT model not available")
    return BLIP3oPatchDiTModel

def get_model_factory(model_type: str = "auto"):
    """
    Get the model factory function (always returns patch-level factory)
    
    Args:
        model_type: Ignored, always returns patch-level factory
        
    Returns:
        create_blip3o_patch_dit_model function
    """
    if not PATCH_MODEL_AVAILABLE:
        raise RuntimeError("BLIP3-o patch-level DiT model not available")
    return create_blip3o_patch_dit_model

def create_model(config=None, **kwargs):
    """
    Create a BLIP3-o model instance (always patch-level DiT for paper alignment)
    
    Args:
        config: Model configuration
        **kwargs: Additional arguments
        
    Returns:
        BLIP3oPatchDiTModel instance
    """
    if not PATCH_MODEL_AVAILABLE:
        raise RuntimeError("BLIP3-o patch-level DiT model not available")
        
    if config is not None:
        return create_blip3o_patch_dit_model(config=config, **kwargs)
    else:
        return create_blip3o_patch_dit_model(**kwargs)

def load_pretrained_model(model_path: str, **kwargs):
    """
    Load a pretrained BLIP3-o patch-level DiT model
    
    Args:
        model_path: Path to pretrained model
        **kwargs: Additional arguments
        
    Returns:
        Loaded BLIP3oPatchDiTModel instance
    """
    from pathlib import Path
    import torch
    import json
    
    model_path = Path(model_path)
    
    # Load config
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        from ..config.blip3o_config import BLIP3oDiTConfig
        config = BLIP3oDiTConfig(**config_dict)
    else:
        # Use default config
        from ..config.blip3o_config import get_default_blip3o_config
        config = get_default_blip3o_config()
        logger.warning(f"No config found at {config_file}, using default")
    
    # Create model
    model = create_blip3o_patch_dit_model(config=config, **kwargs)
    
    # Load weights
    weight_files = [
        model_path / "pytorch_model.bin",
        model_path / "model.safetensors",
        model_path / "pytorch_model.safetensors"
    ]
    
    weight_file = None
    for wf in weight_files:
        if wf.exists():
            weight_file = wf
            break
    
    if weight_file is None:
        logger.warning(f"No weight file found in {model_path}, returning untrained model")
        return model
    
    logger.info(f"Loading weights from: {weight_file}")
    
    # Load weights
    if weight_file.suffix == ".bin":
        state_dict = torch.load(weight_file, map_location='cpu')
    else:
        from safetensors.torch import load_file
        state_dict = load_file(str(weight_file))
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys when loading model: {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading model: {len(unexpected_keys)} keys")
    
    logger.info(f"✅ Model loaded successfully from {model_path}")
    
    return model

def print_model_status():
    """Print status of available models"""
    print("🏗️ BLIP3-o DiT Models Status")
    print("=" * 35)
    print(f"Model type: {DEFAULT_MODEL_TYPE}")
    print()
    print("Available model (Paper-Aligned):")
    
    if PATCH_MODEL_AVAILABLE:
        print("  ✅ BLIP3-o Patch DiT (Primary Model)")
        print("    - 256-token patch-level training")
        print("    - EVA-CLIP conditioning (4096-dim)")
        print("    - CLIP output (1024-dim)")
        print("    - Flow matching training objective")
        print("    - Image-to-text recall optimization")
        print("    - 3D Rotary Position Embedding")
        print("    - Multi-head attention with spatial encoding")
        print("    - Multi-GPU compatible")
        print("    - Paper-aligned architecture")
    else:
        print("  ❌ BLIP3-o Patch DiT (REQUIRED)")
    
    print()
    print("Model components:")
    print("  ✅ RotaryPositionalEmbedding3D (spatial-temporal)")
    print("  ✅ TimestepEmbedder (flow matching)")
    print("  ✅ MultiHeadAttention (with 3D RoPE)")
    print("  ✅ BLIP3oDiTBlock (patch-conditioned)")
    
    print()
    print("Architecture details:")
    print("  📐 Input: EVA-CLIP patches [B, 256, 4096]")
    print("  🎯 Output: CLIP patches [B, 256, 1024]")
    print("  🔄 Conditioning: Cross-attention with EVA features")
    print("  📊 Evaluation: Image-to-text recall metrics")
    
    print("=" * 35)

def estimate_model_memory(
    config=None,
    batch_size: int = 8,
    sequence_length: int = 256,
) -> dict:
    """
    Estimate memory usage for BLIP3-o patch-level DiT model
    
    Args:
        config: Model configuration
        batch_size: Batch size
        sequence_length: Sequence length (should be 256 for patches)
        
    Returns:
        Memory usage estimates
    """
    if config is None:
        from ..config.blip3o_config import get_default_blip3o_config
        config = get_default_blip3o_config()
    
    # Parameter estimation
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    intermediate_size = config.intermediate_size
    
    # Model parameters
    embedding_params = (
        config.clip_embedding_size * hidden_size +  # Input projection
        config.eva_embedding_size * hidden_size +   # EVA projection per layer
        hidden_size * sequence_length                # Position embeddings
    )
    
    layer_params = num_layers * (
        # Self-attention
        3 * hidden_size * hidden_size +             # Q, K, V projections
        hidden_size * hidden_size +                 # Output projection
        # Cross-attention with EVA features
        3 * hidden_size * hidden_size +             # Q, K, V projections
        hidden_size * hidden_size +                 # Output projection
        # FFN
        hidden_size * intermediate_size +           # Up projection
        intermediate_size * hidden_size +           # Down projection
        # LayerNorms and modulation
        hidden_size * 8                             # Various norms and modulations
    )
    
    output_params = hidden_size * config.clip_embedding_size  # Output projection
    
    total_params = embedding_params + layer_params + output_params
    
    # Memory estimates (in GB)
    model_memory = total_params * 4 / (1024**3)  # FP32 parameters
    
    # Activation memory estimation (patch-level specific)
    activation_memory = (
        batch_size * sequence_length * hidden_size * num_layers * 4  # Activations
    ) / (1024**3)
    
    # Training memory (gradients + optimizer)
    gradient_memory = model_memory  # Same as parameters
    optimizer_memory = model_memory * 2  # AdamW states
    
    total_training_memory = model_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'model_memory_gb': model_memory,
        'activation_memory_gb': activation_memory,
        'gradient_memory_gb': gradient_memory,
        'optimizer_memory_gb': optimizer_memory,
        'total_training_memory_gb': total_training_memory,
        'inference_memory_gb': model_memory + activation_memory * 0.5,
        'parameters_millions': total_params / 1e6,
        'config_summary': {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'intermediate_size': intermediate_size,
            'sequence_length': sequence_length,
            'batch_size': batch_size,
            'architecture': 'patch_level_dit',
            'paper_alignment': 'BLIP3-o',
        }
    }

# Add utility functions to exports
__all__.extend([
    "get_model_class",
    "get_model_factory",
    "create_model",
    "load_pretrained_model",
    "print_model_status",
    "estimate_model_memory",
])

# Ensure the patch-level model is available
if not PATCH_MODEL_AVAILABLE:
    logger.error("❌ BLIP3-o patch-level DiT model is required but not available!")
    raise ImportError("BLIP3-o patch-level DiT model is required for this project")

logger.info("BLIP3-o patch-level DiT model loaded successfully - Paper-aligned architecture")