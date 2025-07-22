"""
Inference utilities for BLIP3-o DiT - Enhanced Multi-Model Support

Contains:
- DualSupervisionBLIP3oInference: Dual supervision inference pipeline
- StandardBLIP3oInference: Standard inference pipeline (if available)
- GlobalBLIP3oInference: Global model inference (if available)
- Model loading and generation utilities
- Multi-GPU inference support
"""

import logging

logger = logging.getLogger(__name__)

# Import dual supervision inference
DUAL_SUPERVISION_INFERENCE_AVAILABLE = False
try:
    from .blip3o_inference import (
        DualSupervisionBLIP3oInference,
        load_dual_supervision_blip3o_inference,
    )
    logger.debug("✅ Dual supervision inference loaded")
    DUAL_SUPERVISION_INFERENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Dual supervision inference not available: {e}")
    DualSupervisionBLIP3oInference = None
    load_dual_supervision_blip3o_inference = None

# Try to import other inference modules
STANDARD_INFERENCE_AVAILABLE = False
StandardBLIP3oInference = None
load_standard_blip3o_inference = None

try:
    from .standard_blip3o_inference import (
        StandardBLIP3oInference,
        load_standard_blip3o_inference,
    )
    logger.debug("✅ Standard inference loaded")
    STANDARD_INFERENCE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Standard inference not available: {e}")

GLOBAL_INFERENCE_AVAILABLE = False
GlobalBLIP3oInference = None
load_global_blip3o_inference = None

try:
    from .global_blip3o_inference import (
        GlobalBLIP3oInference,
        load_global_blip3o_inference,
    )
    logger.debug("✅ Global inference loaded")
    GLOBAL_INFERENCE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Global inference not available: {e}")

# Determine the best default inference class
if DUAL_SUPERVISION_INFERENCE_AVAILABLE:
    BLIP3oInference = DualSupervisionBLIP3oInference
    load_blip3o_inference = load_dual_supervision_blip3o_inference
    DEFAULT_INFERENCE_TYPE = "dual_supervision"
    logger.info("✅ Using dual supervision inference as default")
elif GLOBAL_INFERENCE_AVAILABLE:
    BLIP3oInference = GlobalBLIP3oInference
    load_blip3o_inference = load_global_blip3o_inference
    DEFAULT_INFERENCE_TYPE = "global"
    logger.info("✅ Using global inference as default")
elif STANDARD_INFERENCE_AVAILABLE:
    BLIP3oInference = StandardBLIP3oInference
    load_blip3o_inference = load_standard_blip3o_inference
    DEFAULT_INFERENCE_TYPE = "standard"
    logger.info("✅ Using standard inference as default")
else:
    BLIP3oInference = None
    load_blip3o_inference = None
    DEFAULT_INFERENCE_TYPE = None
    logger.warning("❌ No inference pipeline available")

# Build exports list
__all__ = [
    # Availability flags
    "DUAL_SUPERVISION_INFERENCE_AVAILABLE",
    "STANDARD_INFERENCE_AVAILABLE",
    "GLOBAL_INFERENCE_AVAILABLE",
    "DEFAULT_INFERENCE_TYPE",
]

# Export dual supervision components if available
if DUAL_SUPERVISION_INFERENCE_AVAILABLE:
    __all__.extend([
        "DualSupervisionBLIP3oInference",
        "load_dual_supervision_blip3o_inference",
    ])

# Export standard components if available
if STANDARD_INFERENCE_AVAILABLE:
    __all__.extend([
        "StandardBLIP3oInference", 
        "load_standard_blip3o_inference",
    ])

# Export global components if available
if GLOBAL_INFERENCE_AVAILABLE:
    __all__.extend([
        "GlobalBLIP3oInference",
        "load_global_blip3o_inference", 
    ])

# Export default interface if available
if BLIP3oInference is not None:
    __all__.extend([
        "BLIP3oInference",
        "load_blip3o_inference",
    ])

def get_inference_class(inference_type: str = "auto"):
    """
    Get the appropriate inference class
    
    Args:
        inference_type: "auto", "dual_supervision", "global", or "standard"
        
    Returns:
        Inference class
    """
    if inference_type == "auto":
        if BLIP3oInference is None:
            raise ValueError("No inference class available")
        return BLIP3oInference
        
    elif inference_type == "dual_supervision":
        if not DUAL_SUPERVISION_INFERENCE_AVAILABLE:
            raise ValueError("Dual supervision inference not available")
        return DualSupervisionBLIP3oInference
        
    elif inference_type == "global":
        if not GLOBAL_INFERENCE_AVAILABLE:
            raise ValueError("Global inference not available")
        return GlobalBLIP3oInference
        
    elif inference_type == "standard":
        if not STANDARD_INFERENCE_AVAILABLE:
            raise ValueError("Standard inference not available")
        return StandardBLIP3oInference
        
    else:
        raise ValueError(f"Unknown inference type: {inference_type}")

def get_inference_loader(inference_type: str = "auto"):
    """
    Get the appropriate inference loader function
    
    Args:
        inference_type: "auto", "dual_supervision", "global", or "standard"
        
    Returns:
        Inference loader function
    """
    if inference_type == "auto":
        if load_blip3o_inference is None:
            raise ValueError("No inference loader available")
        return load_blip3o_inference
        
    elif inference_type == "dual_supervision":
        if not DUAL_SUPERVISION_INFERENCE_AVAILABLE:
            raise ValueError("Dual supervision inference not available")
        return load_dual_supervision_blip3o_inference
        
    elif inference_type == "global":
        if not GLOBAL_INFERENCE_AVAILABLE:
            raise ValueError("Global inference not available")
        return load_global_blip3o_inference
        
    elif inference_type == "standard":
        if not STANDARD_INFERENCE_AVAILABLE:
            raise ValueError("Standard inference not available")
        return load_standard_blip3o_inference
        
    else:
        raise ValueError(f"Unknown inference type: {inference_type}")

def load_model_for_inference(
    model_path,
    inference_type: str = "auto",
    device: str = "auto",
    **kwargs
):
    """
    Load a BLIP3-o model for inference
    
    Args:
        model_path: Path to the trained model
        inference_type: "auto", "dual_supervision", "global", or "standard"
        device: Device to use for inference
        **kwargs: Additional arguments for inference pipeline
        
    Returns:
        Inference pipeline instance
    """
    loader = get_inference_loader(inference_type)
    return loader(
        model_path=model_path,
        device=device,
        **kwargs
    )

def print_inference_status():
    """Print status of available inference utilities"""
    print("🔮 BLIP3-o Inference Status")
    print("=" * 30)
    print(f"Default inference: {DEFAULT_INFERENCE_TYPE}")
    print()
    print("Available inference pipelines:")
    
    if DUAL_SUPERVISION_INFERENCE_AVAILABLE:
        print("  ✅ Dual Supervision Inference (Recommended)")
    else:
        print("  ❌ Dual Supervision Inference")
        
    if GLOBAL_INFERENCE_AVAILABLE:
        print("  ✅ Global Inference")
    else:
        print("  ❌ Global Inference")
        
    if STANDARD_INFERENCE_AVAILABLE:
        print("  ✅ Standard Inference")
    else:
        print("  ❌ Standard Inference")
    
    if not any([DUAL_SUPERVISION_INFERENCE_AVAILABLE, 
               GLOBAL_INFERENCE_AVAILABLE, 
               STANDARD_INFERENCE_AVAILABLE]):
        print("  ⚠️  No inference pipelines available!")
        print("  💡 Make sure model files are properly implemented")
    
    print("=" * 30)

# Enhanced multi-GPU inference utilities
def create_multi_gpu_inference_pipeline(
    model_path,
    inference_type: str = "auto", 
    num_gpus: int = None,
    **kwargs
):
    """
    Create inference pipeline optimized for multi-GPU
    
    Args:
        model_path: Path to trained model
        inference_type: Type of inference to use
        num_gpus: Number of GPUs to use (None for auto-detect)
        **kwargs: Additional inference arguments
        
    Returns:
        Inference pipeline configured for multi-GPU
    """
    import torch
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if num_gpus > 1:
        logger.info(f"Setting up multi-GPU inference with {num_gpus} GPUs")
        # For multi-GPU inference, we typically use the first GPU as primary
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load inference pipeline
    pipeline = load_model_for_inference(
        model_path=model_path,
        inference_type=inference_type,
        device=device,
        **kwargs
    )
    
    return pipeline

# Add utility functions to exports
__all__.extend([
    "get_inference_class",
    "get_inference_loader",
    "load_model_for_inference",
    "create_multi_gpu_inference_pipeline",
    "print_inference_status",
])

# Log inference module status
if DEFAULT_INFERENCE_TYPE:
    logger.info(f"BLIP3-o inference loaded successfully (default: {DEFAULT_INFERENCE_TYPE})")
else:
    logger.warning("No BLIP3-o inference pipeline available")