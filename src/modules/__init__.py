"""
EVA-CLIP Flow Matching Modules

This package contains the core modules for the EVA-CLIP to CLIP flow matching project:

- embeddings: EVA-CLIP and CLIP ViT-L/14 model implementations
- flow_matching: Lumina-Next DiT implementation for flow matching
- cache: Caching utilities for fast training

Usage:
    from src.modules.embeddings import EmbeddingExtractor
    from src.modules.flow_matching import LuminaNextDiT
    from src.modules.cache import FeatureCache
"""

__version__ = "0.1.0"
__author__ = "EVA-CLIP Flow Matching Team"

# Import main classes for easy access
try:
    from .extract_embeddings import EmbeddingExtractor
    from .cache import FeatureCache
    # from .flow_matching import LuminaNextDiT  # Will add this later
    
    __all__ = [
        "EmbeddingExtractor",
        "FeatureCache",
        # "LuminaNextDiT",  # Will uncomment when implemented
    ]
except ImportError:
    # Modules not yet implemented
    __all__ = []

print("🏗️ EVA-CLIP Flow Matching modules package loaded")