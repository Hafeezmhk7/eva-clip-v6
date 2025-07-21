"""
BLIP3-o DiT Modules - FIXED

This package contains all the core modules for BLIP3-o DiT implementation:
- config: Configuration classes
- models: Model architectures (including dual supervision)
- losses: Loss functions (including dual supervision flow matching)
- datasets: Data loading utilities
- trainers: Training utilities (including dual supervision trainer)
- inference: Inference utilities
"""

# Import all submodules
from .config import *
from .models import *
from .losses import *
from .datasets import *
from .trainers import *
from .inference import *