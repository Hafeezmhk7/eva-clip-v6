"""
Configuration classes for BLIP3-o DiT architecture.
Exact implementation following the BLIP3-o paper and Lumina-Next architecture.
"""

from typing import Optional
from transformers import PretrainedConfig


class BLIP3oDiTConfig(PretrainedConfig):
    """
    Configuration class for BLIP3-o DiT model.
    
    This configuration follows the exact BLIP3-o architecture:
    - Uses NextDiT backbone (Lumina-Next)
    - Cross-attention with EVA-CLIP conditioning
    - Generates 768-dim CLIP embeddings from 1280-dim EVA-CLIP
    - 64 tokens (8x8 grid) input/output format
    """
    
    model_type = "blip3o-dit"

    def __init__(
        self,
        # Spatial configuration
        input_size: int = 8,                    # 8x8 grid = 64 tokens
        patch_size: int = 1,                    # Already tokenized features
        
        # Model dimensions
        in_channels: int = 1024,                 # CLIP feature dimension
        dim: int = 1792,                        # Hidden dimension (from BLIP3-o)
        eva_embedding_size: int = 4096,         # EVA-CLIP conditioning dimension
        
        # Transformer architecture
        n_layers: int = 24,                     # Number of transformer layers
        n_heads: int = 28,                      # Number of attention heads
        n_kv_heads: Optional[int] = None,       # KV heads for GQA (defaults to n_heads)
        
        # FFN configuration
        multiple_of: int = 256,                 # FFN dimension multiple
        ffn_dim_multiplier: Optional[float] = None,  # FFN multiplier
        
        # Normalization
        norm_eps: float = 1e-5,                 # Layer norm epsilon
        qk_norm: bool = True,                   # Query-key normalization
        
        # Training configuration
        learn_sigma: bool = False,              # Don't learn sigma for flow matching
        _gradient_checkpointing: bool = True,   # Memory optimization
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Set n_kv_heads to n_heads if not specified (standard attention)
        if n_kv_heads is None:
            n_kv_heads = n_heads
        
        # Store configuration
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.eva_embedding_size = eva_embedding_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.learn_sigma = learn_sigma
        self._gradient_checkpointing = _gradient_checkpointing
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        assert self.dim % self.n_heads == 0, f"Hidden dim {self.dim} must be divisible by num_heads {self.n_heads}"
        assert self.learn_sigma is False, "BLIP3-o uses flow matching, which doesn't require sigma learning"
        assert self.input_size * self.input_size == 64, f"Input size {self.input_size}x{self.input_size} must equal 64 tokens"
        assert self.in_channels == 1024, "BLIP3-o generates 768-dim CLIP embeddings"
        assert self.eva_embedding_size == 4096, "EVA-CLIP conditioning should be 1280-dim"


class FlowMatchingConfig:
    """
    Configuration for flow matching training objective.
    Based on BLIP3-o's flow matching implementation.
    """
    
    def __init__(
        self,
        # Flow matching parameters
        sigma_min: float = 1e-4,                # Minimum noise level
        sigma_max: float = 1.0,                 # Maximum noise level
        prediction_type: str = "v_prediction",  # "v_prediction" or "epsilon"
        
        # Training parameters
        clip_dim: int = 1024,                    # CLIP embedding dimension
        eva_dim: int = 4096,                    # EVA-CLIP dimension
        
        # Regularization
        regularization_weight: float = 0.0,    # Additional regularization
        
        # Scheduling
        schedule_type: str = "linear",          # Noise schedule type
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prediction_type = prediction_type
        self.clip_dim = clip_dim
        self.eva_dim = eva_dim
        self.regularization_weight = regularization_weight
        self.schedule_type = schedule_type
        
        # Validate
        assert prediction_type in ["v_prediction", "epsilon"], f"Invalid prediction type: {prediction_type}"
        assert 0 <= sigma_min < sigma_max, "Invalid sigma range"


class TrainingConfig:
    """
    Training configuration for BLIP3-o DiT.
    """
    
    def __init__(
        self,
        # Data configuration
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = 4,
        
        # Training parameters
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        
        # Optimization
        optimizer_type: str = "adamw",
        lr_scheduler_type: str = "cosine",
        fp16: bool = True,
        gradient_checkpointing: bool = True,
        
        # Evaluation
        eval_split: float = 0.1,
        eval_steps: int = 1000,
        
        # Logging
        logging_steps: int = 100,
        save_steps: int = 1000,
        
        # Paths
        output_dir: str = "./checkpoints/blip3o-dit",
        embeddings_path: str = "",
        
        # Experiment tracking
        wandb_project: str = "blip3o-dit",
        wandb_run_name: Optional[str] = None,
        use_wandb: bool = True,
        
        # Data subset (for debugging)
        subset_size: Optional[int] = None,
        normalize_embeddings: bool = True,
    ):
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer_type = optimizer_type
        self.lr_scheduler_type = lr_scheduler_type
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing
        self.eval_split = eval_split
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.embeddings_path = embeddings_path
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.use_wandb = use_wandb
        self.subset_size = subset_size
        self.normalize_embeddings = normalize_embeddings


def get_default_blip3o_config() -> BLIP3oDiTConfig:
    """Get default BLIP3-o configuration matching the paper."""
    return BLIP3oDiTConfig(
        input_size=8,                    # 8x8 = 64 tokens
        patch_size=1,                    # Pre-tokenized
        in_channels=1024,                 # CLIP dimension
        dim=1792,                        # Hidden dimension from paper
        eva_embedding_size=4096,         # EVA-CLIP dimension
        n_layers=24,                     # Transformer layers
        n_heads=28,                      # Attention heads
        n_kv_heads=28,                   # KV heads
        multiple_of=256,                 # FFN multiple
        norm_eps=1e-5,                   # Normalization
        qk_norm=True,                    # Query-key norm
        learn_sigma=False,               # Flow matching
        _gradient_checkpointing=True,    # Memory optimization
    )


def get_default_flow_matching_config() -> FlowMatchingConfig:
    """Get default flow matching configuration for BLIP3-o."""
    return FlowMatchingConfig(
        sigma_min=1e-4,
        sigma_max=1.0,
        prediction_type="v_prediction",  # BLIP3-o uses v-prediction
        clip_dim=1024,
        eva_dim=4096,
        regularization_weight=0.0,
        schedule_type="linear",
    )


def get_default_training_config() -> TrainingConfig:
    """Get default training configuration for BLIP3-o."""
    return TrainingConfig(
        batch_size=32,
        eval_batch_size=64,
        num_workers=4,
        num_epochs=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        gradient_accumulation_steps=1,
        optimizer_type="adamw",
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        eval_split=0.1,
        eval_steps=1000,
        logging_steps=100,
        save_steps=1000,
        output_dir="./checkpoints/blip3o-dit",
        embeddings_path="",
        wandb_project="blip3o-dit",
        use_wandb=True,
        normalize_embeddings=True,
    )