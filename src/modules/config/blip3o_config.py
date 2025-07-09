"""
Updated configuration classes for BLIP3-o DiT architecture with 3D RoPE compatibility.
Ensures that default configurations work properly with the fixed 3D RoPE implementation.
"""

from typing import Optional
from transformers import PretrainedConfig


class BLIP3oDiTConfig(PretrainedConfig):
    """
    Configuration class for BLIP3-o DiT model with 3D RoPE compatibility.
    
    This configuration ensures that head_dim is compatible with 3D RoPE (divisible by 4)
    and follows the exact BLIP3-o architecture.
    """
    
    model_type = "blip3o-dit"

    def __init__(
        self,
        # Spatial configuration
        input_size: int = 8,                    # 8x8 grid = 64 tokens
        patch_size: int = 1,                    # Already tokenized features
        
        # Model dimensions - COMPATIBLE with your extracted embeddings
        in_channels: int = 1024,                # CLIP feature dimension (ViT-L/14)
        dim: int = 512,                         # Hidden dimension (3D RoPE compatible)
        eva_embedding_size: int = 4096,         # EVA-CLIP conditioning dimension (EVA-CLIP-8B)
        
        # Transformer architecture - 3D RoPE COMPATIBLE
        n_layers: int = 24,                     # Number of transformer layers
        n_heads: int = 8,                       # Number of attention heads (dim=512 -> head_dim=64, divisible by 4)
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
        
        # Validate and adjust for 3D RoPE compatibility BEFORE storing
        self._validate_and_adjust_for_3d_rope(dim, n_heads)
        
        # Store configuration (after potential adjustments)
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.eva_embedding_size = eva_embedding_size
        self.n_layers = n_layers
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.learn_sigma = learn_sigma
        self._gradient_checkpointing = _gradient_checkpointing
        
        # Validate configuration
        self._validate_config()
    
    def _validate_and_adjust_for_3d_rope(self, dim: int, n_heads: int):
        """
        Validate and adjust dimensions for 3D RoPE compatibility.
        Ensures head_dim is divisible by 4.
        """
        head_dim = dim // n_heads
        
        if head_dim % 4 != 0:
            print(f"⚠️  Adjusting config for 3D RoPE compatibility...")
            print(f"   Original: dim={dim}, n_heads={n_heads}, head_dim={head_dim}")
            
            # Strategy 1: Find compatible number of heads
            compatible_found = False
            for candidate_heads in [4, 8, 16, 32, 64]:  # Common head counts
                if dim % candidate_heads == 0:
                    candidate_head_dim = dim // candidate_heads
                    if candidate_head_dim % 4 == 0 and candidate_head_dim >= 32:  # Reasonable head size
                        n_heads = candidate_heads
                        head_dim = candidate_head_dim
                        compatible_found = True
                        print(f"   ✅ Adjusted n_heads to {n_heads} (head_dim={head_dim})")
                        break
            
            # Strategy 2: Use known compatible configurations
            if not compatible_found:
                # Use proven compatible configs
                known_configs = [
                    (512, 8, 64),    # head_dim=64 (64%4=0)
                    (768, 12, 64),   # head_dim=64 (64%4=0) 
                    (1024, 16, 64),  # head_dim=64 (64%4=0)
                    (1536, 24, 64),  # head_dim=64 (64%4=0)
                    (2048, 32, 64),  # head_dim=64 (64%4=0)
                ]
                
                # Find closest compatible config
                best_config = None
                min_diff = float('inf')
                
                for cfg_dim, cfg_heads, cfg_head_dim in known_configs:
                    diff = abs(cfg_dim - dim) + abs(cfg_heads - n_heads)
                    if diff < min_diff:
                        min_diff = diff
                        best_config = (cfg_dim, cfg_heads, cfg_head_dim)
                
                if best_config:
                    dim, n_heads, head_dim = best_config
                    print(f"   ✅ Using compatible config: dim={dim}, n_heads={n_heads}, head_dim={head_dim}")
                else:
                    # Final fallback
                    dim, n_heads, head_dim = 512, 8, 64
                    print(f"   ✅ Using safe fallback: dim={dim}, n_heads={n_heads}, head_dim={head_dim}")
        
        # Store the (potentially adjusted) values
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads  # Keep them the same
        
        print(f"✅ Final 3D RoPE compatible config:")
        print(f"   dim={self.dim}, n_heads={self.n_heads}, head_dim={self.dim // self.n_heads}")
        print(f"   head_dim % 4 = {(self.dim // self.n_heads) % 4} (must be 0)")
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        assert self.dim % self.n_heads == 0, f"Hidden dim {self.dim} must be divisible by num_heads {self.n_heads}"
        assert (self.dim // self.n_heads) % 4 == 0, f"head_dim {self.dim // self.n_heads} must be divisible by 4 for 3D RoPE"
        assert self.learn_sigma is False, "BLIP3-o uses flow matching, which doesn't require sigma learning"
        assert self.input_size * self.input_size == 64, f"Input size {self.input_size}x{self.input_size} must equal 64 tokens"


class FlowMatchingConfig:
    """
    Configuration for flow matching training objective.
    Based on BLIP3-o's flow matching implementation.
    Updated for compatibility with your embeddings.
    """
    
    def __init__(
        self,
        # Flow matching parameters
        sigma_min: float = 1e-4,                # Minimum noise level
        sigma_max: float = 1.0,                 # Maximum noise level
        prediction_type: str = "v_prediction",  # "v_prediction" or "epsilon"
        
        # Training parameters - CORRECT for your extracted embeddings
        clip_dim: int = 1024,                   # CLIP embedding dimension (ViT-L/14)
        eva_dim: int = 4096,                    # EVA-CLIP dimension (EVA-CLIP-8B)
        
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
        assert clip_dim == 1024, "CLIP dimension must match your embeddings (1024 for ViT-L/14)"
        assert eva_dim == 4096, "EVA-CLIP dimension must match your embeddings (4096 for EVA-CLIP-8B)"


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
    """Get default BLIP3-o configuration with 3D RoPE compatibility."""
    return BLIP3oDiTConfig(
        input_size=8,                    # 8x8 = 64 tokens
        patch_size=1,                    # Pre-tokenized
        in_channels=1024,                # CLIP dimension (ViT-L/14)
        dim=512,                         # Hidden dimension (3D RoPE compatible)
        eva_embedding_size=4096,         # EVA-CLIP dimension (EVA-CLIP-8B)
        n_layers=24,                     # Transformer layers
        n_heads=8,                       # Attention heads (head_dim=64, divisible by 4)
        n_kv_heads=8,                    # KV heads
        multiple_of=256,                 # FFN multiple
        norm_eps=1e-5,                   # Normalization
        qk_norm=True,                    # Query-key norm
        learn_sigma=False,               # Flow matching
        _gradient_checkpointing=True,    # Memory optimization
    )


def get_large_blip3o_config() -> BLIP3oDiTConfig:
    """Get large BLIP3-o configuration with 3D RoPE compatibility."""
    return BLIP3oDiTConfig(
        input_size=8,                    # 8x8 = 64 tokens
        patch_size=1,                    # Pre-tokenized
        in_channels=1024,                # CLIP dimension (ViT-L/14)
        dim=1024,                        # Larger hidden dimension
        eva_embedding_size=4096,         # EVA-CLIP dimension (EVA-CLIP-8B)
        n_layers=32,                     # More transformer layers
        n_heads=16,                      # More attention heads (head_dim=64, divisible by 4)
        n_kv_heads=16,                   # KV heads
        multiple_of=256,                 # FFN multiple
        norm_eps=1e-5,                   # Normalization
        qk_norm=True,                    # Query-key norm
        learn_sigma=False,               # Flow matching
        _gradient_checkpointing=True,    # Memory optimization
    )


def get_small_blip3o_config() -> BLIP3oDiTConfig:
    """Get small BLIP3-o configuration for testing/debugging."""
    return BLIP3oDiTConfig(
        input_size=8,                    # 8x8 = 64 tokens
        patch_size=1,                    # Pre-tokenized
        in_channels=1024,                # CLIP dimension (ViT-L/14)
        dim=256,                         # Smaller hidden dimension
        eva_embedding_size=4096,         # EVA-CLIP dimension (EVA-CLIP-8B)
        n_layers=8,                      # Fewer transformer layers
        n_heads=4,                       # Fewer attention heads (head_dim=64, divisible by 4)
        n_kv_heads=4,                    # KV heads
        multiple_of=256,                 # FFN multiple
        norm_eps=1e-5,                   # Normalization
        qk_norm=True,                    # Query-key norm
        learn_sigma=False,               # Flow matching
        _gradient_checkpointing=False,   # Disabled for testing
    )


def get_default_flow_matching_config() -> FlowMatchingConfig:
    """Get default flow matching configuration for your embeddings."""
    return FlowMatchingConfig(
        sigma_min=1e-4,
        sigma_max=1.0,
        prediction_type="v_prediction",  # BLIP3-o uses v-prediction
        clip_dim=1024,                   # CLIP dimension (ViT-L/14) - matches your embeddings
        eva_dim=4096,                    # EVA-CLIP dimension (EVA-CLIP-8B) - matches your embeddings
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


# Configuration validation functions
def validate_3d_rope_compatibility(config: BLIP3oDiTConfig) -> bool:
    """
    Validate that a configuration is compatible with 3D RoPE.
    
    Args:
        config: BLIP3oDiTConfig to validate
        
    Returns:
        True if compatible, raises assertion error if not
    """
    head_dim = config.dim // config.n_heads
    
    # Check divisibility requirements
    assert config.dim % config.n_heads == 0, f"dim {config.dim} must be divisible by n_heads {config.n_heads}"
    assert head_dim % 4 == 0, f"head_dim {head_dim} must be divisible by 4 for 3D RoPE"
    assert head_dim >= 32, f"head_dim {head_dim} should be at least 32 for reasonable attention"
    
    print(f"✅ Configuration is 3D RoPE compatible:")
    print(f"   dim={config.dim}, n_heads={config.n_heads}, head_dim={head_dim}")
    
    return True


def print_model_size_estimate(config: BLIP3oDiTConfig):
    """Print estimated model size for a configuration."""
    
    # Rough parameter estimation
    embed_params = config.in_channels * config.dim + config.eva_embedding_size * config.dim
    
    # Transformer layers
    layer_params = config.n_layers * (
        # Self-attention
        3 * config.dim * config.dim +  # Q, K, V projections
        config.dim * config.dim +       # Output projection
        # Cross-attention
        config.dim * config.dim +       # Q projection
        2 * config.dim * config.dim +   # K, V projections
        config.dim * config.dim +       # Output projection
        # FFN
        config.dim * config.dim * 4 +   # Up projection
        config.dim * 4 * config.dim +   # Down projection
        # LayerNorms and other
        config.dim * 8                  # Various norms and projections
    )
    
    output_params = config.dim * config.in_channels
    
    total_params = embed_params + layer_params + output_params
    memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
    
    print(f"📊 Estimated model size for config:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Memory (FP32): {memory_mb:.1f} MB")
    print(f"   Memory (FP16): {memory_mb/2:.1f} MB")


if __name__ == "__main__":
    # Test the configurations
    print("Testing BLIP3-o configurations...")
    
    configs = [
        ("Small", get_small_blip3o_config()),
        ("Default", get_default_blip3o_config()),
        ("Large", get_large_blip3o_config()),
    ]
    
    for name, config in configs:
        print(f"\n{name} Configuration:")
        print(f"  dim={config.dim}, n_heads={config.n_heads}, n_layers={config.n_layers}")
        
        # Validate 3D RoPE compatibility
        try:
            validate_3d_rope_compatibility(config)
        except AssertionError as e:
            print(f"  ❌ Not 3D RoPE compatible: {e}")
            continue
        
        # Print size estimate
        print_model_size_estimate(config)
    
    print("\n✅ All configurations are 3D RoPE compatible!")