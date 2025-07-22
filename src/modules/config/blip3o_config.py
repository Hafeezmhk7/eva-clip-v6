"""
FIXED BLIP3-o Configuration with Enhanced Compatibility
Place this file at: src/modules/config/blip3o_config.py

KEY FIXES:
1. Better dimension validation for 3D RoPE compatibility
2. Auto-correction for incompatible configurations
3. Enhanced multi-GPU support
4. Simplified configuration options
"""

from transformers import PretrainedConfig
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import math


class BLIP3oDiTConfig(PretrainedConfig):
    """
    FIXED Configuration class for BLIP3-o Diffusion Transformer
    
    Enhanced with automatic dimension validation and correction
    """
    
    model_type = "blip3o_dit"
    
    def __init__(
        self,
        # ========================
        # Core Model Architecture
        # ========================
        
        # Input configuration
        input_size: int = 16,              # Grid size (16x16 = 256 tokens)
        patch_size: int = 1,               # Patch size (pre-tokenized, so 1)
        in_channels: int = 1024,           # CLIP embedding dimension
        
        # Model dimensions (FIXED: Auto-validated)
        dim: int = 768,                    # Hidden dimension
        n_layers: int = 16,                # Number of transformer layers
        n_heads: int = 12,                 # Number of attention heads
        n_kv_heads: Optional[int] = None,  # Number of key-value heads (defaults to n_heads)
        
        # Attention configuration
        qk_norm: bool = True,              # Query-key normalization
        norm_eps: float = 1e-5,            # Layer norm epsilon
        
        # Cross-attention configuration
        eva_embedding_size: int = 4096,    # EVA-CLIP conditioning dimension
        
        # ========================
        # Global Training Configuration (NEW)
        # ========================
        
        # Global adaptation MLP parameters
        mlp_hidden_dim: int = 2048,        # Hidden dimension for adaptation MLP
        mlp_num_layers: int = 3,           # Number of layers in adaptation MLP  
        mlp_dropout: float = 0.1,          # Dropout rate for adaptation MLP
        mlp_activation: str = "gelu",      # Activation function for MLP
        
        # Global training specific
        global_training: bool = True,      # Enable global training mode
        use_attention_pooling: bool = True, # Use attention pooling vs mean pooling
        
        # ========================
        # Training Configuration
        # ========================
        
        # Diffusion configuration
        learn_sigma: bool = False,         # Must be False for flow matching
        
        # Memory optimization
        _gradient_checkpointing: bool = True,    # Enable gradient checkpointing
        
        # RoPE configuration (FIXED)
        rope_base: float = 10000.0,        # RoPE base frequency
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE scaling configuration
        head_dim_divisible_by: int = 4,    # Head dim must be divisible by this for RoPE
        
        # Initialization
        initializer_range: float = 0.02,   # Standard deviation for weight initialization
        
        # ========================
        # Advanced Configuration
        # ========================
        
        # Attention optimizations
        use_flash_attention: bool = False,     # Use Flash Attention (if available)
        attention_dropout: float = 0.0,       # Attention dropout rate
        
        # Feed-forward configuration
        intermediate_size: Optional[int] = None,  # FFN intermediate size (defaults to 4 * dim)
        hidden_dropout: float = 0.0,          # Hidden layer dropout rate
        
        # Position encoding
        max_position_embeddings: int = 1024,  # Maximum sequence length
        
        **kwargs
    ):
        # Move head_dim_divisible_by assignment BEFORE validation call
        self.head_dim_divisible_by = head_dim_divisible_by

        # FIXED: Now validation can safely access self.head_dim_divisible_by
        dim, n_heads = self._validate_and_fix_dimensions(dim, n_heads)
        
        # FIXED: Auto-validate and correct dimensions
        dim, n_heads = self._validate_and_fix_dimensions(dim, n_heads)
        
        # Set default n_kv_heads if not specified
        if n_kv_heads is None:
            n_kv_heads = n_heads
        
        # Set default intermediate_size if not specified
        if intermediate_size is None:
            intermediate_size = 4 * dim
        
        # Store all parameters
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.qk_norm = qk_norm
        self.norm_eps = norm_eps
        self.eva_embedding_size = eva_embedding_size
        
        # Global training configuration
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.mlp_dropout = mlp_dropout
        self.mlp_activation = mlp_activation
        self.global_training = global_training
        self.use_attention_pooling = use_attention_pooling
        
        # Training configuration
        self.learn_sigma = learn_sigma
        self._gradient_checkpointing = _gradient_checkpointing
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        self.head_dim_divisible_by = head_dim_divisible_by
        self.initializer_range = initializer_range
        
        # Advanced configuration
        self.use_flash_attention = use_flash_attention
        self.attention_dropout = attention_dropout
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings
        
        # Call parent constructor
        super().__init__(**kwargs)
        
        # Final validation
        self._validate_config()
    
    def _validate_and_fix_dimensions(self, dim: int, n_heads: int) -> tuple:
        """
        FIXED: Validate and auto-correct dimensions for RoPE compatibility
        
        Returns: (corrected_dim, corrected_n_heads)
        """
        head_dim = dim // n_heads
        
        # Check if current config is valid
        if dim % n_heads == 0 and head_dim % self.head_dim_divisible_by == 0:
            return dim, n_heads
        
        print(f"⚠️ Incompatible dimensions detected: dim={dim}, n_heads={n_heads}, head_dim={head_dim}")
        
        # Try to find compatible configuration close to the original
        compatible_configs = [
            # (dim, n_heads) pairs that work well
            (768, 12),   # head_dim = 64
            (1024, 16),  # head_dim = 64
            (512, 8),    # head_dim = 64
            (960, 15),   # head_dim = 64
            (640, 10),   # head_dim = 64
            (896, 14),   # head_dim = 64
            (1152, 18), # head_dim = 64
            (576, 9),    # head_dim = 64
            (704, 11),   # head_dim = 64
            (832, 13),   # head_dim = 64
        ]
        
        # Find the closest compatible config
        best_config = None
        min_distance = float('inf')
        
        for candidate_dim, candidate_heads in compatible_configs:
            distance = abs(candidate_dim - dim) + abs(candidate_heads - n_heads) * 10
            if distance < min_distance:
                min_distance = distance
                best_config = (candidate_dim, candidate_heads)
        
        if best_config:
            new_dim, new_heads = best_config
            new_head_dim = new_dim // new_heads
            print(f"✅ Auto-corrected to: dim={new_dim}, n_heads={new_heads}, head_dim={new_head_dim}")
            return new_dim, new_heads
        
        # Fallback: force head_dim = 64
        new_heads = max(8, (n_heads // 4) * 4)  # Round to nearest multiple of 4
        new_dim = new_heads * 64
        print(f"✅ Fallback correction: dim={new_dim}, n_heads={new_heads}, head_dim=64")
        return new_dim, new_heads
    
    def _validate_config(self):
        """Validate model configuration with enhanced checks"""
        
        # Basic dimension checks
        assert self.dim > 0, "Model dimension must be positive"
        assert self.n_layers > 0, "Number of layers must be positive"
        assert self.n_heads > 0, "Number of heads must be positive"
        assert self.n_kv_heads > 0, "Number of key-value heads must be positive"
        
        # FIXED: Check head dimension compatibility for RoPE
        assert self.dim % self.n_heads == 0, f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        head_dim = self.dim // self.n_heads
        assert head_dim % self.head_dim_divisible_by == 0, f"head_dim ({head_dim}) must be divisible by {self.head_dim_divisible_by} for RoPE"
        
        # Input validation
        assert self.input_size > 0, "Input size must be positive"
        assert self.patch_size > 0, "Patch size must be positive"
        assert self.in_channels > 0, "Input channels must be positive"
        assert self.eva_embedding_size > 0, "EVA embedding size must be positive"
        
        # MLP validation
        assert self.mlp_hidden_dim > 0, "MLP hidden dimension must be positive"
        assert self.mlp_num_layers > 0, "MLP number of layers must be positive"
        assert 0.0 <= self.mlp_dropout <= 1.0, "MLP dropout must be between 0 and 1"
        assert self.mlp_activation in ["gelu", "relu", "silu"], f"Unknown activation: {self.mlp_activation}"
        
        # Flow matching validation
        assert self.learn_sigma is False, "BLIP3-o uses flow matching, learn_sigma must be False"
        
        # BLIP3-o specific validation
        assert self.in_channels == 1024, "CLIP embedding dimension must be 1024"
        assert self.eva_embedding_size == 4096, "EVA-CLIP dimension must be 4096"
        assert self.patch_size == 1, "Features are pre-tokenized, patch_size must be 1"
        assert self.input_size == 16, "Input size must be 16 for 256 tokens (16×16)"
        
        print(f"✅ Configuration validated successfully")
        print(f"   Model: {self.dim}D, {self.n_layers}L, {self.n_heads}H")
        print(f"   Head dim: {self.get_head_dim()} (RoPE compatible)")
        print(f"   Global training: {self.global_training}")
    
    def get_num_tokens(self) -> int:
        """Get total number of tokens (patches)."""
        return self.input_size * self.input_size
    
    def get_head_dim(self) -> int:
        """Get attention head dimension."""
        return self.dim // self.n_heads
    
    def is_rope_compatible(self) -> bool:
        """Check if configuration is compatible with 3D RoPE."""
        head_dim = self.get_head_dim()
        return head_dim % self.head_dim_divisible_by == 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'architecture': 'global_blip3o_dit',
            'total_tokens': self.get_num_tokens(),
            'head_dim': self.get_head_dim(),
            'rope_compatible': self.is_rope_compatible(),
            'global_training': self.global_training,
            'parameters_estimate': self._estimate_parameters(),
            'memory_estimate_gb': self._estimate_memory(),
        }
    
    def _estimate_parameters(self) -> int:
        """Estimate total model parameters."""
        # Rough parameter estimation
        embed_params = self.in_channels * self.dim + self.eva_embedding_size * self.dim
        
        # Transformer layers
        layer_params = self.n_layers * (
            # Self-attention
            3 * self.dim * self.dim +  # Q, K, V projections
            self.dim * self.dim +       # Output projection
            # Cross-attention
            self.dim * self.dim +       # Q projection
            2 * self.dim * self.dim +   # K, V projections
            self.dim * self.dim +       # Output projection
            # FFN
            self.dim * self.intermediate_size +
            self.intermediate_size * self.dim +
            # LayerNorms and other
            self.dim * 8
        )
        
        # Global adaptation MLP
        mlp_params = self.dim * self.mlp_hidden_dim + self.mlp_hidden_dim * 1024
        
        # Output and other
        output_params = 1024 * 768  # CLIP projection
        
        total_params = embed_params + layer_params + mlp_params + output_params
        return int(total_params)
    
    def _estimate_memory(self, batch_size: int = 8) -> float:
        """Estimate memory usage in GB."""
        params = self._estimate_parameters()
        
        # Model memory (FP16)
        model_memory = params * 2 / (1024**3)
        
        # Activation memory (rough estimate)
        activation_memory = (
            batch_size * self.get_num_tokens() * self.dim * self.n_layers * 4
        ) / (1024**3)
        
        # Training overhead (gradients + optimizer)
        training_overhead = model_memory * 3
        
        return model_memory + activation_memory + training_overhead


@dataclass
class FlowMatchingConfig:
    """
    Configuration for enhanced flow matching loss computation
    """
    
    # Flow matching parameters
    sigma_min: float = 1e-4
    sigma_max: float = 1.0
    prediction_type: str = "v_prediction"
    schedule_type: str = "linear"
    
    # Embedding dimensions
    clip_dim: int = 1024
    eva_dim: int = 4096
    global_dim: int = 768
    
    # Enhanced loss parameters
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.1
    temperature: float = 0.07
    
    # Additional regularization
    gradient_penalty_weight: float = 0.0
    feature_matching_weight: float = 0.0
    
    def __post_init__(self):
        """Validate flow matching configuration."""
        assert self.prediction_type in ["v_prediction", "epsilon"], f"Invalid prediction type: {self.prediction_type}"
        assert self.schedule_type in ["linear", "cosine", "sigmoid"], f"Invalid schedule type: {self.schedule_type}"
        assert 0 <= self.sigma_min < self.sigma_max <= 10.0, f"Invalid sigma range: [{self.sigma_min}, {self.sigma_max}]"


@dataclass
class TrainingConfig:
    """
    Configuration for training parameters optimized for global training
    """
    
    # Basic training parameters
    num_train_epochs: int = 6  # Fewer epochs needed for global training
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    learning_rate: float = 1e-4  # Slightly higher for global training
    weight_decay: float = 0.01
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    warmup_steps: int = 100
    
    # Training optimization
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True
    bf16: bool = False
    
    # Logging and evaluation
    logging_steps: int = 25
    eval_steps: int = 125
    save_steps: int = 250
    
    # Memory optimization
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    
    # Model selection
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_global_cosine_mean"
    greater_is_better: bool = True
    
    # Multi-GPU specific
    ddp_find_unused_parameters: bool = False
    save_on_each_node: bool = False


# ========================
# Factory Functions (FIXED)
# ========================

def get_default_blip3o_config() -> BLIP3oDiTConfig:
    """Get default BLIP3-o configuration with validation."""
    return BLIP3oDiTConfig()


def get_global_blip3o_config(
    model_size: str = "medium",
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get global BLIP3-o configuration for different model sizes
    
    Args:
        model_size: "small", "medium", "large"
        **kwargs: Additional parameters to override
    """
    configs = {
        "small": {
            "dim": 512,
            "n_layers": 8,
            "n_heads": 8,
            "mlp_hidden_dim": 1024,
        },
        "medium": {
            "dim": 768,
            "n_layers": 12,
            "n_heads": 12,
            "mlp_hidden_dim": 2048,
        },
        "large": {
            "dim": 1024,
            "n_layers": 16,
            "n_heads": 16,
            "mlp_hidden_dim": 3072,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(configs.keys())}")
    
    config_params = configs[model_size]
    config_params.update(kwargs)
    
    return BLIP3oDiTConfig(**config_params)


def get_multi_gpu_config(
    num_gpus: int,
    total_memory_gb: float = 40.0,
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get configuration optimized for multi-GPU training
    
    Args:
        num_gpus: Number of GPUs available
        total_memory_gb: Total memory per GPU
        **kwargs: Additional parameters
    """
    # Choose model size based on memory and GPU count
    if total_memory_gb >= 40 and num_gpus >= 4:
        model_size = "large"
    elif total_memory_gb >= 24 and num_gpus >= 2:
        model_size = "medium"
    else:
        model_size = "small"
    
    print(f"🔧 Auto-selected {model_size} model for {num_gpus} GPUs ({total_memory_gb}GB each)")
    
    return get_global_blip3o_config(model_size, **kwargs)


def get_default_flow_matching_config() -> FlowMatchingConfig:
    """Get default flow matching configuration."""
    return FlowMatchingConfig()


def get_enhanced_flow_matching_config() -> FlowMatchingConfig:
    """Get enhanced flow matching configuration with regularization."""
    return FlowMatchingConfig(
        use_contrastive_loss=True,
        contrastive_weight=0.1,
        gradient_penalty_weight=0.01,
        feature_matching_weight=0.05,
    )


def get_default_training_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


# ========================
# Validation Utilities
# ========================

def validate_config_compatibility(
    model_config: BLIP3oDiTConfig,
    flow_config: FlowMatchingConfig
) -> bool:
    """Validate compatibility between model and flow matching configurations."""
    
    # Check dimension compatibility
    assert model_config.in_channels == flow_config.clip_dim, \
        f"Model input channels ({model_config.in_channels}) must match flow CLIP dim ({flow_config.clip_dim})"
    
    assert model_config.eva_embedding_size == flow_config.eva_dim, \
        f"Model EVA size ({model_config.eva_embedding_size}) must match flow EVA dim ({flow_config.eva_dim})"
    
    # Check RoPE compatibility
    assert model_config.is_rope_compatible(), \
        f"Model configuration is not compatible with 3D RoPE"
    
    print("✅ Configuration compatibility validated")
    return True


def print_config_summary(config: BLIP3oDiTConfig):
    """Print a comprehensive configuration summary."""
    info = config.get_model_info()
    
    print("\n📊 BLIP3-o Configuration Summary")
    print("=" * 50)
    print(f"Architecture: {info['architecture']}")
    print(f"Model size: {config.dim}D, {config.n_layers}L, {config.n_heads}H")
    print(f"Head dimension: {info['head_dim']} (RoPE: {'✅' if info['rope_compatible'] else '❌'})")
    print(f"Total tokens: {info['total_tokens']}")
    print(f"Parameters: ~{info['parameters_estimate']:,}")
    print(f"Memory estimate: {info['memory_estimate_gb']:.1f} GB")
    print(f"Global training: {'✅' if config.global_training else '❌'}")
    print("=" * 50)


if __name__ == "__main__":
    # Test configuration creation and validation
    print("🧪 Testing FIXED BLIP3-o configurations...")
    
    # Test different model sizes
    for size in ["small", "medium", "large"]:
        print(f"\n🔧 Testing {size} model...")
        config = get_global_blip3o_config(size)
        print_config_summary(config)
    
    # Test multi-GPU config
    print(f"\n🔧 Testing multi-GPU config...")
    multi_config = get_multi_gpu_config(num_gpus=4, total_memory_gb=40)
    print_config_summary(multi_config)
    
    # Test flow matching compatibility
    flow_config = get_enhanced_flow_matching_config()
    validate_config_compatibility(multi_config, flow_config)
    
    print("\n🎉 All FIXED configuration tests passed!")