"""
BLIP3-o Configuration for CLIP Reproduction - Aligned with BLIP3-o Paper
Configuration for reproducing CLIP embeddings from EVA embeddings

Key changes from EVA reproduction:
1. eva_embedding_size = 4096 (conditioning)
2. clip_embedding_size = 1024 (target)
3. Updated parameter validation
4. Memory-optimized configurations
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional
from dataclasses import dataclass
import math


class BLIP3oCLIPDiTConfig(PretrainedConfig):
    """
    Configuration class for BLIP3-o CLIP reproduction DiT model.
    
    This configuration follows the BLIP3-o paper architecture with:
    - Patch-level training on 256 EVA tokens (4096-dim)
    - CLIP embedding reproduction (256 tokens, 1024-dim)  
    - Flow matching training objective
    - 3D RoPE and Grouped-Query Attention
    - Sandwich normalization (RMSNorm)
    """
    
    model_type = "blip3o_clip_dit"
    
    def __init__(
        self,
        # Model architecture
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,  # For grouped-query attention
        intermediate_size: int = 3072,
        
        # Input/output dimensions (BLIP3-o specific)
        eva_embedding_size: int = 4096,  # EVA-CLIP dimension (conditioning)
        clip_embedding_size: int = 1024,  # CLIP dimension (target)
        num_tokens: int = 256,  # 16x16 = 256 patches (or 257 with CLS)
        
        # Training configuration
        max_position_embeddings: int = 256,  # 16x16 patches
        dropout_prob: float = 0.0,  # Disabled for better training
        
        # Normalization (BLIP3-o uses RMSNorm)
        rms_norm_eps: float = 1e-6,
        use_rms_norm: bool = True,
        
        # Attention configuration (BLIP3-o specific)
        attention_dropout: float = 0.0,
        use_3d_rope: bool = True,  # 3D Rotary Position Embedding
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict] = None,
        
        # Flow matching parameters
        prediction_type: str = "velocity",
        
        # Training optimizations
        use_gradient_checkpointing: bool = False,
        training_mode: str = "patch_only",
        zero_init_output: bool = True,  # Zero init output for flow matching
        
        # BLIP3-o specific features
        use_sandwich_norm: bool = True,  # Sandwich normalization
        use_grouped_query_attention: bool = True,
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Core architecture
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        
        # Input/output dimensions
        self.eva_embedding_size = eva_embedding_size
        self.clip_embedding_size = clip_embedding_size
        self.num_tokens = num_tokens
        
        # Training configuration
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        
        # Normalization
        self.rms_norm_eps = rms_norm_eps
        self.use_rms_norm = use_rms_norm
        
        # Attention
        self.attention_dropout = attention_dropout
        self.use_3d_rope = use_3d_rope
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        
        # Flow matching
        self.prediction_type = prediction_type
        
        # Training optimizations
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        self.zero_init_output = zero_init_output
        
        # BLIP3-o specific
        self.use_sandwich_norm = use_sandwich_norm
        self.use_grouped_query_attention = use_grouped_query_attention
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check head dimension compatibility
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Check grouped-query attention compatibility
        if self.use_grouped_query_attention:
            if self.num_attention_heads % self.num_key_value_heads != 0:
                raise ValueError(
                    f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                    f"num_key_value_heads ({self.num_key_value_heads}) for grouped-query attention"
                )
        
        # Check number of tokens
        if self.num_tokens not in [256, 257]:
            raise ValueError(f"num_tokens must be 256 or 257, got {self.num_tokens}")
        
        # Check prediction type
        if self.prediction_type not in ["velocity", "epsilon"]:
            raise ValueError(f"prediction_type must be 'velocity' or 'epsilon', got {self.prediction_type}")
        
        # Validate embedding dimensions
        if self.eva_embedding_size <= 0:
            raise ValueError(f"eva_embedding_size must be positive, got {self.eva_embedding_size}")
        if self.clip_embedding_size <= 0:
            raise ValueError(f"clip_embedding_size must be positive, got {self.clip_embedding_size}")
    
    def get_head_dim(self):
        """Get attention head dimension."""
        return self.hidden_size // self.num_attention_heads
    
    def get_num_tokens(self):
        """Get number of tokens."""
        return self.num_tokens
    
    def get_parameter_count_estimate(self):
        """Estimate total parameter count"""
        # Input/output projections (reversed for CLIP reproduction)
        input_params = self.clip_embedding_size * self.hidden_size
        output_params = self.hidden_size * self.clip_embedding_size
        
        # Embeddings
        pos_embed_params = self.max_position_embeddings * self.hidden_size
        timestep_embed_params = 256 * self.hidden_size + self.hidden_size * self.hidden_size
        
        # Transformer layers
        layer_params = self.num_hidden_layers * (
            # Self-attention
            self.hidden_size * self.hidden_size * 3 +  # Q, K, V
            self.hidden_size * self.hidden_size +       # Output projection
            # Cross-attention
            self.hidden_size * self.hidden_size +       # Q projection
            self.eva_embedding_size * self.hidden_size * 2 +  # K, V projections for EVA
            self.hidden_size * self.hidden_size +       # Output projection
            # FFN
            self.hidden_size * self.intermediate_size +   # Up projection
            self.intermediate_size * self.hidden_size +   # Down projection
            # Norms (RMSNorm parameters)
            self.hidden_size * 6                          # Multiple norms per layer
        )
        
        total_params = (
            input_params + output_params + pos_embed_params + 
            timestep_embed_params + layer_params
        )
        
        return total_params
    
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        return output


def get_blip3o_clip_config(
    model_size: str = "base",
    training_mode: str = "patch_only",
    **kwargs
) -> BLIP3oCLIPDiTConfig:
    """
    Get predefined BLIP3-o configuration for CLIP reproduction.
    
    Args:
        model_size: Model size - "tiny", "small", "base", "large"
        training_mode: "patch_only" (256 tokens) or "cls_patch" (257 tokens)
        **kwargs: Additional configuration overrides
        
    Returns:
        BLIP3oCLIPDiTConfig instance
    """
    # Predefined configurations optimized for BLIP3-o architecture
    configs = {
        "tiny": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,  # Grouped-query attention
            "intermediate_size": 1536,
        },
        "small": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 4096,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(configs.keys())}")
    
    # Get base config
    config_dict = configs[model_size].copy()
    
    # Set token count based on training mode
    config_dict["num_tokens"] = 257 if training_mode == "cls_patch" else 256
    config_dict["max_position_embeddings"] = max(config_dict["num_tokens"], 257)
    config_dict["training_mode"] = training_mode
    
    # BLIP3-o specific defaults
    config_dict.update({
        "use_3d_rope": True,
        "use_sandwich_norm": True,
        "use_grouped_query_attention": True,
        "use_rms_norm": True,
        "zero_init_output": True,
        "dropout_prob": 0.0,  # Disable dropout
        "attention_dropout": 0.0,
    })
    
    # Apply overrides
    config_dict.update(kwargs)
    
    return BLIP3oCLIPDiTConfig(**config_dict)


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training for CLIP reproduction"""
    prediction_type: str = "velocity"
    normalize_targets: bool = True
    flow_type: str = "rectified"
    loss_scale: float = 1.0  # Reduced for better stability
    
    # Stability parameters
    min_timestep: float = 1e-3
    max_timestep: float = 1.0 - 1e-3
    clip_norm_max: float = 1.0
    
    # Boundary condition handling
    handle_boundaries: bool = True
    boundary_loss_weight: float = 0.1


@dataclass  
class TrainingConfig:
    """Configuration for training parameters optimized for CLIP reproduction"""
    num_epochs: int = 20
    batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 5e-4  # Higher LR based on feedback
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    dataloader_num_workers: int = 0
    
    # Evaluation parameters
    eval_every_n_steps: int = 50
    eval_num_samples: int = 100
    eval_inference_steps: int = 50
    
    # Debugging
    debug_mode: bool = False
    track_gradients: bool = True
    overfit_test_size: Optional[int] = None
    
    # Robustness
    skip_corrupted_samples: bool = True
    validate_tensor_shapes: bool = True
    max_grad_norm: float = 1.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    eval_every_n_steps: int = 50
    eval_num_samples: int = 100
    eval_batch_size: int = 16
    eval_inference_steps: int = 50
    normalize_embeddings: bool = True
    
    # Quality thresholds for CLIP reproduction
    high_quality_threshold: float = 0.7
    very_high_quality_threshold: float = 0.8
    excellent_quality_threshold: float = 0.9
    
    # Evaluation modes
    use_heun_solver: bool = False  # Use Euler for faster evaluation
    guidance_scale: float = 1.0


def get_default_clip_configs() -> tuple:
    """Get default configurations for all components"""
    model_config = get_blip3o_clip_config("base", "patch_only")
    flow_config = FlowMatchingConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    return model_config, flow_config, training_config, eval_config


def create_config_from_args(args) -> tuple:
    """Create configurations from command line arguments"""
    model_config = get_blip3o_clip_config(
        model_size=getattr(args, 'model_size', 'base'),
        training_mode=getattr(args, 'training_mode', 'patch_only'),
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
    )
    
    flow_config = FlowMatchingConfig(
        prediction_type="velocity",
        normalize_targets=True,
        flow_type="rectified",
        loss_scale=1.0,  # Improved stability
    )
    
    training_config = TrainingConfig(
        num_epochs=getattr(args, 'num_epochs', 20),
        batch_size=getattr(args, 'batch_size', 16),
        learning_rate=getattr(args, 'learning_rate', 5e-4),
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 2),
        fp16=getattr(args, 'fp16', True),
        debug_mode=getattr(args, 'debug_mode', False),
        overfit_test_size=getattr(args, 'overfit_test_size', None),
    )
    
    eval_config = EvaluationConfig(
        eval_every_n_steps=getattr(args, 'eval_every_n_steps', 50),
        eval_num_samples=getattr(args, 'eval_num_samples', 100),
        eval_inference_steps=getattr(args, 'eval_inference_steps', 50),
    )
    
    return model_config, flow_config, training_config, eval_config


def validate_config_compatibility(
    model_config: BLIP3oCLIPDiTConfig, 
    flow_config: FlowMatchingConfig,
    training_config: TrainingConfig
) -> bool:
    """Validate that all configs are compatible"""
    
    # Check flow matching compatibility
    if flow_config.prediction_type not in ["velocity", "epsilon"]:
        raise ValueError(f"Unsupported prediction type: {flow_config.prediction_type}")
    
    # Check model and training compatibility
    if model_config.num_tokens not in [256, 257]:
        raise ValueError(f"Invalid token count: {model_config.num_tokens}")
    
    # Check batch size compatibility
    if training_config.batch_size < 1:
        raise ValueError(f"Invalid batch size: {training_config.batch_size}")
    
    # Check grouped-query attention
    if model_config.use_grouped_query_attention:
        if model_config.num_attention_heads % model_config.num_key_value_heads != 0:
            raise ValueError("Incompatible grouped-query attention configuration")
    
    return True


def print_config_summary(
    model_config: BLIP3oCLIPDiTConfig,
    flow_config: FlowMatchingConfig,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig
):
    """Print comprehensive configuration summary"""
    print("📋 BLIP3-o CLIP Reproduction Configuration Summary")
    print("=" * 60)
    
    print(f"🏗️ Model Configuration (BLIP3-o DiT):")
    print(f"   Architecture: {model_config.hidden_size}D, {model_config.num_hidden_layers}L, {model_config.num_attention_heads}H")
    print(f"   Grouped-Query Attention: {model_config.num_attention_heads}/{model_config.num_key_value_heads} heads")
    print(f"   Tokens: {model_config.num_tokens} ({model_config.training_mode})")
    print(f"   EVA conditioning: {model_config.eva_embedding_size}D")
    print(f"   CLIP target: {model_config.clip_embedding_size}D")
    print(f"   3D RoPE: {model_config.use_3d_rope}")
    print(f"   Sandwich Norm: {model_config.use_sandwich_norm}")
    print(f"   RMS Norm: {model_config.use_rms_norm}")
    print(f"   Parameters: ~{model_config.get_parameter_count_estimate()/1e6:.1f}M")
    
    print(f"\n🌊 Flow Matching Configuration:")
    print(f"   Prediction type: {flow_config.prediction_type}")
    print(f"   Flow type: {flow_config.flow_type}")
    print(f"   Loss scale: {flow_config.loss_scale}")
    print(f"   Timestep range: [{flow_config.min_timestep}, {flow_config.max_timestep}]")
    print(f"   Normalize targets: {flow_config.normalize_targets}")
    
    print(f"\n🏃 Training Configuration:")
    print(f"   Epochs: {training_config.num_epochs}")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Weight decay: {training_config.weight_decay}")
    print(f"   LR scheduler: {training_config.lr_scheduler_type}")
    print(f"   Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"   Mixed precision: {training_config.fp16}")
    print(f"   Debug mode: {training_config.debug_mode}")
    if training_config.overfit_test_size:
        print(f"   Overfitting test: {training_config.overfit_test_size} samples")
    
    print(f"\n📊 Evaluation Configuration:")
    print(f"   Eval every: {eval_config.eval_every_n_steps} steps")
    print(f"   Eval samples: {eval_config.eval_num_samples}")
    print(f"   Inference steps: {eval_config.eval_inference_steps}")
    print(f"   Quality thresholds: {eval_config.high_quality_threshold}/{eval_config.very_high_quality_threshold}/{eval_config.excellent_quality_threshold}")
    print(f"   Normalize embeddings: {eval_config.normalize_embeddings}")
    
    print("=" * 60)


def get_memory_optimized_config(
    available_memory_gb: float,
    target_batch_size: int = None
) -> tuple:
    """
    Get memory-optimized configuration based on available GPU memory
    
    Args:
        available_memory_gb: Available GPU memory in GB
        target_batch_size: Desired batch size (optional)
        
    Returns:
        Tuple of (model_size, config, estimated_memory_usage)
    """
    # Memory usage estimates (rough) for different model sizes
    memory_estimates = {
        "tiny": {
            "base_memory_gb": 2.0,
            "memory_per_batch_item": 0.1,
            "max_batch_size": 32,
        },
        "small": {
            "base_memory_gb": 4.0,
            "memory_per_batch_item": 0.15,
            "max_batch_size": 24,
        },
        "base": {
            "base_memory_gb": 8.0,
            "memory_per_batch_item": 0.25,
            "max_batch_size": 16,
        },
        "large": {
            "base_memory_gb": 16.0,
            "memory_per_batch_item": 0.4,
            "max_batch_size": 8,
        },
    }
    
    # Find the largest model that fits
    for model_size in ["large", "base", "small", "tiny"]:
        estimates = memory_estimates[model_size]
        base_memory = estimates["base_memory_gb"]
        
        if target_batch_size:
            estimated_memory = base_memory + target_batch_size * estimates["memory_per_batch_item"]
            if estimated_memory <= available_memory_gb * 0.9:  # 90% usage
                config = get_blip3o_clip_config(model_size)
                return model_size, config, estimated_memory
        else:
            # Find optimal batch size
            max_batch_size = min(
                estimates["max_batch_size"],
                int((available_memory_gb * 0.9 - base_memory) / estimates["memory_per_batch_item"])
            )
            if max_batch_size >= 4:  # Minimum viable batch size
                config = get_blip3o_clip_config(model_size)
                estimated_memory = base_memory + max_batch_size * estimates["memory_per_batch_item"]
                return model_size, config, estimated_memory
    
    # Fallback to tiny with minimal batch size
    config = get_blip3o_clip_config("tiny")
    return "tiny", config, memory_estimates["tiny"]["base_memory_gb"] + 4 * memory_estimates["tiny"]["memory_per_batch_item"]


def create_overfitting_test_config(
    base_model_size: str = "base",
    test_size: int = 10,
    training_mode: str = "patch_only"
) -> tuple:
    """
    Create configuration optimized for overfitting test
    
    Args:
        base_model_size: Base model size
        test_size: Number of samples for overfitting test
        training_mode: Training mode
        
    Returns:
        Tuple of configurations optimized for overfitting
    """
    # Get base model config
    model_config = get_blip3o_clip_config(base_model_size, training_mode)
    
    # Flow matching config optimized for overfitting
    flow_config = FlowMatchingConfig(
        prediction_type="velocity",
        normalize_targets=True,
        flow_type="rectified",
        loss_scale=1.0,
        min_timestep=1e-3,
        max_timestep=0.999,
    )
    
    # Training config optimized for overfitting
    training_config = TrainingConfig(
        num_epochs=100,  # More epochs for overfitting
        batch_size=min(8, test_size),  # Small batch size
        learning_rate=1e-3,  # Higher learning rate for faster overfitting
        weight_decay=0.0,  # No regularization
        warmup_steps=0,  # No warmup for overfitting
        lr_scheduler_type="constant",
        gradient_accumulation_steps=1,
        fp16=True,
        eval_every_n_steps=20,  # More frequent evaluation
        eval_num_samples=test_size,
        debug_mode=True,  # Enable debugging
        track_gradients=True,
        overfit_test_size=test_size,
        skip_corrupted_samples=False,  # Don't skip any samples
        validate_tensor_shapes=True,
    )
    
    # Evaluation config for overfitting test
    eval_config = EvaluationConfig(
        eval_every_n_steps=20,
        eval_num_samples=test_size,
        eval_batch_size=min(4, test_size),
        eval_inference_steps=20,  # Fewer steps for faster evaluation
        normalize_embeddings=True,
    )
    
    return model_config, flow_config, training_config, eval_config


def validate_blip3o_clip_architecture(config: BLIP3oCLIPDiTConfig) -> Dict[str, bool]:
    """
    Validate that the configuration follows BLIP3-o architecture specifications
    
    Args:
        config: BLIP3oCLIPDiTConfig to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check 3D RoPE
    validation_results["3d_rope_enabled"] = config.use_3d_rope
    
    # Check Grouped-Query Attention
    validation_results["grouped_query_attention"] = (
        config.use_grouped_query_attention and 
        config.num_attention_heads % config.num_key_value_heads == 0
    )
    
    # Check Sandwich Normalization
    validation_results["sandwich_normalization"] = config.use_sandwich_norm
    
    # Check RMS Normalization
    validation_results["rms_normalization"] = config.use_rms_norm
    
    # Check input/output dimensions (reversed for CLIP reproduction)
    validation_results["correct_eva_dim"] = config.eva_embedding_size == 4096
    validation_results["correct_clip_dim"] = config.clip_embedding_size == 1024
    
    # Check token count
    validation_results["valid_token_count"] = config.num_tokens in [256, 257]
    
    # Check flow matching setup
    validation_results["velocity_prediction"] = config.prediction_type == "velocity"
    validation_results["zero_init_output"] = config.zero_init_output
    
    # Check training optimizations
    validation_results["dropout_disabled"] = config.dropout_prob == 0.0
    
    # Overall validation
    validation_results["blip3o_compliant"] = all([
        validation_results["3d_rope_enabled"],
        validation_results["grouped_query_attention"],
        validation_results["sandwich_normalization"],
        validation_results["rms_normalization"],
        validation_results["correct_eva_dim"],
        validation_results["correct_clip_dim"],
        validation_results["valid_token_count"],
        validation_results["velocity_prediction"],
    ])
    
    return validation_results


def print_architecture_validation(config: BLIP3oCLIPDiTConfig):
    """Print BLIP3-o architecture validation results"""
    validation = validate_blip3o_clip_architecture(config)
    
    print("🔍 BLIP3-o CLIP Reproduction Architecture Validation")
    print("=" * 50)
    
    # Core architecture features
    print("Core Architecture Features:")
    print(f"  ✅ 3D RoPE: {'Enabled' if validation['3d_rope_enabled'] else '❌ Disabled'}")
    print(f"  ✅ Grouped-Query Attention: {'Enabled' if validation['grouped_query_attention'] else '❌ Disabled'}")
    print(f"  ✅ Sandwich Normalization: {'Enabled' if validation['sandwich_normalization'] else '❌ Disabled'}")
    print(f"  ✅ RMS Normalization: {'Enabled' if validation['rms_normalization'] else '❌ Disabled'}")
    
    # Dimensions
    print("Input/Output Dimensions:")
    print(f"  ✅ EVA Conditioning (4096): {'Correct' if validation['correct_eva_dim'] else '❌ Incorrect'}")
    print(f"  ✅ CLIP Target (1024): {'Correct' if validation['correct_clip_dim'] else '❌ Incorrect'}")
    print(f"  ✅ Token Count: {'Valid' if validation['valid_token_count'] else '❌ Invalid'}")
    
    # Training setup
    print("Training Configuration:")
    print(f"  ✅ Velocity Prediction: {'Enabled' if validation['velocity_prediction'] else '❌ Disabled'}")
    print(f"  ✅ Zero Init Output: {'Enabled' if validation['zero_init_output'] else '❌ Disabled'}")
    print(f"  ✅ Dropout Disabled: {'Yes' if validation['dropout_disabled'] else '❌ No'}")
    
    # Overall compliance
    compliance_status = "✅ COMPLIANT" if validation['blip3o_compliant'] else "❌ NON-COMPLIANT"
    print(f"\nBLIP3-o CLIP Reproduction Compliance: {compliance_status}")
    
    if not validation['blip3o_compliant']:
        print("\n⚠️ Configuration does not fully comply with BLIP3-o specifications!")
        print("   Please review the failed validation points above.")
    
    print("=" * 50)


# Export commonly used configurations
DEFAULT_MODEL_CONFIG = get_blip3o_clip_config("base", "patch_only")
DEFAULT_FLOW_CONFIG = FlowMatchingConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EVAL_CONFIG = EvaluationConfig()

# Pre-validated configurations for different use cases
OVERFITTING_TEST_CONFIGS = create_overfitting_test_config("base", 10, "patch_only")
MEMORY_EFFICIENT_CONFIGS = get_memory_optimized_config(16.0, 8)  # For 16GB GPU