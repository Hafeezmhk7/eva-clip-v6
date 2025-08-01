"""
Clean BLIP3-o Configuration for CLIP Reproduction
Simple configuration without scale-aware complexities
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


class BLIP3oCLIPDiTConfig(PretrainedConfig):
    """
    Clean configuration class for BLIP3-o CLIP reproduction DiT model.
    
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
        image_size: int = 224,
        patch_size: int = 14,
        
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
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.intermediate_size = int(intermediate_size)
        
        # Input/output dimensions
        self.eva_embedding_size = int(eva_embedding_size)
        self.clip_embedding_size = int(clip_embedding_size)
        self.num_tokens = int(num_tokens)
        
        # Training configuration
        self.max_position_embeddings = int(max_position_embeddings)
        self.dropout_prob = float(dropout_prob)
        
        # Normalization
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_rms_norm = bool(use_rms_norm)
        
        # Attention
        self.attention_dropout = float(attention_dropout)
        self.use_3d_rope = bool(use_3d_rope)
        self.rope_theta = float(rope_theta)
        self.rope_scaling = rope_scaling
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        
        # Flow matching
        self.prediction_type = str(prediction_type)
        
        # Training optimizations
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.training_mode = str(training_mode)
        self.zero_init_output = bool(zero_init_output)
        
        # BLIP3-o specific
        self.use_sandwich_norm = bool(use_sandwich_norm)
        self.use_grouped_query_attention = bool(use_grouped_query_attention)
        
        # Calculate grid size for 3D RoPE
        self.grid_size = self.image_size // self.patch_size  # 224 // 14 = 16
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        validation_errors = []
        
        # Check head dimension compatibility
        if self.hidden_size % self.num_attention_heads != 0:
            validation_errors.append(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        # Check grouped-query attention compatibility
        if self.use_grouped_query_attention:
            if self.num_attention_heads % self.num_key_value_heads != 0:
                validation_errors.append(
                    f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                    f"num_key_value_heads ({self.num_key_value_heads}) for grouped-query attention"
                )
        
        # Check number of tokens
        if self.num_tokens not in [256, 257]:
            validation_errors.append(f"num_tokens must be 256 or 257, got {self.num_tokens}")
        
        # Check prediction type
        if self.prediction_type not in ["velocity", "epsilon"]:
            validation_errors.append(f"prediction_type must be 'velocity' or 'epsilon', got {self.prediction_type}")
        
        # Validate embedding dimensions
        if self.eva_embedding_size <= 0:
            validation_errors.append(f"eva_embedding_size must be positive, got {self.eva_embedding_size}")
        if self.clip_embedding_size <= 0:
            validation_errors.append(f"clip_embedding_size must be positive, got {self.clip_embedding_size}")
        
        # Raise error if validation fails
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  • {err}" for err in validation_errors)
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.info("✅ Configuration validation passed")
    
    def get_head_dim(self):
        """Get attention head dimension."""
        return self.hidden_size // self.num_attention_heads
    
    def get_num_tokens(self):
        """Get number of tokens."""
        return self.num_tokens
    
    def get_parameter_count_estimate(self):
        """Estimate total parameter count"""
        # Input/output projections
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
    
    # Apply additional overrides
    config_dict.update(kwargs)
    
    return BLIP3oCLIPDiTConfig(**config_dict)


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training for CLIP reproduction"""
    prediction_type: str = "velocity"
    flow_type: str = "rectified"
    loss_scale: float = 1.0
    
    # Stability parameters
    min_timestep: float = 1e-3
    max_timestep: float = 1.0 - 1e-3


@dataclass  
class TrainingConfig:
    """Configuration for training parameters"""
    num_epochs: int = 20
    batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 1e-4  # Conservative for stability
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    dataloader_num_workers: int = 0
    
    # Evaluation parameters
    eval_every_n_steps: int = 50
    eval_num_samples: int = 15  # Reduced for faster evaluation
    eval_inference_steps: int = 20  # Reduced for faster evaluation
    
    # Debugging
    debug_mode: bool = False
    track_gradients: bool = True
    overfit_test_size: Optional[int] = 20  # Enable by default for validation
    
    # Robustness
    skip_corrupted_samples: bool = True
    validate_tensor_shapes: bool = True
    max_grad_norm: float = 1.0


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    eval_every_n_steps: int = 50
    eval_num_samples: int = 15
    eval_batch_size: int = 16
    eval_inference_steps: int = 20
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
        flow_type="rectified",
        loss_scale=1.0,
    )
    
    training_config = TrainingConfig(
        num_epochs=getattr(args, 'num_epochs', 10),
        batch_size=getattr(args, 'batch_size', 8),
        learning_rate=getattr(args, 'learning_rate', 1e-4),
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 2),
        fp16=getattr(args, 'fp16', True),
        debug_mode=getattr(args, 'debug_mode', False),
        overfit_test_size=getattr(args, 'overfit_test_size', None),
        eval_every_n_steps=getattr(args, 'eval_every_n_steps', 50),
        eval_num_samples=getattr(args, 'eval_num_samples', 15),
        eval_inference_steps=getattr(args, 'eval_inference_steps', 20),
    )
    
    eval_config = EvaluationConfig(
        eval_every_n_steps=getattr(args, 'eval_every_n_steps', 50),
        eval_num_samples=getattr(args, 'eval_num_samples', 15),
        eval_inference_steps=getattr(args, 'eval_inference_steps', 20),
    )
    
    return model_config, flow_config, training_config, eval_config


def validate_config_compatibility(
    model_config: BLIP3oCLIPDiTConfig, 
    flow_config: FlowMatchingConfig,
    training_config: TrainingConfig
) -> bool:
    """Validate that all configs are compatible"""
    
    validation_errors = []
    
    # Check flow matching compatibility
    if flow_config.prediction_type not in ["velocity", "epsilon"]:
        validation_errors.append(f"Unsupported prediction type: {flow_config.prediction_type}")
    
    # Check model and training compatibility
    if model_config.num_tokens not in [256, 257]:
        validation_errors.append(f"Invalid token count: {model_config.num_tokens}")
    
    # Check batch size compatibility
    if training_config.batch_size < 1:
        validation_errors.append(f"Invalid batch size: {training_config.batch_size}")
    
    # Check grouped-query attention
    if model_config.use_grouped_query_attention:
        if model_config.num_attention_heads % model_config.num_key_value_heads != 0:
            validation_errors.append("Incompatible grouped-query attention configuration")
    
    if validation_errors:
        error_msg = "Configuration compatibility validation failed:\n" + "\n".join(f"  • {err}" for err in validation_errors)
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    logger.info("✅ Configuration compatibility validation passed")
    return True


def print_config_summary(
    model_config: BLIP3oCLIPDiTConfig,
    flow_config: FlowMatchingConfig,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig
):
    """Print comprehensive configuration summary"""
    print("📋 Clean BLIP3-o CLIP Reproduction Configuration Summary")
    print("=" * 80)
    
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
    
    print("=" * 80)


def validate_blip3o_clip_architecture(config: BLIP3oCLIPDiTConfig) -> Dict[str, bool]:
    """
    Validate that the configuration follows BLIP3-o architecture specifications
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
    
    # Check input/output dimensions
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
    
    print("🔍 Clean BLIP3-o CLIP Reproduction Architecture Validation")
    print("=" * 70)
    
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
    compliance_status = "✅ FULLY COMPLIANT" if validation['blip3o_compliant'] else "❌ NON-COMPLIANT"
    print(f"\nBLIP3-o CLIP Reproduction Compliance: {compliance_status}")
    
    if not validation['blip3o_compliant']:
        print("\n⚠️ Configuration does not fully comply with BLIP3-o specifications!")
        print("   Please review the failed validation points above.")
    
    print("=" * 70)


# Export configurations
def create_clean_configs(
    model_size: str = "base",
    training_mode: str = "patch_only",
) -> tuple:
    """Create clean configurations for BLIP3-o"""
    
    # Create model config
    model_config = get_blip3o_clip_config(
        model_size=model_size,
        training_mode=training_mode,
    )
    
    flow_config = FlowMatchingConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    # Validate compatibility
    validate_config_compatibility(model_config, flow_config, training_config)
    
    # Validate architecture
    validation_results = validate_blip3o_clip_architecture(model_config)
    if not validation_results['blip3o_compliant']:
        logger.warning("⚠️ Configuration is not fully compliant - some features may not work as expected")
    
    return model_config, flow_config, training_config, eval_config


# Pre-validated configurations
try:
    DEFAULT_MODEL_CONFIG, DEFAULT_FLOW_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_EVAL_CONFIG = create_clean_configs()
    logger.info("✅ Default clean configurations created and validated")
except Exception as e:
    logger.error(f"❌ Error creating default configurations: {e}")
    # Fallback to basic config
    DEFAULT_MODEL_CONFIG = None
    DEFAULT_FLOW_CONFIG = FlowMatchingConfig()
    DEFAULT_TRAINING_CONFIG = TrainingConfig()
    DEFAULT_EVAL_CONFIG = EvaluationConfig()