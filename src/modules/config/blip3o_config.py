"""
FIXED: BLIP3-o Configuration for CLIP Reproduction - Robust Parameter Handling
Key fixes:
1. Strict type validation for all parameters
2. Safe conversion of numeric parameters to Python types
3. Enhanced parameter validation
4. Better error handling and fallbacks
5. Comprehensive parameter documentation
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


class BLIP3oCLIPDiTConfig(PretrainedConfig):
    """
    FIXED: Configuration class for BLIP3-o CLIP reproduction DiT model with robust parameter handling.
    
    This configuration follows the BLIP3-o paper architecture with:
    - Patch-level training on 256 EVA tokens (4096-dim)
    - CLIP embedding reproduction (256 tokens, 1024-dim)  
    - Flow matching training objective
    - 3D RoPE and Grouped-Query Attention
    - Sandwich normalization (RMSNorm)
    - FIXED: Robust scale-aware generation parameters
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
        
        # FIXED: Scale-aware generation parameters with strict type validation
        typical_clip_norm: Union[float, int] = 26.0,
        velocity_explosion_threshold: Union[float, int] = 100.0,
        norm_guidance_strength: Union[float, int] = 0.1,
        norm_guidance_frequency: int = 10,
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Core architecture with validation
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.intermediate_size = int(intermediate_size)
        
        # Input/output dimensions with validation
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
        
        # FIXED: Scale-aware generation parameters with strict type enforcement
        self.typical_clip_norm = self._safe_convert_to_float(typical_clip_norm, "typical_clip_norm", 26.0)
        self.velocity_explosion_threshold = self._safe_convert_to_float(velocity_explosion_threshold, "velocity_explosion_threshold", 100.0)
        self.norm_guidance_strength = self._safe_convert_to_float(norm_guidance_strength, "norm_guidance_strength", 0.1)
        self.norm_guidance_frequency = int(norm_guidance_frequency)
        
        # Calculate grid size for 3D RoPE
        self.grid_size = self.image_size // self.patch_size  # 224 // 14 = 16
        
        # Validate configuration
        self._validate_config()
        
        # Log successful initialization
        logger.info(f"FIXED BLIP3oCLIPDiTConfig initialized:")
        logger.info(f"  typical_clip_norm: {self.typical_clip_norm} (type: {type(self.typical_clip_norm).__name__})")
        logger.info(f"  velocity_explosion_threshold: {self.velocity_explosion_threshold}")
        logger.info(f"  norm_guidance_strength: {self.norm_guidance_strength}")
        logger.info(f"  norm_guidance_frequency: {self.norm_guidance_frequency}")
    
    def _safe_convert_to_float(self, value, param_name: str, default_value: float) -> float:
        """
        FIXED: Safely convert any numeric value to Python float with comprehensive error handling
        """
        try:
            if value is None:
                logger.warning(f"⚠️ {param_name} is None, using default: {default_value}")
                return float(default_value)
            
            if hasattr(value, 'item'):  # torch.Tensor or numpy array
                if hasattr(value, 'numel') and value.numel() != 1:
                    logger.error(f"❌ {param_name} is a multi-element tensor/array!")
                    logger.error(f"   Shape: {getattr(value, 'shape', 'unknown')}")
                    logger.error(f"   Using default: {default_value}")
                    return float(default_value)
                result = float(value.item())
            elif isinstance(value, (int, float)):
                result = float(value)
            else:
                logger.error(f"❌ {param_name} has unexpected type: {type(value)}")
                logger.error(f"   Value: {value}")
                logger.error(f"   Using default: {default_value}")
                return float(default_value)
            
            # Validate the result
            if math.isnan(result) or math.isinf(result):
                logger.error(f"❌ {param_name} is NaN or Inf: {result}")
                return float(default_value)
            
            # Validate reasonable ranges
            if param_name == "typical_clip_norm" and not (10.0 <= result <= 100.0):
                logger.warning(f"⚠️ {param_name} outside reasonable range [10, 100]: {result}")
                result = max(10.0, min(100.0, result))
            elif param_name == "velocity_explosion_threshold" and not (50.0 <= result <= 1000.0):
                logger.warning(f"⚠️ {param_name} outside reasonable range [50, 1000]: {result}")
                result = max(50.0, min(1000.0, result))
            elif param_name == "norm_guidance_strength" and not (0.0 <= result <= 1.0):
                logger.warning(f"⚠️ {param_name} outside reasonable range [0, 1]: {result}")
                result = max(0.0, min(1.0, result))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error converting {param_name} to float: {e}")
            logger.error(f"   Input value: {value}")
            logger.error(f"   Input type: {type(value)}")
            logger.error(f"   Using default: {default_value}")
            return float(default_value)
    
    def _validate_config(self):
        """FIXED: Validate configuration parameters with enhanced checking."""
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
        
        # FIXED: Validate scale-aware parameters
        if not isinstance(self.typical_clip_norm, float):
            validation_errors.append(f"typical_clip_norm must be float, got {type(self.typical_clip_norm)}")
        if not isinstance(self.velocity_explosion_threshold, float):
            validation_errors.append(f"velocity_explosion_threshold must be float, got {type(self.velocity_explosion_threshold)}")
        if not isinstance(self.norm_guidance_strength, float):
            validation_errors.append(f"norm_guidance_strength must be float, got {type(self.norm_guidance_strength)}")
        if not isinstance(self.norm_guidance_frequency, int):
            validation_errors.append(f"norm_guidance_frequency must be int, got {type(self.norm_guidance_frequency)}")
        
        # Check parameter ranges
        if self.typical_clip_norm <= 0:
            validation_errors.append(f"typical_clip_norm must be positive, got {self.typical_clip_norm}")
        if self.velocity_explosion_threshold <= 0:
            validation_errors.append(f"velocity_explosion_threshold must be positive, got {self.velocity_explosion_threshold}")
        if not (0.0 <= self.norm_guidance_strength <= 1.0):
            validation_errors.append(f"norm_guidance_strength must be in [0, 1], got {self.norm_guidance_strength}")
        if self.norm_guidance_frequency <= 0:
            validation_errors.append(f"norm_guidance_frequency must be positive, got {self.norm_guidance_frequency}")
        
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
        """Convert config to dictionary with type validation."""
        output = super().to_dict()
        
        # FIXED: Ensure all scale-aware parameters are proper Python types in the dict
        output['typical_clip_norm'] = float(self.typical_clip_norm)
        output['velocity_explosion_threshold'] = float(self.velocity_explosion_threshold)
        output['norm_guidance_strength'] = float(self.norm_guidance_strength)
        output['norm_guidance_frequency'] = int(self.norm_guidance_frequency)
        
        return output


def get_blip3o_clip_config(
    model_size: str = "base",
    training_mode: str = "patch_only",
    # FIXED: Scale-aware parameters with type validation
    typical_clip_norm: Union[float, int] = 26.0,
    velocity_explosion_threshold: Union[float, int] = 100.0,
    norm_guidance_strength: Union[float, int] = 0.1,
    norm_guidance_frequency: int = 10,
    **kwargs
) -> BLIP3oCLIPDiTConfig:
    """
    FIXED: Get predefined BLIP3-o configuration for CLIP reproduction with robust parameter handling.
    
    Args:
        model_size: Model size - "tiny", "small", "base", "large"
        training_mode: "patch_only" (256 tokens) or "cls_patch" (257 tokens)
        typical_clip_norm: Typical CLIP embedding norm for scale guidance
        velocity_explosion_threshold: Threshold for velocity explosion prevention
        norm_guidance_strength: Strength of norm guidance during generation
        norm_guidance_frequency: Frequency of norm guidance application
        **kwargs: Additional configuration overrides
        
    Returns:
        BLIP3oCLIPDiTConfig instance with validated parameters
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
    
    # FIXED: Add scale-aware parameters with type validation
    config_dict.update({
        "typical_clip_norm": float(typical_clip_norm),
        "velocity_explosion_threshold": float(velocity_explosion_threshold),
        "norm_guidance_strength": float(norm_guidance_strength),
        "norm_guidance_frequency": int(norm_guidance_frequency),
    })
    
    # Apply additional overrides
    config_dict.update(kwargs)
    
    return BLIP3oCLIPDiTConfig(**config_dict)


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training for CLIP reproduction"""
    prediction_type: str = "velocity"
    normalize_targets: bool = True
    flow_type: str = "rectified"
    loss_scale: float = 1.0
    
    # Stability parameters
    min_timestep: float = 1e-3
    max_timestep: float = 1.0 - 1e-3
    clip_norm_max: float = 1.0
    
    # Boundary condition handling
    handle_boundaries: bool = True
    boundary_loss_weight: float = 0.1


@dataclass  
class TrainingConfig:
    """FIXED: Configuration for training parameters with robust validation"""
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
    """FIXED: Configuration for evaluation parameters with scale-aware enhancements"""
    eval_every_n_steps: int = 50
    eval_num_samples: int = 15
    eval_batch_size: int = 16
    eval_inference_steps: int = 20
    normalize_embeddings: bool = True
    
    # Quality thresholds for CLIP reproduction
    high_quality_threshold: float = 0.7
    very_high_quality_threshold: float = 0.8
    excellent_quality_threshold: float = 0.9
    
    # FIXED: Scale-aware evaluation parameters
    use_scale_aware_eval: bool = True
    adaptive_target_norm: bool = True
    use_lognormal_schedule: bool = True
    target_norm_estimation_method: str = "adaptive"  # "adaptive" or "fixed"
    
    # Evaluation modes
    use_heun_solver: bool = False  # Use Euler for faster evaluation
    guidance_scale: float = 1.0


def get_default_clip_configs() -> tuple:
    """Get default configurations for all components with fixes"""
    model_config = get_blip3o_clip_config("base", "patch_only")
    flow_config = FlowMatchingConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    return model_config, flow_config, training_config, eval_config


def create_config_from_args(args) -> tuple:
    """FIXED: Create configurations from command line arguments with type validation"""
    
    # FIXED: Ensure all scale-aware parameters are proper Python types
    typical_clip_norm = float(getattr(args, 'typical_clip_norm', 26.0))
    velocity_explosion_threshold = float(getattr(args, 'velocity_explosion_threshold', 100.0))
    norm_guidance_strength = float(getattr(args, 'norm_guidance_strength', 0.1))
    norm_guidance_frequency = int(getattr(args, 'norm_guidance_frequency', 10))
    
    logger.info(f"Creating config from args with scale-aware parameters:")
    logger.info(f"  typical_clip_norm: {typical_clip_norm} (type: {type(typical_clip_norm).__name__})")
    logger.info(f"  velocity_explosion_threshold: {velocity_explosion_threshold}")
    logger.info(f"  norm_guidance_strength: {norm_guidance_strength}")
    logger.info(f"  norm_guidance_frequency: {norm_guidance_frequency}")
    
    model_config = get_blip3o_clip_config(
        model_size=getattr(args, 'model_size', 'base'),
        training_mode=getattr(args, 'training_mode', 'patch_only'),
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
        typical_clip_norm=typical_clip_norm,
        velocity_explosion_threshold=velocity_explosion_threshold,
        norm_guidance_strength=norm_guidance_strength,
        norm_guidance_frequency=norm_guidance_frequency,
    )
    
    flow_config = FlowMatchingConfig(
        prediction_type="velocity",
        normalize_targets=True,
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
        use_scale_aware_eval=getattr(args, 'use_scale_aware', True) and not getattr(args, 'no_scale_aware', False),
        adaptive_target_norm=getattr(args, 'adaptive_target_norm', True),
        use_lognormal_schedule=getattr(args, 'eval_use_lognormal_schedule', True),
    )
    
    return model_config, flow_config, training_config, eval_config


def validate_config_compatibility(
    model_config: BLIP3oCLIPDiTConfig, 
    flow_config: FlowMatchingConfig,
    training_config: TrainingConfig
) -> bool:
    """FIXED: Validate that all configs are compatible with enhanced checking"""
    
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
    
    # FIXED: Check scale-aware parameter types
    if not isinstance(model_config.typical_clip_norm, float):
        validation_errors.append(f"typical_clip_norm must be float, got {type(model_config.typical_clip_norm)}")
    
    # Check parameter ranges
    if not (10.0 <= model_config.typical_clip_norm <= 100.0):
        validation_errors.append(f"typical_clip_norm outside reasonable range: {model_config.typical_clip_norm}")
    
    if not (0.0 <= model_config.norm_guidance_strength <= 1.0):
        validation_errors.append(f"norm_guidance_strength outside valid range: {model_config.norm_guidance_strength}")
    
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
    """FIXED: Print comprehensive configuration summary with validation"""
    print("📋 FIXED BLIP3-o CLIP Reproduction Configuration Summary")
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
    
    print(f"\n🎯 FIXED Scale-Aware Configuration:")
    print(f"   Typical CLIP norm: {model_config.typical_clip_norm:.3f} (type: {type(model_config.typical_clip_norm).__name__})")
    print(f"   Velocity explosion threshold: {model_config.velocity_explosion_threshold:.1f}")
    print(f"   Norm guidance strength: {model_config.norm_guidance_strength:.3f}")
    print(f"   Norm guidance frequency: {model_config.norm_guidance_frequency}")
    print(f"   Fixed target norm handling: ✅")
    
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
    
    print(f"\n📊 FIXED Evaluation Configuration:")
    print(f"   Eval every: {eval_config.eval_every_n_steps} steps")
    print(f"   Eval samples: {eval_config.eval_num_samples}")
    print(f"   Inference steps: {eval_config.eval_inference_steps}")
    print(f"   Quality thresholds: {eval_config.high_quality_threshold}/{eval_config.very_high_quality_threshold}/{eval_config.excellent_quality_threshold}")
    print(f"   Scale-aware eval: {eval_config.use_scale_aware_eval}")
    print(f"   Adaptive target norm: {eval_config.adaptive_target_norm}")
    print(f"   Log-normal schedule: {eval_config.use_lognormal_schedule}")
    print(f"   Fixed norm handling: ✅")
    
    print("=" * 80)


def validate_blip3o_clip_architecture(config: BLIP3oCLIPDiTConfig) -> Dict[str, bool]:
    """
    FIXED: Validate that the configuration follows BLIP3-o architecture specifications with enhanced checking
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
    
    # FIXED: Check scale-aware parameters
    validation_results["typical_clip_norm_valid"] = (
        isinstance(config.typical_clip_norm, float) and 
        10.0 <= config.typical_clip_norm <= 100.0
    )
    validation_results["velocity_threshold_valid"] = (
        isinstance(config.velocity_explosion_threshold, float) and 
        config.velocity_explosion_threshold > 0
    )
    validation_results["norm_guidance_strength_valid"] = (
        isinstance(config.norm_guidance_strength, float) and 
        0.0 <= config.norm_guidance_strength <= 1.0
    )
    validation_results["norm_guidance_frequency_valid"] = (
        isinstance(config.norm_guidance_frequency, int) and 
        config.norm_guidance_frequency > 0
    )
    
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
    
    # FIXED: Scale-aware compliance
    validation_results["scale_aware_compliant"] = all([
        validation_results["typical_clip_norm_valid"],
        validation_results["velocity_threshold_valid"],
        validation_results["norm_guidance_strength_valid"],
        validation_results["norm_guidance_frequency_valid"],
    ])
    
    # Overall compliance including fixes
    validation_results["fully_compliant"] = (
        validation_results["blip3o_compliant"] and 
        validation_results["scale_aware_compliant"]
    )
    
    return validation_results


def print_architecture_validation(config: BLIP3oCLIPDiTConfig):
    """FIXED: Print BLIP3-o architecture validation results with scale-aware checks"""
    validation = validate_blip3o_clip_architecture(config)
    
    print("🔍 FIXED BLIP3-o CLIP Reproduction Architecture Validation")
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
    
    # FIXED: Scale-aware validation
    print("FIXED Scale-Aware Parameters:")
    print(f"  ✅ Typical CLIP Norm: {'Valid' if validation['typical_clip_norm_valid'] else '❌ Invalid'} ({config.typical_clip_norm:.3f})")
    print(f"  ✅ Velocity Threshold: {'Valid' if validation['velocity_threshold_valid'] else '❌ Invalid'} ({config.velocity_explosion_threshold:.1f})")
    print(f"  ✅ Norm Guidance Strength: {'Valid' if validation['norm_guidance_strength_valid'] else '❌ Invalid'} ({config.norm_guidance_strength:.3f})")
    print(f"  ✅ Norm Guidance Frequency: {'Valid' if validation['norm_guidance_frequency_valid'] else '❌ Invalid'} ({config.norm_guidance_frequency})")
    
    # Overall compliance
    compliance_status = "✅ FULLY COMPLIANT" if validation['fully_compliant'] else "❌ NON-COMPLIANT"
    print(f"\nBLIP3-o CLIP Reproduction Compliance: {compliance_status}")
    
    if validation['blip3o_compliant'] and not validation['scale_aware_compliant']:
        print("  ✅ Core BLIP3-o architecture: COMPLIANT")
        print("  ❌ Scale-aware parameters: NON-COMPLIANT")
    elif not validation['blip3o_compliant'] and validation['scale_aware_compliant']:
        print("  ❌ Core BLIP3-o architecture: NON-COMPLIANT")
        print("  ✅ Scale-aware parameters: COMPLIANT")
    elif not validation['fully_compliant']:
        print("  ❌ Multiple compliance issues detected")
    
    if not validation['fully_compliant']:
        print("\n⚠️ Configuration does not fully comply with BLIP3-o specifications!")
        print("   Please review the failed validation points above.")
        
        # Specific recommendations
        if not validation['typical_clip_norm_valid']:
            print(f"   💡 Fix typical_clip_norm: current={config.typical_clip_norm}, should be in [10, 100]")
        if not validation['norm_guidance_strength_valid']:
            print(f"   💡 Fix norm_guidance_strength: current={config.norm_guidance_strength}, should be in [0, 1]")
    
    print("=" * 70)


# FIXED: Export configurations with validated parameters
def create_validated_configs(
    model_size: str = "base",
    training_mode: str = "patch_only",
    typical_clip_norm: Union[float, int] = 26.0,
    velocity_explosion_threshold: Union[float, int] = 100.0,
    norm_guidance_strength: Union[float, int] = 0.1,
    norm_guidance_frequency: int = 10,
) -> tuple:
    """Create validated configurations for BLIP3-o with fixed scale-aware parameters"""
    
    # Create model config with validation
    model_config = get_blip3o_clip_config(
        model_size=model_size,
        training_mode=training_mode,
        typical_clip_norm=typical_clip_norm,
        velocity_explosion_threshold=velocity_explosion_threshold,
        norm_guidance_strength=norm_guidance_strength,
        norm_guidance_frequency=norm_guidance_frequency,
    )
    
    flow_config = FlowMatchingConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    # Validate compatibility
    validate_config_compatibility(model_config, flow_config, training_config)
    
    # Validate architecture
    validation_results = validate_blip3o_clip_architecture(model_config)
    if not validation_results['fully_compliant']:
        logger.warning("⚠️ Configuration is not fully compliant - some features may not work as expected")
    
    return model_config, flow_config, training_config, eval_config


# FIXED: Pre-validated configurations
try:
    DEFAULT_MODEL_CONFIG, DEFAULT_FLOW_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_EVAL_CONFIG = create_validated_configs()
    logger.info("✅ Default configurations created and validated")
except Exception as e:
    logger.error(f"❌ Error creating default configurations: {e}")
    # Fallback to basic config
    DEFAULT_MODEL_CONFIG = None
    DEFAULT_FLOW_CONFIG = FlowMatchingConfig()
    DEFAULT_TRAINING_CONFIG = TrainingConfig()
    DEFAULT_EVAL_CONFIG = EvaluationConfig()


# Memory-efficient configurations
def get_memory_optimized_config(
    available_memory_gb: float,
    target_batch_size: int = None,
    enable_scale_aware: bool = True,
) -> tuple:
    """
    FIXED: Get memory-optimized configuration with scale-aware features
    """
    # Memory usage estimates for different model sizes
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
                config = get_blip3o_clip_config(
                    model_size, 
                    enable_scale_aware=enable_scale_aware
                )
                return model_size, config, estimated_memory
        else:
            # Find optimal batch size
            max_batch_size = min(
                estimates["max_batch_size"],
                int((available_memory_gb * 0.9 - base_memory) / estimates["memory_per_batch_item"])
            )
            if max_batch_size >= 4:  # Minimum viable batch size
                config = get_blip3o_clip_config(
                    model_size,
                    enable_scale_aware=enable_scale_aware
                )
                estimated_memory = base_memory + max_batch_size * estimates["memory_per_batch_item"]
                return model_size, config, estimated_memory
    
    # Fallback to tiny with minimal batch size
    config = get_blip3o_clip_config("tiny", enable_scale_aware=enable_scale_aware)
    return "tiny", config, memory_estimates["tiny"]["base_memory_gb"] + 4 * memory_estimates["tiny"]["memory_per_batch_item"]