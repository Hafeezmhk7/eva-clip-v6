"""
BLIP3-o Configuration - Aligned with BLIP3-o Paper
src/modules/config/blip3o_config.py

Configuration classes for BLIP3-o patch-level DiT model following the paper architecture.
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional
from dataclasses import dataclass


class BLIP3oDiTConfig(PretrainedConfig):
    """
    Configuration class for BLIP3-o patch-level DiT model.
    
    This configuration follows the BLIP3-o paper architecture with:
    - Patch-level training on 256 CLIP tokens (1024-dim)
    - EVA-CLIP conditioning (256 tokens, 4096-dim)  
    - Flow matching training objective
    - Image-to-text recall evaluation
    """
    
    model_type = "blip3o_patch_dit"
    
    def __init__(
        self,
        # Model architecture
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        
        # Input/output dimensions (BLIP3-o specific)
        eva_embedding_size: int = 4096,  # EVA-CLIP dimension
        clip_embedding_size: int = 1024,  # CLIP patch dimension
        num_tokens: int = 256,  # 16x16 = 256 patches (or 257 with CLS)
        
        # Training configuration
        max_position_embeddings: int = 256,  # 16x16 patches
        dropout_prob: float = 0.1,
        
        # Flow matching parameters
        prediction_type: str = "velocity",
        
        # Training optimizations
        use_gradient_checkpointing: bool = False,
        training_mode: str = "patch_only",
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Core architecture
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        
        # Input/output dimensions
        self.eva_embedding_size = eva_embedding_size
        self.clip_embedding_size = clip_embedding_size
        self.num_tokens = num_tokens
        
        # Training configuration
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        
        # Flow matching
        self.prediction_type = prediction_type
        
        # Training optimizations
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.training_mode = training_mode
        
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
        
        # Check number of tokens
        if self.num_tokens not in [256, 257]:
            raise ValueError(f"num_tokens must be 256 or 257, got {self.num_tokens}")
        
        # Check prediction type
        if self.prediction_type not in ["velocity", "epsilon"]:
            raise ValueError(f"prediction_type must be 'velocity' or 'epsilon', got {self.prediction_type}")
    
    def get_head_dim(self):
        """Get attention head dimension."""
        return self.hidden_size // self.num_attention_heads
    
    def get_num_tokens(self):
        """Get number of tokens."""
        return self.num_tokens
    
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        return output


def get_blip3o_config(
    model_size: str = "base",
    training_mode: str = "patch_only",
    **kwargs
) -> BLIP3oDiTConfig:
    """
    Get predefined BLIP3-o configuration.
    
    Args:
        model_size: Model size - "tiny", "small", "base", "large"
        training_mode: "patch_only" (256 tokens) or "cls_patch" (257 tokens)
        **kwargs: Additional configuration overrides
        
    Returns:
        BLIP3oDiTConfig instance
    """
    # Predefined configurations optimized for patch-level training
    configs = {
        "tiny": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
        },
        "small": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
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
    
    # Apply overrides
    config_dict.update(kwargs)
    
    return BLIP3oDiTConfig(**config_dict)


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching training"""
    prediction_type: str = "velocity"
    normalize_targets: bool = True
    flow_type: str = "rectified"
    
    # Removed problematic scaling parameters
    clip_norm_max: float = 10.0  # Gradient clipping only


@dataclass  
class TrainingConfig:
    """Configuration for training parameters"""
    num_epochs: int = 10
    batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    dataloader_num_workers: int = 0
    
    # Evaluation parameters
    eval_every_n_steps: int = 100
    eval_num_samples: int = 1000
    eval_inference_steps: int = 50


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    eval_every_n_steps: int = 100
    eval_num_samples: int = 1000
    eval_batch_size: int = 16
    eval_inference_steps: int = 50
    normalize_embeddings: bool = True
    
    # Similarity thresholds
    high_quality_threshold: float = 0.7
    very_high_quality_threshold: float = 0.8
    excellent_quality_threshold: float = 0.9


def get_default_configs() -> tuple:
    """Get default configurations for all components"""
    model_config = get_blip3o_config("base", "patch_only")
    flow_config = FlowMatchingConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()
    
    return model_config, flow_config, training_config, eval_config


def create_config_from_args(args) -> tuple:
    """Create configurations from command line arguments"""
    model_config = get_blip3o_config(
        model_size=getattr(args, 'model_size', 'base'),
        training_mode=getattr(args, 'training_mode', 'patch_only'),
        use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
    )
    
    flow_config = FlowMatchingConfig(
        prediction_type="velocity",
        normalize_targets=True,
        flow_type="rectified",
    )
    
    training_config = TrainingConfig(
        num_epochs=getattr(args, 'num_epochs', 10),
        batch_size=getattr(args, 'batch_size', 32),
        learning_rate=getattr(args, 'learning_rate', 1e-4),
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 2),
        fp16=getattr(args, 'fp16', True),
    )
    
    eval_config = EvaluationConfig(
        eval_every_n_steps=getattr(args, 'eval_every_n_steps', 100),
        eval_num_samples=getattr(args, 'eval_num_samples', 1000),
        eval_inference_steps=getattr(args, 'eval_inference_steps', 50),
    )
    
    return model_config, flow_config, training_config, eval_config


def validate_config_compatibility(
    model_config: BLIP3oDiTConfig, 
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
    
    return True


def print_config_summary(
    model_config: BLIP3oDiTConfig,
    flow_config: FlowMatchingConfig,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig
):
    """Print comprehensive configuration summary"""
    print("📋 BLIP3-o Configuration Summary")
    print("=" * 50)
    
    print(f"🏗️ Model Configuration:")
    print(f"   Architecture: {model_config.hidden_size}D, {model_config.num_hidden_layers}L, {model_config.num_attention_heads}H")
    print(f"   Tokens: {model_config.num_tokens} ({model_config.training_mode})")
    print(f"   EVA input: {model_config.eva_embedding_size}D")
    print(f"   CLIP output: {model_config.clip_embedding_size}D")
    print(f"   Parameters: ~{estimate_parameters(model_config)/1e6:.1f}M")
    
    print(f"\n🌊 Flow Matching Configuration:")
    print(f"   Prediction type: {flow_config.prediction_type}")
    print(f"   Flow type: {flow_config.flow_type}")
    print(f"   Normalize targets: {flow_config.normalize_targets}")
    
    print(f"\n🏃 Training Configuration:")
    print(f"   Epochs: {training_config.num_epochs}")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"   Mixed precision: {training_config.fp16}")
    
    print(f"\n📊 Evaluation Configuration:")
    print(f"   Eval every: {eval_config.eval_every_n_steps} steps")
    print(f"   Eval samples: {eval_config.eval_num_samples}")
    print(f"   Inference steps: {eval_config.eval_inference_steps}")
    print(f"   Normalize embeddings: {eval_config.normalize_embeddings}")
    
    print("=" * 50)


def estimate_parameters(config: BLIP3oDiTConfig) -> int:
    """Estimate number of parameters"""
    # Rough parameter estimation
    embed_params = config.clip_embedding_size * config.hidden_size + config.eva_embedding_size * config.hidden_size
    
    # Transformer layers
    layer_params = config.num_hidden_layers * (
        # Self-attention
        3 * config.hidden_size * config.hidden_size +  # Q, K, V projections
        config.hidden_size * config.hidden_size +       # Output projection
        # Cross-attention  
        config.hidden_size * config.hidden_size +       # Q projection
        2 * config.hidden_size * config.hidden_size +   # K, V projections
        config.hidden_size * config.hidden_size +       # Output projection
        # FFN
        config.hidden_size * config.intermediate_size +   # Up projection
        config.intermediate_size * config.hidden_size +   # Down projection
        # LayerNorms and other
        config.hidden_size * 6                          # Various norms and projections
    )
    
    output_params = config.hidden_size * config.clip_embedding_size
    total_params = embed_params + layer_params + output_params
    
    return total_params


# Export commonly used configurations
DEFAULT_MODEL_CONFIG = get_blip3o_config("base", "patch_only")
DEFAULT_FLOW_CONFIG = FlowMatchingConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EVAL_CONFIG = EvaluationConfig()