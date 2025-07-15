"""
Memory-Optimized Configuration for BLIP3-o Multi-GPU Training
Provides various model sizes and memory optimization strategies
"""

from typing import Dict, Any, Tuple
from ..config.blip3o_config import BLIP3oDiTConfig
from transformers import TrainingArguments


def get_memory_optimized_model_configs() -> Dict[str, BLIP3oDiTConfig]:
    """
    Get memory-optimized model configurations for different GPU setups.
    
    Returns:
        Dictionary of model configurations optimized for different memory constraints
    """
    configs = {}
    
    # Tiny model for testing (fits on single GPU easily)
    configs['tiny'] = BLIP3oDiTConfig(
        input_size=16,  # 16x16 = 256 tokens
        patch_size=1,
        in_channels=1024,  # CLIP dimension
        dim=256,  # Very small hidden dimension
        eva_embedding_size=4096,  # EVA-CLIP dimension
        n_layers=4,  # Few layers
        n_heads=4,  # Few heads (256/4 = 64, divisible by 4)
        norm_eps=1e-5,
        qk_norm=True,
        learn_sigma=False,
        _gradient_checkpointing=True,
    )
    
    # Small model for multi-GPU with limited memory
    configs['small'] = BLIP3oDiTConfig(
        input_size=16,  # 16x16 = 256 tokens
        patch_size=1,
        in_channels=1024,  # CLIP dimension
        dim=512,  # Small hidden dimension
        eva_embedding_size=4096,  # EVA-CLIP dimension
        n_layers=8,  # Moderate layers
        n_heads=8,  # 512/8 = 64, divisible by 4
        norm_eps=1e-5,
        qk_norm=True,
        learn_sigma=False,
        _gradient_checkpointing=True,
    )
    
    # Medium model for 3-4 GPUs
    configs['medium'] = BLIP3oDiTConfig(
        input_size=16,  # 16x16 = 256 tokens
        patch_size=1,
        in_channels=1024,  # CLIP dimension
        dim=768,  # Standard hidden dimension
        eva_embedding_size=4096,  # EVA-CLIP dimension
        n_layers=12,  # Moderate layers
        n_heads=12,  # 768/12 = 64, divisible by 4
        norm_eps=1e-5,
        qk_norm=True,
        learn_sigma=False,
        _gradient_checkpointing=True,
    )
    
    # Large model for 4+ GPUs
    configs['large'] = BLIP3oDiTConfig(
        input_size=16,  # 16x16 = 256 tokens
        patch_size=1,
        in_channels=1024,  # CLIP dimension
        dim=768,  # Standard hidden dimension
        eva_embedding_size=4096,  # EVA-CLIP dimension
        n_layers=16,  # More layers
        n_heads=12,  # 768/12 = 64, divisible by 4
        norm_eps=1e-5,
        qk_norm=True,
        learn_sigma=False,
        _gradient_checkpointing=True,
    )
    
    return configs


def get_memory_optimized_training_args(
    output_dir: str,
    model_size: str = "medium",
    num_gpus: int = 3,
    total_steps: int = 1000,
) -> TrainingArguments:
    """
    Get memory-optimized training arguments based on model size and GPU count.
    
    Args:
        output_dir: Output directory for checkpoints
        model_size: Model size ('tiny', 'small', 'medium', 'large')
        num_gpus: Number of GPUs available
        total_steps: Total training steps
        
    Returns:
        TrainingArguments optimized for memory usage
    """
    
    # Memory-optimized batch sizes based on model size and GPU count
    batch_size_configs = {
        'tiny': {'batch_size': 16, 'eval_batch_size': 32, 'grad_accum': 2},
        'small': {'batch_size': 12, 'eval_batch_size': 24, 'grad_accum': 4},
        'medium': {'batch_size': 8, 'eval_batch_size': 16, 'grad_accum': 4},
        'large': {'batch_size': 6, 'eval_batch_size': 12, 'grad_accum': 6},
    }
    
    # Adjust batch sizes based on GPU count
    config = batch_size_configs.get(model_size, batch_size_configs['medium'])
    
    if num_gpus >= 4:
        # More GPUs = can use slightly larger batch sizes
        config['batch_size'] = min(config['batch_size'] + 2, 16)
    elif num_gpus <= 2:
        # Fewer GPUs = need smaller batch sizes
        config['batch_size'] = max(config['batch_size'] - 2, 4)
        config['grad_accum'] = max(config['grad_accum'] + 2, 4)
    
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=total_steps,
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['grad_accum'],
        
        # Learning rate and optimization
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=min(100, total_steps // 10),
        
        # Memory optimizations
        fp16=True,  # Mixed precision
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        
        # Multi-GPU DDP settings (FIXED)
        ddp_find_unused_parameters=False,  # Better performance
        save_on_each_node=False,  # Only save on main process
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=max(50, total_steps // 20),
        save_steps=max(100, total_steps // 10),
        save_strategy="steps",
        logging_steps=20,
        
        # Memory and performance
        remove_unused_columns=False,
        load_best_model_at_end=False,  # Saves memory
        save_total_limit=3,  # Only keep 3 checkpoints
        prediction_loss_only=False,
        
        # Disable features that can cause issues
        push_to_hub=False,
        report_to=[],
    )


def estimate_memory_usage(config: BLIP3oDiTConfig, batch_size: int) -> Dict[str, float]:
    """
    Estimate memory usage for a given configuration.
    
    Args:
        config: Model configuration
        batch_size: Batch size per GPU
        
    Returns:
        Dictionary with memory estimates in GB
    """
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
    
    # Memory estimates (in GB)
    model_memory = total_params * 4 / (1024**3)  # FP32 parameters
    
    # Activation memory (rough estimate)
    sequence_length = config.input_size * config.input_size  # 256 tokens
    activation_memory = (
        batch_size * sequence_length * config.dim * config.n_layers * 8  # Activations
    ) / (1024**3)
    
    # Gradient memory (same as model for training)
    gradient_memory = model_memory
    
    # Optimizer states (AdamW: 2x gradients)
    optimizer_memory = model_memory * 2
    
    total_training_memory = model_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'model_memory_gb': model_memory,
        'activation_memory_gb': activation_memory,
        'gradient_memory_gb': gradient_memory,
        'optimizer_memory_gb': optimizer_memory,
        'total_training_memory_gb': total_training_memory,
        'inference_memory_gb': model_memory + activation_memory * 0.5,  # Less activation memory
        'parameters_millions': total_params / 1e6,
    }


def recommend_configuration(
    available_gpu_memory_gb: float,
    num_gpus: int,
    target_batch_size: int = None
) -> Tuple[str, BLIP3oDiTConfig, Dict[str, Any]]:
    """
    Recommend the best configuration based on available resources.
    
    Args:
        available_gpu_memory_gb: Available memory per GPU in GB
        num_gpus: Number of available GPUs
        target_batch_size: Desired batch size (optional)
        
    Returns:
        Tuple of (recommended_size, config, memory_info)
    """
    configs = get_memory_optimized_model_configs()
    
    recommendations = []
    
    for size_name, config in configs.items():
        # Test different batch sizes
        test_batch_sizes = [4, 6, 8, 12, 16] if target_batch_size is None else [target_batch_size]
        
        for batch_size in test_batch_sizes:
            memory_info = estimate_memory_usage(config, batch_size)
            
            if memory_info['total_training_memory_gb'] <= available_gpu_memory_gb * 0.9:  # 90% usage
                recommendations.append({
                    'size': size_name,
                    'config': config,
                    'batch_size': batch_size,
                    'memory_usage': memory_info['total_training_memory_gb'],
                    'memory_efficiency': memory_info['total_training_memory_gb'] / available_gpu_memory_gb,
                    'total_effective_batch': batch_size * num_gpus,
                    'memory_info': memory_info,
                })
    
    if not recommendations:
        # If nothing fits, recommend the smallest config
        return 'tiny', configs['tiny'], estimate_memory_usage(configs['tiny'], 4)
    
    # Sort by memory efficiency (higher is better, but not too close to limit)
    recommendations.sort(key=lambda x: (x['memory_efficiency'], x['total_effective_batch']), reverse=True)
    
    best = recommendations[0]
    return best['size'], best['config'], best['memory_info']


def print_memory_recommendations(
    available_gpu_memory_gb: float = 40.0,  # H100 default
    num_gpus: int = 3
):
    """Print memory recommendations for different configurations."""
    
    print(f"🧠 Memory Recommendations for {num_gpus}x GPUs ({available_gpu_memory_gb} GB each)")
    print("=" * 80)
    
    configs = get_memory_optimized_model_configs()
    
    for size_name, config in configs.items():
        print(f"\n📊 {size_name.upper()} Model Configuration:")
        print(f"   Dim: {config.dim}, Layers: {config.n_layers}, Heads: {config.n_heads}")
        
        # Test batch sizes
        for batch_size in [4, 8, 12, 16]:
            memory_info = estimate_memory_usage(config, batch_size)
            
            fits = memory_info['total_training_memory_gb'] <= available_gpu_memory_gb
            status = "✅" if fits else "❌"
            
            print(f"   {status} Batch size {batch_size}: {memory_info['total_training_memory_gb']:.1f} GB "
                  f"({memory_info['parameters_millions']:.1f}M params)")
    
    # Get recommendation
    rec_size, rec_config, rec_memory = recommend_configuration(available_gpu_memory_gb, num_gpus)
    
    print(f"\n🎯 RECOMMENDED: {rec_size.upper()} model")
    print(f"   Memory usage: {rec_memory['total_training_memory_gb']:.1f} GB per GPU")
    print(f"   Parameters: {rec_memory['parameters_millions']:.1f}M")
    print(f"   Efficiency: {rec_memory['total_training_memory_gb']/available_gpu_memory_gb*100:.1f}% GPU memory usage")


if __name__ == "__main__":
    # Print recommendations for H100 setup
    print_memory_recommendations(available_gpu_memory_gb=40.0, num_gpus=3)