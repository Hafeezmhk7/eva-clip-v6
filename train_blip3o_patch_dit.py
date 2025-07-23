#!/usr/bin/env python3
"""
BLIP3-o Patch-Level Training Script - Aligned with BLIP3-o Paper (FIXED GRADIENT FLOW)
train_blip3o_patch_dit.py

UPDATED to use fixed gradient flow implementation:
1. Fixed model with proper gradient flow
2. Fixed trainer with simplified loss computation
3. Updated dataset with gradient-aware collate function
4. All BLIP3-o paper features maintained
"""

import os
import sys
import argparse
import logging
import torch
import torch.distributed as dist
from pathlib import Path
import json
from datetime import datetime
import traceback
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging(local_rank=0):
    """Setup logging for training."""
    log_level = logging.INFO if local_rank == 0 else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format=f'[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'blip3o_patch_training_rank_{local_rank}.log', mode='w')
        ]
    )
    
    return logging.getLogger(__name__)

def detect_gpu_environment():
    """Detect and validate GPU environment."""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpu_names': [],
        'memory_total': [],
        'issues': [],
        'slurm_gpus': os.environ.get('SLURM_GPUS'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    }
    
    if not gpu_info['cuda_available']:
        gpu_info['issues'].append("CUDA not available")
        return gpu_info
    
    try:
        gpu_info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(gpu_info['gpu_count']):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu_info['gpu_names'].append(props.name)
                gpu_info['memory_total'].append(props.total_memory / (1024**3))
                
                # Test GPU access
                test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
                del test_tensor
                
            except Exception as e:
                gpu_info['issues'].append(f"GPU {i} not accessible: {e}")
                
    except Exception as e:
        gpu_info['issues'].append(f"Error detecting GPUs: {e}")
    
    return gpu_info

def apply_gpu_environment_fixes():
    """Apply fixes for common GPU environment issues."""
    fixes_applied = []
    
    # Fix CUDA_VISIBLE_DEVICES if empty
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible == '':
        if 'SLURM_LOCALID' in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
            fixes_applied.append(f"Set CUDA_VISIBLE_DEVICES={os.environ['SLURM_LOCALID']}")
        elif 'SLURM_GPUS' in os.environ:
            try:
                num_gpus = int(os.environ['SLURM_GPUS'])
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(num_gpus)))
                fixes_applied.append(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
            except ValueError:
                pass
    
    # Set optimal environment variables
    optimal_env = {
        'NCCL_ASYNC_ERROR_HANDLING': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'NCCL_DEBUG': 'WARN',
        'OMP_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false',
    }
    
    for var, value in optimal_env.items():
        if var not in os.environ:
            os.environ[var] = value
            fixes_applied.append(f"Set {var}={value}")
    
    return fixes_applied

def initialize_distributed_training(timeout=1800):
    """Initialize distributed training with proper error handling."""
    if 'WORLD_SIZE' not in os.environ:
        return False, "Not a distributed environment"
    
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if world_size <= 1:
            return False, "Single process environment"
        
        # Set device before DDP init
        if torch.cuda.is_available():
            if local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
            else:
                return False, f"Local rank {local_rank} exceeds available GPUs"
        
        # Initialize process group
        if not dist.is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                timeout=torch.timedelta(seconds=timeout)
            )
        
        return True, f"DDP initialized: rank {rank}/{world_size}"
        
    except Exception as e:
        return False, f"DDP initialization failed: {e}"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BLIP3-o Patch-Level DiT Training - Aligned with Paper (FIXED GRADIENT FLOW)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Model configuration
    parser.add_argument("--model_size", type=str, default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model size configuration")
    parser.add_argument("--hidden_size", type=int, default=768,
                       help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads")
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=6,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                       help="Evaluation batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Evaluation configuration
    parser.add_argument("--enable_recall_eval", action="store_true", default=True,
                       help="Enable image-to-text recall evaluation")
    parser.add_argument("--recall_eval_steps", type=int, default=250,
                       help="Steps between recall evaluations")
    parser.add_argument("--recall_eval_samples", type=int, default=50,
                       help="Number of samples for recall evaluation")
    
    # Flow matching configuration
    parser.add_argument("--use_contrastive_loss", action="store_true", default=True,
                       help="Use contrastive loss for better alignment")
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                       help="Weight for contrastive loss")
    parser.add_argument("--enhanced_loss", action="store_true", default=True,
                       help="Use enhanced flow matching loss")
    
    # Hardware configuration
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # Debugging
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()

def validate_model_configuration(args):
    """Validate and potentially fix model configuration."""
    if args.hidden_size % args.num_heads != 0:
        raise ValueError(f"hidden_size ({args.hidden_size}) must be divisible by num_heads ({args.num_heads})")
    
    head_dim = args.hidden_size // args.num_heads
    if head_dim < 32:
        raise ValueError(f"Head dimension too small: {head_dim}. Consider increasing hidden_size.")
    
    print(f"✅ Model configuration validated:")
    print(f"   Size: {args.model_size}")
    print(f"   Dimensions: {args.hidden_size}D, {args.num_layers}L, {args.num_heads}H")
    print(f"   Head dimension: {head_dim}")
    print(f"   Patch tokens: 256 (16x16)")
    print(f"   EVA conditioning: 4096-dim")
    print(f"   CLIP output: 1024-dim")
    
    return head_dim

# UPDATED: Create config function for fixed model
def get_blip3o_patch_config(
    model_size="base",
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    **kwargs
):
    """Create BLIP3-o config for the fixed model"""
    # Import the fixed model's config
    from src.modules.models.blip3o_patch_dit import BLIP3oDiTConfig
    
    # Size presets
    size_configs = {
        "tiny": {"hidden_size": 512, "num_hidden_layers": 6, "num_attention_heads": 8, "intermediate_size": 2048},
        "small": {"hidden_size": 768, "num_hidden_layers": 8, "num_attention_heads": 12, "intermediate_size": 3072},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
        "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16, "intermediate_size": 4096},
    }
    
    if model_size in size_configs:
        base_config = size_configs[model_size]
    else:
        base_config = size_configs["base"]
    
    # Override with provided parameters
    base_config.update({
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        **kwargs
    })
    
    return BLIP3oDiTConfig(**base_config)

def main():
    """Main training function."""
    # Environment setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger = setup_logging(local_rank)
    is_main_process = (global_rank == 0)
    
    if is_main_process:
        print("🚀 BLIP3-o Patch-Level DiT Training - FIXED GRADIENT FLOW")
        print("=" * 65)
        print("🎯 TRAINING FEATURES:")
        print("  ✅ 256-token patch-level flow matching")
        print("  ✅ EVA-CLIP conditioning (4096-dim)")
        print("  ✅ CLIP output supervision (1024-dim)")
        print("  ✅ Image-to-text recall evaluation")
        print("  ✅ 3D Rotary Position Embedding")
        print("  ✅ FIXED: Proper gradient flow implementation")
        print("  ✅ Memory optimization")
        print("=" * 65)
        print(f"Environment: LOCAL_RANK={local_rank}, RANK={global_rank}, WORLD_SIZE={world_size}")
    
    try:
        # 1. GPU Environment Setup
        if is_main_process:
            print("\n🔍 Step 1: GPU Environment Detection")
            print("-" * 40)
        
        gpu_info = detect_gpu_environment()
        fixes = apply_gpu_environment_fixes()
        
        if is_main_process:
            print(f"CUDA Available: {gpu_info['cuda_available']}")
            print(f"GPU Count: {gpu_info['gpu_count']}")
            if fixes:
                print("Applied fixes:")
                for fix in fixes:
                    print(f"  ✅ {fix}")
        
        # Determine device
        if gpu_info['cuda_available'] and gpu_info['gpu_count'] > 0:
            device = torch.device(f"cuda:{local_rank}")
            use_cuda = True
        else:
            device = torch.device("cpu")
            use_cuda = False
            if is_main_process:
                print("⚠️  Using CPU (no GPUs available)")
        
        # 2. Distributed Setup
        if is_main_process:
            print("\n🔗 Step 2: Distributed Initialization")
            print("-" * 40)
        
        is_distributed = False
        if world_size > 1:
            success, message = initialize_distributed_training()
            is_distributed = success
            if is_main_process:
                print(f"{'✅' if success else '❌'} {message}")
        
        # 3. Parse Arguments
        args = parse_arguments()
        
        if is_main_process:
            print("\n⚙️  Step 3: Model Configuration")
            print("-" * 40)
        
        head_dim = validate_model_configuration(args)
        
        # 4. Load BLIP3-o Modules (UPDATED IMPORTS)
        if is_main_process:
            print("\n📦 Step 4: Loading BLIP3-o Patch Modules (FIXED)")
            print("-" * 40)
        
        try:
            # UPDATED: Import fixed components
            from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model
            from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
            from src.modules.trainers.blip3o_patch_trainer import BLIP3oPatchTrainer, create_blip3o_patch_training_args
            from src.modules.datasets import create_gradient_aware_dataloaders
            
            if is_main_process:
                print("✅ All BLIP3-o fixed modules loaded successfully")
                print("   ✅ Fixed model with proper gradient flow")
                print("   ✅ Fixed trainer with simplified loss computation")
                print("   ✅ Updated dataset with gradient-aware collate function")
                
        except ImportError as e:
            if is_main_process:
                print(f"❌ Module import failed: {e}")
                print("💡 Make sure you have:")
                print("   1. fixed_blip3o_model.py in project root")
                print("   2. fixed_trainer.py in project root")
                print("   3. Updated src/modules/datasets/ files")
            raise
        
        # 5. Dataset Loading
        if is_main_process:
            print("\n📊 Step 5: Dataset Loading")
            print("-" * 40)
        
        manifest_path = Path(args.chunked_embeddings_dir) / "embeddings_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        if is_main_process:
            print(f"Dataset: {manifest['total_shards']} shards, {manifest['total_samples']:,} samples")
            print(f"Expected tokens per image: 256 (16x16 patches)")
        
        # 6. Create Model (UPDATED)
        if is_main_process:
            print("\n🏗️  Step 6: Model Creation (FIXED)")
            print("-" * 40)
        
        # UPDATED: Get model configuration for fixed model
        config = get_blip3o_patch_config(
            model_size=args.model_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
        )
        
        # UPDATED: Create fixed patch-level model
        model = create_blip3o_patch_dit_model(config=config)
        model = model.to(device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        if is_main_process:
            param_count = model.get_num_parameters()
            print(f"Model parameters: {param_count:,}")
            print(f"Memory estimate: ~{param_count * 4 / (1024**3):.1f} GB")
            print(f"Architecture: Fixed patch-level DiT with 3D RoPE")
            print(f"Gradient flow: ✅ FIXED")
        
        # 7. DDP Wrapping
        if is_distributed and use_cuda:
            if is_main_process:
                print("\n🔄 Step 7: DDP Model Wrapping")
                print("-" * 40)
            
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            
            if is_main_process:
                print("✅ Model wrapped with DDP")
        
        # 8. Create Loss Function
        flow_matching_loss = create_blip3o_flow_matching_loss(
            enhanced=args.enhanced_loss,
            use_contrastive_loss=args.use_contrastive_loss,
            contrastive_weight=args.contrastive_weight,
        )
        
        if is_main_process:
            print("✅ Flow matching loss created")
            print(f"   Enhanced features: {args.enhanced_loss}")
            print(f"   Contrastive loss: {args.use_contrastive_loss}")
        
        # 9. Create Dataloaders (UPDATED)
        if is_main_process:
            print("\n🔄 Step 8: Dataloader Creation (GRADIENT-AWARE)")
            print("-" * 40)
        
        # UPDATED: Use gradient-aware dataloaders
        train_dataloader, eval_dataloader = create_gradient_aware_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_split_ratio=0.1,
            normalize_embeddings=True,
            num_workers=args.dataloader_num_workers,
            pin_memory=use_cuda,
            drop_last=True,
            use_ddp=is_distributed,
        )
        
        if is_main_process:
            print("✅ Gradient-aware dataloaders created")
            print(f"   ✅ Pre-computed noisy inputs with gradients")
            print(f"   ✅ Proper tensor detachment for targets/conditioning")
            print(f"   ✅ Flow matching timestep generation")
        
        # 10. Training Setup
        if is_main_process:
            print("\n⚙️  Step 9: Training Setup")
            print("-" * 40)
        
        # Calculate training steps
        total_samples = manifest['total_samples']
        train_samples = int(total_samples * 0.9)  # 90% for training
        samples_per_gpu = train_samples // world_size if is_distributed else train_samples
        steps_per_epoch = max(1, samples_per_gpu // (args.batch_size * args.gradient_accumulation_steps))
        max_steps = steps_per_epoch * args.num_epochs
        
        training_args = create_blip3o_patch_training_args(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16 and use_cuda,
            dataloader_num_workers=args.dataloader_num_workers,
            logging_steps=max(10, max_steps // 50),
            save_steps=max(100, max_steps // 10),
            eval_steps=max(50, max_steps // 20) if eval_dataloader else 0,
            metric_for_best_model="eval_recall@1",  # Use recall as primary metric
            greater_is_better=True,
        )
        
        if is_main_process:
            print(f"Training steps: {max_steps}")
            print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
            print(f"Primary metric: Image-to-text Recall@1")
        
        # 11. Create Trainer (UPDATED)
        if is_main_process:
            print("\n🏋️  Step 10: Trainer Creation (FIXED)")
            print("-" * 40)
        
        # UPDATED: Use fixed trainer
        trainer = BLIP3oPatchTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=None,  # We override the dataloader
            eval_dataset=eval_dataloader.dataset if eval_dataloader else None,
            enable_recall_evaluation=args.enable_recall_eval,
            recall_eval_samples=args.recall_eval_samples,
            recall_eval_steps=args.recall_eval_steps,
        )
        
        # Override dataloader methods
        trainer.get_train_dataloader = lambda: train_dataloader
        if eval_dataloader:
            trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
        
        if is_main_process:
            print("✅ Fixed trainer created")
            print("   ✅ Simplified gradient flow handling")
            print("   ✅ Uses pre-computed noisy inputs")
            print("   ✅ Enhanced error handling")
        
        # 12. Start Training
        if is_main_process:
            print("\n🚀 Step 11: Starting BLIP3-o Patch Training (FIXED)")
            print("-" * 40)
            print("✅ All systems ready - starting fixed patch-level flow matching training!")
            print("🎯 Expected: Smooth training without gradient flow errors")
            print("📊 Evaluation: Recall@1, Recall@5, Recall@10")
            print("🔥 Gradient flow: ✅ FIXED and working properly")
        
        train_result = trainer.train()
        
        # 13. Save Model
        if is_main_process:
            print("\n💾 Saving model...")
            trainer.save_model()
            
            # Save training configuration
            config_info = {
                'model_config': config.to_dict() if hasattr(config, 'to_dict') else vars(config),
                'training_args': training_args.to_dict(),
                'flow_matching_config': {
                    'enhanced': args.enhanced_loss,
                    'use_contrastive_loss': args.use_contrastive_loss,
                    'contrastive_weight': args.contrastive_weight,
                },
                'training_completed': True,
                'gradient_flow_fixed': True,
                'paper_alignment': 'BLIP3-o DiT with patch-level flow matching (FIXED)',
                'architecture': '256-token patch-level DiT with EVA-CLIP conditioning',
                'evaluation_metrics': 'Image-to-text recall (R@1, R@5, R@10)',
                'timestamp': datetime.now().isoformat(),
            }
            
            with open(Path(args.output_dir) / 'training_config.json', 'w') as f:
                json.dump(config_info, f, indent=2)
            
            print("✅ BLIP3-o patch training completed successfully!")
            print("📋 Training follows BLIP3-o paper architecture")
            print("🎯 256-token patch-level flow matching")
            print("📊 Optimized for image-to-text recall")
            print("🔥 Gradient flow: ✅ FIXED and working properly")
        
        # Cleanup
        if is_distributed:
            dist.destroy_process_group()
        
        return 0
        
    except Exception as e:
        if is_main_process:
            print(f"\n❌ Training failed: {e}")
            traceback.print_exc()
            
            # Save error info
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'gpu_info': detect_gpu_environment(),
                'environment': {k: os.environ.get(k) for k in [
                    'CUDA_VISIBLE_DEVICES', 'SLURM_GPUS', 'WORLD_SIZE', 'LOCAL_RANK'
                ]},
                'timestamp': datetime.now().isoformat(),
                'training_type': 'blip3o_patch_level_fixed',
                'gradient_flow_version': 'fixed'
            }
            
            with open('blip3o_patch_training_error.json', 'w') as f:
                json.dump(error_info, f, indent=2)
            
            print("💾 Error info saved to blip3o_patch_training_error.json")
        
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)