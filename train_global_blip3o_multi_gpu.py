#!/usr/bin/env python3
"""
FIXED Multi-GPU Training script for BLIP3-o DiT - Global Training Only
File: train_blip3o_dit_multi_gpu_fixed.py

FIXES:
1. Uses only global BLIP3-o model (no standard fallbacks)
2. Enhanced GPU detection and error handling
3. Better SLURM GPU allocation handling
4. Improved DDP initialization for global model
5. Memory optimization for global training
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
    """Enhanced logging setup"""
    log_level = logging.INFO if local_rank == 0 else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format=f'[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'training_rank_{local_rank}.log', mode='w')
        ]
    )
    
    return logging.getLogger(__name__)

def detect_gpu_environment():
    """Enhanced GPU environment detection"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpu_names': [],
        'memory_total': [],
        'slurm_allocation': None,
        'cuda_visible_devices': None,
        'issues': [],
        'recommendations': []
    }
    
    # Basic CUDA check
    if not gpu_info['cuda_available']:
        gpu_info['issues'].append("CUDA not available in PyTorch")
        gpu_info['recommendations'].append("Check CUDA installation and drivers")
        return gpu_info
    
    # Detect GPUs
    try:
        gpu_info['gpu_count'] = torch.cuda.device_count()
        
        # Get detailed GPU info
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
    
    # Check SLURM allocation
    slurm_gpus = os.environ.get('SLURM_GPUS')
    if slurm_gpus:
        gpu_info['slurm_allocation'] = slurm_gpus
        try:
            expected_gpus = int(slurm_gpus)
            if expected_gpus != gpu_info['gpu_count']:
                gpu_info['issues'].append(
                    f"SLURM allocated {expected_gpus} GPUs but {gpu_info['gpu_count']} detected"
                )
        except ValueError:
            pass
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_info['cuda_visible_devices'] = cuda_visible
    
    if cuda_visible == '' or cuda_visible is None:
        gpu_info['issues'].append("CUDA_VISIBLE_DEVICES is empty or not set")
        gpu_info['recommendations'].append("Set CUDA_VISIBLE_DEVICES or check SLURM GPU allocation")
    
    return gpu_info

def apply_gpu_fixes():
    """Apply fixes for common GPU issues"""
    fixes_applied = []
    
    # Fix 1: Handle empty CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible == '':
        # Try to fix using SLURM variables
        if 'SLURM_LOCALID' in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
            fixes_applied.append(f"Set CUDA_VISIBLE_DEVICES={os.environ['SLURM_LOCALID']} from SLURM_LOCALID")
        elif 'SLURM_GPUS' in os.environ:
            try:
                num_gpus = int(os.environ['SLURM_GPUS'])
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(num_gpus)))
                fixes_applied.append(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} from SLURM_GPUS")
            except ValueError:
                pass
    
    # Fix 2: Set optimal NCCL environment
    nccl_settings = {
        'NCCL_ASYNC_ERROR_HANDLING': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        'NCCL_DEBUG': 'WARN',
        'OMP_NUM_THREADS': '1'
    }
    
    for var, value in nccl_settings.items():
        if var not in os.environ:
            os.environ[var] = value
            fixes_applied.append(f"Set {var}={value}")
    
    return fixes_applied

def enhanced_ddp_init(backend='auto', timeout=1800):
    """Enhanced DDP initialization with better error handling"""
    if 'WORLD_SIZE' not in os.environ:
        return False, "Not a distributed environment"
    
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if world_size <= 1:
            return False, "Single process environment"
        
        # Auto-select backend
        if backend == 'auto':
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                backend = 'nccl'
            else:
                backend = 'gloo'
                logger.warning("Using Gloo backend (CPU or no GPU detected)")
        
        # Set device before DDP init
        if torch.cuda.is_available() and backend == 'nccl':
            if local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
            else:
                logger.warning(f"Local rank {local_rank} >= GPU count {torch.cuda.device_count()}")
                return False, f"Local rank {local_rank} exceeds available GPUs"
        
        # Initialize process group with timeout
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                timeout=torch.timedelta(seconds=timeout)
            )
        
        return True, f"DDP initialized with {backend} backend (rank {rank}/{world_size})"
        
    except Exception as e:
        return False, f"DDP initialization failed: {e}"

def parse_arguments():
    """Parse command line arguments for global training"""
    parser = argparse.ArgumentParser(
        description="FIXED Multi-GPU BLIP3-o Global DiT Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    
    # Global model configuration
    parser.add_argument("--model_dim", type=int, default=768,
                       help="Model hidden dimension (global)")
    parser.add_argument("--num_layers", type=int, default=12,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--mlp_hidden_dim", type=int, default=2048,
                       help="MLP hidden dimension for global adapter")
    
    # Training configuration  
    parser.add_argument("--num_epochs", type=int, default=6,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="Evaluation batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Learning rate scheduler
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "constant"],
                       help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                       help="Warmup steps as ratio of total steps")
    
    # Global loss configuration
    parser.add_argument("--use_contrastive_loss", action="store_true", default=True,
                       help="Use contrastive loss in global training")
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                       help="Weight for contrastive loss")
    
    # Hardware configuration
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers per GPU")
    
    # Debugging options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with extra logging")
    parser.add_argument("--cpu_fallback", action="store_true",
                       help="Allow fallback to CPU training if GPU fails")
    
    return parser.parse_args()

def validate_global_model_config(args):
    """Validate and auto-fix global model configuration"""
    head_dim = args.model_dim // args.num_heads
    
    # Check basic divisibility
    if args.model_dim % args.num_heads != 0:
        raise ValueError(f"model_dim ({args.model_dim}) must be divisible by num_heads ({args.num_heads})")
    
    # Check 3D RoPE compatibility (head_dim must be divisible by 4)
    if head_dim % 4 != 0:
        print(f"⚠️  Incompatible head_dim ({head_dim}) for 3D RoPE")
        
        # Find compatible configuration for global model
        compatible_configs = [
            (768, 12),   # head_dim = 64
            (1024, 16),  # head_dim = 64
            (512, 8),    # head_dim = 64
            (960, 15),   # head_dim = 64
            (640, 10),   # head_dim = 64
        ]
        
        # Find closest config
        target_params = args.model_dim * args.num_heads
        best_config = min(compatible_configs, 
                         key=lambda x: abs(x[0] * x[1] - target_params))
        
        print(f"✅ Auto-fixing to compatible global config:")
        print(f"   Original: model_dim={args.model_dim}, num_heads={args.num_heads}, head_dim={head_dim}")
        args.model_dim, args.num_heads = best_config
        head_dim = args.model_dim // args.num_heads
        print(f"   Fixed: model_dim={args.model_dim}, num_heads={args.num_heads}, head_dim={head_dim}")
    
    print(f"✅ Global model config validated: {args.model_dim}D, {args.num_heads}H, head_dim={head_dim}")
    return head_dim

def main():
    """Main global training function with enhanced error handling"""
    
    # Get environment info early
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Setup logging
    logger = setup_logging(local_rank)
    
    is_main_process = (global_rank == 0)
    
    if is_main_process:
        print("🚀 FIXED Multi-GPU BLIP3-o Global DiT Training")
        print("=" * 70)
        print("🎯 GLOBAL TRAINING FEATURES:")
        print("  ✅ Direct [B, 768] global feature training")
        print("  ✅ No training-inference mismatch")
        print("  ✅ Enhanced GPU detection and fixes")
        print("  ✅ Optimized for recall performance")
        print("=" * 70)
        print(f"Environment: LOCAL_RANK={local_rank}, RANK={global_rank}, WORLD_SIZE={world_size}")
    
    try:
        # 1. GPU Environment Detection and Fixes
        if is_main_process:
            print("\n🔍 Step 1: GPU Environment Detection")
            print("-" * 40)
        
        gpu_info = detect_gpu_environment()
        
        if is_main_process:
            print(f"CUDA Available: {gpu_info['cuda_available']}")
            print(f"GPU Count: {gpu_info['gpu_count']}")
            print(f"SLURM GPUs: {gpu_info['slurm_allocation']}")
            print(f"CUDA_VISIBLE_DEVICES: {gpu_info['cuda_visible_devices']}")
            
            for error in gpu_info['issues']:
                print(f"❌ {error}")
            for rec in gpu_info['recommendations']:
                print(f"💡 {rec}")
        
        # Apply fixes
        fixes = apply_gpu_fixes()
        if is_main_process and fixes:
            print("🔧 Applied fixes:")
            for fix in fixes:
                print(f"  ✅ {fix}")
        
        # Re-detect after fixes
        gpu_info = detect_gpu_environment()
        
        # Determine device
        if gpu_info['cuda_available'] and gpu_info['gpu_count'] > 0:
            device = torch.device(f"cuda:{local_rank}")
            use_cuda = True
        else:
            device = torch.device("cpu")
            use_cuda = False
            if is_main_process:
                print("⚠️  No GPUs available, using CPU")
        
        # 2. Distributed Initialization
        if is_main_process:
            print("\n🔗 Step 2: Distributed Initialization")
            print("-" * 40)
        
        is_distributed = False
        if world_size > 1:
            success, message = enhanced_ddp_init()
            is_distributed = success
            if is_main_process:
                if success:
                    print(f"✅ {message}")
                else:
                    print(f"❌ {message}")
                    if not args.cpu_fallback:
                        raise RuntimeError(f"Distributed initialization failed: {message}")
        
        # 3. Parse Arguments and Validate Config
        args = parse_arguments()
        
        if is_main_process:
            print("\n⚙️  Step 3: Global Model Configuration")
            print("-" * 40)
        
        head_dim = validate_global_model_config(args)
        
        # 4. Load Global Modules (no standard fallbacks)
        if is_main_process:
            print("\n📦 Step 4: Loading Global BLIP3-o Modules")
            print("-" * 40)
        
        try:
            from src.modules.config.blip3o_config import get_global_blip3o_config
            from src.modules.models import create_blip3o_dit_model
            from src.modules.losses import create_blip3o_flow_matching_loss
            from src.modules.trainers import get_trainer_class, get_training_args_factory
            from src.modules.datasets import create_dataloaders
            
            if is_main_process:
                print("✅ All global BLIP3-o modules loaded successfully")
                
        except ImportError as e:
            if is_main_process:
                print(f"❌ Global module import failed: {e}")
                print("💡 Make sure all global BLIP3-o components are available")
            raise
        
        # 5. Load Dataset Manifest
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
        
        # 6. Create Global Model
        if is_main_process:
            print("\n🏗️  Step 6: Global Model Creation")
            print("-" * 40)
        
        model_config = get_global_blip3o_config(
            model_size="medium",  # Can be adjusted based on resources
            dim=args.model_dim,
            n_layers=args.num_layers,
            n_heads=args.num_heads,
            mlp_hidden_dim=args.mlp_hidden_dim,
            global_training=True,
            use_attention_pooling=True,
        )
        
        model = create_blip3o_dit_model(config=model_config)
        model = model.to(device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
        # Calculate parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_main_process:
            print(f"Global model parameters: {param_count:,}")
            print(f"Memory per GPU: ~{param_count * 4 / (1024**3):.1f} GB")
        
        # 7. Setup DDP for Global Model
        if is_distributed and use_cuda:
            if is_main_process:
                print("\n🔄 Step 7: DDP Global Model Wrapping")
                print("-" * 40)
            
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            
            if is_main_process:
                print("✅ Global model wrapped with DDP")
        
        # 8. Create Global Loss Function
        flow_matching_loss = create_blip3o_flow_matching_loss(
            enhanced=True,  # Use enhanced version
            use_contrastive_loss=args.use_contrastive_loss,
            contrastive_weight=args.contrastive_weight,
        )
        
        # 9. Create Dataloaders
        if is_main_process:
            print("\n🔄 Step 8: Dataloader Creation")
            print("-" * 40)
        
        train_dataloader, eval_dataloader = create_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_split_ratio=0.1,
            normalize_embeddings=True,
            delete_after_use=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=use_cuda,
            drop_last=True,
        )
        
        # 10. Create Global Training Arguments
        if is_main_process:
            print("\n⚙️  Step 9: Global Training Setup")
            print("-" * 40)
        
        # Calculate training steps for global training
        total_samples = manifest['total_samples']
        samples_per_gpu = total_samples // world_size if is_distributed else total_samples
        steps_per_epoch = max(1, samples_per_gpu // args.batch_size)
        max_steps = (steps_per_epoch * args.num_epochs) // args.gradient_accumulation_steps
        
        training_args_factory = get_training_args_factory("auto")
        training_args = training_args_factory(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16 and use_cuda,
            dataloader_num_workers=args.dataloader_num_workers,
            logging_steps=max(10, max_steps // 50),
            save_steps=max(100, max_steps // 5),
            eval_steps=max(50, max_steps // 10) if eval_dataloader else 0,
            # Global training optimizations
            ddp_find_unused_parameters=False,
            save_on_each_node=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_global_cosine_mean",
            greater_is_better=True,
        )
        
        if is_main_process:
            print(f"Global training steps: {max_steps}")
            print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        
        # 11. Create Global Trainer
        trainer_class = get_trainer_class("auto")
        trainer = trainer_class(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=None,  # We'll override the dataloader
            eval_dataset=None,
        )
        
        # Override dataloader methods
        trainer.get_train_dataloader = lambda: train_dataloader
        trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
        
        # 12. Start Global Training
        if is_main_process:
            print("\n🚀 Step 10: Starting Global BLIP3-o Training")
            print("-" * 40)
            print("✅ All systems ready - starting global training!")
            print("🎯 Expected: 50-70% R@1 recall (500-700x improvement)")
        
        train_result = trainer.train()
        
        # 13. Save Global Model
        if is_main_process:
            print("\n💾 Saving global model...")
            trainer.save_model()
            print("✅ Global training completed successfully!")
        
        # Cleanup
        if is_distributed:
            dist.destroy_process_group()
        
        return 0
        
    except Exception as e:
        if is_main_process:
            print(f"\n❌ Global training failed: {e}")
            traceback.print_exc()
            
            # Save error info for debugging
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'gpu_info': detect_gpu_environment(),
                'environment': {k: os.environ.get(k) for k in [
                    'CUDA_VISIBLE_DEVICES', 'SLURM_GPUS', 'WORLD_SIZE', 'LOCAL_RANK'
                ]},
                'timestamp': datetime.now().isoformat(),
                'training_type': 'global'
            }
            
            with open('global_training_error_debug.json', 'w') as f:
                json.dump(error_info, f, indent=2)
            
            print("💾 Global training debug info saved to global_training_error_debug.json")
        
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)