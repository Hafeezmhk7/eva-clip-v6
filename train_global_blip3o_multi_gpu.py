#!/usr/bin/env python3
"""
FIXED Multi-GPU Global BLIP3-o Training Script
File: train_global_blip3o_multi_gpu.py

KEY FIXES:
1. Fixed import issues and module paths
2. Proper dimension compatibility for 3D RoPE
3. Better multi-GPU memory management
4. Enhanced error handling and logging
5. ✅ CPU fallback with Gloo backend if no GPU available
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

# FIXED: Add src to path before any imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def setup_logging():
    """Setup logging for all ranks"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if local_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    return logging.getLogger(__name__)

def setup_ddp_environment():
    """Setup DDP environment variables based on hardware"""
    if torch.cuda.is_available():
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FIXED Multi-GPU Global BLIP3-o Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--mlp_hidden_dim", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--use_contrastive_loss", action="store_true", default=True)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    return parser.parse_args()

def validate_model_config(args):
    head_dim = args.model_dim // args.num_heads
    if args.model_dim % args.num_heads != 0:
        raise ValueError(f"model_dim ({args.model_dim}) must be divisible by num_heads ({args.num_heads})")

    if head_dim % 4 != 0:
        compatible_configs = [
            (768, 12), (1024, 16), (512, 8), (960, 15), (640, 10),
        ]
        for dim, heads in compatible_configs:
            if dim >= args.model_dim * 0.8 and dim <= args.model_dim * 1.2:
                print(f"⚠️ Auto-fixing incompatible config:")
                print(f"   Original: model_dim={args.model_dim}, num_heads={args.num_heads}, head_dim={head_dim}")
                args.model_dim = dim
                args.num_heads = heads
                head_dim = dim // heads
                print(f"   Fixed: model_dim={args.model_dim}, num_heads={args.num_heads}, head_dim={head_dim}")
                break
        else:
            raise ValueError(f"Cannot auto-fix incompatible config. head_dim ({head_dim}) must be divisible by 4 for 3D RoPE")

    print(f"✅ Model config validated: model_dim={args.model_dim}, num_heads={args.num_heads}, head_dim={head_dim}")
    return head_dim

def main():
    """FIXED main training function with CPU fallback"""
    logger = setup_logging()
    setup_ddp_environment()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        logger.warning("⚠️ CUDA not available! Using CPU for training.")

    if world_size > 1:
        backend = "nccl" if cuda_available else "gloo"
        dist.init_process_group(backend=backend)
        if cuda_available:
            torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if cuda_available else "cpu")
    is_main_process = global_rank == 0

    if is_main_process:
        print("🚀 FIXED Multi-GPU Global BLIP3-o Training")
        print("=" * 70)
        print(f"Hardware: {'GPU' if cuda_available else 'CPU'} training")
        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        print(f"Global rank: {global_rank}")
        print(f"Device: {device}")
        print("=" * 70)

    args = parse_arguments()
    head_dim = validate_model_config(args)

    try:
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.global_blip3o_dit import create_global_blip3o_dit_model
        from src.modules.losses.global_flow_matching_loss import create_global_flow_matching_loss
        from src.modules.trainers.global_blip3o_trainer import GlobalBLIP3oTrainer, create_global_training_args
        from src.modules.datasets.blip3o_dataset import create_chunked_dataloaders

        manifest_path = Path(args.chunked_embeddings_dir) / "embeddings_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        model_config = BLIP3oDiTConfig(
            input_size=16,
            patch_size=1,
            in_channels=1024,
            dim=args.model_dim,
            eva_embedding_size=4096,
            n_layers=args.num_layers,
            n_heads=args.num_heads,
            norm_eps=1e-5,
            learn_sigma=False,
            _gradient_checkpointing=True,
            mlp_hidden_dim=args.mlp_hidden_dim,
        )

        model = create_global_blip3o_dit_model(
            config=model_config,
            load_clip_projection=True,
        )
        model = model.to(device)

        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if cuda_available else None,
                output_device=local_rank if cuda_available else None,
                find_unused_parameters=False,
            )

        flow_matching_loss = create_global_flow_matching_loss(
            use_contrastive_loss=args.use_contrastive_loss,
            contrastive_weight=args.contrastive_weight,
        )

        train_dataloader, eval_dataloader = create_chunked_dataloaders(
            chunked_embeddings_dir=args.chunked_embeddings_dir,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_split_ratio=0.1,
            normalize_embeddings=True,
            delete_after_use=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=cuda_available,
            drop_last=True,
        )

        class CompatibleDataset:
            def __init__(self, estimated_samples):
                self.estimated_samples = max(1, estimated_samples // world_size if world_size > 1 else estimated_samples)
            def __len__(self):
                return self.estimated_samples
            def __getitem__(self, idx):
                raise NotImplementedError("Use custom dataloader")

        total_samples = manifest['total_samples']
        train_dataset = CompatibleDataset(int(total_samples * 0.9))
        eval_dataset = CompatibleDataset(int(total_samples * 0.1)) if eval_dataloader else None

        effective_batch_size = args.batch_size * world_size * args.gradient_accumulation_steps
        steps_per_epoch = max(1, total_samples // effective_batch_size)
        max_steps = steps_per_epoch * args.num_epochs

        training_args = create_global_training_args(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            dataloader_num_workers=args.dataloader_num_workers,
            logging_steps=max(10, steps_per_epoch // 20),
            save_steps=max(50, steps_per_epoch // 4),
            eval_steps=max(25, steps_per_epoch // 8) if eval_dataloader else 0,
            ddp_find_unused_parameters=False,
            save_on_each_node=False,
        )

        trainer = GlobalBLIP3oTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.get_train_dataloader = lambda: train_dataloader
        trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader if eval_dataloader else None

        train_result = trainer.train()

        if is_main_process:
            trainer.save_model()
            if hasattr(trainer.flow_matching_loss, 'ema_cosine'):
                final_cosine = trainer.flow_matching_loss.ema_cosine.item()
                predicted_recall = min(final_cosine * 70, 70)
                success_file = Path(args.output_dir) / "training_success.json"
                with open(success_file, 'w') as f:
                    json.dump({
                        'training_successful': final_cosine > 0.7,
                        'final_cosine': final_cosine,
                        'predicted_recall': predicted_recall,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                print(f"\n📊 FINAL COSINE: {final_cosine:.4f} — Predicted Recall: {predicted_recall:.1f}%")

        if world_size > 1:
            dist.destroy_process_group()

        return 0

    except Exception as e:
        if is_main_process:
            print(f"❌ Training failed on rank {global_rank}: {e}")
            traceback.print_exc()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
