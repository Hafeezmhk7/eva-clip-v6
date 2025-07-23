#!/usr/bin/env python3
"""
FIXED: Enhanced BLIP3-o Training Script with CLS+Patch Support and Flexible Training
train_blip3o_enhanced.py

FIXES:
1. Proper gradient checkpointing handling
2. Better error handling and recovery
3. Improved alignment with BLIP3-o paper
4. Fixed device and memory management

Features:
1. Support for both patch-only (256) and CLS+patch (257) modes
2. Flexible shard selection for training (overfitting tests)
3. Same-data evaluation (overfitting verification)
4. Pure flow matching loss (BLIP3-o paper aligned)
5. Detailed evaluation with cosine similarity analysis and plots
6. Comprehensive logging and progress tracking
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
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging(local_rank=0, log_dir=None):
    """Setup comprehensive logging"""
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'blip3o_training_rank_{local_rank}.log'
    else:
        log_file = f'blip3o_training_rank_{local_rank}.log'
    
    log_level = logging.INFO if local_rank == 0 else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format=f'[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments with all new features"""
    parser = argparse.ArgumentParser(
        description="FIXED: Enhanced BLIP3-o Training with CLS+Patch Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output paths
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints and results")
    
    # Training mode selection
    parser.add_argument("--training_mode", type=str, default="cls_patch",
                       choices=["cls_patch", "patch_only"],
                       help="Training mode: cls_patch (257 tokens) or patch_only (256 tokens)")
    
    # Flexible shard selection
    parser.add_argument("--max_training_shards", type=int, default=None,
                       help="Maximum number of shards to use for training (None for all)")
    parser.add_argument("--overfitting_test", action="store_true",
                       help="Run overfitting test with single shard")
    
    # Evaluation options
    parser.add_argument("--enable_same_data_eval", action="store_true", default=True,
                       help="Enable evaluation on same training data (overfitting check)")
    parser.add_argument("--same_data_eval_frequency", type=int, default=100,
                       help="Frequency of same-data evaluation (steps)")
    parser.add_argument("--enable_detailed_eval", action="store_true", default=True,
                       help="Enable detailed cosine similarity evaluation")
    parser.add_argument("--detailed_eval_frequency", type=int, default=500,
                       help="Frequency of detailed evaluation (steps)")
    parser.add_argument("--max_eval_batches", type=int, default=50,
                       help="Maximum batches for detailed evaluation")
    
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
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=2,
                       help="Evaluation batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=50,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       choices=["linear", "cosine", "constant"],
                       help="Learning rate scheduler")
    
    # Loss configuration
    parser.add_argument("--normalize_targets", action="store_true", default=True,
                       help="Normalize target embeddings")
    parser.add_argument("--prediction_type", type=str, default="velocity",
                       choices=["velocity", "epsilon"],
                       help="Flow matching prediction type")
    
    # Hardware configuration
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                       help="Enable gradient checkpointing (if supported)")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                       help="Number of dataloader workers")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=200,
                       help="Model saving frequency")
    parser.add_argument("--detailed_logging", action="store_true", default=True,
                       help="Enable detailed progress logging")
    
    # Debugging and testing
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--test_gradient_flow", action="store_true",
                       help="Test gradient flow before training")
    parser.add_argument("--cpu_fallback", action="store_true",
                       help="Use CPU if GPU fails")
    parser.add_argument("--safe_mode", action="store_true",
                       help="Enable safe mode with extra error checking")
    
    return parser.parse_args()

def setup_distributed_training():
    """Setup distributed training environment with error handling"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        try:
            if not dist.is_initialized():
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                dist.init_process_group(backend=backend, init_method='env://')
            
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            return True, local_rank, global_rank, world_size
        except Exception as e:
            print(f"⚠️ Distributed training setup failed: {e}")
            print("   Falling back to single GPU/CPU training")
            return False, 0, 0, 1
    
    return False, local_rank, global_rank, world_size

def validate_training_setup(args, logger):
    """Validate training setup and configuration with better error handling"""
    logger.info("🔍 Validating training setup...")
    
    # Check embeddings directory
    embeddings_path = Path(args.chunked_embeddings_dir)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_path}")
    
    manifest_path = embeddings_path / "embeddings_manifest.json"
    if not manifest_path.exists():
        logger.warning(f"Embeddings manifest not found: {manifest_path}")
        logger.warning("Proceeding without manifest validation")
        manifest = {}
    else:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    
    # Validate mode compatibility
    extraction_mode = manifest.get('extraction_mode', '')
    tokens_per_sample = manifest.get('tokens_per_sample', 0)
    
    expected_tokens = 257 if args.training_mode == "cls_patch" else 256
    
    if tokens_per_sample and tokens_per_sample != expected_tokens:
        logger.warning(f"Token mismatch: training mode expects {expected_tokens}, "
                      f"but embeddings have {tokens_per_sample}")
        logger.warning("Training will attempt to handle this automatically")
    
    # Validate shard selection
    total_shards = manifest.get('total_shards', 0)
    if args.max_training_shards:
        if total_shards > 0 and args.max_training_shards > total_shards:
            logger.warning(f"Requested {args.max_training_shards} shards but only {total_shards} available")
            args.max_training_shards = total_shards
    
    # Overfitting test setup
    if args.overfitting_test:
        args.max_training_shards = 1
        args.enable_same_data_eval = True
        args.same_data_eval_frequency = 50
        logger.info("🧪 Overfitting test mode: Using 1 shard with frequent evaluation")
    
    logger.info("✅ Training setup validated")
    logger.info(f"   📊 Dataset: {total_shards} total shards, {manifest.get('total_samples', 0):,} samples")
    logger.info(f"   🎯 Training mode: {args.training_mode} ({expected_tokens} tokens)")
    logger.info(f"   📦 Using shards: {args.max_training_shards or 'all'}")
    logger.info(f"   🔄 Same-data eval: {args.enable_same_data_eval}")
    
    return manifest

def setup_device_safely(local_rank, cpu_fallback, logger):
    """Setup device with fallback options"""
    try:
        if torch.cuda.is_available() and not cpu_fallback:
            device = torch.device(f"cuda:{local_rank}")
            logger.info(f"🎮 Using GPU: {torch.cuda.get_device_name(local_rank)}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GB")
            return device, False
        else:
            device = torch.device("cpu")
            logger.info("🖥️ Using CPU (GPU not available or fallback enabled)")
            return device, True
    except Exception as e:
        logger.warning(f"⚠️ GPU setup failed: {e}")
        if cpu_fallback:
            device = torch.device("cpu")
            logger.info("🖥️ Falling back to CPU")
            return device, True
        else:
            raise

def create_model_safely(args, device, is_cpu, logger):
    """Create model with proper error handling and gradient checkpointing"""
    logger.info("🏗️ Creating model...")
    
    try:
        # Import the FIXED model
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
        
        # Size configurations
        size_configs = {
            "tiny": {"hidden_size": 512, "num_hidden_layers": 6, "num_attention_heads": 8},
            "small": {"hidden_size": 768, "num_hidden_layers": 8, "num_attention_heads": 12},
            "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
            "large": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 16},
        }
        
        base_config = size_configs.get(args.model_size, size_configs["base"])
        
        # FIXED: Create config with gradient checkpointing support
        config = BLIP3oDiTConfig(
            hidden_size=args.hidden_size or base_config["hidden_size"],
            num_hidden_layers=args.num_layers or base_config["num_hidden_layers"],
            num_attention_heads=args.num_heads or base_config["num_attention_heads"],
            training_mode=args.training_mode,
            num_tokens=257 if args.training_mode == "cls_patch" else 256,
            max_position_embeddings=257,
            use_gradient_checkpointing=args.gradient_checkpointing and not is_cpu,  # Disable on CPU
        )
        
        model = create_blip3o_patch_dit_model(config=config)
        model = model.to(device)
        
        # FIXED: Proper gradient checkpointing handling
        if args.gradient_checkpointing and not is_cpu:
            try:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logger.info("✅ Gradient checkpointing enabled")
                else:
                    logger.warning("⚠️ Model doesn't support gradient_checkpointing_enable")
            except Exception as e:
                logger.warning(f"⚠️ Gradient checkpointing failed: {e}")
                logger.warning("   Continuing without gradient checkpointing")
        
        param_count = model.get_num_parameters()
        logger.info(f"   Parameters: {param_count:,}")
        logger.info(f"   Mode: {args.training_mode} ({config.num_tokens} tokens)")
        logger.info(f"   Device: {device}")
        
        return model, config
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        if args.safe_mode:
            logger.info("🛡️ Safe mode: Trying fallback model configuration...")
            # Try a smaller, safer configuration
            try:
                config = BLIP3oDiTConfig(
                    hidden_size=256,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    training_mode=args.training_mode,
                    num_tokens=257 if args.training_mode == "cls_patch" else 256,
                    use_gradient_checkpointing=False,  # Disable in safe mode
                )
                model = create_blip3o_patch_dit_model(config=config)
                model = model.to(device)
                logger.info("✅ Fallback model created successfully")
                return model, config
            except Exception as e2:
                logger.error(f"❌ Even fallback model failed: {e2}")
                raise
        else:
            raise

def test_gradient_flow_safely(model, dataloader, flow_matching_loss, device, logger):
    """Test gradient flow with better error handling"""
    logger.info("🧪 Testing gradient flow...")
    
    try:
        model.train()
        batch = next(iter(dataloader))
        
        # Move to device safely
        eva_embeddings = batch['encoder_hidden_states'].to(device)
        clip_embeddings = batch['clip_embeddings'].to(device)
        timesteps = batch['timestep'].to(device)
        noisy_input = batch['hidden_states'].to(device)
        noise = batch.get('noise', torch.randn_like(clip_embeddings)).to(device)
        
        # Check input gradients
        logger.info(f"   EVA embeddings requires_grad: {eva_embeddings.requires_grad}")
        logger.info(f"   CLIP embeddings requires_grad: {clip_embeddings.requires_grad}")
        logger.info(f"   Noisy input requires_grad: {noisy_input.requires_grad}")
        
        if not noisy_input.requires_grad:
            logger.warning("⚠️ Noisy input doesn't require gradients - fixing...")
            noisy_input = noisy_input.requires_grad_(True)
        
        # Forward pass
        model_output = model(
            hidden_states=noisy_input,
            timestep=timesteps,
            encoder_hidden_states=eva_embeddings,
            return_dict=True
        )
        
        velocity_pred = model_output['velocity_prediction']
        logger.info(f"   Model output requires_grad: {velocity_pred.requires_grad}")
        
        if not velocity_pred.requires_grad:
            raise RuntimeError("Model output doesn't require gradients!")
        
        # Compute loss
        loss, metrics = flow_matching_loss(
            model_output=velocity_pred,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            noise=noise,
            return_metrics=True
        )
        
        logger.info(f"   Loss value: {loss.item():.6f}")
        logger.info(f"   Loss requires_grad: {loss.requires_grad}")
        
        if not loss.requires_grad:
            raise RuntimeError("Loss doesn't require gradients!")
        
        # Test backward pass
        loss.backward()
        
        # Check parameter gradients
        param_with_grad = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    param_with_grad += 1
        
        logger.info(f"   Parameters with gradients: {param_with_grad}/{total_params}")
        
        if param_with_grad == 0:
            raise RuntimeError("No parameters received gradients!")
        
        logger.info("✅ Gradient flow test passed!")
        
        # Zero gradients for actual training
        model.zero_grad()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gradient flow test failed: {e}")
        if args.safe_mode:
            logger.warning("🛡️ Safe mode: Continuing despite gradient flow issues")
            return False
        else:
            raise

def main():
    """FIXED: Main training function"""
    args = parse_arguments()
    
    # Setup distributed training
    is_distributed, local_rank, global_rank, world_size = setup_distributed_training()
    is_main_process = (global_rank == 0)
    
    # Setup logging
    log_dir = Path(args.output_dir) / "logs"
    logger = setup_logging(local_rank, log_dir if is_main_process else None)
    
    if is_main_process:
        print("🚀 FIXED: Enhanced BLIP3-o Training with CLS+Patch Support")
        print("=" * 60)
        print("🎯 FEATURES:")
        print(f"  ✅ Training mode: {args.training_mode}")
        print(f"  ✅ Expected tokens: {257 if args.training_mode == 'cls_patch' else 256}")
        print(f"  ✅ Max training shards: {args.max_training_shards or 'All'}")
        print(f"  ✅ Same-data evaluation: {args.enable_same_data_eval}")
        print(f"  ✅ Gradient checkpointing: {args.gradient_checkpointing}")
        print(f"  ✅ Safe mode: {args.safe_mode}")
        print(f"  ✅ CPU fallback: {args.cpu_fallback}")
        print("=" * 60)
    
    try:
        # 1. Validate setup
        if is_main_process:
            logger.info("🔍 Step 1: Validating training setup")
        
        manifest = validate_training_setup(args, logger)
        
        # 2. Setup device
        device, is_cpu = setup_device_safely(local_rank, args.cpu_fallback, logger)
        
        # 3. Load modules with error handling
        if is_main_process:
            logger.info("📦 Step 2: Loading enhanced modules")
        
        try:
            from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
            from src.modules.trainers.blip3o_flexible_trainer import (
                BLIP3oFlexibleTrainer, create_blip3o_flexible_training_args
            )
            from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
            from src.modules.evaluation.blip3o_detailed_evaluator import create_detailed_evaluator
        except ImportError as e:
            logger.error(f"❌ Failed to import modules: {e}")
            if args.safe_mode:
                logger.info("🛡️ Safe mode: Please check your src module installation")
                return 1
            else:
                raise
        
        # 4. Create model
        if is_main_process:
            logger.info("🏗️ Step 3: Creating model")
        
        model, config = create_model_safely(args, device, is_cpu, logger)
        
        # 5. Wrap with DDP if needed
        if is_distributed and not is_cpu:
            try:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,
                )
                if is_main_process:
                    logger.info("✅ Model wrapped with DDP")
            except Exception as e:
                logger.warning(f"⚠️ DDP wrapping failed: {e}")
                if not args.safe_mode:
                    raise
        
        # 6. Create loss function
        flow_matching_loss = create_blip3o_flow_matching_loss(
            prediction_type=args.prediction_type,
            normalize_targets=args.normalize_targets,
        )
        
        if is_main_process:
            logger.info("✅ Pure flow matching loss created (BLIP3-o paper aligned)")
        
        # 7. Create dataloaders
        if is_main_process:
            logger.info("🔄 Step 4: Creating flexible dataloaders")
        
        try:
            train_dataloader, eval_dataloader = create_flexible_dataloaders(
                chunked_embeddings_dir=args.chunked_embeddings_dir,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                eval_split_ratio=0.1,
                normalize_embeddings=True,
                training_mode=args.training_mode,
                max_shards=args.max_training_shards,
                use_same_data_for_eval=args.enable_same_data_eval,
                delete_after_use=False,  # Keep for evaluation
                num_workers=min(args.dataloader_num_workers, 2) if is_cpu else args.dataloader_num_workers,
                pin_memory=torch.cuda.is_available() and not is_cpu,
            )
            
            if is_main_process:
                logger.info(f"✅ Dataloaders created for {args.training_mode} mode")
                logger.info(f"   Training shards: {args.max_training_shards or 'all'}")
        except Exception as e:
            logger.error(f"❌ Dataloader creation failed: {e}")
            if args.safe_mode:
                logger.info("🛡️ Safe mode: Cannot continue without dataloaders")
                return 1
            else:
                raise
        
        # 8. Test gradient flow
        if args.test_gradient_flow and is_main_process:
            gradient_flow_ok = test_gradient_flow_safely(model, train_dataloader, flow_matching_loss, device, logger)
            if not gradient_flow_ok and not args.safe_mode:
                logger.error("❌ Gradient flow test failed and safe mode is disabled")
                return 1
        
        # 9. Create detailed evaluator
        evaluator = None
        if args.enable_detailed_eval and is_main_process:
            try:
                evaluator = create_detailed_evaluator(
                    model=model,
                    training_mode=args.training_mode,
                    device=device,
                )
                logger.info("✅ Detailed evaluator created")
            except Exception as e:
                logger.warning(f"⚠️ Detailed evaluator creation failed: {e}")
                if not args.safe_mode:
                    raise
        
        # 10. Create training args
        if is_main_process:
            logger.info("⚙️ Step 5: Setting up training")
        
        training_args = create_blip3o_flexible_training_args(
            output_dir=args.output_dir,
            training_mode=args.training_mode,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16 and not is_cpu,  # Disable FP16 on CPU
            dataloader_num_workers=min(args.dataloader_num_workers, 2) if is_cpu else args.dataloader_num_workers,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            enable_evaluation=args.enable_same_data_eval,
            eval_steps=args.same_data_eval_frequency if args.enable_same_data_eval else None,
        )
        
        # 11. Create enhanced trainer
        trainer = BLIP3oFlexibleTrainer(
            model=model,
            args=training_args,
            flow_matching_loss=flow_matching_loss,
            train_dataset=None,  # We override the dataloader
            eval_dataset=None,
            training_mode=args.training_mode,
            max_training_shards=args.max_training_shards,
            enable_same_data_eval=args.enable_same_data_eval,
            eval_frequency=args.same_data_eval_frequency,
            detailed_logging=args.detailed_logging,
        )
        
        # Override dataloaders
        trainer.get_train_dataloader = lambda: train_dataloader
        if eval_dataloader:
            trainer.get_eval_dataloader = lambda eval_dataset=None: eval_dataloader
        
        if is_main_process:
            logger.info("✅ Enhanced flexible trainer created")
        
        # 12. Start training
        if is_main_process:
            logger.info("🚀 Step 6: Starting enhanced training")
            logger.info(f"   Mode: {args.training_mode} ({config.num_tokens} tokens)")
            logger.info(f"   Shards: {args.max_training_shards or 'all'}")
            logger.info(f"   Same-data eval: {args.enable_same_data_eval}")
            logger.info(f"   Overfitting test: {args.overfitting_test}")
            logger.info(f"   Device: {device}")
        
        # Training with error handling
        try:
            train_result = trainer.train()
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            if args.safe_mode:
                logger.info("🛡️ Safe mode: Attempting to save partial results...")
                try:
                    trainer.save_model()
                    logger.info("✅ Partial model saved")
                except Exception as save_e:
                    logger.error(f"❌ Failed to save partial model: {save_e}")
            raise
        
        # 13. Save model and results
        if is_main_process:
            logger.info("💾 Saving model and results...")
            try:
                trainer.save_model()
                
                # Save comprehensive training info
                training_info = {
                    'training_completed': True,
                    'training_mode': args.training_mode,
                    'expected_tokens': config.num_tokens,
                    'max_training_shards': args.max_training_shards,
                    'overfitting_test': args.overfitting_test,
                    'same_data_evaluation': args.enable_same_data_eval,
                    'detailed_evaluation': args.enable_detailed_eval,
                    'gradient_checkpointing_used': args.gradient_checkpointing,
                    'device_used': str(device),
                    'model_config': config.to_dict() if hasattr(config, 'to_dict') else vars(config),
                    'training_args': training_args.to_dict(),
                    'training_statistics': trainer.get_training_statistics(),
                    'manifest_info': manifest,
                    'architecture': f'BLIP3-o DiT ({args.training_mode} mode)',
                    'paper_alignment': 'Pure flow matching loss (BLIP3-o paper)',
                    'fixes_applied': [
                        'Gradient checkpointing support',
                        'Better error handling',
                        'CPU fallback option',
                        'Safe mode operation',
                        'Memory optimization',
                    ],
                    'timestamp': datetime.now().isoformat(),
                }
                
                with open(Path(args.output_dir) / 'comprehensive_training_info.json', 'w') as f:
                    json.dump(training_info, f, indent=2)
                
                logger.info("✅ FIXED Enhanced BLIP3-o training completed successfully!")
                logger.info("📋 Key results:")
                logger.info(f"   🎯 Training mode: {args.training_mode}")
                logger.info(f"   📊 Total steps: {trainer.training_step_count}")
                logger.info(f"   📦 Shards used: {args.max_training_shards or 'all'}")
                logger.info(f"   🎮 Device: {device}")
                
                # Final statistics
                stats = trainer.get_training_statistics()
                if 'loss_statistics' in stats:
                    logger.info(f"   📉 Final loss: {stats['loss_statistics']['current_loss']:.6f}")
                
                if args.enable_same_data_eval and 'latest_evaluation_metrics' in stats:
                    eval_metrics = stats['latest_evaluation_metrics']
                    logger.info(f"   📈 Same-data similarity: {eval_metrics.get('per_image_mean_cosine', 0):.4f}")
                    if stats.get('overfitting_detected', False):
                        logger.info("   🎉 Overfitting successfully detected!")
                
            except Exception as e:
                logger.error(f"❌ Failed to save results: {e}")
                if not args.safe_mode:
                    raise
        
        # Cleanup
        if is_distributed:
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup distributed training: {e}")
        
        return 0
        
    except Exception as e:
        if is_main_process:
            logger.error(f"❌ Training failed: {e}")
            if args.debug:
                traceback.print_exc()
            
            # Save error info
            error_info = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'training_mode': getattr(args, 'training_mode', 'unknown'),
                'training_args': vars(args),
                'environment': {
                    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'WORLD_SIZE': os.environ.get('WORLD_SIZE'),
                    'LOCAL_RANK': os.environ.get('LOCAL_RANK'),
                    'CUDA_AVAILABLE': torch.cuda.is_available(),
                    'CUDA_DEVICE_COUNT': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                },
                'fixes_available': [
                    'Try --gradient_checkpointing=False',
                    'Try --cpu_fallback',
                    'Try --safe_mode',
                    'Reduce --batch_size',
                    'Use smaller --model_size',
                ],
                'timestamp': datetime.now().isoformat(),
            }
            
            try:
                with open('blip3o_enhanced_training_error.json', 'w') as f:
                    json.dump(error_info, f, indent=2)
                logger.error("💾 Error info saved to blip3o_enhanced_training_error.json")
            except Exception:
                pass
        
        if is_distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)