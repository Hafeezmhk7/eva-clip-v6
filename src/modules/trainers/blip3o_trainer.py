#!/usr/bin/env python3
"""
BLIP3-o Trainer for CLIP Reproduction from EVA Embeddings
FIXED: WandB initialization order issue

Key fixes:
1. Fixed WandB setup order - estimate steps before WandB initialization
2. Proper attribute initialization sequence
3. Better error handling for WandB setup
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
import numpy as np
from pathlib import Path
import json
import gc
from collections import deque
import math
import os

# WandB import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class BLIP3oCLIPTrainer:
    """
    Trainer for CLIP reproduction with WandB integration and minimal normalization approach
    FIXED: Proper initialization order
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        train_dataloader,
        eval_dataloader=None,
        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        # Evaluation
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 500,
        eval_inference_steps: int = 50,
        # Debugging
        debug_mode: bool = False,
        overfit_test_size: Optional[int] = None,
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 500,
        # Output
        output_dir: str = "./checkpoints",
        # Device
        device: Optional[torch.device] = None,
        # WandB configuration
        use_wandb: bool = True,
        wandb_project: str = "blip3o-clip-reproduction",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        wandb_api_key: Optional[str] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        
        # Evaluation config
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_num_samples = eval_num_samples
        self.eval_inference_steps = eval_inference_steps
        
        # Debugging config
        self.debug_mode = debug_mode
        self.overfit_test_size = overfit_test_size
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = self.model.to(self.device)
        
        # WandB configuration (store for later setup)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self.wandb_api_key = wandb_api_key
        
        # Initialize tracking variables early
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_similarity = 0.0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        
        # FIXED: Estimate steps per epoch BEFORE WandB setup
        self.estimated_steps_per_epoch = self._estimate_steps_per_epoch()
        
        # Setup optimizer and scheduler BEFORE WandB (needs estimated_steps_per_epoch)
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # FIXED: Setup WandB AFTER all required attributes are initialized
        if self.use_wandb:
            self._setup_wandb()
        elif use_wandb and not WANDB_AVAILABLE:
            logger.warning("WandB requested but not available. Install with: pip install wandb")
        
        # Overfitting test data (setup after WandB to log properly)
        self.overfit_batch = None
        if self.overfit_test_size:
            self._prepare_overfit_test()
        
        logger.info("BLIP3-o CLIP Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Estimated steps per epoch: {self.estimated_steps_per_epoch}")
        logger.info(f"  Overfit test: {self.overfit_test_size if self.overfit_test_size else 'Disabled'}")
        logger.info(f"  Mixed precision: {self.fp16}")
        logger.info(f"  WandB logging: {self.use_wandb}")
        logger.info(f"  🚫 Minimal normalization: Only for evaluation similarity")

    def _estimate_steps_per_epoch(self) -> int:
        """Estimate steps per epoch for IterableDataset (WandB-safe)"""
        try:
            # Try to get exact length
            length = len(self.train_dataloader)
            logger.info(f"Got exact dataloader length: {length}")
            return length
        except TypeError:
            # For IterableDataset, estimate from dataset length and batch size
            try:
                dataset_length = len(self.train_dataloader.dataset)
                batch_size = getattr(self.train_dataloader, 'batch_size', 1)
                estimated_steps = max(1, dataset_length // batch_size)
                logger.info(f"Estimated steps per epoch from dataset length: {estimated_steps}")
                return estimated_steps
            except (TypeError, AttributeError):
                # Final fallback - reasonable default for WandB
                logger.warning("Could not estimate steps per epoch, using default: 100")
                return 100

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        # Use AdamW with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup learning rate scheduler with warmup
        total_steps = self.estimated_steps_per_epoch * self.num_epochs
        
        if self.warmup_steps > 0:
            # Warmup + Cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=self.learning_rate * 0.01
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
            # Just cosine decay
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01
            )
        
        logger.info(f"Optimizer and scheduler setup complete")
        logger.info(f"  Total estimated steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")

    def _setup_wandb(self):
        """Setup WandB with proper configuration - FIXED"""
        try:
            # Set API key if provided
            if self.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.wandb_api_key
            elif "WANDB_API_KEY" not in os.environ:
                # Use the provided API key from user
                os.environ["WANDB_API_KEY"] = "0d9895af249ee18e4fa141e8a2350e0f4adb920f"
            
            # Get model config if available
            model_config = {}
            if hasattr(self.model, 'config'):
                model_config = {
                    'model_type': getattr(self.model.config, 'model_type', 'blip3o_clip_dit'),
                    'hidden_size': getattr(self.model.config, 'hidden_size', 768),
                    'num_hidden_layers': getattr(self.model.config, 'num_hidden_layers', 12),
                    'num_attention_heads': getattr(self.model.config, 'num_attention_heads', 12),
                    'use_3d_rope': getattr(self.model.config, 'use_3d_rope', True),
                    'use_sandwich_norm': getattr(self.model.config, 'use_sandwich_norm', True),
                    'training_mode': getattr(self.model.config, 'training_mode', 'patch_only'),
                    'eva_embedding_size': getattr(self.model.config, 'eva_embedding_size', 4096),
                    'clip_embedding_size': getattr(self.model.config, 'clip_embedding_size', 1024),
                }
            
            # Create comprehensive WandB config
            wandb_config = {
                # Training hyperparameters
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps,
                'max_grad_norm': self.max_grad_norm,
                'fp16': self.fp16,
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,  # Now available!
                
                # Evaluation config
                'eval_every_n_steps': self.eval_every_n_steps,
                'eval_num_samples': self.eval_num_samples,
                'eval_inference_steps': self.eval_inference_steps,
                
                # Model architecture
                **model_config,
                
                # Experiment details
                'experiment_type': 'clip_reproduction',
                'task': 'EVA_to_CLIP_embedding_reproduction',
                'method': 'BLIP3o_DiT_with_rectified_flow',
                'normalization_approach': 'minimal',
                'overfit_test_size': self.overfit_test_size,
                'debug_mode': self.debug_mode,
                
                # Architecture features
                'uses_3d_rope': model_config.get('use_3d_rope', True),
                'uses_sandwich_norm': model_config.get('use_sandwich_norm', True),
                'uses_grouped_query_attention': True,
                'flow_matching_type': 'rectified_flow',
                'prediction_type': 'velocity',
                
                # Dataset info (will be updated dynamically)
                'dataset_type': 'IterableDataset',
                'handles_iterable_length': True,
                
                # Additional config from user
                **self.wandb_config,
            }
            
            # Safely get dataset info
            try:
                if hasattr(self.train_dataloader, 'batch_size'):
                    wandb_config['batch_size'] = self.train_dataloader.batch_size
                if hasattr(self.train_dataloader, 'dataset'):
                    dataset = self.train_dataloader.dataset
                    if hasattr(dataset, 'training_mode'):
                        wandb_config['training_mode'] = dataset.training_mode
                    if hasattr(dataset, 'normalize_embeddings'):
                        wandb_config['normalize_embeddings'] = dataset.normalize_embeddings
                    if hasattr(dataset, 'expected_tokens'):
                        wandb_config['expected_tokens'] = dataset.expected_tokens
            except Exception as e:
                logger.warning(f"Could not extract dataset info for WandB: {e}")
            
            # Initialize WandB run
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
                dir=str(self.output_dir),
                resume="allow",
                tags=["blip3o", "clip_reproduction", "eva_conditioning", "3d_rope", "sandwich_norm"]
            )
            
            # Log model architecture
            if hasattr(self.model, 'get_num_parameters'):
                wandb.log({"model/total_parameters": self.model.get_num_parameters()})
            
            # Watch model for gradients and parameters
            wandb.watch(self.model, log="all", log_freq=self.log_every_n_steps)
            
            logger.info(f"✅ WandB initialized: {self.wandb_project}")
            logger.info(f"   Run ID: {self.wandb_run.id}")
            logger.info(f"   Run URL: {self.wandb_run.url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup WandB: {e}")
            logger.error("Continuing without WandB logging...")
            self.use_wandb = False

    def _prepare_overfit_test(self):
        """Prepare overfitting test batch"""
        logger.info(f"Preparing overfitting test with {self.overfit_test_size} samples...")
        
        try:
            # Get first batch and repeat it
            first_batch = next(iter(self.train_dataloader))
            
            # Trim to desired size
            actual_size = min(self.overfit_test_size, first_batch['batch_size'])
            
            self.overfit_batch = {}
            for key, value in first_batch.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    self.overfit_batch[key] = value[:actual_size].clone().detach()
                elif isinstance(value, list):
                    self.overfit_batch[key] = value[:actual_size]
                else:
                    self.overfit_batch[key] = value
            
            # Update batch size
            self.overfit_batch['batch_size'] = actual_size
            
            logger.info(f"Overfitting test prepared with {actual_size} samples")
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "overfit_test/enabled": True,
                    "overfit_test/size": actual_size,
                    "overfit_test/step": 0
                })
            
        except Exception as e:
            logger.error(f"Failed to prepare overfitting test: {e}")
            self.overfit_batch = None

    def _compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch"""
        # Move batch to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        
        # Use overfit batch if specified
        if self.overfit_batch is not None:
            # Move overfit batch to device
            for key, value in self.overfit_batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
                else:
                    batch[key] = value
        
        # Extract inputs
        hidden_states = batch['hidden_states']          # [B, N, 1024] - Noisy CLIP
        timestep = batch['timestep']                    # [B] - Timesteps
        encoder_hidden_states = batch['encoder_hidden_states']  # [B, N, 4096] - EVA
        clip_embeddings = batch['clip_embeddings']      # [B, N, 1024] - Clean CLIP (target)
        noise = batch.get('noise')                      # [B, N, 1024] - Noise
        
        # Forward pass
        if self.fp16:
            with torch.cuda.amp.autocast():
                model_output = self.model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False
                )
                
                # Compute loss
                loss, metrics = self.loss_fn(
                    model_output=model_output,
                    target_samples=clip_embeddings,
                    timesteps=timestep,
                    eva_conditioning=encoder_hidden_states,
                    noise=noise,
                    return_metrics=True
                )
        else:
            model_output = self.model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )
            
            loss, metrics = self.loss_fn(
                model_output=model_output,
                target_samples=clip_embeddings,
                timesteps=timestep,
                eva_conditioning=encoder_hidden_states,
                noise=noise,
                return_metrics=True
            )
        
        return loss, metrics

    def _backward_and_step(self, loss: torch.Tensor) -> float:
        """Backward pass and optimizer step"""
        # Backward pass
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Compute gradient norm before clipping
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            if self.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return grad_norm

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """Run evaluation (normalize only for similarity)"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        self.model.eval()
        
        all_similarities = []
        all_mse_losses = []
        samples_processed = 0
        
        eval_start_time = time.time()
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                if samples_processed >= num_samples:
                    break
                
                # Move to device
                eva_features = batch['encoder_hidden_states'].to(self.device)
                target_clip = batch['clip_embeddings'].to(self.device)
                
                # Generate CLIP embeddings (no normalization during generation)
                generated_clip = self.model.generate(
                    eva_features=eva_features,
                    num_inference_steps=self.eval_inference_steps,
                    normalize_output=False  # No normalization during generation
                )
                
                # Compute similarity (normalize ONLY for similarity computation)
                target_norm = F.normalize(target_clip, p=2, dim=-1)
                generated_norm = F.normalize(generated_clip, p=2, dim=-1)
                similarity = F.cosine_similarity(generated_norm, target_norm, dim=-1)
                per_image_similarity = similarity.mean(dim=1)
                
                # Compute MSE loss in raw space
                mse_loss = F.mse_loss(generated_clip, target_clip, reduction='none').mean(dim=(1, 2))
                
                all_similarities.append(per_image_similarity.cpu())
                all_mse_losses.append(mse_loss.cpu())
                samples_processed += eva_features.shape[0]
        
        self.model.train()
        
        if not all_similarities:
            return {}
        
        all_sims = torch.cat(all_similarities)
        all_mse = torch.cat(all_mse_losses)
        
        eval_time = time.time() - eval_start_time
        
        eval_metrics = {
            'eval_clip_similarity': all_sims.mean().item(),
            'eval_clip_similarity_std': all_sims.std().item(),
            'eval_clip_similarity_min': all_sims.min().item(),
            'eval_clip_similarity_max': all_sims.max().item(),
            'eval_mse_loss': all_mse.mean().item(),
            'eval_mse_loss_std': all_mse.std().item(),
            'eval_high_quality': (all_sims > 0.7).float().mean().item(),
            'eval_very_high_quality': (all_sims > 0.8).float().mean().item(),
            'eval_excellent_quality': (all_sims > 0.9).float().mean().item(),
            'eval_samples': samples_processed,
            'eval_time_seconds': eval_time,
            'eval_samples_per_second': samples_processed / eval_time if eval_time > 0 else 0,
        }
        
        return eval_metrics

    def _log_metrics(self, loss: float, metrics: Dict[str, float], grad_norm: float):
        """Log training metrics to console and WandB"""
        # Store metrics
        self.loss_history.append(loss)
        if 'velocity_similarity' in metrics:
            self.similarity_history.append(metrics['velocity_similarity'])
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.grad_norm_history.append(grad_norm)
        
        # Update best metrics
        if 'velocity_similarity' in metrics:
            if metrics['velocity_similarity'] > self.best_eval_similarity:
                self.best_eval_similarity = metrics['velocity_similarity']
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Prepare WandB metrics
        wandb_metrics = {}
        if self.use_wandb:
            # Training metrics
            wandb_metrics.update({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                "train/epoch": self.current_epoch,
                "train/step": self.global_step,
            })
            
            # Loss function specific metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        if key.startswith('eval_'):
                            wandb_metrics[f"eval/{key[5:]}"] = value
                        else:
                            wandb_metrics[f"train/{key}"] = value
            
            # Moving averages
            if len(self.loss_history) > 0:
                wandb_metrics["train/loss_ma"] = np.mean(list(self.loss_history))
            if len(self.similarity_history) > 0:
                wandb_metrics["train/similarity_ma"] = np.mean(list(self.similarity_history))
            if len(self.grad_norm_history) > 0:
                wandb_metrics["train/grad_norm_ma"] = np.mean(list(self.grad_norm_history))
            
            # Best metrics
            wandb_metrics["train/best_loss"] = self.best_loss
            wandb_metrics["train/best_similarity"] = self.best_eval_similarity
            
            # Overfit test specific metrics
            if self.overfit_batch is not None:
                wandb_metrics["overfit_test/active"] = True
                if 'velocity_similarity' in metrics:
                    wandb_metrics["overfit_test/similarity"] = metrics['velocity_similarity']
                wandb_metrics["overfit_test/loss"] = loss
            
            # System metrics
            if torch.cuda.is_available():
                wandb_metrics["system/gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
                wandb_metrics["system/gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
            
            # Log to WandB
            wandb.log(wandb_metrics, step=self.global_step)
        
        # Log to console
        if self.global_step % self.log_every_n_steps == 0:
            log_msg = f"Step {self.global_step}: Loss={loss:.6f}"
            
            if 'velocity_similarity' in metrics:
                sim = metrics['velocity_similarity']
                quality = metrics.get('quality_assessment', 'unknown')
                log_msg += f", VelSim={sim:.4f} ({quality})"
            
            log_msg += f", GradNorm={grad_norm:.3f}"
            log_msg += f", LR={self.optimizer.param_groups[0]['lr']:.2e}"
            
            # Show raw norms (no normalization applied)
            if 'pred_norm' in metrics and 'target_norm' in metrics:
                log_msg += f", PredNorm={metrics['pred_norm']:.3f}, TargetNorm={metrics['target_norm']:.3f}"
            
            if self.overfit_batch is not None:
                log_msg += " [OVERFIT TEST]"
            
            logger.info(log_msg)
            
            # Detailed logging in debug mode
            if self.debug_mode:
                logger.info(f"  Detailed metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"    {key}: {value:.6f}")

    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_eval_similarity': self.best_eval_similarity,
            'best_loss': self.best_loss,
            'loss_history': list(self.loss_history),
            'similarity_history': list(self.similarity_history),
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Log checkpoint to WandB
        if self.use_wandb:
            wandb.log({
                "checkpoint/saved": True,
                "checkpoint/step": self.global_step,
                "checkpoint/path": str(checkpoint_path),
            }, step=self.global_step)

    def train(self) -> Dict[str, Any]:
        """Main training loop with WandB integration"""
        logger.info("Starting CLIP reproduction training with WandB...")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Estimated training steps per epoch: {self.estimated_steps_per_epoch}")
        logger.info(f"  Total estimated training steps: {self.estimated_steps_per_epoch * self.num_epochs}")
        logger.info(f"  🚫 Minimal normalization: Only for evaluation similarity")
        logger.info(f"  📊 WandB logging: {self.use_wandb}")
        
        if self.overfit_batch is not None:
            logger.info(f"  OVERFITTING TEST MODE: Using {self.overfit_batch['batch_size']} samples")
        
        # Log initial setup to WandB
        if self.use_wandb:
            wandb.log({
                "setup/total_parameters": sum(p.numel() for p in self.model.parameters()),
                "setup/estimated_steps_per_epoch": self.estimated_steps_per_epoch,
                "setup/total_estimated_steps": self.estimated_steps_per_epoch * self.num_epochs,
                "setup/training_started": True,
            }, step=0)
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                # Log epoch start to WandB
                if self.use_wandb:
                    wandb.log({
                        "epoch/started": epoch + 1,
                        "epoch/total_epochs": self.num_epochs,
                    }, step=self.global_step)
                
                epoch_loss = 0.0
                epoch_steps = 0
                epoch_start_time = time.time()
                
                # Handle both regular DataLoader and IterableDataset
                try:
                    dataloader_iter = iter(self.train_dataloader)
                    batch_count = 0
                    
                    # For IterableDataset, we'll iterate until StopIteration
                    while True:
                        try:
                            batch = next(dataloader_iter)
                            batch_count += 1
                        except StopIteration:
                            logger.info(f"Epoch {epoch + 1} completed: {batch_count} batches processed")
                            break
                        
                        step_start_time = time.time()
                        
                        # Compute loss
                        try:
                            loss, metrics = self._compute_loss(batch)
                        except Exception as e:
                            logger.error(f"Error computing loss at step {self.global_step}: {e}")
                            continue
                        
                        # Backward pass
                        try:
                            grad_norm = self._backward_and_step(loss)
                        except Exception as e:
                            logger.error(f"Error in backward pass at step {self.global_step}: {e}")
                            continue
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        epoch_steps += 1
                        self.global_step += 1
                        
                        # Add timing metrics
                        step_time = time.time() - step_start_time
                        if self.use_wandb:
                            wandb.log({
                                "timing/step_time": step_time,
                                "timing/samples_per_second": batch.get('batch_size', 1) / step_time if step_time > 0 else 0,
                            }, step=self.global_step)
                        
                        # Log metrics
                        self._log_metrics(loss.item(), metrics or {}, grad_norm)
                        
                        # Run evaluation
                        if self.global_step % self.eval_every_n_steps == 0:
                            logger.info(f"Running evaluation at step {self.global_step}...")
                            eval_metrics = self._evaluate()
                            
                            if eval_metrics:
                                logger.info(f"Evaluation results (similarity uses normalization):")
                                for key, value in eval_metrics.items():
                                    logger.info(f"  {key}: {value:.4f}")
                                
                                # Log evaluation metrics to WandB
                                if self.use_wandb:
                                    wandb_eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                                    wandb.log(wandb_eval_metrics, step=self.global_step)
                                
                                # Update best eval similarity
                                if eval_metrics.get('eval_clip_similarity', 0) > self.best_eval_similarity:
                                    self.best_eval_similarity = eval_metrics['eval_clip_similarity']
                                    logger.info(f"New best CLIP similarity: {self.best_eval_similarity:.4f}")
                                    
                                    if self.use_wandb:
                                        wandb.log({
                                            "eval/new_best_similarity": self.best_eval_similarity,
                                            "eval/best_similarity_step": self.global_step,
                                        }, step=self.global_step)
                        
                        # Save checkpoint
                        if self.global_step % self.save_every_n_steps == 0:
                            self._save_checkpoint()
                        
                        # Check for early success in overfitting test
                        if (self.overfit_batch is not None and 
                            metrics and 
                            metrics.get('velocity_similarity', 0) > 0.9):
                            logger.info("🎉 OVERFITTING TEST PASSED! Model can learn effectively.")
                            if self.use_wandb:
                                wandb.log({
                                    "overfit_test/passed": True,
                                    "overfit_test/final_similarity": metrics['velocity_similarity'],
                                    "overfit_test/steps_to_pass": self.global_step,
                                }, step=self.global_step)
                            break
                
                except Exception as e:
                    logger.error(f"Error during epoch {epoch + 1}: {e}")
                    # Try to continue with next epoch
                    continue
                
                # End of epoch logging
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                
                logger.info(f"Epoch {epoch + 1} completed:")
                logger.info(f"  Average loss: {avg_epoch_loss:.6f}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  Steps in epoch: {epoch_steps}")
                logger.info(f"  Epoch time: {epoch_time:.1f}s")
                
                # Log epoch summary to WandB
                if self.use_wandb:
                    wandb.log({
                        "epoch/completed": epoch + 1,
                        "epoch/avg_loss": avg_epoch_loss,
                        "epoch/steps": epoch_steps,
                        "epoch/time_seconds": epoch_time,
                        "epoch/steps_per_second": epoch_steps / epoch_time if epoch_time > 0 else 0,
                    }, step=self.global_step)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if self.use_wandb:
                wandb.log({"training/interrupted": True}, step=self.global_step)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            if self.use_wandb:
                wandb.log({"training/failed": True, "training/error": str(e)}, step=self.global_step)
            raise
        
        finally:
            # Final checkpoint
            self._save_checkpoint()
            
            # Final evaluation
            logger.info("Running final evaluation...")
            final_eval = self._evaluate(num_samples=self.eval_num_samples * 2)
            
            total_time = time.time() - start_time
            
            # Training summary
            summary = {
                'training_completed': True,
                'total_time_seconds': total_time,
                'total_steps': self.global_step,
                'final_epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'best_eval_similarity': self.best_eval_similarity,
                'final_eval': final_eval,
                'overfit_test': self.overfit_batch is not None,
                'overfit_success': (self.overfit_batch is not None and 
                                  len(self.similarity_history) > 0 and 
                                  max(self.similarity_history) > 0.8),
                'loss_history': list(self.loss_history),
                'similarity_history': list(self.similarity_history),
                'lr_history': list(self.lr_history),
                'grad_norm_history': list(self.grad_norm_history),
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,
                'minimal_normalization': True,
                'wandb_enabled': self.use_wandb,
            }
            
            # Log final summary to WandB
            if self.use_wandb:
                final_wandb_metrics = {
                    "final/training_completed": True,
                    "final/total_time_seconds": total_time,
                    "final/total_steps": self.global_step,
                    "final/best_loss": self.best_loss,
                    "final/best_eval_similarity": self.best_eval_similarity,
                }
                
                if final_eval:
                    for key, value in final_eval.items():
                        final_wandb_metrics[f"final/{key}"] = value
                
                if self.overfit_batch is not None:
                    final_wandb_metrics["final/overfit_test_success"] = summary['overfit_success']
                
                wandb.log(final_wandb_metrics, step=self.global_step)
                
                # Finish WandB run
                wandb.finish()
            
            # Save training summary
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Training completed!")
            logger.info(f"  Total time: {total_time:.1f} seconds")
            logger.info(f"  Total steps: {self.global_step}")
            logger.info(f"  Best loss: {self.best_loss:.6f}")
            logger.info(f"  Best CLIP similarity: {self.best_eval_similarity:.4f}")
            logger.info(f"  🚫 Used minimal normalization approach")
            logger.info(f"  📊 WandB run: {self.use_wandb}")
            
            if final_eval:
                logger.info(f"  Final evaluation:")
                for key, value in final_eval.items():
                    logger.info(f"    {key}: {value:.4f}")
            
            if self.overfit_batch is not None:
                success = summary['overfit_success']
                logger.info(f"  Overfitting test: {'✅ PASSED' if success else '❌ FAILED'}")
            
            return summary


def create_clip_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./checkpoints",
    overfit_test_size: Optional[int] = None,
    debug_mode: bool = False,
    # WandB parameters
    use_wandb: bool = True,
    wandb_project: str = "blip3o-clip-reproduction",
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    **kwargs
) -> BLIP3oCLIPTrainer:
    """Factory function to create CLIP trainer with WandB integration"""
    
    return BLIP3oCLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir,
        overfit_test_size=overfit_test_size,
        debug_mode=debug_mode,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config,
        **kwargs
    )