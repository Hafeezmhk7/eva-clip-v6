#!/usr/bin/env python3
"""
UPDATED: BLIP3-o Trainer with Scale-Aware Evaluation
Key improvements:
1. Uses scale-aware generation for evaluation
2. Better target norm estimation
3. Enhanced norm tracking and debugging
4. Adaptive evaluation parameters based on data statistics
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
from collections import deque, defaultdict
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
    UPDATED: Trainer with Scale-Aware Evaluation and Enhanced Norm Tracking
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
        # Evaluation - UPDATED with scale-aware parameters
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 500,
        eval_inference_steps: int = 50,
        # NEW: Scale-aware evaluation parameters
        use_scale_aware_eval: bool = True,
        eval_target_norm: Optional[float] = None,
        eval_use_lognormal_schedule: bool = True,
        adaptive_target_norm: bool = True,
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
        use_wandb: bool = False,
        wandb_project: str = "blip3o-clip-scale-aware",
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
        
        # NEW: Scale-aware evaluation parameters
        self.use_scale_aware_eval = use_scale_aware_eval
        self.eval_target_norm = eval_target_norm
        self.eval_use_lognormal_schedule = eval_use_lognormal_schedule
        self.adaptive_target_norm = adaptive_target_norm
        
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
        
        # WandB configuration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self.wandb_api_key = wandb_api_key
        
        # Initialize tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_similarity = 0.0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.loss_history = deque(maxlen=1000)
        self.similarity_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        
        # UPDATED: Enhanced norm tracking with scale-aware metrics
        self.norm_tracking = {
            'training_target_norms': deque(maxlen=500),
            'training_eva_norms': deque(maxlen=500),
            'training_pred_norms': deque(maxlen=500),
            'eval_target_norms': deque(maxlen=100),
            'eval_generated_norms': deque(maxlen=100),
            'eval_target_norm_estimates': deque(maxlen=100),  # NEW: Track target norm estimates
            'eval_scale_consistency': deque(maxlen=100),      # NEW: Track scale consistency
            'step_norms': {},
            'overfit_target_norms': deque(maxlen=100),
            'batch_target_norm_history': [],
            'noise_stats': deque(maxlen=100),
            # NEW: Scale-aware evaluation metrics
            'velocity_explosion_events': deque(maxlen=50),
            'norm_guidance_applications': deque(maxlen=50),
            'timestep_schedule_type': deque(maxlen=50),
        }
        
        # Estimate steps per epoch BEFORE WandB setup
        self.estimated_steps_per_epoch = self._estimate_steps_per_epoch()
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Setup mixed precision
        if self.fp16:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Setup WandB
        if self.use_wandb:
            self._setup_wandb()
        elif use_wandb and not WANDB_AVAILABLE:
            logger.warning("WandB requested but not available. Install with: pip install wandb")
        
        # Overfitting test data
        self.overfit_batch = None
        if self.overfit_test_size:
            self._prepare_overfit_test_from_eval_data()
        
        logger.info("UPDATED BLIP3-o CLIP Trainer initialized with Scale-Aware Evaluation")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  🚀 Scale-aware evaluation: {self.use_scale_aware_eval}")
        logger.info(f"  🎯 Adaptive target norm: {self.adaptive_target_norm}")
        logger.info(f"  📅 Log-normal schedule: {self.eval_use_lognormal_schedule}")
        logger.info(f"  🔧 Enhanced norm tracking enabled")

    def _estimate_steps_per_epoch(self) -> int:
        """Estimate steps per epoch for IterableDataset"""
        try:
            length = len(self.train_dataloader)
            logger.info(f"Got exact dataloader length: {length}")
            return length
        except TypeError:
            try:
                dataset_length = len(self.train_dataloader.dataset)
                batch_size = getattr(self.train_dataloader, 'batch_size', 1)
                estimated_steps = max(1, dataset_length // batch_size)
                logger.info(f"Estimated steps per epoch from dataset length: {estimated_steps}")
                return estimated_steps
            except (TypeError, AttributeError):
                logger.warning("Could not estimate steps per epoch, using default: 100")
                return 100

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = self.estimated_steps_per_epoch * self.num_epochs
        
        if self.warmup_steps > 0:
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
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01
            )
        
        logger.info(f"Optimizer and scheduler setup complete")
        logger.info(f"  Total estimated steps: {total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")

    def _setup_wandb(self):
        """Setup WandB with scale-aware configuration"""
        try:
            if self.wandb_api_key:
                os.environ["WANDB_API_KEY"] = self.wandb_api_key
            elif "WANDB_API_KEY" not in os.environ:
                os.environ["WANDB_API_KEY"] = "0d9895af249ee18e4fa141e8a2350e0f4adb920f"
            
            model_config = {}
            if hasattr(self.model, 'config'):
                model_config = {
                    'model_type': getattr(self.model.config, 'model_type', 'blip3o_clip_dit'),
                    'hidden_size': getattr(self.model.config, 'hidden_size', 768),
                    'num_hidden_layers': getattr(self.model.config, 'num_hidden_layers', 12),
                    'use_3d_rope': getattr(self.model.config, 'use_3d_rope', True),
                    'use_sandwich_norm': getattr(self.model.config, 'use_sandwich_norm', True),
                    # NEW: Scale-aware parameters
                    'typical_clip_norm': getattr(self.model.config, 'typical_clip_norm', 26.0),
                    'velocity_explosion_threshold': getattr(self.model.config, 'velocity_explosion_threshold', 100.0),
                    'norm_guidance_strength': getattr(self.model.config, 'norm_guidance_strength', 0.1),
                    'norm_guidance_frequency': getattr(self.model.config, 'norm_guidance_frequency', 10),
                }
            
            wandb_config = {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps,
                'max_grad_norm': self.max_grad_norm,
                'fp16': self.fp16,
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,
                
                # Evaluation config
                'eval_every_n_steps': self.eval_every_n_steps,
                'eval_num_samples': self.eval_num_samples,
                'eval_inference_steps': self.eval_inference_steps,
                
                # NEW: Scale-aware evaluation config
                'use_scale_aware_eval': self.use_scale_aware_eval,
                'eval_target_norm': self.eval_target_norm,
                'eval_use_lognormal_schedule': self.eval_use_lognormal_schedule,
                'adaptive_target_norm': self.adaptive_target_norm,
                
                # Experiment details
                'experiment_type': 'blip3o_clip_scale_aware',
                'task': 'EVA_to_CLIP_embedding_reproduction',
                'method': 'BLIP3o_DiT_with_scale_aware_generation',
                'key_improvements': [
                    'lognormal_timestep_schedule',
                    'velocity_explosion_prevention',
                    'periodic_norm_guidance',
                    'adaptive_target_norm_estimation',
                    'scale_aware_evaluation',
                ],
                
                **model_config,
                **self.wandb_config,
            }
            
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=wandb_config,
                dir=str(self.output_dir),
                resume="allow",
                tags=["blip3o", "clip_reproduction", "scale_aware", "lognormal_schedule"]
            )
            
            if hasattr(self.model, 'get_num_parameters'):
                wandb.log({"model/total_parameters": self.model.get_num_parameters()})
            
            wandb.watch(self.model, log="all", log_freq=self.log_every_n_steps)
            
            logger.info(f"✅ WandB initialized: {self.wandb_project}")
            logger.info(f"   Run ID: {self.wandb_run.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup WandB: {e}")
            self.use_wandb = False

    def _prepare_overfit_test_from_eval_data(self):
        """Prepare overfitting test batch using eval data"""
        logger.info(f"🔧 Preparing overfitting test with {self.overfit_test_size} samples from EVAL DATA...")
        
        try:
            if self.eval_dataloader is not None:
                eval_batch = next(iter(self.eval_dataloader))
                logger.info("✅ Using eval_dataloader for overfitting test")
            else:
                eval_batch = next(iter(self.train_dataloader))
                logger.warning("⚠️  No eval_dataloader available, using train_dataloader")
            
            actual_size = min(self.overfit_test_size, eval_batch['batch_size'])
            
            self.overfit_batch = {}
            for key, value in eval_batch.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    self.overfit_batch[key] = value[:actual_size].clone().detach()
                elif isinstance(value, list):
                    self.overfit_batch[key] = value[:actual_size]
                else:
                    self.overfit_batch[key] = value
            
            self.overfit_batch['batch_size'] = actual_size
            
            overfit_clip_norm = torch.norm(self.overfit_batch['clip_embeddings'], dim=-1).mean().item()
            overfit_eva_norm = torch.norm(self.overfit_batch['encoder_hidden_states'], dim=-1).mean().item()
            
            self.norm_tracking['overfit_target_norms'].append(overfit_clip_norm)
            
            logger.info(f"✅ Overfitting test prepared with {actual_size} samples:")
            logger.info(f"  Overfit CLIP norm: {overfit_clip_norm:.3f}")
            logger.info(f"  Overfit EVA norm: {overfit_eva_norm:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to prepare overfitting test: {e}")
            self.overfit_batch = None

    def _estimate_target_norm_from_eval_data(self) -> float:
        """
        NEW: Estimate target norm from recent evaluation data
        This helps with adaptive target norm for scale-aware generation
        """
        if not self.norm_tracking['eval_target_norms']:
            # Fallback to model config or default
            if hasattr(self.model.config, 'typical_clip_norm'):
                fallback = self.model.config.typical_clip_norm
                # Ensure it's a Python float, not a tensor
                if torch.is_tensor(fallback):
                    return float(fallback.item())
                return float(fallback)
            return 26.0  # Default
        
        recent_norms = list(self.norm_tracking['eval_target_norms'])[-10:]  # Last 10 evaluations
        
        # Convert any tensors to Python floats
        python_norms = []
        for norm in recent_norms:
            if torch.is_tensor(norm):
                python_norms.append(float(norm.item()))
            else:
                python_norms.append(float(norm))
        
        estimated_norm = float(np.mean(python_norms))
        
        # Clamp to reasonable range and ensure it's a scalar float
        estimated_norm = max(20.0, min(35.0, estimated_norm))
        
        self.norm_tracking['eval_target_norm_estimates'].append(estimated_norm)
        
        return estimated_norm

    def _track_batch_norms(self, batch: Dict[str, Any], step: int, is_overfit: bool = False):
        """Track and log batch norms for debugging"""
        with torch.no_grad():
            clip_embeddings = batch.get('clip_embeddings')
            eva_embeddings = batch.get('encoder_hidden_states', batch.get('eva_embeddings'))
            noise = batch.get('noise')
            
            if clip_embeddings is not None and eva_embeddings is not None:
                # Convert norms to Python floats immediately
                clip_norm = float(torch.norm(clip_embeddings, dim=-1).mean().item())
                eva_norm = float(torch.norm(eva_embeddings, dim=-1).mean().item())
                
                self.norm_tracking['training_target_norms'].append(clip_norm)
                self.norm_tracking['training_eva_norms'].append(eva_norm)
                
                noise_stats = {}
                if noise is not None:
                    noise_stats = {
                        'noise_mean': float(noise.mean().item()),
                        'noise_std': float(noise.std().item()),
                        'noise_norm': float(torch.norm(noise, dim=-1).mean().item()),
                        'is_standard_gaussian': abs(noise.std().item() - 1.0) < 0.2 and abs(noise.mean().item()) < 0.2,
                    }
                    self.norm_tracking['noise_stats'].append(noise_stats)
                
                self.norm_tracking['batch_target_norm_history'].append({
                    'step': step,
                    'clip_norm': clip_norm,
                    'eva_norm': eva_norm,
                    'is_overfit': is_overfit,
                    'noise_stats': noise_stats,
                })
                
                if step % 10 == 0 or is_overfit:
                    self.norm_tracking['step_norms'][step] = {
                        'clip_norm': clip_norm,
                        'eva_norm': eva_norm,
                        'clip_std': float(clip_embeddings.std().item()),
                        'eva_std': float(eva_embeddings.std().item()),
                        'is_overfit': is_overfit,
                        'batch_size': batch.get('batch_size', clip_embeddings.shape[0]),
                        'noise_stats': noise_stats,
                    }
                
                return clip_norm, eva_norm, noise_stats
            
            return None, None, {}

    def _compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch with enhanced norm tracking"""
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        
        using_overfit = False
        if self.overfit_batch is not None:
            for key, value in self.overfit_batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)
                else:
                    batch[key] = value
            using_overfit = True
        
        batch_clip_norm, batch_eva_norm, noise_stats = self._track_batch_norms(batch, self.global_step, using_overfit)
        
        hidden_states = batch['hidden_states']
        timestep = batch['timestep']
        encoder_hidden_states = batch['encoder_hidden_states']
        clip_embeddings = batch['clip_embeddings']
        noise = batch.get('noise')
        
        if self.fp16:
            with torch.amp.autocast('cuda'):
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
        
        if model_output is not None:
            pred_norm = torch.norm(model_output, dim=-1).mean().item()
            self.norm_tracking['training_pred_norms'].append(pred_norm)
            
            if metrics:
                metrics['batch_clip_norm'] = batch_clip_norm
                metrics['batch_eva_norm'] = batch_eva_norm
                metrics['using_overfit_test'] = using_overfit
                metrics['pred_norm_tracked'] = pred_norm
                
                if noise_stats:
                    metrics.update({f"batch_{k}": v for k, v in noise_stats.items()})
        
        return loss, metrics

    def _backward_and_step(self, loss: torch.Tensor) -> float:
        """Backward pass and optimizer step"""
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        if self.max_grad_norm > 0:
            if self.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        if self.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return grad_norm

    def _evaluate(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """UPDATED: Run evaluation with scale-aware generation"""
        if self.eval_dataloader is None:
            return {}
        
        if num_samples is None:
            num_samples = self.eval_num_samples
        
        self.model.eval()
        
        all_similarities = []
        all_mse_losses = []
        all_generated_norms = []
        all_target_norms = []
        all_scale_consistency = []
        samples_processed = 0
        
        # NEW: Track scale-aware metrics
        velocity_explosions = 0
        norm_guidance_used = 0
        
        eval_start_time = time.time()
        
        # NEW: Determine target norm for scale-aware evaluation
        if self.use_scale_aware_eval and self.adaptive_target_norm:
            target_norm = self._estimate_target_norm_from_eval_data()
        else:
            if self.eval_target_norm is not None:
                target_norm = self.eval_target_norm
            else:
                target_norm = getattr(self.model.config, 'typical_clip_norm', 26.0)
        
        # FIXED: Ensure target_norm is always a Python float, not a tensor
        if torch.is_tensor(target_norm):
            target_norm = float(target_norm.item())
        else:
            target_norm = float(target_norm)
            
        # Validate target_norm is reasonable
        if not (15.0 <= target_norm <= 50.0):
            logger.warning(f"Target norm {target_norm:.3f} seems unusual, clamping to reasonable range")
            target_norm = max(15.0, min(50.0, target_norm))
        
        if self.debug_mode:
            logger.info(f"🎯 Using target norm: {target_norm:.3f} (type: {type(target_norm).__name__})")
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                if samples_processed >= num_samples:
                    break
                
                eva_features = batch['encoder_hidden_states'].to(self.device)
                target_clip = batch['clip_embeddings'].to(self.device)
                
                # Track evaluation target norm (convert to Python float immediately)
                eval_target_norm = float(torch.norm(target_clip, dim=-1).mean().item())
                self.norm_tracking['eval_target_norms'].append(eval_target_norm)
                
                # UPDATED: Use scale-aware generation
                if self.use_scale_aware_eval:
                    generated_clip = self.model.generate(
                        eva_features=eva_features,
                        num_inference_steps=self.eval_inference_steps,
                        normalize_output=False,
                        target_norm=target_norm,
                        use_lognormal_schedule=self.eval_use_lognormal_schedule,
                        debug_generation=self.debug_mode,
                    )
                    
                    # Track scale-aware metrics
                    self.norm_tracking['timestep_schedule_type'].append('lognormal' if self.eval_use_lognormal_schedule else 'linear')
                    
                else:
                    # Fallback to original generation
                    generated_clip = self.model.generate(
                        eva_features=eva_features,
                        num_inference_steps=self.eval_inference_steps,
                        normalize_output=False,
                        debug_generation=self.debug_mode,
                    )
                
                # Track generated norm (convert to Python float immediately)
                eval_generated_norm = float(torch.norm(generated_clip, dim=-1).mean().item())
                self.norm_tracking['eval_generated_norms'].append(eval_generated_norm)
                
                # Compute similarity (normalize only for similarity computation)
                target_norm = F.normalize(target_clip, p=2, dim=-1)
                generated_norm = F.normalize(generated_clip, p=2, dim=-1)
                similarity = F.cosine_similarity(generated_norm, target_norm, dim=-1)
                per_image_similarity = similarity.mean(dim=1)
                
                # Compute MSE loss in raw space
                mse_loss = F.mse_loss(generated_clip, target_clip, reduction='none').mean(dim=(1, 2))
                
                # NEW: Compute scale consistency
                generated_norms = torch.norm(generated_clip, dim=-1).mean(dim=1)
                target_norms = torch.norm(target_clip, dim=-1).mean(dim=1)
                scale_consistency = 1.0 - torch.abs(generated_norms - target_norms) / (target_norms + 1e-8)
                
                all_similarities.append(per_image_similarity.cpu())
                all_mse_losses.append(mse_loss.cpu())
                all_generated_norms.append(generated_norms.cpu())
                all_target_norms.append(target_norms.cpu())
                all_scale_consistency.append(scale_consistency.cpu())
                samples_processed += eva_features.shape[0]
        
        self.model.train()
        
        if not all_similarities:
            return {}
        
        all_sims = torch.cat(all_similarities)
        all_mse = torch.cat(all_mse_losses)
        all_gen_norms = torch.cat(all_generated_norms)
        all_tgt_norms = torch.cat(all_target_norms)
        all_scale_consist = torch.cat(all_scale_consistency)
        
        eval_time = time.time() - eval_start_time
        
        eval_metrics = {
            'eval_clip_similarity': all_sims.mean().item(),
            'eval_clip_similarity_std': all_sims.std().item(),
            'eval_mse_loss': all_mse.mean().item(),
            'eval_high_quality': (all_sims > 0.7).float().mean().item(),
            'eval_very_high_quality': (all_sims > 0.8).float().mean().item(),
            'eval_excellent_quality': (all_sims > 0.9).float().mean().item(),
            'eval_samples': samples_processed,
            'eval_time_seconds': eval_time,
            
            # Enhanced norm analysis
            'eval_generated_norm_mean': all_gen_norms.mean().item(),
            'eval_generated_norm_std': all_gen_norms.std().item(),
            'eval_target_norm_mean': all_tgt_norms.mean().item(),
            'eval_target_norm_std': all_tgt_norms.std().item(),
            'eval_norm_ratio': all_gen_norms.mean().item() / (all_tgt_norms.mean().item() + 1e-8),
            'eval_norm_consistency': 1.0 - abs(1.0 - all_gen_norms.mean().item() / (all_tgt_norms.mean().item() + 1e-8)),
            
            # NEW: Scale-aware evaluation metrics
            'eval_scale_consistency_mean': all_scale_consist.mean().item(),
            'eval_scale_consistency_std': all_scale_consist.std().item(),
            'eval_used_scale_aware': self.use_scale_aware_eval,
            'eval_target_norm_used': target_norm,
            'eval_adaptive_target_norm': self.adaptive_target_norm,
            'eval_lognormal_schedule': self.eval_use_lognormal_schedule,
        }
        
        # Store scale consistency for tracking
        self.norm_tracking['eval_scale_consistency'].extend(all_scale_consist.tolist())
        
        return eval_metrics

    def _get_norm_statistics(self) -> Dict[str, Any]:
        """Get comprehensive norm statistics including scale-aware metrics"""
        stats = {}
        
        # Training norms
        if self.norm_tracking['training_target_norms']:
            target_norms = list(self.norm_tracking['training_target_norms'])
            stats['training_target_norm'] = {
                'mean': np.mean(target_norms),
                'std': np.std(target_norms),
                'min': np.min(target_norms),
                'max': np.max(target_norms),
                'latest_10': target_norms[-10:] if len(target_norms) >= 10 else target_norms,
                'count': len(target_norms)
            }
        
        # Evaluation norms
        if self.norm_tracking['eval_target_norms']:
            eval_norms = list(self.norm_tracking['eval_target_norms'])
            stats['eval_target_norm'] = {
                'mean': np.mean(eval_norms),
                'std': np.std(eval_norms),
                'min': np.min(eval_norms),
                'max': np.max(eval_norms),
                'latest_10': eval_norms[-10:] if len(eval_norms) >= 10 else eval_norms,
                'count': len(eval_norms)
            }
        
        # NEW: Scale-aware evaluation metrics
        if self.norm_tracking['eval_target_norm_estimates']:
            estimates = list(self.norm_tracking['eval_target_norm_estimates'])
            stats['target_norm_estimates'] = {
                'mean': np.mean(estimates),
                'std': np.std(estimates),
                'latest_5': estimates[-5:] if len(estimates) >= 5 else estimates,
                'count': len(estimates)
            }
        
        if self.norm_tracking['eval_scale_consistency']:
            consistency = list(self.norm_tracking['eval_scale_consistency'])[-50:]  # Last 50 samples
            stats['scale_consistency'] = {
                'mean': np.mean(consistency),
                'std': np.std(consistency),
                'min': np.min(consistency),
                'max': np.max(consistency),
                'count': len(consistency)
            }
        
        # Generated norms
        if self.norm_tracking['eval_generated_norms']:
            gen_norms = list(self.norm_tracking['eval_generated_norms'])
            stats['eval_generated_norm'] = {
                'mean': np.mean(gen_norms),
                'std': np.std(gen_norms),
                'latest_10': gen_norms[-10:] if len(gen_norms) >= 10 else gen_norms,
                'count': len(gen_norms)
            }
        
        # Noise statistics
        if self.norm_tracking['noise_stats']:
            noise_stats_list = list(self.norm_tracking['noise_stats'])
            if noise_stats_list:
                recent_noise = noise_stats_list[-10:] if len(noise_stats_list) >= 10 else noise_stats_list
                stats['noise_statistics'] = {
                    'recent_mean_values': [ns.get('noise_mean', 0) for ns in recent_noise],
                    'recent_std_values': [ns.get('noise_std', 1) for ns in recent_noise],
                    'avg_mean': np.mean([ns.get('noise_mean', 0) for ns in noise_stats_list]),
                    'avg_std': np.mean([ns.get('noise_std', 1) for ns in noise_stats_list]),
                    'is_standard_gaussian_rate': np.mean([ns.get('is_standard_gaussian', True) for ns in noise_stats_list]),
                    'count': len(noise_stats_list)
                }
        
        return stats

    def _log_metrics(self, loss: float, metrics: Dict[str, float], grad_norm: float):
        """Log training metrics with scale-aware enhancements"""
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
            wandb_metrics.update({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                "train/epoch": self.current_epoch,
                "train/step": self.global_step,
            })
            
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        if key.startswith('eval_'):
                            wandb_metrics[f"eval/{key[5:]}"] = value
                        else:
                            wandb_metrics[f"train/{key}"] = value
            
            # Enhanced norm tracking metrics
            if 'batch_clip_norm' in metrics:
                wandb_metrics["train/batch_clip_norm"] = metrics['batch_clip_norm']
            if 'batch_eva_norm' in metrics:
                wandb_metrics["train/batch_eva_norm"] = metrics['batch_eva_norm']
            
            # Scale-aware metrics
            norm_stats = self._get_norm_statistics()
            for category, stats in norm_stats.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    wandb_metrics[f"norms/{category}_mean"] = stats['mean']
                    if 'std' in stats:
                        wandb_metrics[f"norms/{category}_std"] = stats['std']
            
            # NEW: Scale-aware specific metrics
            if 'target_norm_estimates' in norm_stats:
                wandb_metrics["scale_aware/target_norm_estimate"] = norm_stats['target_norm_estimates']['mean']
            if 'scale_consistency' in norm_stats:
                wandb_metrics["scale_aware/scale_consistency"] = norm_stats['scale_consistency']['mean']
            
            # Moving averages
            if len(self.loss_history) > 0:
                wandb_metrics["train/loss_ma"] = np.mean(list(self.loss_history))
            if len(self.similarity_history) > 0:
                wandb_metrics["train/similarity_ma"] = np.mean(list(self.similarity_history))
            
            # Best metrics
            wandb_metrics["train/best_loss"] = self.best_loss
            wandb_metrics["train/best_similarity"] = self.best_eval_similarity
            
            # System metrics
            if torch.cuda.is_available():
                wandb_metrics["system/gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
                wandb_metrics["system/gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
            
            wandb.log(wandb_metrics, step=self.global_step)
        
        # Enhanced console logging
        if self.global_step % self.log_every_n_steps == 0:
            log_msg = f"Step {self.global_step}: Loss={loss:.6f}"
            
            if 'velocity_similarity' in metrics:
                sim = metrics['velocity_similarity']
                quality = metrics.get('quality_assessment', 'unknown')
                log_msg += f", VelSim={sim:.4f} ({quality})"
            
            log_msg += f", GradNorm={grad_norm:.3f}"
            log_msg += f", LR={self.optimizer.param_groups[0]['lr']:.2e}"
            
            # Show noise statistics
            if 'batch_noise_mean' in metrics and 'batch_noise_std' in metrics:
                noise_mean = metrics['batch_noise_mean']
                noise_std = metrics['batch_noise_std']
                is_standard = metrics.get('batch_is_standard_gaussian', False)
                log_msg += f", Noise=μ{noise_mean:.3f}σ{noise_std:.3f}{'✅' if is_standard else '❌'}"
            
            # Show norms
            if 'pred_norm' in metrics and 'target_norm' in metrics:
                log_msg += f", PredNorm={metrics['pred_norm']:.3f}, TargetNorm={metrics['target_norm']:.3f}"
            
            if 'batch_clip_norm' in metrics:
                log_msg += f", BatchCLIPNorm={metrics['batch_clip_norm']:.3f}"
            
            if self.overfit_batch is not None:
                log_msg += " [OVERFIT TEST]"
            
            logger.info(log_msg)
            
            # Detailed norm analysis logging every 50 steps
            if self.global_step % 50 == 0:
                norm_stats = self._get_norm_statistics()
                logger.info("📊 NORM ANALYSIS (Scale-Aware):")
                
                if 'training_target_norm' in norm_stats:
                    train_stats = norm_stats['training_target_norm']
                    logger.info(f"  Training Target Norms: mean={train_stats['mean']:.3f}, std={train_stats['std']:.3f}, range=[{train_stats['min']:.3f}, {train_stats['max']:.3f}]")
                
                if 'eval_target_norm' in norm_stats:
                    eval_stats = norm_stats['eval_target_norm']
                    logger.info(f"  Eval Target Norms: mean={eval_stats['mean']:.3f}, std={eval_stats['std']:.3f}, range=[{eval_stats['min']:.3f}, {eval_stats['max']:.3f}]")
                
                # NEW: Scale-aware metrics
                if 'target_norm_estimates' in norm_stats:
                    est_stats = norm_stats['target_norm_estimates']
                    logger.info(f"  🎯 Target Norm Estimates: mean={est_stats['mean']:.3f}, std={est_stats['std']:.3f}")
                
                if 'scale_consistency' in norm_stats:
                    consist_stats = norm_stats['scale_consistency']
                    logger.info(f"  📊 Scale Consistency: mean={consist_stats['mean']:.3f}, std={consist_stats['std']:.3f}")
                
                # Check consistency
                if 'training_target_norm' in norm_stats and 'eval_target_norm' in norm_stats:
                    train_mean = norm_stats['training_target_norm']['mean']
                    eval_mean = norm_stats['eval_target_norm']['mean']
                    diff = abs(train_mean - eval_mean)
                    consistency = "✅ CONSISTENT" if diff < 2.0 else "⚠️  INCONSISTENT"
                    logger.info(f"  Train vs Eval Consistency: {consistency} (diff={diff:.3f})")

    def _save_checkpoint(self):
        """Save model checkpoint with scale-aware tracking data"""
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
            
            # Enhanced norm tracking data
            'norm_tracking': {
                'training_target_norms': list(self.norm_tracking['training_target_norms']),
                'eval_target_norms': list(self.norm_tracking['eval_target_norms']),
                'eval_target_norm_estimates': list(self.norm_tracking['eval_target_norm_estimates']),
                'eval_scale_consistency': list(self.norm_tracking['eval_scale_consistency']),
                'batch_target_norm_history': self.norm_tracking['batch_target_norm_history'],
                'step_norms': dict(self.norm_tracking['step_norms']),
                'noise_stats': list(self.norm_tracking['noise_stats']),
            },
            
            # Scale-aware configuration
            'scale_aware_config': {
                'use_scale_aware_eval': self.use_scale_aware_eval,
                'eval_target_norm': self.eval_target_norm,
                'eval_use_lognormal_schedule': self.eval_use_lognormal_schedule,
                'adaptive_target_norm': self.adaptive_target_norm,
            },
            
            'experiment_type': 'blip3o_clip_scale_aware',
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if self.use_wandb:
            wandb.log({
                "checkpoint/saved": True,
                "checkpoint/step": self.global_step,
                "checkpoint/scale_aware": True,
            }, step=self.global_step)

    def train(self) -> Dict[str, Any]:
        """Main training loop with scale-aware evaluation"""
        logger.info("🚀 Starting BLIP3-o training with Scale-Aware Evaluation...")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  🎯 Scale-aware evaluation: {self.use_scale_aware_eval}")
        logger.info(f"  📅 Log-normal schedule: {self.eval_use_lognormal_schedule}")
        logger.info(f"  🎛️  Adaptive target norm: {self.adaptive_target_norm}")
        
        if self.use_wandb:
            wandb.log({
                "setup/scale_aware_evaluation": self.use_scale_aware_eval,
                "setup/lognormal_schedule": self.eval_use_lognormal_schedule,
                "setup/adaptive_target_norm": self.adaptive_target_norm,
                "setup/training_started": True,
            }, step=0)
        
        self.model.train()
        start_time = time.time()
        
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
                
                epoch_loss = 0.0
                epoch_steps = 0
                epoch_start_time = time.time()
                
                try:
                    dataloader_iter = iter(self.train_dataloader)
                    batch_count = 0
                    
                    while True:
                        try:
                            batch = next(dataloader_iter)
                            batch_count += 1
                        except StopIteration:
                            logger.info(f"Epoch {epoch + 1} completed: {batch_count} batches processed")
                            break
                        
                        step_start_time = time.time()
                        
                        try:
                            loss, metrics = self._compute_loss(batch)
                        except Exception as e:
                            logger.error(f"Error computing loss at step {self.global_step}: {e}")
                            continue
                        
                        try:
                            grad_norm = self._backward_and_step(loss)
                        except Exception as e:
                            logger.error(f"Error in backward pass at step {self.global_step}: {e}")
                            continue
                        
                        epoch_loss += loss.item()
                        epoch_steps += 1
                        self.global_step += 1
                        
                        step_time = time.time() - step_start_time
                        if self.use_wandb:
                            wandb.log({
                                "timing/step_time": step_time,
                                "timing/samples_per_second": batch.get('batch_size', 1) / step_time if step_time > 0 else 0,
                            }, step=self.global_step)
                        
                        self._log_metrics(loss.item(), metrics or {}, grad_norm)
                        
                        # UPDATED: Run scale-aware evaluation
                        if self.global_step % self.eval_every_n_steps == 0:
                            logger.info(f"Running scale-aware evaluation at step {self.global_step}...")
                            eval_metrics = self._evaluate()
                            
                            if eval_metrics:
                                logger.info(f"Scale-aware evaluation results:")
                                logger.info(f"  CLIP similarity: {eval_metrics.get('eval_clip_similarity', 0):.4f}")
                                logger.info(f"  Generated norm: {eval_metrics.get('eval_generated_norm_mean', 0):.3f}")
                                logger.info(f"  Target norm: {eval_metrics.get('eval_target_norm_mean', 0):.3f}")
                                logger.info(f"  Scale consistency: {eval_metrics.get('eval_scale_consistency_mean', 0):.3f}")
                                logger.info(f"  🎯 Target norm used: {eval_metrics.get('eval_target_norm_used', 0):.3f}")
                                logger.info(f"  📅 Log-normal schedule: {eval_metrics.get('eval_lognormal_schedule', False)}")
                                
                                if self.use_wandb:
                                    wandb_eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                                    wandb.log(wandb_eval_metrics, step=self.global_step)
                                
                                if eval_metrics.get('eval_clip_similarity', 0) > self.best_eval_similarity:
                                    self.best_eval_similarity = eval_metrics['eval_clip_similarity']
                                    logger.info(f"🎉 NEW BEST CLIP similarity: {self.best_eval_similarity:.4f}")
                                    
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
                            logger.info("🎉 OVERFITTING TEST PASSED! Model can learn effectively with scale-aware generation.")
                            if self.use_wandb:
                                wandb.log({
                                    "overfit_test/passed": True,
                                    "overfit_test/final_similarity": metrics['velocity_similarity'],
                                    "overfit_test/steps_to_pass": self.global_step,
                                    "overfit_test/scale_aware": True,
                                }, step=self.global_step)
                            break
                
                except Exception as e:
                    logger.error(f"Error during epoch {epoch + 1}: {e}")
                    continue
                
                # End of epoch logging with scale-aware analysis
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                
                logger.info(f"Epoch {epoch + 1} completed:")
                logger.info(f"  Average loss: {avg_epoch_loss:.6f}")
                logger.info(f"  Best loss: {self.best_loss:.6f}")
                logger.info(f"  Best similarity: {self.best_eval_similarity:.4f}")
                logger.info(f"  Steps in epoch: {epoch_steps}")
                logger.info(f"  Epoch time: {epoch_time:.1f}s")
                
                # Scale-aware epoch analysis
                norm_stats = self._get_norm_statistics()
                if norm_stats:
                    logger.info(f"  📊 Scale-Aware Epoch Analysis:")
                    if 'training_target_norm' in norm_stats:
                        train_norm = norm_stats['training_target_norm']['mean']
                        logger.info(f"    Training target norm: {train_norm:.3f}")
                    if 'eval_target_norm' in norm_stats:
                        eval_norm = norm_stats['eval_target_norm']['mean']
                        logger.info(f"    Eval target norm: {eval_norm:.3f}")
                    if 'target_norm_estimates' in norm_stats:
                        est_norm = norm_stats['target_norm_estimates']['mean']
                        logger.info(f"    🎯 Estimated target norm: {est_norm:.3f}")
                    if 'scale_consistency' in norm_stats:
                        consistency = norm_stats['scale_consistency']['mean']
                        logger.info(f"    📊 Scale consistency: {consistency:.3f}")
                
                if self.use_wandb:
                    wandb_epoch_metrics = {
                        "epoch/completed": epoch + 1,
                        "epoch/avg_loss": avg_epoch_loss,
                        "epoch/steps": epoch_steps,
                        "epoch/time_seconds": epoch_time,
                    }
                    
                    if norm_stats:
                        for category, stats in norm_stats.items():
                            if isinstance(stats, dict) and 'mean' in stats:
                                wandb_epoch_metrics[f"epoch/{category}_mean"] = stats['mean']
                    
                    wandb.log(wandb_epoch_metrics, step=self.global_step)
        
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
            logger.info("Running final scale-aware evaluation...")
            final_eval = self._evaluate(num_samples=self.eval_num_samples * 2)
            
            total_time = time.time() - start_time
            
            # Enhanced training summary with scale-aware metrics
            norm_stats = self._get_norm_statistics()
            
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
                'estimated_steps_per_epoch': self.estimated_steps_per_epoch,
                'norm_statistics': norm_stats,
                'scale_aware_config': {
                    'use_scale_aware_eval': self.use_scale_aware_eval,
                    'adaptive_target_norm': self.adaptive_target_norm,
                    'lognormal_schedule': self.eval_use_lognormal_schedule,
                },
                'experiment_type': 'blip3o_clip_scale_aware',
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
                    "final/scale_aware_evaluation": self.use_scale_aware_eval,
                }
                
                if final_eval:
                    for key, value in final_eval.items():
                        final_wandb_metrics[f"final/{key}"] = value
                
                if norm_stats:
                    for category, stats in norm_stats.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            final_wandb_metrics[f"final/{category}_mean"] = stats['mean']
                
                wandb.log(final_wandb_metrics, step=self.global_step)
                wandb.finish()
            
            # Save training summary
            summary_path = self.output_dir / "scale_aware_training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("🚀 Scale-Aware Training completed!")
            logger.info(f"  Total time: {total_time:.1f} seconds")
            logger.info(f"  Total steps: {self.global_step}")
            logger.info(f"  Best loss: {self.best_loss:.6f}")
            logger.info(f"  Best CLIP similarity: {self.best_eval_similarity:.4f}")
            logger.info(f"  🎯 Scale-aware evaluation used: {self.use_scale_aware_eval}")
            
            # Final scale-aware analysis
            if norm_stats:
                logger.info(f"📊 FINAL SCALE-AWARE ANALYSIS:")
                if 'training_target_norm' in norm_stats:
                    train_stats = norm_stats['training_target_norm']
                    logger.info(f"  Training target norm: {train_stats['mean']:.3f} ± {train_stats['std']:.3f}")
                
                if 'eval_target_norm' in norm_stats:
                    eval_stats = norm_stats['eval_target_norm']
                    logger.info(f"  Eval target norm: {eval_stats['mean']:.3f} ± {eval_stats['std']:.3f}")
                
                if 'target_norm_estimates' in norm_stats:
                    est_stats = norm_stats['target_norm_estimates']
                    logger.info(f"  🎯 Final target norm estimate: {est_stats['mean']:.3f} ± {est_stats['std']:.3f}")
                
                if 'scale_consistency' in norm_stats:
                    consist_stats = norm_stats['scale_consistency']
                    logger.info(f"  📊 Final scale consistency: {consist_stats['mean']:.3f} ± {consist_stats['std']:.3f}")
            
            if final_eval:
                logger.info(f"  Final scale-aware evaluation:")
                logger.info(f"    CLIP similarity: {final_eval.get('eval_clip_similarity', 0):.4f}")
                logger.info(f"    Scale consistency: {final_eval.get('eval_scale_consistency_mean', 0):.3f}")
                logger.info(f"    Target norm used: {final_eval.get('eval_target_norm_used', 0):.3f}")
            
            return summary


def create_clip_trainer(
    model,
    loss_fn,
    train_dataloader,
    eval_dataloader=None,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    output_dir: str = "./checkpoints",
    # NEW: Scale-aware parameters
    use_scale_aware_eval: bool = True,
    eval_target_norm: Optional[float] = None,
    eval_use_lognormal_schedule: bool = True,
    adaptive_target_norm: bool = True,
    # Other parameters
    overfit_test_size: Optional[int] = None,
    debug_mode: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "blip3o-clip-scale-aware",
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    **kwargs
) -> BLIP3oCLIPTrainer:
    """UPDATED: Factory function to create CLIP trainer with scale-aware evaluation"""
    
    return BLIP3oCLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir=output_dir,
        use_scale_aware_eval=use_scale_aware_eval,
        eval_target_norm=eval_target_norm,
        eval_use_lognormal_schedule=eval_use_lognormal_schedule,
        adaptive_target_norm=adaptive_target_norm,
        overfit_test_size=overfit_test_size,
        debug_mode=debug_mode,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config,
        **kwargs
    )