#!/usr/bin/env python3
"""
FIXED: BLIP3-o Training Only Trainer - No Evaluation During Training
src/modules/trainers/blip3o_training_only_trainer.py

FIXES:
1. Fixed eval_strategy parameter (was evaluation_strategy)
2. Proper gradient flow handling
3. Clean training metrics reporting
4. No evaluation during training
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, Union, Tuple
import logging
import numpy as np
import json
import time
from pathlib import Path
import gc
import os

logger = logging.getLogger(__name__)


class BLIP3oTrainingOnlyTrainer(Trainer):
    """
    FIXED: BLIP3-o Trainer that ONLY does training - No evaluation during training
    
    Reports:
    - Training loss
    - Learning rate
    - Training metrics
    - Gradient norms
    - Training speed
    """
    
    def __init__(
        self,
        model,
        args: TrainingArguments,
        flow_matching_loss,
        train_dataset=None,
        training_mode: str = "patch_only",
        **kwargs
    ):
        # Remove evaluation-related arguments
        kwargs.pop('eval_dataset', None)
        kwargs.pop('compute_metrics', None)
        
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=None,  # No evaluation dataset
            compute_metrics=None,  # No metrics computation
            **kwargs
        )
        
        self.flow_matching_loss = flow_matching_loss
        self.training_mode = training_mode
        
        # Expected token count based on mode
        self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        # Training metrics tracking
        self.training_step_count = 0
        self.loss_history = []
        self.lr_history = []
        self.training_speed_history = []
        
        # Distributed training setup
        self.is_distributed = dist.is_initialized()
        self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        
        if self.is_main_process:
            logger.info("✅ FIXED BLIP3-o Training-Only Trainer initialized")
            logger.info(f"🎯 Training mode: {self.training_mode} ({self.expected_tokens} tokens)")
            logger.info(f"🚫 Evaluation during training: DISABLED")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """FIXED: Compute training loss with proper gradient handling"""
        model.train()
        
        # Start timing
        step_start_time = time.time()
        
        # Extract inputs with proper error handling
        try:
            eva_embeddings = inputs['encoder_hidden_states']
            clip_embeddings = inputs['clip_embeddings']
            timesteps = inputs['timestep']
        except KeyError as e:
            logger.error(f"Missing input key: {e}")
            logger.error(f"Available keys: {list(inputs.keys())}")
            raise
        
        # Handle multiprocessing-safe inputs (FIXED gradient flow)
        if 'hidden_states' in inputs:
            noisy_clip_base = inputs['hidden_states']
            noise = inputs.get('noise', torch.randn_like(clip_embeddings))
        else:
            # Create noisy input here
            device = eva_embeddings.device
            noise = torch.randn_like(clip_embeddings)
            alpha = timesteps.view(-1, 1, 1)
            noisy_clip_base = (1 - alpha) * noise + alpha * clip_embeddings.detach()

        # CRITICAL FIX: Add gradients to noisy input for training
        if not noisy_clip_base.requires_grad:
            noisy_clip = noisy_clip_base.detach().requires_grad_(True)
        else:
            noisy_clip = noisy_clip_base
        
        # Validate shapes
        batch_size, seq_len, eva_dim = eva_embeddings.shape
        assert seq_len == self.expected_tokens, f"Expected {self.expected_tokens} tokens, got {seq_len}"
        assert eva_dim == 4096, f"Expected EVA 4096-dim, got {eva_dim}"
        assert clip_embeddings.shape[2] == 1024, f"Expected CLIP 1024-dim, got {clip_embeddings.shape[2]}"
        
        # Forward pass
        try:
            model_outputs = model(
                hidden_states=noisy_clip,
                timestep=timesteps,
                encoder_hidden_states=eva_embeddings,
                return_dict=True
            )
        except Exception as e:
            logger.error(f"Model forward pass failed: {e}")
            raise
        
        # Extract velocity prediction
        if isinstance(model_outputs, dict):
            velocity_pred = model_outputs.get('velocity_prediction', 
                                            model_outputs.get('last_hidden_state'))
        else:
            velocity_pred = model_outputs
        
        # Validate model output
        if velocity_pred is None:
            raise ValueError("Model output is None")
        
        # During training, model output should have gradients
        if not velocity_pred.requires_grad:
            raise RuntimeError("Model output doesn't require gradients during training!")
        
        # Compute flow matching loss
        try:
            loss, metrics = self.flow_matching_loss(
                model_output=velocity_pred,
                target_samples=clip_embeddings,
                timesteps=timesteps,
                eva_conditioning=eva_embeddings,
                noise=noise,
                return_metrics=True
            )
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            raise
        
        # Track training metrics
        step_time = time.time() - step_start_time
        
        if self.is_main_process:
            # Get current learning rate
            current_lr = self.get_lr()
            
            # Compute gradient norm
            grad_norm = self._compute_grad_norm()
            
            # Store metrics
            self.loss_history.append(loss.item())
            if isinstance(current_lr, list):
                self.lr_history.append(current_lr[0] if current_lr else 0.0)
            else:
                self.lr_history.append(float(current_lr))
            self.training_speed_history.append(batch_size / step_time if step_time > 0 else 0)
            
            # Log detailed progress
            if self.training_step_count % self.args.logging_steps == 0:
                self._log_training_progress(loss, metrics, current_lr, grad_norm, step_time, batch_size)
        
        self.training_step_count += 1
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'velocity_prediction': velocity_pred,
            'target_samples': clip_embeddings,
            'training_metrics': metrics,
            'step_time': step_time,
        } if return_outputs else None
        
        return (loss, outputs) if return_outputs else loss

    def _log_training_progress(
        self,
        loss: torch.Tensor,
        metrics: Optional[Dict[str, float]],
        learning_rate: float,
        grad_norm: float,
        step_time: float,
        batch_size: int
    ):
        """Log detailed training progress - training metrics only"""
        if not self.is_main_process:
            return
        
        loss_value = loss.item()
        samples_per_sec = batch_size / step_time if step_time > 0 else 0
        
        # Basic training info
        if isinstance(learning_rate, list):
            lr_val = learning_rate[0] if learning_rate else 0.0
        else:
            lr_val = float(learning_rate)
            
        progress_msg = (
            f"Step {self.training_step_count}: "
            f"Loss={loss_value:.6f}, "
            f"LR={lr_val:.2e}, "
            f"Speed={samples_per_sec:.1f} samples/s"
        )
        
        if grad_norm > 0:
            progress_msg += f", GradNorm={grad_norm:.3f}"
        
        # Add flow matching specific metrics
        if metrics:
            if 'velocity_cosine_sim' in metrics:
                progress_msg += f", VelCos={metrics['velocity_cosine_sim']:.3f}"
            if 'prediction_norm' in metrics:
                progress_msg += f", PredNorm={metrics['prediction_norm']:.3f}"
            if 'training_quality' in metrics:
                progress_msg += f", Quality={metrics['training_quality']}"
        
        # Add training mode info
        progress_msg += f" [{self.training_mode}]"
        
        logger.info(progress_msg)
        
        # Log additional metrics every N steps
        if self.training_step_count % (self.args.logging_steps * 5) == 0:
            self._log_extended_metrics(metrics)

    def _log_extended_metrics(self, metrics: Optional[Dict[str, float]]):
        """Log extended training metrics"""
        if not metrics or not self.is_main_process:
            return
        
        logger.info("📊 Extended Training Metrics:")
        
        # Flow matching metrics
        if 'flow_matching_loss' in metrics:
            logger.info(f"   Flow Matching Loss: {metrics['flow_matching_loss']:.6f}")
        
        # Prediction quality
        if 'prediction_norm' in metrics and 'target_norm' in metrics:
            logger.info(f"   Prediction Norm: {metrics['prediction_norm']:.3f}")
            logger.info(f"   Target Norm: {metrics['target_norm']:.3f}")
        
        # Training quality assessment
        if 'training_quality' in metrics:
            logger.info(f"   Training Quality: {metrics['training_quality']}")
        
        # Token-specific metrics
        if 'num_tokens' in metrics:
            logger.info(f"   Token Count: {metrics['num_tokens']} ({metrics.get('mode', 'unknown')})")
        
        # Recent loss trend
        if len(self.loss_history) >= 10:
            recent_losses = self.loss_history[-10:]
            loss_trend = "decreasing" if recent_losses[-1] < recent_losses[0] else "increasing"
            logger.info(f"   Loss Trend (last 10): {loss_trend}")
            logger.info(f"   Loss Range: {min(recent_losses):.6f} - {max(recent_losses):.6f}")

    def get_lr(self):
        """Get current learning rate"""
        try:
            if self.lr_scheduler is None:
                return [group['lr'] for group in self.optimizer.param_groups]
            return self.lr_scheduler.get_last_lr()
        except:
            return [0.0]

    def _compute_grad_norm(self) -> float:
        """Compute gradient norm for monitoring"""
        try:
            total_norm = 0.0
            param_count = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                return total_norm
            return 0.0
        except:
            return 0.0

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            'training_mode': self.training_mode,
            'expected_tokens': self.expected_tokens,
            'total_steps': self.training_step_count,
            'evaluation_during_training': False,
        }
        
        if self.loss_history:
            stats['loss_statistics'] = {
                'current_loss': self.loss_history[-1],
                'min_loss': min(self.loss_history),
                'max_loss': max(self.loss_history),
                'avg_loss': sum(self.loss_history) / len(self.loss_history),
                'loss_trend': 'decreasing' if len(self.loss_history) > 10 and self.loss_history[-1] < self.loss_history[-11] else 'stable'
            }
        
        if self.lr_history:
            stats['learning_rate_statistics'] = {
                'current_lr': self.lr_history[-1],
                'max_lr': max(self.lr_history),
                'min_lr': min(self.lr_history),
            }
        
        if self.training_speed_history:
            stats['speed_statistics'] = {
                'current_speed_samples_per_sec': self.training_speed_history[-1],
                'avg_speed_samples_per_sec': sum(self.training_speed_history) / len(self.training_speed_history),
                'max_speed_samples_per_sec': max(self.training_speed_history),
            }
        
        return stats

    def log_training_summary(self):
        """Log training summary"""
        if not self.is_main_process:
            return
        
        stats = self.get_training_statistics()
        
        logger.info("=" * 60)
        logger.info("📊 TRAINING SUMMARY (No Evaluation)")
        logger.info("=" * 60)
        logger.info(f"🎯 Training Mode: {self.training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"📈 Total Steps: {self.training_step_count}")
        
        if 'loss_statistics' in stats:
            loss_stats = stats['loss_statistics']
            logger.info(f"📉 Loss - Current: {loss_stats['current_loss']:.6f}, Min: {loss_stats['min_loss']:.6f}")
            logger.info(f"📊 Loss Trend: {loss_stats['loss_trend'].upper()}")
        
        if 'learning_rate_statistics' in stats:
            lr_stats = stats['learning_rate_statistics']
            logger.info(f"📚 Learning Rate: {lr_stats['current_lr']:.2e}")
        
        if 'speed_statistics' in stats:
            speed_stats = stats['speed_statistics']
            logger.info(f"⚡ Training Speed: {speed_stats['avg_speed_samples_per_sec']:.1f} samples/sec")
        
        logger.info("🚫 Evaluation: Disabled during training")
        logger.info("✅ Ready for post-training evaluation")
        logger.info("=" * 60)

    # Override evaluation methods to disable them
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override to disable evaluation"""
        logger.warning("🚫 Evaluation disabled during training")
        return {}

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to disable prediction steps"""
        logger.warning("🚫 Prediction steps disabled during training")
        return None


def create_training_only_args(
    output_dir: str,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 5e-5,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 200,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    dataloader_num_workers: int = 0,  # FIXED: Default to 0 for stability
    **kwargs
) -> TrainingArguments:
    """FIXED: Create training arguments with NO EVALUATION - Fixed parameter names"""
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        
        # FIXED: Use correct parameter names for newer transformers
        eval_strategy="no",  # FIXED: was evaluation_strategy
        eval_steps=None,  # No evaluation steps
        per_device_eval_batch_size=None,  # No eval batch size
        eval_accumulation_steps=None,  # No eval accumulation
        
        save_strategy="steps",
        remove_unused_columns=False,
        load_best_model_at_end=False,  # Cannot load best model without evaluation
        metric_for_best_model=None,  # No metrics
        greater_is_better=None,  # No comparison
        save_total_limit=3,
        prediction_loss_only=True,  # Only compute loss, no other metrics
        report_to=[],  # No reporting to wandb/tensorboard
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        ignore_data_skip=True,
        **kwargs
    )