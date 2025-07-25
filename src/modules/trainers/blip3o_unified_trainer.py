#!/usr/bin/env python3
"""
UNIFIED BLIP3-o Trainer - Handles Training-Only and Training+Evaluation
src/modules/trainers/blip3o_unified_trainer.py

FEATURES:
- Single trainer that can do training-only OR training+evaluation
- All scaling fixes applied
- Proper gradient flow handling
- Comprehensive metrics reporting
- Overfitting test support
- Production training support
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import numpy as np
from collections import defaultdict
import json
import time
from pathlib import Path
import traceback
import gc
import os

logger = logging.getLogger(__name__)


class BLIP3oUnifiedTrainer(Trainer):
    """
    UNIFIED BLIP3-o Trainer - Handles Everything with All Fixes Applied
    
    Features:
    - Training-only mode (no evaluation during training)
    - Training+evaluation mode (periodic evaluation)
    - All scaling fixes applied
    - Proper gradient flow
    - Comprehensive metrics
    - Overfitting test support
    """
    
    def __init__(
        self,
        model,
        args: TrainingArguments,
        flow_matching_loss,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        
        # BLIP3-o specific parameters
        training_mode: str = "patch_only",
        enable_evaluation: bool = False,  # NEW: Control evaluation
        enable_same_data_eval: bool = False,
        eval_frequency: int = 100,
        detailed_logging: bool = True,
        
        # Scaling parameters (for monitoring)
        expected_velocity_scale: float = 0.1,
        expected_output_scale: float = 0.1,
        
        **kwargs
    ):
        
        # Set up evaluation dataset based on mode
        if not enable_evaluation:
            eval_dataset = None
            compute_metrics = None
        
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        
        self.flow_matching_loss = flow_matching_loss
        self.training_mode = training_mode
        self.enable_evaluation = enable_evaluation
        self.enable_same_data_eval = enable_same_data_eval
        self.eval_frequency = eval_frequency
        self.detailed_logging = detailed_logging
        
        # Expected scaling parameters for monitoring
        self.expected_velocity_scale = expected_velocity_scale
        self.expected_output_scale = expected_output_scale
        
        # Expected token count based on mode
        self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        # Training metrics tracking
        self.training_step_count = 0
        self.loss_history = []
        self.metric_history = []
        self.eval_history = []
        self.lr_history = []
        self.training_speed_history = []
        
        # Distributed training setup
        self.is_distributed = dist.is_initialized()
        self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        
        # Scaling monitoring
        self.norm_mismatch_warnings = 0
        self.scaling_issues_detected = []
        
        if self.is_main_process:
            logger.info("✅ UNIFIED BLIP3-o Trainer initialized with ALL FIXES")
            logger.info(f"🎯 Training mode: {self.training_mode} ({self.expected_tokens} tokens)")
            logger.info(f"📊 Evaluation enabled: {self.enable_evaluation}")
            logger.info(f"🔧 Expected velocity scale: {self.expected_velocity_scale}")
            logger.info(f"🔧 Expected output scale: {self.expected_output_scale}")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """UNIFIED compute_loss with all fixes and monitoring"""
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
        
        # Compute flow matching loss with all fixes
        try:
            loss, metrics = self.flow_matching_loss(
                model_output=velocity_pred,
                target_samples=clip_embeddings,
                timesteps=timesteps,
                eva_conditioning=eva_embeddings,
                noise=noise,
                return_metrics=True,
                training_mode=self.training_mode,
            )
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            raise
        
        # Monitor scaling issues
        if metrics and self.is_main_process:
            self._monitor_scaling_issues(metrics)
        
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
            
            if metrics:
                self.metric_history.append({
                    **metrics,
                    'step': self.training_step_count,
                    'timestamp': time.time(),
                    'learning_rate': current_lr,
                    'grad_norm': grad_norm,
                    'step_time': step_time,
                })
            
            # Log detailed progress
            if (self.detailed_logging and 
                self.training_step_count % self.args.logging_steps == 0):
                self._log_detailed_progress(loss, metrics, current_lr, grad_norm, step_time, batch_size)
        
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

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        UNIFIED evaluation loop - only runs if evaluation is enabled
        """
        if not self.enable_evaluation:
            logger.info("🚫 Evaluation disabled - skipping")
            return EvalLoopOutput(
                predictions=None,
                label_ids=None,
                metrics={f"{metric_key_prefix}_loss": 0.0},
                num_samples=0,
            )
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        batch_size = dataloader.batch_size
        num_samples = self.num_examples(dataloader)
        
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_samples}")
        logger.info(f"  Batch size = {batch_size}")
        
        all_losses = []
        all_metrics = []
        
        for step, inputs in enumerate(dataloader):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():  # CRITICAL: This disables gradients for evaluation
                # Compute loss using the same logic as training but without gradients
                eva_embeddings = inputs['encoder_hidden_states']
                clip_embeddings = inputs['clip_embeddings']
                timesteps = inputs['timestep']
                
                # FIXED: Create noisy input for evaluation (no gradients needed)
                if 'hidden_states' in inputs:
                    noisy_clip = inputs['hidden_states'].detach()  # Ensure no gradients
                else:
                    device = eva_embeddings.device
                    noise = torch.randn_like(clip_embeddings)
                    alpha = timesteps.view(-1, 1, 1)
                    noisy_clip = (1 - alpha) * noise + alpha * clip_embeddings.detach()
                    noisy_clip = noisy_clip.detach()  # Ensure no gradients
                
                # Forward pass (no gradients)
                model_outputs = model(
                    hidden_states=noisy_clip,
                    timestep=timesteps,
                    encoder_hidden_states=eva_embeddings,
                    return_dict=True
                )
                
                velocity_pred = model_outputs.get('velocity_prediction', 
                                                model_outputs.get('last_hidden_state'))
                
                # FIXED: Loss computation in evaluation mode (no gradients expected)
                loss, metrics = self.flow_matching_loss(
                    model_output=velocity_pred,  # This won't have gradients in eval mode
                    target_samples=clip_embeddings,
                    timesteps=timesteps,
                    eva_conditioning=eva_embeddings,
                    noise=inputs.get('noise', torch.randn_like(clip_embeddings)),
                    return_metrics=True,
                    training_mode=self.training_mode,
                )
                
                all_losses.append(loss.item())
                if metrics:
                    # Add step info to metrics
                    metrics['eval_step'] = step
                    all_metrics.append(metrics)
        
        # Compute average metrics
        avg_loss = np.mean(all_losses)
        
        eval_metrics = {
            f"{metric_key_prefix}_loss": avg_loss,  # This is the key fix!
            f"{metric_key_prefix}_runtime": len(all_losses) * batch_size,
            f"{metric_key_prefix}_samples_per_second": len(all_losses) * batch_size / max(1, len(all_losses)),
            f"{metric_key_prefix}_steps_per_second": len(all_losses) / max(1, len(all_losses)),
        }
        
        # Add detailed metrics if available
        if all_metrics:
            for key in all_metrics[0].keys():
                if isinstance(all_metrics[0][key], (int, float)) and key != 'eval_step':
                    values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
                    if values:  # Only add if we have values
                        eval_metrics[f"{metric_key_prefix}_{key}"] = np.mean(values)
        
        logger.info(f"***** {description} results *****")
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key} = {value:.6f}" if key.endswith('_loss') else f"  {key} = {value}")
        
        # Store evaluation results for analysis
        if self.is_main_process:
            eval_result = {
                'step': self.training_step_count,
                'metrics': eval_metrics,
                'detailed_metrics': all_metrics[-3:] if all_metrics else [],  # Keep last 3 for analysis
                'timestamp': time.time(),
            }
            self.eval_history.append(eval_result)
            
            # Log overfitting progress
            if 'eval_final_embedding_similarity' in eval_metrics:
                global_cos = eval_metrics['eval_final_embedding_similarity']
                if global_cos and global_cos > 0.8:
                    logger.info("🎉 EXCELLENT OVERFITTING: Model successfully learning training data!")
                elif global_cos and global_cos > 0.6:
                    logger.info("✅ GOOD OVERFITTING: Strong same-data performance")
                elif global_cos and global_cos > 0.4:
                    logger.info("🔄 MODERATE OVERFITTING: Some learning detected")
                else:
                    logger.info("⚠️ LOW OVERFITTING: Model needs more training")
        
        # CRITICAL FIX: Return EvalLoopOutput object instead of dict
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=eval_metrics,
            num_samples=num_samples,
        )

    def _monitor_scaling_issues(self, metrics: Dict[str, float]):
        """Monitor for scaling issues and log warnings"""
        if not metrics:
            return
        
        # Check norm ratios
        norm_ratio = metrics.get('norm_ratio', 1.0)
        if norm_ratio > 5.0 or norm_ratio < 0.2:
            self.norm_mismatch_warnings += 1
            if self.norm_mismatch_warnings <= 5:  # Only warn first few times
                logger.warning(f"⚠️ Norm mismatch detected: ratio = {norm_ratio:.2f}")
                logger.warning(f"   Prediction norm: {metrics.get('prediction_norm', 'unknown'):.3f}")
                logger.warning(f"   Target norm: {metrics.get('target_norm', 'unknown'):.3f}")
                if norm_ratio > 5.0:
                    logger.warning("   Consider reducing velocity_scale or increasing output_scale")
                else:
                    logger.warning("   Consider increasing velocity_scale or reducing output_scale")
        
        # Check velocity scale
        if hasattr(self.flow_matching_loss, 'velocity_scale'):
            actual_velocity_scale = self.flow_matching_loss.velocity_scale
            if abs(actual_velocity_scale - self.expected_velocity_scale) > 0.01:
                warning = f"Velocity scale mismatch: expected {self.expected_velocity_scale}, got {actual_velocity_scale}"
                if warning not in self.scaling_issues_detected:
                    self.scaling_issues_detected.append(warning)
                    logger.warning(f"⚠️ {warning}")
        
        # Check adaptive scaling
        adaptive_scale = metrics.get('adaptive_scale', 1.0)
        if adaptive_scale > 2.0 or adaptive_scale < 0.5:
            warning = f"Adaptive scale out of range: {adaptive_scale:.3f}"
            if warning not in self.scaling_issues_detected:
                self.scaling_issues_detected.append(warning)
                logger.warning(f"⚠️ {warning}")

    def _log_detailed_progress(
        self,
        loss: torch.Tensor,
        metrics: Optional[Dict[str, float]],
        learning_rate: float,
        grad_norm: float,
        step_time: float,
        batch_size: int
    ):
        """Log detailed training progress with scaling monitoring"""
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
        
        # Add scaling-specific metrics
        if metrics:
            if 'velocity_cosine_sim' in metrics:
                progress_msg += f", VelCos={metrics['velocity_cosine_sim']:.3f}"
            if 'prediction_norm' in metrics:
                progress_msg += f", PredNorm={metrics['prediction_norm']:.3f}"
            if 'target_norm' in metrics:
                progress_msg += f", TargetNorm={metrics['target_norm']:.3f}"
            if 'adaptive_scale' in metrics:
                progress_msg += f", AdaptScale={metrics['adaptive_scale']:.3f}"
            if 'final_embedding_similarity' in metrics and metrics['final_embedding_similarity']:
                progress_msg += f", FinalSim={metrics['final_embedding_similarity']:.3f}"
            if 'training_quality' in metrics:
                progress_msg += f", Quality={metrics['training_quality']}"
        
        # Add training mode info
        progress_msg += f" [{self.training_mode}]"
        
        logger.info(progress_msg)
        
        # Log extended metrics every N steps
        if self.training_step_count % (self.args.logging_steps * 5) == 0:
            self._log_extended_metrics(metrics)

    def _log_extended_metrics(self, metrics: Optional[Dict[str, float]]):
        """Log extended training metrics with scaling focus"""
        if not metrics or not self.is_main_process:
            return
        
        logger.info("📊 Extended Training Metrics:")
        
        # Flow matching metrics
        if 'flow_matching_loss' in metrics:
            logger.info(f"   Flow Matching Loss: {metrics['flow_matching_loss']:.6f}")
        
        # FIXED: Scaling metrics (critical for debugging)
        if 'prediction_norm' in metrics and 'target_norm' in metrics:
            logger.info(f"   Prediction Norm: {metrics['prediction_norm']:.3f}")
            logger.info(f"   Target Norm: {metrics['target_norm']:.3f}")
            logger.info(f"   Norm Ratio: {metrics.get('norm_ratio', 0):.3f}")
        
        if 'adaptive_scale' in metrics:
            logger.info(f"   Adaptive Scale: {metrics['adaptive_scale']:.3f}")
        
        # Evaluation metrics if available
        if 'final_embedding_similarity' in metrics and metrics['final_embedding_similarity']:
            logger.info(f"   Final Embedding Similarity: {metrics['final_embedding_similarity']:.3f}")
        
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
        
        # Scaling warnings summary
        if self.norm_mismatch_warnings > 0:
            logger.info(f"   Norm Mismatch Warnings: {self.norm_mismatch_warnings}")
        
        if self.scaling_issues_detected:
            logger.info(f"   Scaling Issues: {len(self.scaling_issues_detected)}")

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
        """Get comprehensive training statistics with scaling info"""
        stats = {
            'training_mode': self.training_mode,
            'expected_tokens': self.expected_tokens,
            'total_steps': self.training_step_count,
            'evaluation_enabled': self.enable_evaluation,
            'evaluation_during_training': self.enable_evaluation,
            
            # Scaling monitoring
            'expected_velocity_scale': self.expected_velocity_scale,
            'expected_output_scale': self.expected_output_scale,
            'norm_mismatch_warnings': self.norm_mismatch_warnings,
            'scaling_issues_detected': self.scaling_issues_detected,
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
        
        if self.metric_history:
            latest_metrics = self.metric_history[-1]
            stats['latest_training_metrics'] = latest_metrics
            
            # Overfitting analysis for same-data training
            if self.enable_same_data_eval:
                final_sim_history = [m.get('final_embedding_similarity', 0) for m in self.metric_history[-10:] if m.get('final_embedding_similarity')]
                if final_sim_history:
                    current_sim = final_sim_history[-1]
                    avg_recent_sim = sum(final_sim_history) / len(final_sim_history)
                    
                    stats['overfitting_analysis'] = {
                        'current_final_similarity': current_sim,
                        'recent_average_similarity': avg_recent_sim,
                        'overfitting_status': (
                            'excellent' if current_sim > 0.8 else
                            'very_good' if current_sim > 0.7 else
                            'good' if current_sim > 0.6 else
                            'moderate' if current_sim > 0.4 else
                            'early'
                        ),
                        'expected_overfitting': self.enable_same_data_eval,
                        'similarity_trend': 'improving' if len(final_sim_history) > 5 and final_sim_history[-1] > final_sim_history[-6] else 'stable'
                    }
        
        if self.eval_history:
            latest_eval = self.eval_history[-1]
            stats['latest_evaluation'] = {
                'eval_loss': latest_eval['metrics'].get('eval_loss', 'unknown'),
                'eval_final_similarity': latest_eval['metrics'].get('eval_final_embedding_similarity', 'unknown'),
                'eval_timestamp': latest_eval['timestamp']
            }
        
        return stats

    def log_training_summary(self):
        """Log comprehensive training summary"""
        if not self.is_main_process:
            return
        
        stats = self.get_training_statistics()
        
        logger.info("=" * 60)
        if self.enable_evaluation:
            logger.info("📊 UNIFIED TRAINING SUMMARY (Training + Evaluation)")
        else:
            logger.info("📊 UNIFIED TRAINING SUMMARY (Training Only)")
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
        
        # Scaling info
        logger.info(f"🔧 Scaling Info:")
        logger.info(f"   Expected velocity scale: {self.expected_velocity_scale}")
        logger.info(f"   Expected output scale: {self.expected_output_scale}")
        logger.info(f"   Norm mismatch warnings: {self.norm_mismatch_warnings}")
        if self.scaling_issues_detected:
            logger.info(f"   Scaling issues: {len(self.scaling_issues_detected)}")
        
        logger.info(f"📊 Evaluation: {'Enabled' if self.enable_evaluation else 'Disabled during training'}")
        
        if self.enable_evaluation:
            logger.info("✅ Comprehensive training and evaluation completed")
        else:
            logger.info("✅ Training completed - ready for separate evaluation")
        
        logger.info("=" * 60)


def create_unified_training_args(
    output_dir: str,
    training_mode: str = "patch_only",
    enable_evaluation: bool = False,  # NEW: Control evaluation
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 1e-4,  # FIXED: Better default
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,  # FIXED: More warmup
    logging_steps: int = 10,
    save_steps: int = 200,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    dataloader_num_workers: int = 0,  # FIXED: Conservative default
    eval_steps: int = 50,
    **kwargs
) -> TrainingArguments:
    """
    Create training arguments for UNIFIED trainer
    
    Args:
        enable_evaluation: If True, enables evaluation during training
        All other args: Standard training parameters
    """
    
    # Set evaluation strategy based on enable_evaluation
    if enable_evaluation:
        eval_strategy = "steps"
        metric_for_best_model = "eval_loss"
        load_best_model_at_end = True
        greater_is_better = False
    else:
        eval_strategy = "no"
        metric_for_best_model = None
        load_best_model_at_end = False
        greater_is_better = None
        eval_steps = None
        per_device_eval_batch_size = None
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=3,
        prediction_loss_only=False,  # We want all metrics
        report_to=[],
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        ignore_data_skip=True,
        **kwargs
    )


# Convenience functions for different training modes
def create_training_only_args(**kwargs):
    """Create args for training-only mode (no evaluation during training)"""
    return create_unified_training_args(enable_evaluation=False, **kwargs)


def create_training_with_eval_args(**kwargs):
    """Create args for training with evaluation mode"""
    return create_unified_training_args(enable_evaluation=True, **kwargs)


def create_overfitting_training_args(output_dir: str, **kwargs):
    """Create args optimized for overfitting tests"""
    defaults = {
        'enable_evaluation': True,
        'num_train_epochs': 15,
        'per_device_train_batch_size': 16,
        'learning_rate': 5e-5,
        'weight_decay': 0.0,  # No regularization for overfitting
        'eval_steps': 25,
        'logging_steps': 5,
    }
    defaults.update(kwargs)
    return create_unified_training_args(output_dir=output_dir, **defaults)


def create_production_training_args(output_dir: str, **kwargs):
    """Create args optimized for production training"""
    defaults = {
        'enable_evaluation': False,  # Usually disable for production
        'num_train_epochs': 10,
        'per_device_train_batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2,
        'logging_steps': 10,
        'save_steps': 500,
    }
    defaults.update(kwargs)
    return create_unified_training_args(output_dir=output_dir, **defaults)