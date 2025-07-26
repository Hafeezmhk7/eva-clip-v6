#!/usr/bin/env python3
"""
Fixed BLIP3-o Trainer for EVA-CLIP Reproduction Testing
src/modules/trainers/blip3o_eva_trainer.py

MAJOR FIXES:
1. Better debugging and monitoring based on feedback
2. Fixed gradient tracking and numerical stability
3. Overfitting test capability
4. Improved evaluation with proper error handling
5. Better training health monitoring
"""

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import time
import numpy as np
from pathlib import Path
import json
import gc
from collections import deque

logger = logging.getLogger(__name__)


class BLIP3oEVATrainer(Trainer):
    """
    Fixed trainer with comprehensive debugging and monitoring based on feedback
    """
    
    def __init__(
        self,
        model,
        args: TrainingArguments,
        flow_matching_loss,
        train_dataset=None,
        eval_dataset=None,
        eval_dataloader=None,
        training_mode: str = "patch_only",
        # Evaluation parameters
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 1000,
        eval_batch_size: int = 16,
        eval_inference_steps: int = 50,
        # Debugging parameters
        debug_mode: bool = False,
        track_gradients: bool = True,
        log_gradient_norms_every: int = 10,
        overfit_test_size: Optional[int] = None,
        # Health monitoring
        loss_window_size: int = 100,
        similarity_window_size: int = 50,
        # WandB
        wandb_instance=None,
        use_wandb: bool = False,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        
        self.flow_matching_loss = flow_matching_loss
        self.training_mode = training_mode
        self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        # Evaluation
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_num_samples = eval_num_samples
        self.eval_batch_size = eval_batch_size
        self.eval_inference_steps = eval_inference_steps
        self.eval_dataloader = eval_dataloader
        
        # Debugging
        self.debug_mode = debug_mode
        self.track_gradients = track_gradients
        self.log_gradient_norms_every = log_gradient_norms_every
        self.overfit_test_size = overfit_test_size
        
        # Health monitoring
        self.loss_window = deque(maxlen=loss_window_size)
        self.similarity_window = deque(maxlen=similarity_window_size)
        
        # WandB
        self.wandb_instance = wandb_instance
        self.use_wandb = use_wandb and wandb_instance is not None
        
        # Metrics tracking
        self.step_count = 0
        self.gradient_norms_history = deque(maxlen=1000)
        self.learning_rate_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.velocity_sim_history = deque(maxlen=1000)
        self.eva_sim_history = deque(maxlen=500)
        
        # Best metrics
        self.best_velocity_sim = 0.0
        self.best_eva_sim = 0.0
        self.best_loss = float('inf')
        
        # Training health indicators
        self.training_health_status = "initializing"
        self.last_improvement_step = 0
        self.plateau_patience = 200  # Steps to wait before considering plateau
        
        # Gradient statistics
        self.gradient_stats = {
            'min_norm': float('inf'),
            'max_norm': 0.0,
            'zero_grad_steps': 0,
            'exploding_grad_steps': 0,
            'nan_grad_steps': 0,
        }
        
        # Store overfit test samples if requested
        self.overfit_samples = None
        if self.overfit_test_size:
            self._prepare_overfit_test()
        
        logger.info(f"✅ Fixed EVA Trainer initialized")
        logger.info(f"   Debug mode: {debug_mode}")
        logger.info(f"   Track gradients: {track_gradients}")
        logger.info(f"   Overfit test size: {overfit_test_size}")
        logger.info(f"   Health monitoring enabled with windows: loss={loss_window_size}, sim={similarity_window_size}")
    
    def _prepare_overfit_test(self):
        """Prepare small subset for overfitting test"""
        logger.info(f"Preparing overfit test with {self.overfit_test_size} samples...")
        
        try:
            # Get first N samples from dataloader
            samples = []
            sample_count = 0
            
            for batch in self.get_train_dataloader():
                batch_size = batch['encoder_hidden_states'].shape[0]
                
                # Store batch (deep copy to avoid memory issues)
                samples.append({
                    k: v.clone().detach() if torch.is_tensor(v) else v
                    for k, v in batch.items()
                })
                
                sample_count += batch_size
                if sample_count >= self.overfit_test_size:
                    break
            
            self.overfit_samples = samples
            logger.info(f"✅ Stored {len(samples)} batches ({sample_count} samples) for overfitting test")
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare overfit test: {e}")
            self.overfit_samples = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with comprehensive debugging and health monitoring"""
        model.train()
        
        # Extract inputs with error handling
        try:
            clip_embeddings = inputs['encoder_hidden_states']  # [B, N, 1024]
            eva_embeddings = inputs['eva_embeddings']          # [B, N, 4096]
            timesteps = inputs['timestep']                     # [B]
            noisy_eva = inputs.get('hidden_states')           # [B, N, 4096]
            noise = inputs.get('noise')                        # [B, N, 4096]
            
            # Validate input shapes
            assert clip_embeddings.dim() == 3, f"CLIP embeddings wrong shape: {clip_embeddings.shape}"
            assert eva_embeddings.dim() == 3, f"EVA embeddings wrong shape: {eva_embeddings.shape}"
            assert timesteps.dim() == 1, f"Timesteps wrong shape: {timesteps.shape}"
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            raise
        
        # Use overfit samples if in overfit test mode
        if self.overfit_samples and self.training:
            try:
                # Cycle through overfit samples
                batch_idx = self.step_count % len(self.overfit_samples)
                overfit_batch = self.overfit_samples[batch_idx]
                
                # Replace inputs with overfit samples
                clip_embeddings = overfit_batch['encoder_hidden_states'].to(clip_embeddings.device)
                eva_embeddings = overfit_batch['eva_embeddings'].to(eva_embeddings.device)
                timesteps = overfit_batch['timestep'].to(timesteps.device)
                noisy_eva = overfit_batch.get('hidden_states')
                if noisy_eva is not None:
                    noisy_eva = noisy_eva.to(clip_embeddings.device)
                noise = overfit_batch.get('noise')
                if noise is not None:
                    noise = noise.to(clip_embeddings.device)
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to use overfit samples: {e}, using original batch")
        
        # Debug: Check input statistics
        if self.debug_mode and self.step_count % 50 == 0:
            logger.info(f"[Step {self.step_count}] Input Statistics:")
            logger.info(f"  CLIP norm: {torch.norm(clip_embeddings, dim=-1).mean():.3f}")
            logger.info(f"  EVA norm: {torch.norm(eva_embeddings, dim=-1).mean():.3f}")
            logger.info(f"  Timesteps: {timesteps.mean():.3f} ± {timesteps.std():.3f}")
            if noisy_eva is not None:
                logger.info(f"  Noisy EVA norm: {torch.norm(noisy_eva, dim=-1).mean():.3f}")
        
        # Forward pass with error handling
        try:
            model_outputs = model(
                hidden_states=noisy_eva,
                timestep=timesteps,
                encoder_hidden_states=clip_embeddings,
                return_dict=True
            )
            
            velocity_pred = model_outputs.get('velocity_prediction')
            
            if velocity_pred is None:
                raise ValueError("Model did not return velocity_prediction")
                
        except Exception as e:
            logger.error(f"❌ Model forward pass failed: {e}")
            logger.error(f"   Input shapes: CLIP {clip_embeddings.shape}, EVA {eva_embeddings.shape}, t {timesteps.shape}")
            raise
        
        # Debug: Check output statistics
        if self.debug_mode and self.step_count % 50 == 0:
            logger.info(f"  Prediction norm: {torch.norm(velocity_pred, dim=-1).mean():.3f}")
            logger.info(f"  Prediction std: {velocity_pred.std():.3f}")
            
            # Check for NaN/Inf
            if torch.isnan(velocity_pred).any():
                logger.error(f"  ❌ NaN detected in predictions!")
            if torch.isinf(velocity_pred).any():
                logger.error(f"  ❌ Inf detected in predictions!")
        
        # Compute loss with error handling
        try:
            loss, metrics = self.flow_matching_loss(
                model_output=velocity_pred,
                target_samples=eva_embeddings,
                timesteps=timesteps,
                clip_conditioning=clip_embeddings,
                noise=noise,
                return_metrics=True,
                training_mode=self.training_mode,
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"❌ Invalid loss detected: {loss}")
                raise ValueError(f"Invalid loss: {loss}")
                
        except Exception as e:
            logger.error(f"❌ Loss computation failed: {e}")
            raise
        
        # Track metrics and health
        self._track_training_metrics(loss, metrics, model)
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'velocity_prediction': velocity_pred,
            'metrics': metrics,
        } if return_outputs else None
        
        self.step_count += 1
        
        return (loss, outputs) if return_outputs else loss
    
    def _track_training_metrics(self, loss: torch.Tensor, metrics: Dict[str, float], model):
        """Track comprehensive training metrics with health monitoring"""
        # Basic metrics
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.loss_window.append(loss_value)
        
        if metrics:
            velocity_sim = metrics.get('velocity_similarity', 0.0)
            self.velocity_sim_history.append(velocity_sim)
            self.similarity_window.append(velocity_sim)
            
            # Update best metrics
            if velocity_sim > self.best_velocity_sim:
                self.best_velocity_sim = velocity_sim
                self.last_improvement_step = self.step_count
                
            if loss_value < self.best_loss:
                self.best_loss = loss_value
        
        # Track gradients if enabled
        if self.track_gradients and model.training:
            self._track_gradient_statistics(model)
        
        # Track learning rate
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rate_history.append(current_lr)
        
        # Health monitoring
        self._update_training_health()
        
        # Log to WandB
        if self.use_wandb:
            self._log_to_wandb(loss_value, metrics)
        
        # Periodic logging
        if self.step_count % self.args.logging_steps == 0:
            self._log_training_progress(loss_value, metrics)
        
        # Run evaluation
        if self.step_count > 0 and self.step_count % self.eval_every_n_steps == 0:
            self._run_eva_evaluation_during_training()
    
    def _update_training_health(self):
        """Update training health status based on recent metrics"""
        if len(self.loss_window) < 10:
            self.training_health_status = "warming_up"
            return
        
        # Check for improvement
        steps_since_improvement = self.step_count - self.last_improvement_step
        is_improving = steps_since_improvement < self.plateau_patience
        
        # Check loss trend
        recent_losses = list(self.loss_window)[-50:]
        if len(recent_losses) >= 10:
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            is_decreasing = loss_trend < 0
        else:
            is_decreasing = True
        
        # Check similarity trend
        recent_sims = list(self.similarity_window)[-20:]
        if len(recent_sims) >= 5:
            sim_trend = np.polyfit(range(len(recent_sims)), recent_sims, 1)[0]
            sim_improving = sim_trend > 0 or recent_sims[-1] > 0.1
        else:
            sim_improving = True
        
        # Check for NaN/explosion
        has_nan = any(torch.isnan(torch.tensor(recent_losses)))
        has_explosion = any(l > 100.0 for l in recent_losses[-5:])
        
        # Determine health status
        if has_nan or has_explosion:
            self.training_health_status = "unstable"
        elif not is_improving and steps_since_improvement > self.plateau_patience:
            self.training_health_status = "plateau"
        elif is_decreasing and sim_improving:
            self.training_health_status = "healthy"
        elif self.best_velocity_sim > 0.3:
            self.training_health_status = "converged"
        else:
            self.training_health_status = "learning"
    
    def _track_gradient_statistics(self, model):
        """Track gradient norms and statistics with better error handling"""
        total_norm = 0.0
        param_count = 0
        nan_count = 0
        
        try:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    
                    if torch.isnan(param.grad).any():
                        nan_count += 1
                        self.gradient_stats['nan_grad_steps'] += 1
                        continue
                        
                    total_norm += param_norm ** 2
                    param_count += 1
                    
                    # Track specific layer gradients
                    if self.debug_mode and self.step_count % self.log_gradient_norms_every == 0:
                        if any(key in name for key in ['output_proj', 'input_proj', 'timestep_embedder']):
                            logger.info(f"  Gradient norm {name}: {param_norm:.6f}")
            
            if param_count > 0:
                total_norm = (total_norm ** 0.5) / param_count  # Average norm
            else:
                total_norm = 0.0
                
            self.gradient_norms_history.append(total_norm)
            
            # Update statistics
            if total_norm < self.gradient_stats['min_norm']:
                self.gradient_stats['min_norm'] = total_norm
            if total_norm > self.gradient_stats['max_norm']:
                self.gradient_stats['max_norm'] = total_norm
            if total_norm < 1e-8:
                self.gradient_stats['zero_grad_steps'] += 1
            if total_norm > 10.0:
                self.gradient_stats['exploding_grad_steps'] += 1
            
            # Log warnings
            if nan_count > 0:
                logger.warning(f"[Step {self.step_count}] NaN gradients in {nan_count} parameters!")
            if total_norm < 1e-8:
                logger.warning(f"[Step {self.step_count}] Very small gradients: {total_norm:.2e}")
            elif total_norm > 10.0:
                logger.warning(f"[Step {self.step_count}] Large gradient norm: {total_norm:.2f}")
                
        except Exception as e:
            logger.warning(f"Error tracking gradients: {e}")
    
    def _log_to_wandb(self, loss_value: float, metrics: Optional[Dict[str, float]]):
        """Log metrics to WandB"""
        try:
            wandb_data = {
                "train/step": self.step_count,
                "train/loss": loss_value,
                "train/learning_rate": self.learning_rate_history[-1] if self.learning_rate_history else 0,
                "train/training_health": self.training_health_status,
            }
            
            if metrics:
                wandb_data.update({
                    "train/velocity_similarity": metrics.get('velocity_similarity', 0),
                    "train/pred_norm": metrics.get('pred_norm', 0),
                    "train/velocity_norm": metrics.get('velocity_norm', 0),
                    "train/error_norm": metrics.get('error_norm', 0),
                    "train/relative_error": metrics.get('relative_error', 0),
                    "train/quality_assessment": 0 if metrics.get('quality_assessment') == 'poor' else 
                                              1 if metrics.get('quality_assessment') == 'fair' else
                                              2 if metrics.get('quality_assessment') == 'good' else 3,
                })
            
            if self.gradient_norms_history:
                wandb_data["train/gradient_norm"] = self.gradient_norms_history[-1]
            
            self.wandb_instance.log(wandb_data, step=self.step_count)
            
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")
    
    def _log_training_progress(self, loss: float, metrics: Optional[Dict[str, float]]):
        """Log detailed training progress"""
        msg = f"Step {self.step_count}: Loss={loss:.6f}"
        
        if metrics:
            velocity_sim = metrics.get('velocity_similarity', 0.0)
            quality = metrics.get('quality_assessment', 'unknown')
            msg += f", VelSim={velocity_sim:.4f} ({quality})"
            msg += f", PredNorm={metrics.get('pred_norm', 0):.3f}"
            msg += f", Error={metrics.get('relative_error', 0):.3f}"
        
        if self.gradient_norms_history:
            msg += f", GradNorm={self.gradient_norms_history[-1]:.3f}"
        
        if self.learning_rate_history:
            msg += f", LR={self.learning_rate_history[-1]:.2e}"
        
        msg += f" | Best: Loss={self.best_loss:.6f}, VelSim={self.best_velocity_sim:.4f}"
        msg += f" | Health: {self.training_health_status}"
        
        if self.overfit_samples:
            msg += " [OVERFIT TEST]"
        
        logger.info(msg)
        
        # Detailed summary every N steps
        if self.step_count % (self.args.logging_steps * 10) == 0:
            self._log_detailed_summary()
    
    def _log_detailed_summary(self):
        """Log comprehensive training summary"""
        logger.info("=" * 80)
        logger.info(f"📊 TRAINING SUMMARY - Step {self.step_count}")
        logger.info("=" * 80)
        
        # Training health
        logger.info(f"🏥 Training Health: {self.training_health_status}")
        steps_since_improvement = self.step_count - self.last_improvement_step
        logger.info(f"   Steps since improvement: {steps_since_improvement}")
        
        # Loss statistics
        if len(self.loss_history) > 10:
            recent_losses = list(self.loss_history)[-100:]
            logger.info(f"📉 Loss Statistics:")
            logger.info(f"   Current: {self.loss_history[-1]:.6f}")
            logger.info(f"   Recent mean: {np.mean(recent_losses):.6f}")
            logger.info(f"   Recent std: {np.std(recent_losses):.6f}")
            logger.info(f"   Best: {self.best_loss:.6f}")
        
        # Gradient statistics
        if self.gradient_norms_history:
            recent_grads = list(self.gradient_norms_history)[-100:]
            logger.info(f"📈 Gradient Statistics:")
            logger.info(f"   Current norm: {self.gradient_norms_history[-1]:.3f}")
            logger.info(f"   Recent mean: {np.mean(recent_grads):.3f}")
            logger.info(f"   Min/Max: {self.gradient_stats['min_norm']:.3f} / {self.gradient_stats['max_norm']:.3f}")
            logger.info(f"   Issues: Zero={self.gradient_stats['zero_grad_steps']}, "
                       f"Exploding={self.gradient_stats['exploding_grad_steps']}, "
                       f"NaN={self.gradient_stats['nan_grad_steps']}")
        
        # Velocity similarity
        if self.velocity_sim_history:
            recent_sims = list(self.velocity_sim_history)[-100:]
            logger.info(f"🎯 Velocity Similarity:")
            logger.info(f"   Current: {self.velocity_sim_history[-1]:.4f}")
            logger.info(f"   Recent mean: {np.mean(recent_sims):.4f}")
            logger.info(f"   Best: {self.best_velocity_sim:.4f}")
        
        # Learning rate
        if self.learning_rate_history:
            logger.info(f"📚 Learning Rate: {self.learning_rate_history[-1]:.2e}")
        
        # Overfit test status
        if self.overfit_samples:
            logger.info(f"🔬 Overfitting Test: Training on {len(self.overfit_samples)} batches")
            if self.best_velocity_sim > 0.9:
                logger.info("   ✅ Successfully overfitting! Model can learn.")
            elif self.best_velocity_sim > 0.5:
                logger.info("   📈 Making progress on overfitting.")
            elif self.best_velocity_sim > 0.1:
                logger.info("   📊 Some learning detected.")
            else:
                logger.info("   ⚠️ Struggling to overfit. Check architecture/loss.")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"💾 GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        
        logger.info("=" * 80)
    
    def _run_eva_evaluation_during_training(self):
        """Run evaluation during training with improved error handling"""
        if self.eval_dataloader is None:
            return
        
        logger.info(f"🔍 Running evaluation at step {self.step_count}...")
        
        try:
            self.model.eval()
            eval_results = self._evaluate_eva_similarity(
                num_samples=self.eval_num_samples,
                batch_size=self.eval_batch_size,
                inference_steps=self.eval_inference_steps
            )
            self.model.train()
            
            if eval_results:
                eva_sim = eval_results['overall_eva_similarity']
                self.eva_sim_history.append(eva_sim)
                
                if eva_sim > self.best_eva_sim:
                    self.best_eva_sim = eva_sim
                    logger.info(f"🎉 New best EVA similarity: {eva_sim:.4f}")
                
                logger.info(f"📊 Evaluation Results:")
                logger.info(f"   EVA Similarity: {eva_sim:.4f}")
                logger.info(f"   High Quality: {eval_results['high_quality_images']*100:.1f}%")
                logger.info(f"   Very High Quality: {eval_results['very_high_quality_images']*100:.1f}%")
                
                if self.use_wandb:
                    self.wandb_instance.log({
                        "eval/eva_similarity": eva_sim,
                        "eval/high_quality_ratio": eval_results['high_quality_images'],
                        "eval/very_high_quality_ratio": eval_results['very_high_quality_images'],
                    }, step=self.step_count)
            else:
                logger.warning("⚠️ Evaluation failed")
                
        except Exception as e:
            logger.error(f"❌ Evaluation error: {e}")
            self.model.train()  # Ensure model is back in training mode
    
    def _evaluate_eva_similarity(
        self, 
        num_samples: int = 1000,
        batch_size: int = 16,
        inference_steps: int = 50
    ) -> Optional[Dict[str, float]]:
        """Evaluate EVA similarity with robust error handling"""
        try:
            all_similarities = []
            samples_processed = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.eval_dataloader):
                    if samples_processed >= num_samples:
                        break
                    
                    try:
                        # Move to device
                        clip_features = batch['encoder_hidden_states'].to(self.model.device)
                        target_eva = batch['eva_embeddings'].to(self.model.device)
                        
                        # Generate with error handling
                        generated = self.model.generate(
                            clip_features=clip_features,
                            num_inference_steps=inference_steps,
                            normalize_output=True,
                            solver="euler"  # Use simple solver for evaluation
                        )
                        
                        # Compute similarity
                        target_norm = F.normalize(target_eva, p=2, dim=-1)
                        sim = F.cosine_similarity(generated, target_norm, dim=-1)
                        per_image_sim = sim.mean(dim=1)
                        
                        all_similarities.append(per_image_sim.cpu())
                        samples_processed += clip_features.shape[0]
                        
                        # Clean up GPU memory
                        del clip_features, target_eva, generated, target_norm, sim
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.warning(f"Error in evaluation batch {batch_idx}: {e}")
                        continue
            
            if not all_similarities:
                return None
            
            all_sims = torch.cat(all_similarities)
            
            return {
                'overall_eva_similarity': all_sims.mean().item(),
                'high_quality_images': (all_sims > 0.7).float().mean().item(),
                'very_high_quality_images': (all_sims > 0.8).float().mean().item(),
                'excellent_quality_images': (all_sims > 0.9).float().mean().item(),
                'similarity_std': all_sims.std().item(),
                'samples_evaluated': samples_processed,
            }
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return None
    
    def get_final_evaluation(self) -> Dict[str, Any]:
        """Get comprehensive final evaluation"""
        logger.info("🔍 Running final evaluation...")
        
        # Training summary
        training_summary = {
            'total_steps': self.step_count,
            'final_loss': self.loss_history[-1] if self.loss_history else float('inf'),
            'best_loss': self.best_loss,
            'final_velocity_sim': self.velocity_sim_history[-1] if self.velocity_sim_history else 0.0,
            'best_velocity_sim': self.best_velocity_sim,
            'final_eva_sim': self.eva_sim_history[-1] if self.eva_sim_history else 0.0,
            'best_eva_sim': self.best_eva_sim,
            'gradient_stats': self.gradient_stats,
            'training_health': self.training_health_status,
            'evaluations_performed': len(self.eva_sim_history),
            'overfit_test': self.overfit_test_size is not None,
            'overfit_success': self.best_velocity_sim > 0.9 if self.overfit_test_size else None,
        }
        
        # Final evaluation
        try:
            self.model.eval()
            final_results = self._evaluate_eva_similarity(
                num_samples=min(5000, self.eval_num_samples * 5),
                batch_size=self.eval_batch_size,
                inference_steps=self.eval_inference_steps * 2  # More steps for final eval
            )
            self.model.train()
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            final_results = None
        
        return {
            'training_summary': training_summary,
            'final_evaluation': final_results,
            'loss_history': list(self.loss_history),
            'velocity_sim_history': list(self.velocity_sim_history),
            'eva_sim_history': list(self.eva_sim_history),
            'gradient_norms_history': list(self.gradient_norms_history),
        }


def create_eva_training_args(
    output_dir: str,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 5e-4,  # Increased based on feedback
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    fp16: bool = True,
    dataloader_num_workers: int = 0,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    report_to: list = None,
    **kwargs
) -> TrainingArguments:
    """Create training arguments with better defaults based on feedback"""
    
    if report_to is None:
        report_to = []
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_total_limit=save_total_limit,
        eval_strategy="no",  # We handle evaluation manually
        prediction_loss_only=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=report_to,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        **kwargs
    )