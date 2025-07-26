#!/usr/bin/env python3
"""
Fixed BLIP3-o Trainer for EVA-CLIP Reproduction Testing
src/modules/trainers/blip3o_eva_trainer.py

Key fixes:
- Better debugging and monitoring
- Gradient tracking
- Overfitting test capability
- Improved evaluation
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

logger = logging.getLogger(__name__)


class BLIP3oEVATrainer(Trainer):
    """
    Fixed trainer with comprehensive debugging and monitoring
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
        
        # WandB
        self.wandb_instance = wandb_instance
        self.use_wandb = use_wandb and wandb_instance is not None
        
        # Metrics tracking
        self.step_count = 0
        self.gradient_norms_history = []
        self.learning_rate_history = []
        self.loss_history = []
        self.velocity_sim_history = []
        self.eva_sim_history = []
        
        # Best metrics
        self.best_velocity_sim = 0.0
        self.best_eva_sim = 0.0
        self.best_loss = float('inf')
        
        # Gradient statistics
        self.gradient_stats = {
            'min_norm': float('inf'),
            'max_norm': 0.0,
            'zero_grad_steps': 0,
            'exploding_grad_steps': 0,
        }
        
        # Store overfit test samples if requested
        self.overfit_samples = None
        if self.overfit_test_size:
            self._prepare_overfit_test()
        
        logger.info(f"✅ Fixed EVA Trainer initialized")
        logger.info(f"   Debug mode: {debug_mode}")
        logger.info(f"   Track gradients: {track_gradients}")
        logger.info(f"   Overfit test size: {overfit_test_size}")
    
    def _prepare_overfit_test(self):
        """Prepare small subset for overfitting test"""
        logger.info(f"Preparing overfit test with {self.overfit_test_size} samples...")
        
        # Get first N samples from dataloader
        samples = []
        sample_count = 0
        
        for batch in self.get_train_dataloader():
            batch_size = batch['encoder_hidden_states'].shape[0]
            
            # Store batch
            samples.append({
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in batch.items()
            })
            
            sample_count += batch_size
            if sample_count >= self.overfit_test_size:
                break
        
        self.overfit_samples = samples
        logger.info(f"✅ Stored {len(samples)} batches ({sample_count} samples) for overfitting test")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with comprehensive debugging"""
        model.train()
        
        # Extract inputs
        clip_embeddings = inputs['encoder_hidden_states']  # [B, N, 1024]
        eva_embeddings = inputs['eva_embeddings']          # [B, N, 4096]
        timesteps = inputs['timestep']                     # [B]
        noisy_eva = inputs.get('hidden_states')           # [B, N, 4096]
        noise = inputs.get('noise')                        # [B, N, 4096]
        
        # Use overfit samples if in overfit test mode
        if self.overfit_samples and self.training:
            # Cycle through overfit samples
            batch_idx = self.step_count % len(self.overfit_samples)
            overfit_batch = self.overfit_samples[batch_idx]
            
            # Replace inputs with overfit samples
            clip_embeddings = overfit_batch['encoder_hidden_states'].to(clip_embeddings.device)
            eva_embeddings = overfit_batch['eva_embeddings'].to(eva_embeddings.device)
            timesteps = overfit_batch['timestep'].to(timesteps.device)
            noisy_eva = overfit_batch.get('hidden_states', noisy_eva)
            if noisy_eva is not None:
                noisy_eva = noisy_eva.to(clip_embeddings.device)
            noise = overfit_batch.get('noise', noise)
            if noise is not None:
                noise = noise.to(clip_embeddings.device)
        
        # Debug: Check input statistics
        if self.debug_mode and self.step_count % 50 == 0:
            logger.info(f"[Step {self.step_count}] Input Statistics:")
            logger.info(f"  CLIP norm: {torch.norm(clip_embeddings, dim=-1).mean():.3f}")
            logger.info(f"  EVA norm: {torch.norm(eva_embeddings, dim=-1).mean():.3f}")
            logger.info(f"  Timesteps: {timesteps.mean():.3f} ± {timesteps.std():.3f}")
            if noisy_eva is not None:
                logger.info(f"  Noisy EVA norm: {torch.norm(noisy_eva, dim=-1).mean():.3f}")
        
        # Forward pass
        model_outputs = model(
            hidden_states=noisy_eva,
            timestep=timesteps,
            encoder_hidden_states=clip_embeddings,
            return_dict=True
        )
        
        velocity_pred = model_outputs.get('velocity_prediction')
        
        # Debug: Check output statistics
        if self.debug_mode and self.step_count % 50 == 0:
            logger.info(f"  Prediction norm: {torch.norm(velocity_pred, dim=-1).mean():.3f}")
            logger.info(f"  Prediction std: {velocity_pred.std():.3f}")
        
        # Compute loss
        loss, metrics = self.flow_matching_loss(
            model_output=velocity_pred,
            target_samples=eva_embeddings,
            timesteps=timesteps,
            clip_conditioning=clip_embeddings,
            noise=noise,
            return_metrics=True,
            training_mode=self.training_mode,
        )
        
        # Track metrics
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
        """Track comprehensive training metrics"""
        # Basic metrics
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        if metrics:
            velocity_sim = metrics.get('velocity_similarity', 0.0)
            self.velocity_sim_history.append(velocity_sim)
            
            # Update best metrics
            if velocity_sim > self.best_velocity_sim:
                self.best_velocity_sim = velocity_sim
            if loss_value < self.best_loss:
                self.best_loss = loss_value
        
        # Track gradients if enabled
        if self.track_gradients and model.training:
            self._track_gradient_statistics(model)
        
        # Track learning rate
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rate_history.append(current_lr)
        
        # Log to WandB
        if self.use_wandb:
            wandb_data = {
                "train/step": self.step_count,
                "train/loss": loss_value,
                "train/learning_rate": self.learning_rate_history[-1] if self.learning_rate_history else 0,
            }
            
            if metrics:
                wandb_data.update({
                    "train/velocity_similarity": metrics.get('velocity_similarity', 0),
                    "train/pred_norm": metrics.get('pred_norm', 0),
                    "train/velocity_norm": metrics.get('velocity_norm', 0),
                    "train/error_norm": metrics.get('error_norm', 0),
                    "train/relative_error": metrics.get('relative_error', 0),
                })
            
            if self.gradient_norms_history:
                wandb_data["train/gradient_norm"] = self.gradient_norms_history[-1]
            
            self.wandb_instance.log(wandb_data, step=self.step_count)
        
        # Periodic logging
        if self.step_count % self.args.logging_steps == 0:
            self._log_training_progress(loss_value, metrics)
        
        # Run evaluation
        if self.step_count > 0 and self.step_count % self.eval_every_n_steps == 0:
            self._run_eva_evaluation_during_training()
    
    def _track_gradient_statistics(self, model):
        """Track gradient norms and statistics"""
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                
                # Track specific layer gradients
                if self.debug_mode and self.step_count % self.log_gradient_norms_every == 0:
                    if 'output_proj' in name or 'input_proj' in name:
                        logger.info(f"  Gradient norm {name}: {param_norm:.6f}")
        
        total_norm = total_norm ** 0.5
        self.gradient_norms_history.append(total_norm)
        
        # Update statistics
        if total_norm < self.gradient_stats['min_norm']:
            self.gradient_stats['min_norm'] = total_norm
        if total_norm > self.gradient_stats['max_norm']:
            self.gradient_stats['max_norm'] = total_norm
        if total_norm < 1e-8:
            self.gradient_stats['zero_grad_steps'] += 1
        if total_norm > 100.0:
            self.gradient_stats['exploding_grad_steps'] += 1
        
        # Log warnings
        if total_norm < 1e-8:
            logger.warning(f"[Step {self.step_count}] Zero gradients detected!")
        elif total_norm > 100.0:
            logger.warning(f"[Step {self.step_count}] Large gradient norm: {total_norm:.2f}")
    
    def _log_training_progress(self, loss: float, metrics: Optional[Dict[str, float]]):
        """Log detailed training progress"""
        msg = f"Step {self.step_count}: Loss={loss:.6f}"
        
        if metrics:
            velocity_sim = metrics.get('velocity_similarity', 0.0)
            msg += f", VelSim={velocity_sim:.4f}"
            msg += f", PredNorm={metrics.get('pred_norm', 0):.3f}"
            msg += f", Error={metrics.get('relative_error', 0):.3f}"
        
        if self.gradient_norms_history:
            msg += f", GradNorm={self.gradient_norms_history[-1]:.3f}"
        
        if self.learning_rate_history:
            msg += f", LR={self.learning_rate_history[-1]:.2e}"
        
        msg += f" | Best: Loss={self.best_loss:.6f}, VelSim={self.best_velocity_sim:.4f}"
        
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
        
        # Loss statistics
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-100:]
            logger.info(f"📉 Loss Statistics:")
            logger.info(f"   Current: {self.loss_history[-1]:.6f}")
            logger.info(f"   Recent mean: {np.mean(recent_losses):.6f}")
            logger.info(f"   Recent std: {np.std(recent_losses):.6f}")
            logger.info(f"   Best: {self.best_loss:.6f}")
        
        # Gradient statistics
        if self.gradient_norms_history:
            recent_grads = self.gradient_norms_history[-100:]
            logger.info(f"📈 Gradient Statistics:")
            logger.info(f"   Current norm: {self.gradient_norms_history[-1]:.3f}")
            logger.info(f"   Recent mean: {np.mean(recent_grads):.3f}")
            logger.info(f"   Min/Max: {self.gradient_stats['min_norm']:.3f} / {self.gradient_stats['max_norm']:.3f}")
            logger.info(f"   Zero grad steps: {self.gradient_stats['zero_grad_steps']}")
            logger.info(f"   Exploding grad steps: {self.gradient_stats['exploding_grad_steps']}")
        
        # Velocity similarity
        if self.velocity_sim_history:
            recent_sims = self.velocity_sim_history[-100:]
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
            else:
                logger.info("   ⚠️ Struggling to overfit. Check architecture/loss.")
        
        logger.info("=" * 80)
    
    def _run_eva_evaluation_during_training(self):
        """Run evaluation during training"""
        if self.eval_dataloader is None:
            return
        
        logger.info(f"🔍 Running evaluation at step {self.step_count}...")
        
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
            
            if self.use_wandb:
                self.wandb_instance.log({
                    "eval/eva_similarity": eva_sim,
                    "eval/high_quality_ratio": eval_results['high_quality_images'],
                }, step=self.step_count)
    
    def _evaluate_eva_similarity(
        self, 
        num_samples: int = 1000,
        batch_size: int = 16,
        inference_steps: int = 50
    ) -> Optional[Dict[str, float]]:
        """Evaluate EVA similarity"""
        try:
            all_similarities = []
            samples_processed = 0
            
            with torch.no_grad():
                for batch in self.eval_dataloader:
                    if samples_processed >= num_samples:
                        break
                    
                    # Move to device
                    clip_features = batch['encoder_hidden_states'].to(self.model.device)
                    target_eva = batch['eva_embeddings'].to(self.model.device)
                    
                    # Generate
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
            
            if not all_similarities:
                return None
            
            all_sims = torch.cat(all_similarities)
            
            return {
                'overall_eva_similarity': all_sims.mean().item(),
                'high_quality_images': (all_sims > 0.7).float().mean().item(),
                'very_high_quality_images': (all_sims > 0.8).float().mean().item(),
            }
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return None
    
    def get_final_evaluation(self) -> Dict[str, Any]:
        """Get comprehensive final evaluation"""
        logger.info("🔍 Running final evaluation...")
        
        self.model.eval()
        final_results = self._evaluate_eva_similarity(
            num_samples=min(5000, self.eval_num_samples * 5),
            batch_size=self.eval_batch_size,
            inference_steps=self.eval_inference_steps * 2  # More steps for final eval
        )
        
        # Training summary
        training_summary = {
            'total_steps': self.step_count,
            'best_loss': self.best_loss,
            'best_velocity_sim': self.best_velocity_sim,
            'best_eva_sim': self.best_eva_sim,
            'gradient_stats': self.gradient_stats,
            'overfit_test': self.overfit_test_size is not None,
            'overfit_success': self.best_velocity_sim > 0.9 if self.overfit_test_size else None,
        }
        
        return {
            'training_summary': training_summary,
            'final_evaluation': final_results,
            'loss_history': self.loss_history[-1000:],  # Last 1000 steps
            'velocity_sim_history': self.velocity_sim_history[-1000:],
            'eva_sim_history': self.eva_sim_history,
            'gradient_norms_history': self.gradient_norms_history[-1000:],
        }


def create_eva_training_args(
    output_dir: str,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.1,  # Use ratio instead of steps
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
    """Create training arguments with better defaults"""
    
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