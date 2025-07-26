#!/usr/bin/env python3
"""
FIXED: BLIP3-o Trainer with Proper L2 Normalization Tracking and WandB Integration
src/modules/trainers/blip3o_trainer.py

KEY FIX:
1. Track and log proper norm values (CLIP targets should be ~1.0)
2. Monitor normalization status throughout training
3. Report both CLIP target norms and velocity target norms
4. Warn if normalization is not working correctly
5. Comprehensive WandB integration for all metrics
"""

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, Union, Tuple
import logging
import time
import os
import random

logger = logging.getLogger(__name__)


class BLIP3oTrainer(Trainer):
    """
    FIXED: BLIP3-o trainer with proper L2 normalization tracking and WandB integration
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
        # EVALUATION PARAMETERS
        eval_every_n_steps: int = 100,
        eval_num_samples: int = 1000,
        eval_batch_size: int = 16,
        eval_inference_steps: int = 50,
        # WANDB INTEGRATION
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
        
        # Evaluation parameters
        self.eval_every_n_steps = eval_every_n_steps
        self.eval_num_samples = eval_num_samples
        self.eval_batch_size = eval_batch_size
        self.eval_inference_steps = eval_inference_steps
        self.eval_dataloader = eval_dataloader
        
        # WandB integration
        self.wandb_instance = wandb_instance
        self.use_wandb = use_wandb and wandb_instance is not None
        
        # Training metrics tracking
        self.step_count = 0
        self.loss_history = []
        self.velocity_similarity_history = []
        self.embedding_similarity_history = []
        
        # FIXED: Norm tracking
        self.clip_target_norm_history = []
        self.prediction_norm_history = []
        self.velocity_target_norm_history = []
        
        # Best metrics tracking
        self.best_velocity_sim = 0.0
        self.best_embedding_sim = 0.0
        self.best_loss = float('inf')
        
        # Steps without improvement (for monitoring)
        self.steps_without_velocity_improvement = 0
        self.steps_without_embedding_improvement = 0
        
        # Normalization tracking
        self.normalization_warnings = 0
        self.last_clip_norm = None
        
        # WandB logging intervals
        self.wandb_log_interval = max(1, args.logging_steps // 2)  # Log to WandB more frequently
        
        logger.info(f"✅ FIXED BLIP3o Trainer initialized with L2 normalization tracking and WandB")
        logger.info(f"   Training mode: {training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"   Evaluation every: {eval_every_n_steps} steps")
        logger.info(f"   Evaluation samples: {eval_num_samples}")
        logger.info(f"   Evaluation inference steps: {eval_inference_steps}")
        logger.info(f"   Expected CLIP target norms: ~1.0 (L2 normalized)")
        logger.info(f"   WandB tracking: {'Enabled' if self.use_wandb else 'Disabled'}")
        if self.use_wandb:
            logger.info(f"   WandB URL: {self.wandb_instance.run.url}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with FIXED L2 normalization tracking and WandB logging"""
        model.train()
        
        # Extract inputs with better error handling
        try:
            eva_embeddings = inputs['encoder_hidden_states']  # [B, N, 4096]
            clip_embeddings = inputs['clip_embeddings']       # [B, N, 1024] - Should be normalized ~1.0
            timesteps = inputs['timestep']                    # [B]
        except KeyError as e:
            logger.error(f"Missing input key: {e}")
            logger.error(f"Available keys: {list(inputs.keys())}")
            raise
        
        # FIXED: Check normalization status from collate function
        clip_norm_from_collate = inputs.get('clip_norm_mean', None)
        initial_clip_norm = inputs.get('initial_clip_norm', None)
        
        if clip_norm_from_collate is not None:
            if abs(clip_norm_from_collate - 1.0) > 0.1:
                self.normalization_warnings += 1
                if self.normalization_warnings <= 5:  # Limit warnings
                    logger.warning(f"CLIP embeddings not properly normalized: {clip_norm_from_collate:.3f} (should be ~1.0)")
        
        # Get or create noisy input
        if 'hidden_states' in inputs:
            noisy_clip = inputs['hidden_states']  # [B, N, 1024]
        else:
            # Create noisy input for flow matching (this should be in the collate function)
            device = eva_embeddings.device
            noise = torch.randn_like(clip_embeddings)
            alpha = timesteps.view(-1, 1, 1)
            noisy_clip = (1 - alpha) * noise + alpha * clip_embeddings.detach()
        
        # Ensure gradients are enabled for training
        if not noisy_clip.requires_grad:
            noisy_clip = noisy_clip.detach().requires_grad_(True)
        
        # Validate shapes
        batch_size, seq_len, eva_dim = eva_embeddings.shape
        assert seq_len == self.expected_tokens, f"Expected {self.expected_tokens} tokens, got {seq_len}"
        assert eva_dim == 4096, f"Expected EVA 4096-dim, got {eva_dim}"
        assert clip_embeddings.shape[2] == 1024, f"Expected CLIP 1024-dim, got {clip_embeddings.shape[2]}"
        assert noisy_clip.shape == clip_embeddings.shape, f"Noisy input shape mismatch"
        
        # Forward pass
        model_outputs = model(
            hidden_states=noisy_clip,
            timestep=timesteps,
            encoder_hidden_states=eva_embeddings,
            return_dict=True
        )
        
        # Extract velocity prediction
        if isinstance(model_outputs, dict):
            velocity_pred = model_outputs.get('velocity_prediction', 
                                            model_outputs.get('last_hidden_state'))
        else:
            velocity_pred = model_outputs
        
        # Verify gradients
        if not velocity_pred.requires_grad:
            raise RuntimeError("Model output doesn't require gradients during training!")
        
        # Compute FIXED flow matching loss
        loss, metrics = self.flow_matching_loss(
            model_output=velocity_pred,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            return_metrics=True,
            training_mode=self.training_mode,
        )
        
        # Track training metrics with FIXED normalization and WandB
        self._track_training_metrics_fixed_with_wandb(loss, metrics)
        
        # Run evaluation every N steps
        if self.step_count > 0 and self.step_count % self.eval_every_n_steps == 0:
            self._run_evaluation_during_training()
        
        # Log progress with normalization info
        if self.step_count % self.args.logging_steps == 0:
            self._log_training_progress_fixed(loss, metrics)
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'velocity_prediction': velocity_pred,
            'metrics': metrics,
        } if return_outputs else None
        
        return (loss, outputs) if return_outputs else loss
    
    def _track_training_metrics_fixed_with_wandb(self, loss: torch.Tensor, metrics: Optional[Dict[str, float]]):
        """Track training metrics with FIXED L2 normalization monitoring and WandB logging"""
        self.step_count += 1
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        # Prepare WandB logging data
        wandb_data = {
            "train/step": self.step_count,
            "train/loss": loss_value,
            "train/epoch": getattr(self, '_current_epoch', 0),
        }
        
        # Get learning rate from optimizer
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            current_lr = self.lr_scheduler.get_last_lr()[0]
            wandb_data["train/learning_rate"] = current_lr
        elif hasattr(self, 'optimizer') and self.optimizer is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb_data["train/learning_rate"] = current_lr
        
        if metrics:
            velocity_sim = metrics.get('velocity_cosine_sim', 0.0)
            self.velocity_similarity_history.append(velocity_sim)
            
            # FIXED: Track norm values properly
            clip_target_norm = metrics.get('clip_target_norm', 1.0)  # CLIP embeddings (should be ~1.0)
            prediction_norm = metrics.get('prediction_norm', 1.0)    # Model predictions
            velocity_target_norm = metrics.get('target_norm', 1.0)   # Velocity targets
            
            self.clip_target_norm_history.append(clip_target_norm)
            self.prediction_norm_history.append(prediction_norm)
            self.velocity_target_norm_history.append(velocity_target_norm)
            
            self.last_clip_norm = clip_target_norm
            
            # Add detailed metrics to WandB
            wandb_data.update({
                # Velocity similarity metrics
                "train/velocity_cosine_sim": velocity_sim,
                "train/velocity_per_patch_mean": metrics.get('velocity_per_patch_mean', 0.0),
                "train/velocity_per_patch_std": metrics.get('velocity_per_patch_std', 0.0),
                "train/velocity_high_quality_patches": metrics.get('velocity_high_quality_patches', 0.0),
                "train/velocity_very_high_quality_patches": metrics.get('velocity_very_high_quality_patches', 0.0),
                "train/velocity_high_quality_images": metrics.get('velocity_high_quality_images', 0.0),
                
                # Norm tracking (FIXED)
                "norms/clip_target_norm": clip_target_norm,
                "norms/prediction_norm": prediction_norm,
                "norms/velocity_target_norm": velocity_target_norm,
                "norms/norm_ratio": metrics.get('norm_ratio', 1.0),
                "norms/clip_normalized": metrics.get('clip_normalized', True),
                "norms/targets_properly_normalized": metrics.get('targets_properly_normalized', True),
                
                # EMA metrics
                "train/ema_velocity_cosine": metrics.get('ema_velocity_cosine', 0.0),
                "train/ema_pred_norm": metrics.get('ema_pred_norm', 1.0),
                "train/ema_target_norm": metrics.get('ema_target_norm', 1.0),
                
                # Training progress
                "train/training_steps": metrics.get('training_steps', self.step_count),
                "train/best_velocity_sim": metrics.get('best_velocity_sim', 0.0),
                
                # Implementation status
                "status/normalize_targets": metrics.get('normalize_targets', True),
                "status/double_normalization_avoided": metrics.get('double_normalization_avoided', True),
                "status/l2_normalization_enabled": True,
            })
            
            # Check normalization status
            if abs(clip_target_norm - 1.0) > 0.1:
                self.normalization_warnings += 1
                wandb_data["warnings/normalization_warnings"] = self.normalization_warnings
                if self.normalization_warnings <= 10:  # Limit warnings
                    logger.warning(f"Step {self.step_count}: CLIP target norm {clip_target_norm:.3f} (should be ~1.0)")
            
            # Update best metrics
            if velocity_sim > self.best_velocity_sim:
                self.best_velocity_sim = velocity_sim
                self.steps_without_velocity_improvement = 0
                wandb_data["train/new_best_velocity"] = True
            else:
                self.steps_without_velocity_improvement += 1
                wandb_data["train/new_best_velocity"] = False
                
            wandb_data["train/steps_without_velocity_improvement"] = self.steps_without_velocity_improvement
                
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                wandb_data["train/new_best_loss"] = True
            else:
                wandb_data["train/new_best_loss"] = False
            
            wandb_data["train/best_loss"] = self.best_loss
        
        # Log to WandB
        if self.use_wandb and self.step_count % self.wandb_log_interval == 0:
            self.wandb_instance.log(wandb_data, step=self.step_count)
    
    def _run_evaluation_during_training(self):
        """Run evaluation during training to compute embedding similarity with WandB logging"""
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader available for evaluation")
            return
        
        logger.info(f"🔍 Running evaluation at step {self.step_count}...")
        
        self.model.eval()
        eval_results = self._evaluate_embedding_similarity(
            num_samples=self.eval_num_samples,
            batch_size=self.eval_batch_size,
            inference_steps=self.eval_inference_steps
        )
        self.model.train()
        
        if eval_results:
            embedding_sim = eval_results['overall_embedding_similarity']
            self.embedding_similarity_history.append(embedding_sim)
            
            # Update best embedding similarity
            new_best_embedding = False
            if embedding_sim > self.best_embedding_sim:
                self.best_embedding_sim = embedding_sim
                self.steps_without_embedding_improvement = 0
                new_best_embedding = True
                logger.info(f"🎉 New best embedding similarity: {embedding_sim:.4f}")
            else:
                self.steps_without_embedding_improvement += 1
            
            # Log evaluation results
            logger.info(f"📊 Evaluation Results:")
            logger.info(f"   Overall Embedding Similarity: {embedding_sim:.4f}")
            logger.info(f"   High Quality Images (>0.7): {eval_results['high_quality_images']*100:.1f}%")
            logger.info(f"   Very High Quality Images (>0.8): {eval_results['very_high_quality_images']*100:.1f}%")
            logger.info(f"   Best so far: {self.best_embedding_sim:.4f}")
            
            # Log comprehensive evaluation metrics to WandB
            if self.use_wandb:
                eval_wandb_data = {
                    "eval/step": self.step_count,
                    "eval/overall_embedding_similarity": embedding_sim,
                    "eval/per_image_mean": eval_results.get('per_image_mean', 0.0),
                    "eval/per_image_std": eval_results.get('per_image_std', 0.0),
                    "eval/per_patch_mean": eval_results.get('per_patch_mean', 0.0),
                    "eval/per_patch_std": eval_results.get('per_patch_std', 0.0),
                    "eval/high_quality_images": eval_results.get('high_quality_images', 0.0),
                    "eval/very_high_quality_images": eval_results.get('very_high_quality_images', 0.0),
                    "eval/excellent_quality_images": eval_results.get('excellent_quality_images', 0.0),
                    "eval/samples_evaluated": eval_results.get('samples_evaluated', 0),
                    "eval/inference_steps": eval_results.get('inference_steps', self.eval_inference_steps),
                    "eval/best_embedding_sim": self.best_embedding_sim,
                    "eval/new_best_embedding": new_best_embedding,
                    "eval/steps_without_embedding_improvement": self.steps_without_embedding_improvement,
                }
                
                self.wandb_instance.log(eval_wandb_data, step=self.step_count)
                logger.info(f"✅ Evaluation metrics logged to WandB")
    
    def _evaluate_embedding_similarity(
        self, 
        num_samples: int = 1000,
        batch_size: int = 16,
        inference_steps: int = 50
    ) -> Optional[Dict[str, float]]:
        """Evaluate embedding similarity using the model's generation capability"""
        try:
            all_similarities = []
            all_per_image_sims = []
            samples_processed = 0
            
            eval_iterator = iter(self.eval_dataloader)
            
            with torch.no_grad():
                while samples_processed < num_samples:
                    try:
                        batch = next(eval_iterator)
                    except StopIteration:
                        break
                    
                    # Move batch to device
                    eva_features = batch['encoder_hidden_states'].to(self.model.device)
                    target_embeddings = batch['clip_embeddings'].to(self.model.device)
                    
                    current_batch_size = eva_features.shape[0]
                    
                    # Evaluate this batch
                    batch_results = self.model.evaluate_similarity(
                        eva_features=eva_features,
                        target_embeddings=target_embeddings,
                        num_inference_steps=inference_steps,
                        normalize_embeddings=True
                    )
                    
                    # Collect results
                    # Compute per-patch similarities for this batch
                    generated = self.model.generate(
                        eva_features=eva_features,
                        num_inference_steps=inference_steps,
                        normalize_output=True
                    )
                    
                    targets_norm = F.normalize(target_embeddings, p=2, dim=-1)
                    per_patch_sim = F.cosine_similarity(generated, targets_norm, dim=-1)
                    per_image_sim = per_patch_sim.mean(dim=1)
                    
                    all_similarities.append(per_patch_sim.cpu())
                    all_per_image_sims.append(per_image_sim.cpu())
                    
                    samples_processed += current_batch_size
                    
                    if samples_processed >= num_samples:
                        break
            
            if not all_similarities:
                logger.warning("No evaluation samples processed")
                return None
            
            # Aggregate results
            all_patch_sims = torch.cat(all_similarities, dim=0)
            all_image_sims = torch.cat(all_per_image_sims, dim=0)
            
            # Compute final metrics
            results = {
                'overall_embedding_similarity': all_image_sims.mean().item(),
                'per_image_mean': all_image_sims.mean().item(),
                'per_image_std': all_image_sims.std().item(),
                'per_patch_mean': all_patch_sims.mean().item(),
                'per_patch_std': all_patch_sims.std().item(),
                'high_quality_images': (all_image_sims > 0.7).float().mean().item(),
                'very_high_quality_images': (all_image_sims > 0.8).float().mean().item(),
                'excellent_quality_images': (all_image_sims > 0.9).float().mean().item(),
                'samples_evaluated': samples_processed,
                'inference_steps': inference_steps,
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return None
    
    def _log_training_progress_fixed(self, loss: torch.Tensor, metrics: Optional[Dict[str, float]]):
        """Log detailed training progress with FIXED normalization info"""
        loss_value = loss.item()
        
        # Basic progress
        progress_msg = f"Step {self.step_count}: Loss={loss_value:.6f}"
        
        if metrics:
            # Velocity similarity (training metric)
            velocity_sim = metrics.get('velocity_cosine_sim', 0.0)
            progress_msg += f", VelSim={velocity_sim:.4f}"
            
            # Embedding similarity (if available from recent evaluation)
            if self.embedding_similarity_history:
                latest_emb_sim = self.embedding_similarity_history[-1]
                progress_msg += f", EmbSim={latest_emb_sim:.4f}"
            
            # FIXED: Norm tracking with proper labels
            clip_norm = metrics.get('clip_target_norm', 1.0)       # CLIP embeddings (should be ~1.0)
            pred_norm = metrics.get('prediction_norm', 1.0)        # Model predictions
            velocity_norm = metrics.get('target_norm', 1.0)        # Velocity targets
            
            progress_msg += f", ClipNorm={clip_norm:.3f}, PredNorm={pred_norm:.3f}, VelNorm={velocity_norm:.3f}"
            
            # Normalization status
            norm_status = "✅" if abs(clip_norm - 1.0) < 0.1 else "⚠️"
            progress_msg += f" {norm_status}"
            
            # Best metrics
            progress_msg += f" | Best: Vel={self.best_velocity_sim:.4f}, Emb={self.best_embedding_sim:.4f}"
        
        progress_msg += f" [{self.training_mode}]"
        if self.use_wandb:
            progress_msg += f" [WandB: {self.wandb_instance.run.url}]"
        
        logger.info(progress_msg)
        
        # Detailed summary every 50 steps
        if self.step_count % (self.args.logging_steps * 5) == 0:
            self._log_detailed_summary_fixed()
    
    def _log_detailed_summary_fixed(self):
        """Log detailed training summary with FIXED normalization tracking"""
        logger.info("=" * 80)
        logger.info(f"📊 TRAINING SUMMARY - Step {self.step_count}")
        logger.info("=" * 80)
        
        # Loss summary
        if len(self.loss_history) > 0:
            recent_loss = sum(self.loss_history[-10:]) / min(10, len(self.loss_history))
            logger.info(f"📉 Loss: Current={self.loss_history[-1]:.6f}, Recent avg={recent_loss:.6f}, Best={self.best_loss:.6f}")
        
        # Velocity similarity summary
        if len(self.velocity_similarity_history) > 0:
            recent_vel = sum(self.velocity_similarity_history[-10:]) / min(10, len(self.velocity_similarity_history))
            logger.info(f"🎯 Velocity Similarity: Current={self.velocity_similarity_history[-1]:.4f}, Recent avg={recent_vel:.4f}, Best={self.best_velocity_sim:.4f}")
        
        # Embedding similarity summary
        if len(self.embedding_similarity_history) > 0:
            recent_emb = sum(self.embedding_similarity_history[-5:]) / min(5, len(self.embedding_similarity_history))
            logger.info(f"🎯 Embedding Similarity: Recent avg={recent_emb:.4f}, Best={self.best_embedding_sim:.4f}")
            logger.info(f"   Evaluations performed: {len(self.embedding_similarity_history)}")
        
        # FIXED: Normalization summary
        if len(self.clip_target_norm_history) > 0:
            recent_clip_norm = sum(self.clip_target_norm_history[-10:]) / min(10, len(self.clip_target_norm_history))
            recent_pred_norm = sum(self.prediction_norm_history[-10:]) / min(10, len(self.prediction_norm_history))
            recent_vel_norm = sum(self.velocity_target_norm_history[-10:]) / min(10, len(self.velocity_target_norm_history))
            
            logger.info(f"📏 Normalization Status:")
            logger.info(f"   CLIP targets: {recent_clip_norm:.3f} (should be ~1.0) {'✅' if abs(recent_clip_norm - 1.0) < 0.1 else '⚠️'}")
            logger.info(f"   Predictions:  {recent_pred_norm:.3f}")
            logger.info(f"   Vel targets:  {recent_vel_norm:.3f}")
            logger.info(f"   Warnings issued: {self.normalization_warnings}")
        
        # Improvement tracking
        logger.info(f"📈 Steps without improvement:")
        logger.info(f"   Velocity: {self.steps_without_velocity_improvement}")
        logger.info(f"   Embedding: {self.steps_without_embedding_improvement}")
        
        # Training health assessment
        health = self._assess_training_health_fixed()
        logger.info(f"🏥 Training Health: {health}")
        
        # WandB status
        if self.use_wandb:
            logger.info(f"📊 WandB Run: {self.wandb_instance.run.url}")
            logger.info(f"📊 All metrics logged to WandB dashboard")
        
        logger.info("=" * 80)
    
    def _assess_training_health_fixed(self) -> str:
        """Assess training health including normalization status"""
        if len(self.velocity_similarity_history) < 10:
            return "STARTING - Not enough data yet"
        
        recent_vel = sum(self.velocity_similarity_history[-10:]) / 10
        recent_emb = sum(self.embedding_similarity_history[-3:]) / max(3, len(self.embedding_similarity_history[-3:])) if self.embedding_similarity_history else 0
        
        # Check normalization
        norm_ok = True
        if len(self.clip_target_norm_history) > 0:
            recent_clip_norm = sum(self.clip_target_norm_history[-10:]) / min(10, len(self.clip_target_norm_history))
            norm_ok = abs(recent_clip_norm - 1.0) < 0.2
        
        if not norm_ok:
            return "NORMALIZATION ISSUE - CLIP targets not properly normalized"
        elif recent_vel < 0.01:
            return "POOR - Very low velocity similarity"
        elif recent_vel > 0.1 and recent_emb < 0.05:
            return "CONCERNING - Good velocity but poor embeddings"
        elif recent_vel > 0.1 and recent_emb > 0.1:
            return "EXCELLENT - Both metrics improving well"
        elif recent_vel > 0.05 and recent_emb > 0.05:
            return "GOOD - Steady improvement"
        elif recent_vel > 0.02:
            return "LEARNING - Making progress"
        else:
            return "SLOW - May need hyperparameter tuning"
    
    def get_final_evaluation(self) -> Dict[str, Any]:
        """Get comprehensive final evaluation with WandB logging"""
        logger.info("🔍 Running final comprehensive evaluation...")
        
        self.model.eval()
        final_results = self._evaluate_embedding_similarity(
            num_samples=min(5000, self.eval_num_samples * 5),  # More samples for final eval
            batch_size=self.eval_batch_size,
            inference_steps=self.eval_inference_steps
        )
        self.model.train()
        
        # Training summary with FIXED normalization info
        training_summary = {
            'total_steps': self.step_count,
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'best_loss': self.best_loss,
            'final_velocity_sim': self.velocity_similarity_history[-1] if self.velocity_similarity_history else 0,
            'best_velocity_sim': self.best_velocity_sim,
            'final_embedding_sim': self.embedding_similarity_history[-1] if self.embedding_similarity_history else 0,
            'best_embedding_sim': self.best_embedding_sim,
            'training_health': self._assess_training_health_fixed(),
            'evaluations_performed': len(self.embedding_similarity_history),
            
            # FIXED: Normalization summary
            'final_clip_norm': self.clip_target_norm_history[-1] if self.clip_target_norm_history else 1.0,
            'final_prediction_norm': self.prediction_norm_history[-1] if self.prediction_norm_history else 1.0,
            'normalization_warnings': self.normalization_warnings,
            'clip_norms_properly_normalized': self.last_clip_norm is not None and abs(self.last_clip_norm - 1.0) < 0.1,
            
            # WandB info
            'wandb_enabled': self.use_wandb,
            'wandb_url': self.wandb_instance.run.url if self.use_wandb else None,
        }
        
        # Log final evaluation to WandB
        if self.use_wandb and final_results:
            final_wandb_data = {
                "final_eval/overall_embedding_similarity": final_results.get('overall_embedding_similarity', 0),
                "final_eval/high_quality_images": final_results.get('high_quality_images', 0),
                "final_eval/very_high_quality_images": final_results.get('very_high_quality_images', 0),
                "final_eval/excellent_quality_images": final_results.get('excellent_quality_images', 0),
                "final_eval/samples_evaluated": final_results.get('samples_evaluated', 0),
                "final_eval/per_image_mean": final_results.get('per_image_mean', 0),
                "final_eval/per_image_std": final_results.get('per_image_std', 0),
                "final_eval/per_patch_mean": final_results.get('per_patch_mean', 0),
                "final_eval/per_patch_std": final_results.get('per_patch_std', 0),
            }
            
            self.wandb_instance.log(final_wandb_data, step=self.step_count)
            logger.info("✅ Final evaluation results logged to WandB")
        
        return {
            'training_summary': training_summary,
            'final_evaluation': final_results,
            'loss_history': self.loss_history,
            'velocity_similarity_history': self.velocity_similarity_history,
            'embedding_similarity_history': self.embedding_similarity_history,
            'clip_target_norm_history': self.clip_target_norm_history,
            'prediction_norm_history': self.prediction_norm_history,
            'velocity_target_norm_history': self.velocity_target_norm_history,
        }


def create_training_args(
    output_dir: str,
    num_train_epochs: int = 15,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    dataloader_num_workers: int = 0,
    logging_steps: int = 10,
    save_steps: int = 200,
    report_to: list = None,
    **kwargs
) -> TrainingArguments:
    """Create training arguments for BLIP3-o training with WandB support"""
    
    if report_to is None:
        report_to = []
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=3,
        eval_strategy="no",  # We handle evaluation manually
        prediction_loss_only=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=report_to,
        **kwargs
    )