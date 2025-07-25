#!/usr/bin/env python3
"""
FIXED: BLIP3-o Trainer with Proper Evaluation During Training
src/modules/trainers/blip3o_trainer.py

KEY FIXES:
1. Proper evaluation every N steps
2. Both velocity and embedding similarity tracking
3. Clean training metrics aligned with BLIP3-o paper
4. Evaluation matches training metrics
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
    FIXED: BLIP3-o trainer with proper evaluation during training
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
        
        # Training metrics tracking
        self.step_count = 0
        self.loss_history = []
        self.velocity_similarity_history = []
        self.embedding_similarity_history = []
        
        # Best metrics tracking
        self.best_velocity_sim = 0.0
        self.best_embedding_sim = 0.0
        self.best_loss = float('inf')
        
        # Steps without improvement (for monitoring)
        self.steps_without_velocity_improvement = 0
        self.steps_without_embedding_improvement = 0
        
        logger.info(f"✅ FIXED BLIP3o Trainer initialized")
        logger.info(f"   Training mode: {training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"   Evaluation every: {eval_every_n_steps} steps")
        logger.info(f"   Evaluation samples: {eval_num_samples}")
        logger.info(f"   Evaluation inference steps: {eval_inference_steps}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with clean flow matching"""
        model.train()
        
        # Extract inputs with better error handling
        try:
            eva_embeddings = inputs['encoder_hidden_states']  # [B, N, 4096]
            clip_embeddings = inputs['clip_embeddings']       # [B, N, 1024]
            timesteps = inputs['timestep']                    # [B]
        except KeyError as e:
            logger.error(f"Missing input key: {e}")
            logger.error(f"Available keys: {list(inputs.keys())}")
            raise
        
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
        
        # Compute clean flow matching loss
        loss, metrics = self.flow_matching_loss(
            model_output=velocity_pred,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            return_metrics=True,
            training_mode=self.training_mode,
        )
        
        # Track training metrics
        self._track_training_metrics(loss, metrics)
        
        # Run evaluation every N steps
        if self.step_count > 0 and self.step_count % self.eval_every_n_steps == 0:
            self._run_evaluation_during_training()
        
        # Log progress
        if self.step_count % self.args.logging_steps == 0:
            self._log_training_progress(loss, metrics)
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'velocity_prediction': velocity_pred,
            'metrics': metrics,
        } if return_outputs else None
        
        return (loss, outputs) if return_outputs else loss
    
    def _track_training_metrics(self, loss: torch.Tensor, metrics: Optional[Dict[str, float]]):
        """Track training metrics"""
        self.step_count += 1
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        if metrics:
            velocity_sim = metrics.get('velocity_cosine_sim', 0.0)
            self.velocity_similarity_history.append(velocity_sim)
            
            # Update best metrics
            if velocity_sim > self.best_velocity_sim:
                self.best_velocity_sim = velocity_sim
                self.steps_without_velocity_improvement = 0
            else:
                self.steps_without_velocity_improvement += 1
                
            if loss_value < self.best_loss:
                self.best_loss = loss_value
    
    def _run_evaluation_during_training(self):
        """Run evaluation during training to compute embedding similarity"""
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
            if embedding_sim > self.best_embedding_sim:
                self.best_embedding_sim = embedding_sim
                self.steps_without_embedding_improvement = 0
                logger.info(f"🎉 New best embedding similarity: {embedding_sim:.4f}")
            else:
                self.steps_without_embedding_improvement += 1
            
            # Log evaluation results
            logger.info(f"📊 Evaluation Results:")
            logger.info(f"   Overall Embedding Similarity: {embedding_sim:.4f}")
            logger.info(f"   High Quality Images (>0.7): {eval_results['high_quality_images']*100:.1f}%")
            logger.info(f"   Very High Quality Images (>0.8): {eval_results['very_high_quality_images']*100:.1f}%")
            logger.info(f"   Best so far: {self.best_embedding_sim:.4f}")
    
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
    
    def _log_training_progress(self, loss: torch.Tensor, metrics: Optional[Dict[str, float]]):
        """Log detailed training progress"""
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
            
            # Norm tracking
            pred_norm = metrics.get('prediction_norm', 0.0)
            target_norm = metrics.get('target_norm', 0.0)
            progress_msg += f", PredNorm={pred_norm:.3f}, TargetNorm={target_norm:.3f}"
            
            # Best metrics
            progress_msg += f" | Best: Vel={self.best_velocity_sim:.4f}, Emb={self.best_embedding_sim:.4f}"
        
        progress_msg += f" [{self.training_mode}]"
        logger.info(progress_msg)
        
        # Detailed summary every 50 steps
        if self.step_count % (self.args.logging_steps * 5) == 0:
            self._log_detailed_summary()
    
    def _log_detailed_summary(self):
        """Log detailed training summary"""
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
        
        # Improvement tracking
        logger.info(f"📈 Steps without improvement:")
        logger.info(f"   Velocity: {self.steps_without_velocity_improvement}")
        logger.info(f"   Embedding: {self.steps_without_embedding_improvement}")
        
        # Training health assessment
        health = self._assess_training_health()
        logger.info(f"🏥 Training Health: {health}")
        
        logger.info("=" * 80)
    
    def _assess_training_health(self) -> str:
        """Assess training health"""
        if len(self.velocity_similarity_history) < 10:
            return "STARTING - Not enough data yet"
        
        recent_vel = sum(self.velocity_similarity_history[-10:]) / 10
        recent_emb = sum(self.embedding_similarity_history[-3:]) / max(3, len(self.embedding_similarity_history[-3:])) if self.embedding_similarity_history else 0
        
        if recent_vel < 0.01:
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
        """Get comprehensive final evaluation"""
        logger.info("🔍 Running final comprehensive evaluation...")
        
        self.model.eval()
        final_results = self._evaluate_embedding_similarity(
            num_samples=min(5000, self.eval_num_samples * 5),  # More samples for final eval
            batch_size=self.eval_batch_size,
            inference_steps=self.eval_inference_steps
        )
        self.model.train()
        
        # Training summary
        training_summary = {
            'total_steps': self.step_count,
            'final_loss': self.loss_history[-1] if self.loss_history else 0,
            'best_loss': self.best_loss,
            'final_velocity_sim': self.velocity_similarity_history[-1] if self.velocity_similarity_history else 0,
            'best_velocity_sim': self.best_velocity_sim,
            'final_embedding_sim': self.embedding_similarity_history[-1] if self.embedding_similarity_history else 0,
            'best_embedding_sim': self.best_embedding_sim,
            'training_health': self._assess_training_health(),
            'evaluations_performed': len(self.embedding_similarity_history),
        }
        
        return {
            'training_summary': training_summary,
            'final_evaluation': final_results,
            'loss_history': self.loss_history,
            'velocity_similarity_history': self.velocity_similarity_history,
            'embedding_similarity_history': self.embedding_similarity_history,
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
    **kwargs
) -> TrainingArguments:
    """Create training arguments for BLIP3-o training"""
    
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
        report_to=[],
        **kwargs
    )