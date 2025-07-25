#!/usr/bin/env python3
"""
FIXED: BLIP3-o Trainer with Both Velocity and Embedding Similarity Tracking
src/modules/trainers/blip3o_trainer.py
"""

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, Union, Tuple
import logging
import time
import os

logger = logging.getLogger(__name__)


class BLIP3oTrainer(Trainer):
    """
    FIXED: BLIP3-o trainer that tracks BOTH velocity and embedding similarity
    """
    
    def __init__(
        self,
        model,
        args: TrainingArguments,
        flow_matching_loss,
        train_dataset=None,
        eval_dataset=None,
        training_mode: str = "patch_only",
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
        
        # Training metrics tracking
        self.step_count = 0
        self.loss_history = []
        self.velocity_similarity_history = []
        self.embedding_similarity_history = []
        
        # Best metrics tracking
        self.best_velocity_sim = 0.0
        self.best_embedding_sim = 0.0
        self.best_loss = float('inf')
        
        # Overfitting detection
        self.steps_without_velocity_improvement = 0
        self.steps_without_embedding_improvement = 0
        self.overfitting_threshold = 50  # Steps
        
        logger.info(f"✅ FIXED BLIP3o Trainer initialized")
        logger.info(f"   Training mode: {training_mode} ({self.expected_tokens} tokens)")
        logger.info(f"   Tracking: Velocity similarity + Embedding similarity")
        logger.info(f"   Overfitting detection: {self.overfitting_threshold} steps")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with both velocity and embedding tracking"""
        model.train()
        
        # Extract inputs
        try:
            eva_embeddings = inputs['encoder_hidden_states']
            clip_embeddings = inputs['clip_embeddings'] 
            timesteps = inputs['timestep']
        except KeyError as e:
            logger.error(f"Missing input key: {e}")
            logger.error(f"Available keys: {list(inputs.keys())}")
            raise
        
        # Handle noisy input
        if 'hidden_states' in inputs:
            noisy_clip_base = inputs['hidden_states']
        else:
            # Create noisy input for flow matching
            device = eva_embeddings.device
            noise = torch.randn_like(clip_embeddings)
            alpha = timesteps.view(-1, 1, 1)
            noisy_clip_base = (1 - alpha) * noise + alpha * clip_embeddings.detach()
        
        # Ensure gradients
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
        
        # Check gradients
        if not velocity_pred.requires_grad:
            raise RuntimeError("Model output doesn't require gradients during training!")
        
        # Compute flow matching loss with BOTH velocity and embedding tracking
        loss, metrics = self.flow_matching_loss(
            model_output=velocity_pred,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            return_metrics=True,
            training_mode=self.training_mode,
            model_ref=model,  # Pass model reference for embedding testing
        )
        
        # Track metrics and detect improvements
        self.step_count += 1
        self.loss_history.append(loss.item())
        
        if metrics:
            velocity_sim = metrics.get('velocity_cosine_sim', 0.0)
            embedding_sim = metrics.get('embedding_cosine_sim', 0.0)
            
            self.velocity_similarity_history.append(velocity_sim)
            self.embedding_similarity_history.append(embedding_sim)
            
            # Update best metrics and detect overfitting
            if velocity_sim > self.best_velocity_sim:
                self.best_velocity_sim = velocity_sim
                self.steps_without_velocity_improvement = 0
            else:
                self.steps_without_velocity_improvement += 1
            
            if embedding_sim > self.best_embedding_sim:
                self.best_embedding_sim = embedding_sim
                self.steps_without_embedding_improvement = 0
            elif embedding_sim > 0:  # Only count if embedding test was performed
                self.steps_without_embedding_improvement += 1
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
        
        # Log progress
        if self.step_count % self.args.logging_steps == 0:
            self._log_detailed_progress(loss, metrics)
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'velocity_prediction': velocity_pred,
            'metrics': metrics,
        } if return_outputs else None
        
        return (loss, outputs) if return_outputs else loss
    
    def _log_detailed_progress(self, loss: torch.Tensor, metrics: Optional[Dict[str, float]]):
        """Log detailed training progress with both velocity and embedding similarity"""
        loss_value = loss.item()
        
        # Basic progress
        progress_msg = f"Step {self.step_count}: Loss={loss_value:.6f}"
        
        if metrics:
            # VELOCITY metrics (what we're training on)
            velocity_sim = metrics.get('velocity_cosine_sim', 0.0)
            progress_msg += f", VelCos={velocity_sim:.3f}"
            
            # EMBEDDING metrics (what we actually care about)
            embedding_sim = metrics.get('embedding_cosine_sim', 0.0)
            if embedding_sim > 0:
                progress_msg += f", EmbCos={embedding_sim:.3f}"
            else:
                progress_msg += f", EmbCos=pending"
            
            # Quality indicators
            velocity_hq_images = metrics.get('velocity_high_quality_images', 0.0) * 100
            progress_msg += f", VelHQ={velocity_hq_images:.1f}%"
            
            # Norm tracking for debugging
            pred_norm = metrics.get('prediction_norm', 0.0)
            target_norm = metrics.get('target_norm', 0.0)
            progress_msg += f", PredNorm={pred_norm:.3f}, TargetNorm={target_norm:.3f}"
            
            # Scaling info
            adaptive_scale = metrics.get('adaptive_scale', 1.0)
            velocity_scale = metrics.get('velocity_scale', 0.1)
            progress_msg += f", Scale={adaptive_scale:.3f}, VelScale={velocity_scale}"
            
            # Best metrics so far
            progress_msg += f" | Best: VelCos={self.best_velocity_sim:.3f}, EmbCos={self.best_embedding_sim:.3f}"
            
            # Overfitting warning
            if self.steps_without_velocity_improvement > self.overfitting_threshold:
                progress_msg += " ⚠️OVERFITTING?"
        
        progress_msg += f" [{self.training_mode}]"
        logger.info(progress_msg)
        
        # Detailed progress every 50 steps
        if self.step_count % (self.args.logging_steps * 5) == 0:
            self._log_training_summary()
    
    def _log_training_summary(self):
        """Log detailed training summary"""
        logger.info("=" * 80)
        logger.info(f"📊 TRAINING SUMMARY - Step {self.step_count}")
        logger.info("=" * 80)
        
        if len(self.loss_history) > 0:
            recent_loss = sum(self.loss_history[-10:]) / min(10, len(self.loss_history))
            logger.info(f"📉 Loss: Current={self.loss_history[-1]:.6f}, Recent avg={recent_loss:.6f}, Best={self.best_loss:.6f}")
        
        if len(self.velocity_similarity_history) > 0:
            recent_vel_sim = sum(self.velocity_similarity_history[-10:]) / min(10, len(self.velocity_similarity_history))
            logger.info(f"🎯 Velocity Similarity: Current={self.velocity_similarity_history[-1]:.4f}, Recent avg={recent_vel_sim:.4f}, Best={self.best_velocity_sim:.4f}")
        
        if len(self.embedding_similarity_history) > 0:
            valid_embedding_sims = [x for x in self.embedding_similarity_history[-10:] if x > 0]
            if valid_embedding_sims:
                recent_emb_sim = sum(valid_embedding_sims) / len(valid_embedding_sims)
                logger.info(f"🎯 Embedding Similarity: Recent avg={recent_emb_sim:.4f}, Best={self.best_embedding_sim:.4f}")
            else:
                logger.info(f"🎯 Embedding Similarity: No recent tests, Best={self.best_embedding_sim:.4f}")
        
        # Overfitting analysis
        logger.info(f"📈 Improvement Status:")
        logger.info(f"   Velocity: {self.steps_without_velocity_improvement} steps without improvement")
        logger.info(f"   Embedding: {self.steps_without_embedding_improvement} steps without improvement")
        
        if self.steps_without_velocity_improvement > self.overfitting_threshold:
            logger.warning(f"⚠️  VELOCITY OVERFITTING DETECTED: {self.steps_without_velocity_improvement} steps without improvement")
        
        if self.steps_without_embedding_improvement > self.overfitting_threshold:
            logger.warning(f"⚠️  EMBEDDING OVERFITTING DETECTED: {self.steps_without_embedding_improvement} steps without improvement")
        
        # Training health check
        if self.best_velocity_sim > 0.1 and self.best_embedding_sim < 0.05:
            logger.warning("⚠️  TRAINING ISSUE: Good velocity prediction but poor embedding generation!")
            logger.warning("     This suggests generation process bugs.")
        elif self.best_velocity_sim > 0.1 and self.best_embedding_sim > 0.08:
            logger.info("✅ HEALTHY TRAINING: Both velocity and embedding similarities improving")
        
        logger.info("=" * 80)
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics"""
        return {
            'step_count': self.step_count,
            'loss_history': self.loss_history,
            'velocity_similarity_history': self.velocity_similarity_history,
            'embedding_similarity_history': self.embedding_similarity_history,
            'best_velocity_sim': self.best_velocity_sim,
            'best_embedding_sim': self.best_embedding_sim,
            'best_loss': self.best_loss,
            'steps_without_velocity_improvement': self.steps_without_velocity_improvement,
            'steps_without_embedding_improvement': self.steps_without_embedding_improvement,
            'overfitting_detected': {
                'velocity': self.steps_without_velocity_improvement > self.overfitting_threshold,
                'embedding': self.steps_without_embedding_improvement > self.overfitting_threshold,
            },
            'training_health': self._assess_training_health(),
        }
    
    def _assess_training_health(self) -> str:
        """Assess overall training health"""
        if self.best_velocity_sim < 0.05:
            return "POOR - Low velocity similarity"
        elif self.best_velocity_sim > 0.1 and self.best_embedding_sim < 0.05:
            return "PROBLEMATIC - Good velocity but poor embeddings"
        elif self.best_velocity_sim > 0.1 and self.best_embedding_sim > 0.08:
            return "HEALTHY - Both metrics improving"
        elif self.steps_without_velocity_improvement > self.overfitting_threshold:
            return "OVERFITTING - No velocity improvement"
        else:
            return "LEARNING - Making progress"


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
    logging_steps: int = 5,  # More frequent logging to see both metrics
    save_steps: int = 200,
    **kwargs
) -> TrainingArguments:
    """Create training arguments optimized for tracking both metrics"""
    
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
        eval_strategy="no",
        prediction_loss_only=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=[],
        **kwargs
    )