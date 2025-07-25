#!/usr/bin/env python3
"""
BLIP3-o Trainer - Simple and Working Implementation
src/modules/trainers/blip3o_trainer.py

Simple trainer that actually works without complex dependencies
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
    Simple BLIP3-o trainer that works
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
        
        # Training metrics
        self.step_count = 0
        self.loss_history = []
        
        logger.info(f"✅ BLIP3o Trainer initialized")
        logger.info(f"   Training mode: {training_mode} ({self.expected_tokens} tokens)")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with flow matching"""
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
        
        # Compute flow matching loss
        loss, metrics = self.flow_matching_loss(
            model_output=velocity_pred,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            return_metrics=True,
            training_mode=self.training_mode,
        )
        
        # Track metrics
        self.step_count += 1
        self.loss_history.append(loss.item())
        
        # Log progress
        if self.step_count % self.args.logging_steps == 0:
            self._log_progress(loss, metrics)
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'velocity_prediction': velocity_pred,
            'metrics': metrics,
        } if return_outputs else None
        
        return (loss, outputs) if return_outputs else loss
    
    def _log_progress(self, loss: torch.Tensor, metrics: Optional[Dict[str, float]]):
        """Log training progress with patch-wise cosine similarity"""
        loss_value = loss.item()
        
        progress_msg = f"Step {self.step_count}: Loss={loss_value:.6f}"
        
        if metrics:
            # Use new patch-wise cosine similarity (matches evaluation methodology)
            if 'patch_wise_cosine_sim' in metrics:
                progress_msg += f", PatchCos={metrics['patch_wise_cosine_sim']:.3f}"
            
            # Add quality metrics for quick assessment
            if 'high_quality_images_ratio' in metrics:
                hq_images_pct = metrics['high_quality_images_ratio'] * 100
                progress_msg += f", HQ_Images={hq_images_pct:.1f}%"
            
            # Keep norm tracking for debugging
            if 'prediction_norm' in metrics:
                progress_msg += f", PredNorm={metrics['prediction_norm']:.3f}"
            if 'target_norm' in metrics:
                progress_msg += f", TargetNorm={metrics['target_norm']:.3f}"
            
            # Add adaptive scale info
            if 'adaptive_scale' in metrics:
                progress_msg += f", Scale={metrics['adaptive_scale']:.3f}"
        
        progress_msg += f" [{self.training_mode}]"
        logger.info(progress_msg)


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
    """Create training arguments"""
    
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
        eval_strategy="no",  # No evaluation during training
        prediction_loss_only=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=[],
        **kwargs
    )