#!/usr/bin/env python3
"""
FIXED: BLIP3-o Unified Trainer - Core Training Logic
src/modules/trainers/blip3o_unified_trainer.py

FIXES:
- Fixed eval_strategy parameter name
- Streamlined training logic
- Removed redundant code
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import numpy as np
import json
import time
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class BLIP3oUnifiedTrainer(Trainer):
    """
    BLIP3-o Unified Trainer for DiT training with flow matching
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
        enable_evaluation: bool = False,
        detailed_logging: bool = True,
        expected_velocity_scale: float = 0.1,
        expected_output_scale: float = 0.1,
        **kwargs
    ):
        
        # Disable evaluation if not enabled
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
        self.detailed_logging = detailed_logging
        self.expected_velocity_scale = expected_velocity_scale
        self.expected_output_scale = expected_output_scale
        
        # Expected token count
        self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        # Training metrics
        self.training_step_count = 0
        self.loss_history = []
        self.metric_history = []
        
        # Distributed training
        self.is_distributed = dist.is_initialized()
        self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        
        if self.is_main_process:
            logger.info("✅ BLIP3-o Unified Trainer initialized")
            logger.info(f"🎯 Training mode: {self.training_mode} ({self.expected_tokens} tokens)")
            logger.info(f"📊 Evaluation enabled: {self.enable_evaluation}")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
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
            # Create noisy input
            device = eva_embeddings.device
            noise = torch.randn_like(clip_embeddings)
            alpha = timesteps.view(-1, 1, 1)
            noisy_clip_base = (1 - alpha) * noise + alpha * clip_embeddings.detach()

        # Ensure gradients for training
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
            noise=inputs.get('noise', torch.randn_like(clip_embeddings)),
            return_metrics=True,
            training_mode=self.training_mode,
        )
        
        # Track metrics
        if self.is_main_process:
            self.loss_history.append(loss.item())
            if metrics:
                self.metric_history.append({
                    **metrics,
                    'step': self.training_step_count,
                    'timestamp': time.time(),
                })
            
            # Log progress
            if (self.detailed_logging and 
                self.training_step_count % self.args.logging_steps == 0):
                self._log_progress(loss, metrics)
        
        self.training_step_count += 1
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'velocity_prediction': velocity_pred,
            'target_samples': clip_embeddings,
            'metrics': metrics,
        } if return_outputs else None
        
        return (loss, outputs) if return_outputs else loss

    def _log_progress(self, loss: torch.Tensor, metrics: Optional[Dict[str, float]]):
        """Log training progress"""
        loss_value = loss.item()
        
        progress_msg = f"Step {self.training_step_count}: Loss={loss_value:.6f}"
        
        if metrics:
            if 'velocity_cosine_sim' in metrics:
                progress_msg += f", VelCos={metrics['velocity_cosine_sim']:.3f}"
            if 'prediction_norm' in metrics:
                progress_msg += f", PredNorm={metrics['prediction_norm']:.3f}"
            if 'target_norm' in metrics:
                progress_msg += f", TargetNorm={metrics['target_norm']:.3f}"
            if 'final_embedding_similarity' in metrics and metrics['final_embedding_similarity']:
                progress_msg += f", FinalSim={metrics['final_embedding_similarity']:.3f}"
        
        progress_msg += f" [{self.training_mode}]"
        logger.info(progress_msg)

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = {
            'training_mode': self.training_mode,
            'expected_tokens': self.expected_tokens,
            'total_steps': self.training_step_count,
            'evaluation_enabled': self.enable_evaluation,
        }
        
        if self.loss_history:
            stats['loss_statistics'] = {
                'current_loss': self.loss_history[-1],
                'min_loss': min(self.loss_history),
                'max_loss': max(self.loss_history),
                'avg_loss': sum(self.loss_history) / len(self.loss_history),
            }
        
        if self.metric_history:
            latest_metrics = self.metric_history[-1]
            stats['latest_training_metrics'] = latest_metrics
        
        return stats


def create_unified_training_args(
    output_dir: str,
    enable_evaluation: bool = False,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 200,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    dataloader_num_workers: int = 0,
    eval_steps: int = 50,
    **kwargs
) -> TrainingArguments:
    """
    Create training arguments for unified trainer
    FIXED: Use eval_strategy instead of evaluation_strategy
    """
    
    # Set evaluation strategy
    if enable_evaluation:
        eval_strategy_val = "steps"
        metric_for_best_model = "eval_loss"
        load_best_model_at_end = True
        greater_is_better = False
    else:
        eval_strategy_val = "no"
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
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy=eval_strategy_val,  # FIXED: Use eval_strategy
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
        prediction_loss_only=False,
        report_to=[],
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        ignore_data_skip=True,
        **kwargs
    )