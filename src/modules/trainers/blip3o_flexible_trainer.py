"""
FINAL FIXED: Enhanced BLIP3-o Trainer - Training & Evaluation Compatible
src/modules/trainers/blip3o_flexible_trainer.py

KEY FIXES:
1. Proper evaluation loop that handles gradient-free evaluation
2. Fixed metric_for_best_model configuration
3. Better learning rate defaults
4. Compatible with both training and evaluation modes
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Trainer, TrainingArguments
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


class BLIP3oFlexibleTrainer(Trainer):
    """
    FINAL FIXED: Enhanced BLIP3-o Trainer with Training & Evaluation Compatibility
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
        training_mode: str = "cls_patch",
        max_training_shards: Optional[int] = None,
        enable_same_data_eval: bool = False,
        eval_frequency: int = 100,
        detailed_logging: bool = True,
        **kwargs
    ):
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
        self.max_training_shards = max_training_shards
        self.enable_same_data_eval = enable_same_data_eval
        self.eval_frequency = eval_frequency
        self.detailed_logging = detailed_logging
        
        # Expected token count based on mode
        self.expected_tokens = 257 if training_mode == "cls_patch" else 256
        
        # Training metrics tracking
        self.training_step_count = 0
        self.loss_history = []
        self.metric_history = []
        self.eval_history = []
        
        # Distributed training setup
        self.is_distributed = dist.is_initialized()
        self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        
        if self.is_main_process:
            logger.info("✅ FINAL FIXED: BLIP3-o Trainer (training & evaluation compatible)")
            logger.info(f"🎯 Training mode: {self.training_mode} ({self.expected_tokens} tokens)")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Enhanced compute_loss with proper gradient handling"""
        model.train()
        
        # Extract inputs
        eva_embeddings = inputs['encoder_hidden_states']
        clip_embeddings = inputs['clip_embeddings']
        timesteps = inputs['timestep']
        
        # Handle multiprocessing-safe inputs
        if 'hidden_states' in inputs:
            noisy_clip_base = inputs['hidden_states']
            noise = inputs.get('noise', torch.randn_like(clip_embeddings))
        else:
            # Create noisy input here
            device = eva_embeddings.device
            noise = torch.randn_like(clip_embeddings)
            alpha = timesteps.view(-1, 1, 1)
            noisy_clip_base = (1 - alpha) * noise + alpha * clip_embeddings.detach()

        # Add gradients to noisy input for training
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
        
        # Validate model output
        if velocity_pred is None:
            raise ValueError("Model output is None")
        
        # During training, model output should have gradients
        if not velocity_pred.requires_grad:
            raise RuntimeError("Model output doesn't require gradients during training!")
        
        # Compute flow matching loss
        loss, metrics = self.flow_matching_loss(
            model_output=velocity_pred,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            noise=noise,
            return_metrics=True
        )
        
        # Store metrics
        if metrics and self.is_main_process:
            self.metric_history.append({
                **metrics,
                'step': self.training_step_count,
                'timestamp': time.time(),
            })
            self.loss_history.append(loss.item())
        
        # Progress logging
        if (self.is_main_process and self.detailed_logging and 
            self.training_step_count % self.args.logging_steps == 0):
            self._log_detailed_progress(loss, metrics, velocity_pred, clip_embeddings)
        
        self.training_step_count += 1
        
        # Prepare outputs
        outputs = {
            'loss': loss,
            'velocity_prediction': velocity_pred,
            'target_samples': clip_embeddings,
            'training_metrics': metrics,
        } if return_outputs else None
        
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        FINAL FIXED: Custom evaluation loop that handles gradient-free evaluation properly
        """
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
                    return_metrics=True
                )
                
                all_losses.append(loss.item())
                if metrics:
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
                if isinstance(all_metrics[0][key], (int, float)):
                    values = [m[key] for m in all_metrics if key in m]
                    if values:  # Only add if we have values
                        eval_metrics[f"{metric_key_prefix}_{key}"] = np.mean(values)
        
        logger.info(f"***** {description} results *****")
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key} = {value:.6f}" if key.endswith('_loss') else f"  {key} = {value}")
        
        return eval_metrics

    def _log_detailed_progress(
        self,
        loss: torch.Tensor,
        metrics: Optional[Dict[str, float]],
        velocity_pred: torch.Tensor,
        target_samples: torch.Tensor
    ):
        """Log detailed training progress"""
        if not self.is_main_process:
            return
        
        loss_value = loss.item()
        seq_len = target_samples.shape[1]
        
        mode_info = f"Mode: {self.training_mode} ({seq_len} tokens)"
        if seq_len == 257:
            mode_info += " [CLS+Patches]"
        else:
            mode_info += " [Patches Only]"
        
        progress_msg = f"Step {self.training_step_count}: Loss={loss_value:.4f}, {mode_info}"
        
        if metrics:
            if 'velocity_cosine_sim' in metrics:
                progress_msg += f", VelCos={metrics['velocity_cosine_sim']:.3f}"
            if 'global_mean_cosine' in metrics:
                progress_msg += f", GlobalCos={metrics['global_mean_cosine']:.3f}"
            if 'per_image_mean_cosine' in metrics:
                progress_msg += f", ImageCos={metrics['per_image_mean_cosine']:.3f}"
            if 'training_quality' in metrics:
                progress_msg += f", Quality={metrics['training_quality']}"
            if metrics.get('multiprocessing_safe', False):
                progress_msg += " [MP-Safe]"
        
        logger.info(progress_msg)
        
        # Quality assessment
        if metrics:
            global_cos = metrics.get('global_mean_cosine', metrics.get('per_image_mean_cosine', 0))
            if global_cos > 0.8:
                logger.info("🎉 EXCELLENT: Very strong alignment achieved!")
            elif global_cos > 0.6:
                logger.info("✅ GOOD: Strong training progress detected")
            elif global_cos > 0.4:
                logger.info("🔄 FAIR: Making steady progress")
            elif global_cos > 0.2:
                logger.info("📈 EARLY: Training is progressing")

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            'training_mode': self.training_mode,
            'expected_tokens': self.expected_tokens,
            'total_steps': self.training_step_count,
            'loss_history_length': len(self.loss_history),
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


def create_blip3o_flexible_training_args(
    output_dir: str,
    training_mode: str = "cls_patch",
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 5e-5,  # FIXED: Lower learning rate for flow matching
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.1,  # FIXED: More warmup for stability
    weight_decay: float = 0.01,
    warmup_steps: int = 100,  # FIXED: More warmup steps
    logging_steps: int = 10,
    save_steps: int = 200,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    dataloader_num_workers: int = 2,  # FIXED: Conservative default
    enable_evaluation: bool = True,
    eval_steps: int = 50,
    **kwargs
) -> TrainingArguments:
    """FINAL FIXED: Create training arguments with proper evaluation setup"""
    
    # Use eval_loss as metric (which we now properly compute)
    eval_strategy = "steps" if enable_evaluation else "no"
    metric_for_best_model = "eval_loss"  # This now works!
    
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
        eval_steps=eval_steps if enable_evaluation else None,
        save_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        load_best_model_at_end=enable_evaluation,
        metric_for_best_model=metric_for_best_model if enable_evaluation else None,
        greater_is_better=False,  # Lower loss is better
        save_total_limit=3,
        prediction_loss_only=False,  # We want all metrics
        report_to=[],
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        ignore_data_skip=True,
        **kwargs
    )