"""
Custom HuggingFace Trainer for BLIP3-o DiT training with flow matching.
FIXED: Updated parameter validation and removed invalid lr_end parameter
ADDED: Proper cosine learning rate scheduler support
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import wandb
from pathlib import Path
import json
import numpy as np
import math
from collections import defaultdict

from ..models.blip3o_dit import BLIP3oDiTModel
from ..losses.flow_matching_loss import BLIP3oFlowMatchingLoss
from ..config.blip3o_config import BLIP3oDiTConfig, FlowMatchingConfig

logger = logging.getLogger(__name__)


class BLIP3oTrainer(Trainer):
    """
    Custom trainer for BLIP3-o DiT training with flow matching.
    
    This trainer implements the exact BLIP3-o training methodology:
    - Flow matching loss computation with velocity prediction
    - EVA-CLIP conditioning for cross-attention
    - Proper handling of 256-token embeddings (16x16 grid)
    - Detailed metrics and logging
    - Memory-efficient training with gradient checkpointing
    """
    
    def __init__(
        self,
        model: BLIP3oDiTModel,
        args: TrainingArguments,
        flow_matching_loss: BLIP3oFlowMatchingLoss,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        **kwargs
    ):
        """
        Initialize the BLIP3-o trainer.
        
        Args:
            model: BLIP3oDiTModel instance
            args: TrainingArguments for training configuration
            flow_matching_loss: Flow matching loss function
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            data_collator: Data collator (not used, we handle collation in dataset)
            tokenizer: Not used for embedding training
            model_init: Model initialization function
            compute_metrics: Custom metrics computation function
            callbacks: Training callbacks
            optimizers: (optimizer, lr_scheduler) tuple
            preprocess_logits_for_metrics: Not used for flow matching
            **kwargs: Additional arguments
        """
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
        self.training_step_count = 0
        
        # Metrics tracking
        self.train_metrics_history = []
        self.eval_metrics_history = []
        
        # Loss components tracking
        self.loss_components = defaultdict(list)
        
        logger.info("BLIP3-o trainer initialized")
    
    def compute_loss(
        self,
        model: BLIP3oDiTModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,  # FIXED: Compatibility parameter
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute flow matching loss for BLIP3-o training.
        
        This implements the exact BLIP3-o flow matching training procedure:
        1. Extract EVA-CLIP conditioning and CLIP targets
        2. Sample random timesteps for flow matching
        3. Add noise according to flow matching schedule
        4. Forward through DiT model
        5. Compute velocity prediction loss
        
        Args:
            model: The BLIP3oDiTModel
            inputs: Batch inputs from dataloader
            return_outputs: Whether to return model outputs
            num_items_in_batch: Compatibility parameter (unused)
            
        Returns:
            Loss tensor, optionally with additional outputs
        """
        # Extract inputs from batch
        eva_embeddings = inputs['eva_embeddings']      # [B, 256, 4096]
        clip_embeddings = inputs['clip_embeddings']    # [B, 256, 1024]
        
        batch_size = eva_embeddings.shape[0]
        device = eva_embeddings.device
        
        # Sample random timesteps for flow matching [0, 1]
        timesteps = self.flow_matching_loss.sample_timesteps(batch_size, device)
        
        # Sample noise for flow matching
        noise = torch.randn_like(clip_embeddings)
        
        # Create noisy samples according to flow matching interpolation
        # x_t = (1-t)*x_0 + t*x_1 + sigma_t*epsilon
        x_0 = torch.randn_like(clip_embeddings)  # Source distribution
        noisy_clip = self.flow_matching_loss.interpolate_data(
            x_0=x_0,
            x_1=clip_embeddings,
            t=timesteps,
            noise=noise
        )
        
        # Forward pass through DiT model
        model_output = model(
            hidden_states=noisy_clip,
            timestep=timesteps,
            encoder_hidden_states=eva_embeddings,
            return_dict=False
        )
        
        # Compute flow matching loss with detailed metrics
        loss, metrics = self.flow_matching_loss(
            model_output=model_output,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            noise=noise,
            return_metrics=True
        )
        
        # Store metrics for logging
        if metrics is not None:
            for key, value in metrics.items():
                self.loss_components[key].append(value)
        
        # Log metrics periodically
        if self.training_step_count % self.args.logging_steps == 0:
            self._log_training_metrics(metrics, timesteps, model_output, clip_embeddings)
        
        self.training_step_count += 1
        
        # Prepare outputs
        outputs = {
            'model_output': model_output,
            'noisy_clip': noisy_clip,
            'timesteps': timesteps,
            'metrics': metrics,
            'eva_embeddings': eva_embeddings,
            'clip_embeddings': clip_embeddings,
        } if return_outputs else None
        
        if return_outputs:
            return loss, outputs
        else:
            return loss
    
    def _log_training_metrics(
        self,
        metrics: Optional[Dict[str, float]],
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
        target_clip: torch.Tensor
    ):
        """Log detailed training metrics."""
        
        if metrics is None:
            return
        
        # Create logging dictionary
        log_dict = {}
        
        # Add flow matching metrics
        for key, value in metrics.items():
            log_dict[f"train/{key}"] = value
        
        # Add timestep statistics
        with torch.no_grad():
            log_dict.update({
                "train/timestep_mean": timesteps.mean().item(),
                "train/timestep_std": timesteps.std().item(),
                "train/timestep_min": timesteps.min().item(),
                "train/timestep_max": timesteps.max().item(),
            })
            
            # Model output statistics
            output_mean = model_output.mean().item()
            output_std = model_output.std().item()
            output_abs_mean = model_output.abs().mean().item()
            
            log_dict.update({
                "train/output_mean": output_mean,
                "train/output_std": output_std,
                "train/output_abs_mean": output_abs_mean,
            })
            
            # Target statistics
            target_mean = target_clip.mean().item()
            target_std = target_clip.std().item()
            
            log_dict.update({
                "train/target_mean": target_mean,
                "train/target_std": target_std,
            })
        
        # Add step information
        log_dict["train/step"] = self.training_step_count
        log_dict["train/epoch"] = self.state.epoch
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(log_dict, step=self.training_step_count)
        
        # Store in history
        self.train_metrics_history.append({
            'step': self.training_step_count,
            'epoch': self.state.epoch,
            **log_dict
        })
        
        # Print progress periodically
        if self.training_step_count % (self.args.logging_steps * 5) == 0:
            logger.info(
                f"Step {self.training_step_count}: "
                f"Loss={metrics.get('total_loss', 0):.4f}, "
                f"FM_Loss={metrics.get('flow_matching_loss', 0):.4f}, "
                f"Cosine_Sim={metrics.get('cosine_similarity', 0):.4f}, "
                f"SNR={metrics.get('snr_db', 0):.1f}dB"
            )
    
    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.
        
        This implements evaluation specific to flow matching:
        - Computes flow matching loss on evaluation set
        - Generates sample embeddings for quality assessment
        - Computes detailed metrics for monitoring
        """
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}
        
        # Set model to evaluation mode
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        # Create evaluation dataloader
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Collect evaluation metrics
        eval_losses = []
        all_metrics = defaultdict(list)
        
        logger.info(f"Running evaluation on {len(eval_dataloader)} batches")
        
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                # Move inputs to device
                inputs = self._prepare_inputs(inputs)
                
                # Compute loss and metrics
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                
                eval_losses.append(loss.item())
                
                # Collect detailed metrics
                if outputs and outputs.get('metrics'):
                    for key, value in outputs['metrics'].items():
                        all_metrics[key].append(value)
                
                # Log progress
                if step % max(1, len(eval_dataloader) // 10) == 0:
                    logger.info(f"Evaluation step {step}/{len(eval_dataloader)}")
        
        # Aggregate metrics
        eval_results = {
            f'{metric_key_prefix}_loss': np.mean(eval_losses),
            f'{metric_key_prefix}_loss_std': np.std(eval_losses),
        }
        
        # Aggregate detailed metrics
        for key, values in all_metrics.items():
            eval_results[f'{metric_key_prefix}_{key}'] = np.mean(values)
            eval_results[f'{metric_key_prefix}_{key}_std'] = np.std(values)
        
        # Generate sample embeddings for quality assessment
        sample_metrics = self._evaluate_generation_quality(model, eval_dataloader)
        eval_results.update({f'{metric_key_prefix}_{k}': v for k, v in sample_metrics.items()})
        
        # Log evaluation results
        if wandb.run is not None:
            wandb.log(eval_results, step=self.training_step_count)
        
        # Store in history
        self.eval_metrics_history.append({
            'step': self.training_step_count,
            'epoch': self.state.epoch,
            **eval_results
        })
        
        logger.info(f"Evaluation results: {eval_results}")
        
        return eval_results
    
    def _evaluate_generation_quality(
        self,
        model: BLIP3oDiTModel,
        eval_dataloader: torch.utils.data.DataLoader,
        num_samples: int = 4,
        num_inference_steps: int = 20,
    ) -> Dict[str, float]:
        """
        Evaluate generation quality by generating samples and comparing to targets.
        """
        try:
            # Get a batch for generation
            sample_batch = next(iter(eval_dataloader))
            eva_conditioning = sample_batch['eva_embeddings'][:num_samples]
            target_clip = sample_batch['clip_embeddings'][:num_samples]
            
            # Generate samples
            generated_clip = model.generate(
                encoder_hidden_states=eva_conditioning,
                num_inference_steps=num_inference_steps,
            )
            
            # Compute generation metrics
            with torch.no_grad():
                # Cosine similarity between generated and target
                gen_flat = generated_clip.flatten(1)
                target_flat = target_clip.flatten(1)
                cosine_sim = nn.functional.cosine_similarity(gen_flat, target_flat, dim=1).mean().item()
                
                # L2 distance
                l2_dist = torch.norm(generated_clip - target_clip, dim=-1).mean().item()
                
                # Generated embedding statistics
                gen_norm = torch.norm(generated_clip, dim=-1).mean().item()
                target_norm = torch.norm(target_clip, dim=-1).mean().item()
                
            return {
                'generation_cosine_similarity': cosine_sim,
                'generation_l2_distance': l2_dist,
                'generation_norm': gen_norm,
                'generation_target_norm': target_norm,
                'generation_norm_ratio': gen_norm / (target_norm + 1e-8),
            }
        
        except Exception as e:
            logger.warning(f"Failed to evaluate generation quality: {e}")
            return {}
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save the model and training state with BLIP3-o specific information."""
        
        output_dir = output_dir or self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using parent class
        super().save_model(output_dir, _internal_call)
        
        # Save BLIP3-o specific configurations
        self._save_blip3o_configs(output_dir)
        
        # Save training metrics history
        self._save_metrics_history(output_dir)
        
        logger.info(f"BLIP3-o model and configs saved to {output_dir}")
    
    def _save_blip3o_configs(self, output_dir: Path):
        """Save BLIP3-o specific configurations."""
        
        # Save flow matching configuration
        flow_config = {
            'sigma_min': self.flow_matching_loss.sigma_min,
            'sigma_max': self.flow_matching_loss.sigma_max,
            'prediction_type': self.flow_matching_loss.prediction_type,
            'schedule_type': self.flow_matching_loss.schedule_type,
            'clip_dim': self.flow_matching_loss.clip_dim,
            'eva_dim': self.flow_matching_loss.eva_dim,
            'regularization_weight': self.flow_matching_loss.regularization_weight,
        }
        
        with open(output_dir / 'flow_matching_config.json', 'w') as f:
            json.dump(flow_config, f, indent=2)
        
        # Save model configuration (if not already saved by parent)
        if hasattr(self.model, 'config'):
            model_config = self.model.config.to_dict()
            with open(output_dir / 'blip3o_model_config.json', 'w') as f:
                json.dump(model_config, f, indent=2)
        
        # FIXED: Create training summary without invalid lr_end
        training_summary = {
            'total_steps': self.training_step_count,
            'num_parameters': self.model.get_num_parameters(),
            'num_trainable_parameters': self.model.get_num_parameters(trainable_only=True),
            'memory_footprint': self.model.get_memory_footprint(),
            'gradient_checkpointing': self.model._gradient_checkpointing,
            'lr_scheduler_type': self.args.lr_scheduler_type,
            'learning_rate': self.args.learning_rate,
            'warmup_ratio': self.args.warmup_ratio,
            'warmup_steps': self.args.warmup_steps,
        }
        
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
    
    def _save_metrics_history(self, output_dir: Path):
        """Save training and evaluation metrics history."""
        
        # Save training metrics
        if self.train_metrics_history:
            with open(output_dir / 'train_metrics_history.json', 'w') as f:
                json.dump(self.train_metrics_history, f, indent=2)
        
        # Save evaluation metrics
        if self.eval_metrics_history:
            with open(output_dir / 'eval_metrics_history.json', 'w') as f:
                json.dump(self.eval_metrics_history, f, indent=2)
        
        # Save loss components summary
        loss_summary = {}
        for key, values in self.loss_components.items():
            if values:
                loss_summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1] if values else 0,
                    'count': len(values),
                }
        
        with open(output_dir / 'loss_components_summary.json', 'w') as f:
            json.dump(loss_summary, f, indent=2)
    
    def create_optimizer(self):
        """Create optimizer with BLIP3-o specific settings."""
        # Use the parent class implementation but log the configuration
        optimizer = super().create_optimizer()
        
        logger.info(f"Created optimizer: {type(optimizer).__name__}")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Weight decay: {self.args.weight_decay}")
        logger.info(f"LR Scheduler: {self.args.lr_scheduler_type}")
        logger.info(f"Warmup Ratio: {self.args.warmup_ratio}")
        
        return optimizer
    
    def get_train_dataloader(self):
        """Get training dataloader with logging."""
        dataloader = super().get_train_dataloader()
        
        logger.info(f"Training dataloader: {len(dataloader)} batches")
        
        # Handle case where batch_size might be None (common with custom datasets)
        batch_size = getattr(dataloader, 'batch_size', None)
        if batch_size is None:
            # Fall back to training arguments
            batch_size = self.args.per_device_train_batch_size
            logger.info(f"Batch size: {batch_size} (from training args)")
        else:
            logger.info(f"Batch size: {batch_size}")
        
        # Calculate total samples if we have a valid batch size
        if batch_size is not None:
            total_samples = len(dataloader) * batch_size
            logger.info(f"Total training samples per epoch: {total_samples}")
        else:
            logger.info("Unable to determine total training samples (batch_size unknown)")
        
        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        """Get evaluation dataloader with logging."""
        dataloader = super().get_eval_dataloader(eval_dataset)
        
        if dataloader is not None:
            logger.info(f"Evaluation dataloader: {len(dataloader)} batches")
            
            # Handle case where batch_size might be None
            batch_size = getattr(dataloader, 'batch_size', None)
            if batch_size is None:
                # Fall back to training arguments
                batch_size = self.args.per_device_eval_batch_size
                logger.info(f"Eval batch size: {batch_size} (from training args)")
            else:
                logger.info(f"Eval batch size: {batch_size}")
        
        return dataloader


def create_blip3o_training_args(
    output_dir: str,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",   # Scheduler type parameter
    warmup_ratio: float = 0.05,          # Warmup ratio parameter
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    logging_steps: int = 100,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    gradient_accumulation_steps: int = 1,
    fp16: bool = True,
    bf16: bool = False,
    dataloader_num_workers: int = 4,
    remove_unused_columns: bool = False,  # Important for custom data
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    **kwargs
) -> TrainingArguments:
    """
    Create TrainingArguments optimized for BLIP3-o DiT training.
    FIXED: Removed invalid lr_end parameter and proper scheduler configuration.
    
    Args:
        output_dir: Output directory for checkpoints and logs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        learning_rate: Learning rate
        lr_scheduler_type: Learning rate scheduler type
        warmup_ratio: Warmup steps as ratio of total steps
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for learning rate
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        gradient_accumulation_steps: Gradient accumulation steps
        fp16: Use mixed precision training
        bf16: Use bfloat16 (alternative to fp16)
        dataloader_num_workers: Number of dataloader workers
        remove_unused_columns: Remove unused columns from dataset
        load_best_model_at_end: Load best model at end of training
        metric_for_best_model: Metric to use for best model selection
        greater_is_better: Whether higher metric values are better
        **kwargs: Additional TrainingArguments parameters
        
    Returns:
        TrainingArguments configured for BLIP3-o training
    """
    # FIXED: Ensure save_steps is compatible with eval_steps
    if load_best_model_at_end and eval_steps > 0:
        # Adjust save_steps to be a multiple of eval_steps
        if save_steps % eval_steps != 0:
            # Round save_steps to nearest multiple of eval_steps
            adjusted_save_steps = ((save_steps // eval_steps) + 1) * eval_steps
            logger.warning(f"Adjusting save_steps from {save_steps} to {adjusted_save_steps} to be compatible with eval_steps ({eval_steps})")
            save_steps = adjusted_save_steps
    
    # Determine evaluation strategy based on eval_steps
    if eval_steps > 0:
        eval_strategy = "steps"
        eval_steps_value = eval_steps
    else:
        eval_strategy = "no"
        eval_steps_value = None
        load_best_model_at_end = False  # Can't load best model if no evaluation
        logger.info("Evaluation disabled, setting load_best_model_at_end=False")
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        # FIXED: Proper cosine scheduler parameters (removed invalid lr_end)
        lr_scheduler_type=lr_scheduler_type,  # Set scheduler type
        warmup_ratio=warmup_ratio,             # Set warmup ratio
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy=eval_strategy,  # FIXED: was evaluation_strategy
        eval_steps=eval_steps_value,
        save_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16 and not bf16,  # Don't use both fp16 and bf16
        bf16=bf16,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=remove_unused_columns,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=3,  # Keep only last 3 checkpoints
        prediction_loss_only=False,  # We want to compute custom metrics
        report_to=["wandb"] if wandb.run is not None else [],
        run_name=f"blip3o-dit-{output_dir.split('/')[-1]}" if wandb.run is not None else None,
        # Additional compatibility fixes
        push_to_hub=False,  # Avoid any hub-related issues
        **kwargs
    )