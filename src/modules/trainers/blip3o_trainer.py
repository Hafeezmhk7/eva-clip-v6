"""
UPDATED: BLIP3-o Trainer with Dual Supervision Support
Handles both patch-level and global-level supervision for improved recall performance.

Key Changes:
1. Extracts both CLIP patch and global targets during training
2. Handles dual model outputs (patch + global)
3. Uses dual supervision loss function
4. Updated logging for new metrics
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, CLIPModel, CLIPProcessor
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
from ..losses.flow_matching_loss import DualSupervisionFlowMatchingLoss
from ..config.blip3o_config import BLIP3oDiTConfig, FlowMatchingConfig

logger = logging.getLogger(__name__)


class BLIP3oTrainer(Trainer):
    """
    UPDATED: Custom trainer for BLIP3-o DiT training with dual supervision.
    
    New training methodology:
    1. Extract both patch and global targets from CLIP
    2. Forward through DiT to get patch and global outputs
    3. Apply dual supervision loss (patch + global + flow matching)
    4. Detailed logging for both supervision levels
    """
    
    def __init__(
        self,
        model: BLIP3oDiTModel,
        args: TrainingArguments,
        flow_matching_loss: DualSupervisionFlowMatchingLoss,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        **kwargs
    ):
        """
        Initialize the dual supervision BLIP3-o trainer.
        
        Args:
            model: BLIP3oDiTModel instance with dual outputs
            args: TrainingArguments for training configuration
            flow_matching_loss: Dual supervision flow matching loss function
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            data_collator: Data collator (not used)
            tokenizer: Not used for embedding training
            model_init: Model initialization function
            compute_metrics: Custom metrics computation function
            callbacks: Training callbacks
            optimizers: (optimizer, lr_scheduler) tuple
            preprocess_logits_for_metrics: Not used
            clip_model_name: CLIP model name for target extraction
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
        self.clip_model_name = clip_model_name
        
        # Load CLIP model for target extraction during training
        self._load_clip_model()
        
        # Ensure model has frozen CLIP projection loaded
        if not hasattr(model, 'frozen_clip_visual_proj') or model.frozen_clip_visual_proj is None:
            logger.info("Loading frozen CLIP projection into model...")
            model.load_frozen_clip_projection(clip_model_name)
        
        # Metrics tracking
        self.train_metrics_history = []
        self.eval_metrics_history = []
        
        # Loss components tracking (expanded for dual supervision)
        self.loss_components = defaultdict(list)
        
        logger.info("Dual Supervision BLIP3-o trainer initialized")
        logger.info(f"Using CLIP model: {clip_model_name}")
    
    def _load_clip_model(self):
        """Load CLIP model for extracting ground truth targets during training."""
        try:
            logger.info(f"Loading CLIP model for target extraction: {self.clip_model_name}")
            
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            
            # Move to same device as main model
            if hasattr(self.model, 'device'):
                self.clip_model = self.clip_model.to(self.model.device)
            
            # Set to eval mode and freeze
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            logger.info("✅ CLIP model loaded and frozen for target extraction")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError("CLIP model required for dual supervision training")
    
    def extract_clip_targets(self, images):
        """
        Extract both patch and global CLIP targets from images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dict containing:
            - patch_targets: [B, 256, 1024] CLIP patch embeddings
            - global_targets: [B, 768] CLIP global embeddings
        """
        if not hasattr(self, 'clip_model') or self.clip_model is None:
            raise ValueError("CLIP model not available for target extraction")
        
        batch_size = len(images)
        device = next(self.clip_model.parameters()).device
        
        patch_targets = []
        global_targets = []
        
        with torch.no_grad():
            for img in images:
                # Process image
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get vision model outputs
                vision_outputs = self.clip_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract patch embeddings (remove CLS token)
                # vision_outputs.last_hidden_state: [1, 257, 1024]
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 1024]
                patch_targets.append(patch_embeddings.squeeze(0))
                
                # Extract global embeddings (CLS token + visual projection)
                cls_token = vision_outputs.last_hidden_state[:, 0, :]  # [1, 1024]
                global_embedding = self.clip_model.visual_projection(cls_token)  # [1, 768]
                global_embedding = torch.nn.functional.normalize(global_embedding, p=2, dim=-1)
                global_targets.append(global_embedding.squeeze(0))
        
        return {
            'patch_targets': torch.stack(patch_targets),    # [B, 256, 1024]
            'global_targets': torch.stack(global_targets),  # [B, 768]
        }
    
    def compute_loss(
        self,
        model: BLIP3oDiTModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        UPDATED: Compute dual supervision loss for BLIP3-o training.
        
        Training procedure:
        1. Extract EVA-CLIP conditioning and CLIP targets (patch + global)
        2. Sample timesteps and create noisy samples for flow matching
        3. Forward through DiT model to get patch and global outputs
        4. Compute dual supervision loss (flow matching + patch + global)
        
        Args:
            model: The BLIP3oDiTModel with dual outputs
            inputs: Batch inputs from dataloader
            return_outputs: Whether to return model outputs
            num_items_in_batch: Compatibility parameter (unused)
            
        Returns:
            Loss tensor, optionally with additional outputs
        """
        # Extract inputs from batch
        eva_embeddings = inputs['eva_embeddings']      # [B, 256, 4096]
        clip_embeddings = inputs['clip_embeddings']    # [B, 256, 1024] - patch targets
        
        batch_size = eva_embeddings.shape[0]
        device = eva_embeddings.device
        
        # For dual supervision, we need both patch and global CLIP targets
        # In the updated dataset, we should have both, but as fallback we'll extract global from patch
        if 'clip_global_embeddings' in inputs:
            clip_global_targets = inputs['clip_global_embeddings']  # [B, 768]
        else:
            # Fallback: extract global targets from patch embeddings using CLIP
            # Average pool patch embeddings and apply CLIP visual projection
            with torch.no_grad():
                pooled_patches = clip_embeddings.mean(dim=1)  # [B, 1024]
                clip_global_targets = self.clip_model.visual_projection(pooled_patches)  # [B, 768]
                clip_global_targets = torch.nn.functional.normalize(clip_global_targets, p=2, dim=-1)
        
        # Sample random timesteps for flow matching
        timesteps = self.flow_matching_loss.sample_timesteps(batch_size, device)
        
        # Sample noise for flow matching
        noise = torch.randn_like(clip_embeddings)
        
        # Create noisy samples according to flow matching interpolation
        x_0 = torch.randn_like(clip_embeddings)  # Source distribution
        noisy_clip = self.flow_matching_loss.interpolate_data(
            x_0=x_0,
            x_1=clip_embeddings,  # Use patch targets for flow matching
            t=timesteps,
            noise=noise
        )
        
        # Forward pass through DiT model (dual outputs)
        model_outputs = model(
            hidden_states=noisy_clip,
            timestep=timesteps,
            encoder_hidden_states=eva_embeddings,
            return_dict=True
        )
        
        dit_patch_output = model_outputs['patch_output']    # [B, 256, 1024]
        dit_global_output = model_outputs['global_output']  # [B, 768] or None
        
        # Compute dual supervision loss
        loss, metrics = self.flow_matching_loss(
            dit_patch_output=dit_patch_output,
            dit_global_output=dit_global_output,
            clip_patch_targets=clip_embeddings,
            clip_global_targets=clip_global_targets,
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
            self._log_training_metrics(
                metrics, timesteps, dit_patch_output, dit_global_output, 
                clip_embeddings, clip_global_targets
            )
        
        self.training_step_count += 1
        
        # Prepare outputs
        outputs = {
            'dit_patch_output': dit_patch_output,
            'dit_global_output': dit_global_output,
            'clip_patch_targets': clip_embeddings,
            'clip_global_targets': clip_global_targets,
            'noisy_clip': noisy_clip,
            'timesteps': timesteps,
            'metrics': metrics,
            'eva_embeddings': eva_embeddings,
            'pooled_features': model_outputs.get('pooled_features'),
            'adapted_features': model_outputs.get('adapted_features'),
        } if return_outputs else None
        
        if return_outputs:
            return loss, outputs
        else:
            return loss
    
    def _log_training_metrics(
        self,
        metrics: Optional[Dict[str, float]],
        timesteps: torch.Tensor,
        dit_patch_output: torch.Tensor,
        dit_global_output: Optional[torch.Tensor],
        clip_patch_targets: torch.Tensor,
        clip_global_targets: torch.Tensor,
    ):
        """Log detailed training metrics for dual supervision."""
        
        if metrics is None:
            return
        
        # Create logging dictionary
        log_dict = {}
        
        # Add dual supervision metrics
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
            
            # DiT patch output statistics
            patch_mean = dit_patch_output.mean().item()
            patch_std = dit_patch_output.std().item()
            patch_norm = torch.norm(dit_patch_output, dim=-1).mean().item()
            
            log_dict.update({
                "train/dit_patch_mean": patch_mean,
                "train/dit_patch_std": patch_std,
                "train/dit_patch_norm": patch_norm,
            })
            
            # DiT global output statistics (if available)
            if dit_global_output is not None:
                global_mean = dit_global_output.mean().item()
                global_std = dit_global_output.std().item()
                global_norm = torch.norm(dit_global_output, dim=-1).mean().item()
                
                log_dict.update({
                    "train/dit_global_mean": global_mean,
                    "train/dit_global_std": global_std,
                    "train/dit_global_norm": global_norm,
                })
                
                # Global alignment quality
                global_cosine = torch.nn.functional.cosine_similarity(
                    dit_global_output, clip_global_targets, dim=1
                ).mean().item()
                log_dict["train/global_cosine_realtime"] = global_cosine
            
            # CLIP target statistics
            clip_patch_norm = torch.norm(clip_patch_targets, dim=-1).mean().item()
            clip_global_norm = torch.norm(clip_global_targets, dim=-1).mean().item()
            
            log_dict.update({
                "train/clip_patch_norm": clip_patch_norm,
                "train/clip_global_norm": clip_global_norm,
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
        
        # Print progress periodically with dual supervision info
        if self.training_step_count % (self.args.logging_steps * 5) == 0:
            total_loss = metrics.get('total_loss', 0)
            flow_loss = metrics.get('flow_matching_loss', 0)
            patch_loss = metrics.get('patch_reconstruction_loss', 0)
            global_loss = metrics.get('global_alignment_loss', 0)
            
            patch_cosine = metrics.get('patch_cosine_similarity', 0)
            global_cosine = metrics.get('global_cosine_similarity', 0)
            
            logger.info(
                f"Step {self.training_step_count}: "
                f"Total={total_loss:.4f} "
                f"(Flow={flow_loss:.4f}, Patch={patch_loss:.4f}, Global={global_loss:.4f}) "
                f"Cosine: Patch={patch_cosine:.3f}, Global={global_cosine:.3f}"
            )
    
    def evaluate(
        self,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        UPDATED: Run evaluation with dual supervision metrics.
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
        
        logger.info(f"Running dual supervision evaluation on {len(eval_dataloader)} batches")
        
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
        
        # Aggregate detailed dual supervision metrics
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
        
        logger.info(f"Dual supervision evaluation results: {eval_results}")
        
        return eval_results
    
    def _evaluate_generation_quality(
        self,
        model: BLIP3oDiTModel,
        eval_dataloader: torch.utils.data.DataLoader,
        num_samples: int = 4,
        num_inference_steps: int = 20,
    ) -> Dict[str, float]:
        """
        UPDATED: Evaluate generation quality with dual outputs.
        """
        try:
            # Get a batch for generation
            sample_batch = next(iter(eval_dataloader))
            eva_conditioning = sample_batch['eva_embeddings'][:num_samples]
            target_clip_patches = sample_batch['clip_embeddings'][:num_samples]
            
            # Extract global targets if available
            if 'clip_global_embeddings' in sample_batch:
                target_clip_global = sample_batch['clip_global_embeddings'][:num_samples]
            else:
                # Fallback: create global targets from patches
                with torch.no_grad():
                    pooled_patches = target_clip_patches.mean(dim=1)
                    target_clip_global = self.clip_model.visual_projection(pooled_patches)
                    target_clip_global = torch.nn.functional.normalize(target_clip_global, p=2, dim=-1)
            
            # Generate samples (get global output)
            generated_global = model.generate(
                encoder_hidden_states=eva_conditioning,
                num_inference_steps=num_inference_steps,
                return_global_only=True,  # Get global embeddings for retrieval evaluation
            )
            
            # Compute generation metrics
            metrics = {}
            
            with torch.no_grad():
                if generated_global is not None:
                    # Global generation metrics (most important for retrieval)
                    global_cosine = nn.functional.cosine_similarity(
                        generated_global, target_clip_global, dim=1
                    ).mean().item()
                    
                    global_l2 = torch.norm(generated_global - target_clip_global, dim=-1).mean().item()
                    
                    # Generated embedding statistics
                    gen_global_norm = torch.norm(generated_global, dim=-1).mean().item()
                    target_global_norm = torch.norm(target_clip_global, dim=-1).mean().item()
                    
                    metrics.update({
                        'generation_global_cosine': global_cosine,
                        'generation_global_l2': global_l2,
                        'generation_global_norm': gen_global_norm,
                        'target_global_norm': target_global_norm,
                        'generation_global_norm_ratio': gen_global_norm / (target_global_norm + 1e-8),
                    })
                    
                    # Retrieval simulation (cosine similarity distribution)
                    similarities = nn.functional.cosine_similarity(
                        generated_global.unsqueeze(1), target_clip_global.unsqueeze(0), dim=2
                    )
                    diagonal_similarities = similarities.diag().mean().item()
                    off_diagonal_similarities = similarities.fill_diagonal_(0).mean().item()
                    
                    metrics.update({
                        'retrieval_self_similarity': diagonal_similarities,
                        'retrieval_cross_similarity': off_diagonal_similarities,
                        'retrieval_separation': diagonal_similarities - off_diagonal_similarities,
                    })
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Failed to evaluate generation quality: {e}")
            return {}
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """UPDATED: Save model with dual supervision configurations."""
        
        output_dir = output_dir or self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using parent class
        super().save_model(output_dir, _internal_call)
        
        # Save dual supervision specific configurations
        self._save_dual_supervision_configs(output_dir)
        
        # Save training metrics history
        self._save_metrics_history(output_dir)
        
        logger.info(f"Dual supervision BLIP3-o model and configs saved to {output_dir}")
    
    def _save_dual_supervision_configs(self, output_dir: Path):
        """Save dual supervision specific configurations."""
        
        # Save dual supervision loss configuration
        dual_loss_config = {
            'sigma_min': self.flow_matching_loss.sigma_min,
            'sigma_max': self.flow_matching_loss.sigma_max,
            'prediction_type': self.flow_matching_loss.prediction_type,
            'schedule_type': self.flow_matching_loss.schedule_type,
            'clip_dim': self.flow_matching_loss.clip_dim,
            'eva_dim': self.flow_matching_loss.eva_dim,
            'clip_global_dim': self.flow_matching_loss.clip_global_dim,
            'patch_loss_weight': self.flow_matching_loss.patch_loss_weight,
            'global_loss_weight': self.flow_matching_loss.global_loss_weight,
            'flow_matching_loss_weight': self.flow_matching_loss.flow_matching_loss_weight,
            'use_cosine_similarity': self.flow_matching_loss.use_cosine_similarity,
            'clip_model_name': self.flow_matching_loss.clip_model_name,
        }
        
        with open(output_dir / 'dual_supervision_loss_config.json', 'w') as f:
            json.dump(dual_loss_config, f, indent=2)
        
        # Save model configuration
        if hasattr(self.model, 'config'):
            model_config = self.model.config.to_dict()
            with open(output_dir / 'blip3o_dual_model_config.json', 'w') as f:
                json.dump(model_config, f, indent=2)
        
        # Training summary
        training_summary = {
            'architecture': 'dual_supervision',
            'total_steps': self.training_step_count,
            'num_parameters': self.model.get_num_parameters(),
            'num_trainable_parameters': self.model.get_num_parameters(trainable_only=True),
            'memory_footprint': self.model.get_memory_footprint(),
            'gradient_checkpointing': self.model._gradient_checkpointing,
            'lr_scheduler_type': self.args.lr_scheduler_type,
            'learning_rate': self.args.learning_rate,
            'warmup_ratio': self.args.warmup_ratio,
            'warmup_steps': self.args.warmup_steps,
            'dual_supervision': {
                'patch_supervision': True,
                'global_supervision': True,
                'frozen_clip_projection': True,
                'custom_mlp_layers': True,
            }
        }
        
        with open(output_dir / 'dual_supervision_training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
    
    def _save_metrics_history(self, output_dir: Path):
        """Save dual supervision training metrics history."""
        
        # Save training metrics
        if self.train_metrics_history:
            with open(output_dir / 'dual_supervision_train_metrics.json', 'w') as f:
                json.dump(self.train_metrics_history, f, indent=2)
        
        # Save evaluation metrics
        if self.eval_metrics_history:
            with open(output_dir / 'dual_supervision_eval_metrics.json', 'w') as f:
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
        
        with open(output_dir / 'dual_supervision_loss_summary.json', 'w') as f:
            json.dump(loss_summary, f, indent=2)
    
    def create_optimizer(self):
        """Create optimizer for dual supervision training."""
        optimizer = super().create_optimizer()
        
        logger.info(f"Created optimizer for dual supervision: {type(optimizer).__name__}")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Weight decay: {self.args.weight_decay}")
        logger.info(f"LR Scheduler: {self.args.lr_scheduler_type}")
        logger.info(f"Warmup Ratio: {self.args.warmup_ratio}")
        
        # Log parameter groups for dual supervision
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Frozen CLIP projection: {self.model.frozen_clip_visual_proj is not None}")
        
        return optimizer


def create_blip3o_training_args(
    output_dir: str,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 8,  # Smaller for dual supervision
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 5e-5,  # Lower LR for dual supervision
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    logging_steps: int = 50,  # More frequent logging for dual supervision
    save_steps: int = 1000,
    eval_steps: int = 500,  # More frequent evaluation
    gradient_accumulation_steps: int = 4,  # Higher accumulation for smaller batches
    fp16: bool = True,
    bf16: bool = False,
    dataloader_num_workers: int = 4,
    remove_unused_columns: bool = False,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_global_cosine_similarity",  # Focus on global alignment
    greater_is_better: bool = True,  # Higher cosine similarity is better
    **kwargs
) -> TrainingArguments:
    """
    Create TrainingArguments optimized for dual supervision BLIP3-o training.
    
    Args:
        output_dir: Output directory for checkpoints and logs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device (smaller for dual supervision)
        per_device_eval_batch_size: Evaluation batch size per device
        learning_rate: Learning rate (lower for dual supervision stability)
        lr_scheduler_type: Learning rate scheduler type
        warmup_ratio: Warmup steps as ratio of total steps
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps (more frequent for dual supervision)
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps (more frequent)
        gradient_accumulation_steps: Gradient accumulation steps (higher for smaller batches)
        fp16: Use mixed precision training
        bf16: Use bfloat16 (alternative to fp16)
        dataloader_num_workers: Number of dataloader workers
        remove_unused_columns: Remove unused columns from dataset
        load_best_model_at_end: Load best model at end of training
        metric_for_best_model: Metric to use for best model selection (global cosine similarity)
        greater_is_better: Whether higher metric values are better
        **kwargs: Additional TrainingArguments parameters
        
    Returns:
        TrainingArguments configured for dual supervision BLIP3-o training
    """
    # Ensure save_steps is compatible with eval_steps
    if load_best_model_at_end and eval_steps > 0:
        if save_steps % eval_steps != 0:
            adjusted_save_steps = ((save_steps // eval_steps) + 1) * eval_steps
            logger.warning(f"Adjusting save_steps from {save_steps} to {adjusted_save_steps}")
            save_steps = adjusted_save_steps
    
    # Determine evaluation strategy
    if eval_steps > 0:
        eval_strategy = "steps"
        eval_steps_value = eval_steps
    else:
        eval_strategy = "no"
        eval_steps_value = None
        load_best_model_at_end = False
        logger.info("Evaluation disabled for dual supervision training")
    
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
        eval_steps=eval_steps_value,
        save_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=fp16 and not bf16,
        bf16=bf16,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=remove_unused_columns,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=3,
        prediction_loss_only=False,
        report_to=["wandb"] if wandb.run is not None else [],
        run_name=f"dual-supervision-blip3o-{output_dir.split('/')[-1]}" if wandb.run is not None else None,
        push_to_hub=False,
        **kwargs
    )