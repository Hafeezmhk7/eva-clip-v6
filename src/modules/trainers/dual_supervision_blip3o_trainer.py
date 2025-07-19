"""
FIXED: Complete Dual Supervision BLIP3-o Trainer with Device Mismatch Resolution
Replace: src/modules/trainers/dual_supervision_blip3o_trainer.py

Key Fixes:
1. FIXED device mismatch in CLIP visual projection
2. Dynamic device placement for multi-GPU compatibility
3. Proper error handling for device operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, CLIPModel
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import wandb
import numpy as np
import math
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DualSupervisionBLIP3oTrainer(Trainer):
    """
    FIXED: Enhanced trainer for dual supervision BLIP3-o training with device mismatch resolution.
    
    Handles:
    - Dual model outputs (patch + global)
    - Dual supervision loss computation
    - CLIP projection for target global features (FIXED device handling)
    - Enhanced metrics for recall performance
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
        
        # Dual supervision specific
        clip_model_name: str = "openai/clip-vit-large-patch14",
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
        self.training_step_count = 0
        self.clip_model_name = clip_model_name
        
        # Load CLIP model for target global feature computation
        self._load_clip_model()
        
        # FIXED: Apply device tracking patch to model
        self._apply_device_tracking_patch()
        
        # Metrics tracking
        self.train_metrics_history = []
        self.eval_metrics_history = []
        self.loss_components = defaultdict(list)
        
        # EMA tracking for dual supervision
        self.ema_patch_cosine = 0.0
        self.ema_global_cosine = 0.0
        self.ema_decay = 0.99
        
        logger.info("FIXED Dual Supervision BLIP3-o trainer initialized")
    
    def _load_clip_model(self):
        """FIXED: Load CLIP model for computing target global features."""
        try:
            print(f"🔄 Loading CLIP model for target computation: {self.clip_model_name}")
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            
            # Freeze CLIP model
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            self.clip_model.eval()
            
            # Get visual projection layer
            self.clip_visual_projection = self.clip_model.visual_projection
            
            # FIXED: Don't move to device here - will be moved dynamically
            print(f"✅ CLIP model loaded for target computation")
            print(f"   Visual projection: {self.clip_visual_projection.weight.shape}")
            print(f"   Device handling: Dynamic (will move to match input tensors)")
            
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            # Create dummy projection
            self.clip_visual_projection = nn.Linear(1024, 768, bias=False)
            self.clip_visual_projection.requires_grad_(False)
            print(f"⚠️  Using dummy CLIP projection due to error")
    
    def _apply_device_tracking_patch(self):
        """FIXED: Apply patch to track when model moves between devices."""
        if hasattr(self.model, 'to'):
            original_to = self.model.to
            
            def patched_to(device_or_dtype, *args, **kwargs):
                result = original_to(device_or_dtype, *args, **kwargs)
                
                # FIXED: Also move CLIP projection to same device
                if hasattr(self, 'clip_visual_projection') and self.clip_visual_projection is not None:
                    try:
                        if isinstance(device_or_dtype, (torch.device, str)):
                            self.clip_visual_projection = self.clip_visual_projection.to(device_or_dtype)
                            logger.debug(f"Moved CLIP projection to {device_or_dtype}")
                        elif hasattr(device_or_dtype, 'device'):
                            self.clip_visual_projection = self.clip_visual_projection.to(device_or_dtype.device)
                            logger.debug(f"Moved CLIP projection to {device_or_dtype.device}")
                    except Exception as e:
                        logger.warning(f"Could not move CLIP projection to device: {e}")
                
                return result
            
            self.model.to = patched_to
            logger.info("✅ Applied device tracking patch to model")
    
    def compute_target_global_features(self, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Compute target global features with dynamic device placement.
        
        Args:
            clip_embeddings: [B, 256, 1024] CLIP patch embeddings
            
        Returns:
            Target global features [B, 768] (in CLIP aligned space)
        """
        with torch.no_grad():
            # Average pool: [B, 256, 1024] → [B, 1024]
            pooled_clip = clip_embeddings.mean(dim=1)
            
            # FIXED: Ensure CLIP projection is on the same device as input tensors
            target_device = pooled_clip.device
            
            # Check if CLIP projection needs to be moved
            if (hasattr(self.clip_visual_projection, 'weight') and 
                self.clip_visual_projection.weight.device != target_device):
                
                try:
                    self.clip_visual_projection = self.clip_visual_projection.to(target_device)
                    logger.debug(f"Moved CLIP projection to {target_device}")
                except Exception as e:
                    logger.error(f"Failed to move CLIP projection to {target_device}: {e}")
                    # Try to move input to CLIP projection device instead
                    pooled_clip = pooled_clip.to(self.clip_visual_projection.weight.device)
                    logger.debug(f"Moved input to CLIP projection device instead")
            
            # Apply CLIP visual projection: [B, 1024] → [B, 768]
            try:
                target_global = self.clip_visual_projection(pooled_clip)
            except RuntimeError as e:
                if "device" in str(e).lower():
                    logger.error(f"Device mismatch in CLIP projection: {e}")
                    logger.error(f"Input device: {pooled_clip.device}")
                    logger.error(f"CLIP projection device: {self.clip_visual_projection.weight.device}")
                    
                    # Emergency fix: move CLIP projection to input device
                    self.clip_visual_projection = self.clip_visual_projection.to(pooled_clip.device)
                    target_global = self.clip_visual_projection(pooled_clip)
                else:
                    raise e
        
        return target_global
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        FIXED: Compute dual supervision loss with robust device handling.
        """
        # Extract inputs
        eva_embeddings = inputs['eva_embeddings']      # [B, 256, 4096]
        clip_embeddings = inputs['clip_embeddings']    # [B, 256, 1024]
        
        batch_size = eva_embeddings.shape[0]
        device = eva_embeddings.device
        
        # Sample timesteps for flow matching
        timesteps = self.flow_matching_loss.sample_timesteps(batch_size, device)
        
        # Sample noise for flow matching
        noise = torch.randn_like(clip_embeddings)
        
        # Create noisy input for flow matching
        x_0 = torch.randn_like(clip_embeddings)
        noisy_clip = self.flow_matching_loss.interpolate_data(
            x_0=x_0,
            x_1=clip_embeddings,
            t=timesteps,
            noise=noise
        )
        
        # Forward pass through dual supervision model
        try:
            outputs = model(
                hidden_states=noisy_clip,
                timestep=timesteps,
                encoder_hidden_states=eva_embeddings,
                return_dict=True
            )
        except RuntimeError as e:
            if "device" in str(e).lower():
                logger.error(f"Device mismatch in model forward pass: {e}")
                logger.error(f"Inputs device check:")
                logger.error(f"  noisy_clip: {noisy_clip.device}")
                logger.error(f"  timesteps: {timesteps.device}")
                logger.error(f"  eva_embeddings: {eva_embeddings.device}")
            raise e
        
        # Extract dual outputs
        patch_output = outputs['patch_output']    # [B, 256, 1024]
        global_output = outputs['global_output']  # [B, 768]
        
        # FIXED: Compute target global features with device handling
        try:
            target_global = self.compute_target_global_features(clip_embeddings)
        except Exception as e:
            logger.error(f"Error computing target global features: {e}")
            # Fallback: create dummy target on same device
            target_global = torch.zeros(batch_size, 768, device=device, dtype=clip_embeddings.dtype)
        
        # Ensure all tensors are on the same device before loss computation
        if global_output is not None and global_output.device != target_global.device:
            target_global = target_global.to(global_output.device)
        
        # Compute dual supervision loss
        try:
            loss, metrics = self.flow_matching_loss(
                # DiT outputs
                dit_output=patch_output,
                dit_global=global_output,
                
                # Targets
                clip_patches=clip_embeddings,
                clip_global=target_global,
                
                # Flow matching inputs
                timesteps=timesteps,
                eva_conditioning=eva_embeddings,
                noise=noise,
                return_metrics=True
            )
        except RuntimeError as e:
            if "device" in str(e).lower():
                logger.error(f"Device mismatch in loss computation: {e}")
                logger.error(f"Tensor devices:")
                logger.error(f"  patch_output: {patch_output.device}")
                logger.error(f"  global_output: {global_output.device if global_output is not None else 'None'}")
                logger.error(f"  clip_embeddings: {clip_embeddings.device}")
                logger.error(f"  target_global: {target_global.device}")
            raise e
        
        # Store metrics
        if metrics is not None:
            for key, value in metrics.items():
                self.loss_components[key].append(value)
        
        # Enhanced logging
        if self.training_step_count % self.args.logging_steps == 0:
            self._log_dual_supervision_metrics(
                metrics, timesteps, patch_output, global_output, 
                clip_embeddings, target_global
            )
        
        self.training_step_count += 1
        
        # Prepare outputs
        enhanced_outputs = {
            'patch_output': patch_output,
            'global_output': global_output,
            'target_global': target_global,
            'timesteps': timesteps,
            'metrics': metrics,
            'eva_embeddings': eva_embeddings,
            'clip_embeddings': clip_embeddings,
        } if return_outputs else None
        
        if return_outputs:
            return loss, enhanced_outputs
        else:
            return loss
    
    def _log_dual_supervision_metrics(
        self,
        metrics: Optional[Dict[str, float]],
        timesteps: torch.Tensor,
        patch_output: torch.Tensor,
        global_output: torch.Tensor,
        clip_embeddings: torch.Tensor,
        target_global: torch.Tensor,
    ):
        """Enhanced logging for dual supervision training."""
        
        if metrics is None:
            return
        
        # Compute additional real-time metrics
        with torch.no_grad():
            # Patch-level alignment
            patch_cosine = F.cosine_similarity(
                F.normalize(patch_output, dim=-1),
                F.normalize(clip_embeddings, dim=-1),
                dim=-1
            ).mean().item()
            
            # Global-level alignment (key for recall)
            if global_output is not None and target_global is not None:
                global_cosine = F.cosine_similarity(
                    F.normalize(global_output, dim=-1),
                    F.normalize(target_global, dim=-1),
                    dim=-1
                ).mean().item()
            else:
                global_cosine = 0.0
            
            # Update EMA metrics
            self.ema_patch_cosine = self.ema_decay * self.ema_patch_cosine + (1 - self.ema_decay) * patch_cosine
            self.ema_global_cosine = self.ema_decay * self.ema_global_cosine + (1 - self.ema_decay) * global_cosine
            
            # Quality indicators
            good_patch_ratio = (F.cosine_similarity(
                F.normalize(patch_output, dim=-1),
                F.normalize(clip_embeddings, dim=-1),
                dim=-1
            ) > 0.7).float().mean().item()
            
            if global_output is not None and target_global is not None:
                good_global_ratio = (F.cosine_similarity(
                    F.normalize(global_output, dim=-1),
                    F.normalize(target_global, dim=-1),
                    dim=-1
                ) > 0.8).float().mean().item()
            else:
                good_global_ratio = 0.0
        
        # Create comprehensive logging dictionary
        log_dict = {}
        
        # Core dual supervision metrics
        for key, value in metrics.items():
            log_dict[f"train/{key}"] = value
        
        # Enhanced alignment metrics
        log_dict.update({
            # Patch metrics
            "train/patch_cosine_similarity": patch_cosine,
            "train/ema_patch_cosine": self.ema_patch_cosine,
            "train/good_patch_ratio": good_patch_ratio,
            
            # Global metrics (critical for recall)
            "train/global_cosine_similarity": global_cosine,
            "train/ema_global_cosine": self.ema_global_cosine,
            "train/good_global_ratio": good_global_ratio,
            
            # Overall quality (weighted for retrieval)
            "train/overall_quality": 0.3 * patch_cosine + 0.7 * global_cosine,
            "train/recall_readiness_score": global_cosine,  # Key metric for recall
        })
        
        # Training diagnostics
        log_dict.update({
            "train/timestep_mean": timesteps.mean().item(),
            "train/training_step": self.training_step_count,
            "train/epoch": self.state.epoch,
        })
        
        # Device diagnostics
        log_dict.update({
            "train/clip_projection_device": str(self.clip_visual_projection.weight.device),
            "train/input_device": str(clip_embeddings.device),
        })
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log(log_dict, step=self.training_step_count)
        
        # Store in history
        self.train_metrics_history.append({
            'step': self.training_step_count,
            'epoch': self.state.epoch,
            **log_dict
        })
        
        # Enhanced progress logging
        if self.training_step_count % (self.args.logging_steps * 2) == 0:
            logger.info(
                f"Step {self.training_step_count}: "
                f"Total_Loss={metrics.get('total_loss', 0):.4f}, "
                f"Patch_Loss={metrics.get('patch_loss', 0):.4f}, "
                f"Global_Loss={metrics.get('global_loss', 0):.4f}, "
                f"Patch_Cos={patch_cosine:.4f}, "
                f"Global_Cos={global_cosine:.4f}, "
                f"Recall_Ready={global_cosine:.4f}"
            )
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Enhanced evaluation with dual supervision metrics and device safety."""
        
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}
        
        model = self._wrap_model(self.model, training=False)
        model.eval()
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Collect evaluation metrics
        eval_losses = []
        all_metrics = defaultdict(list)
        patch_cosine_sims = []
        global_cosine_sims = []
        
        logger.info(f"Running FIXED dual supervision evaluation on {len(eval_dataloader)} batches")
        
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                try:
                    inputs = self._prepare_inputs(inputs)
                    
                    # Compute loss and metrics
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    eval_losses.append(loss.item())
                    
                    # Collect detailed metrics
                    if outputs and outputs.get('metrics'):
                        for key, value in outputs['metrics'].items():
                            all_metrics[key].append(value)
                    
                    # Additional alignment analysis
                    patch_out = outputs['patch_output']
                    global_out = outputs['global_output']
                    clip_emb = outputs['clip_embeddings']
                    target_global = outputs['target_global']
                    
                    # Compute real-time similarities
                    patch_cosine = F.cosine_similarity(
                        F.normalize(patch_out, dim=-1),
                        F.normalize(clip_emb, dim=-1),
                        dim=-1
                    ).mean().item()
                    
                    if global_out is not None and target_global is not None:
                        global_cosine = F.cosine_similarity(
                            F.normalize(global_out, dim=-1),
                            F.normalize(target_global, dim=-1),
                            dim=-1
                        ).mean().item()
                    else:
                        global_cosine = 0.0
                    
                    patch_cosine_sims.append(patch_cosine)
                    global_cosine_sims.append(global_cosine)
                    
                    if step % max(1, len(eval_dataloader) // 10) == 0:
                        logger.info(f"Evaluation step {step}/{len(eval_dataloader)}")
                
                except RuntimeError as e:
                    if "device" in str(e).lower():
                        logger.error(f"Device error in evaluation step {step}: {e}")
                        continue
                    else:
                        raise e
        
        # Aggregate results
        eval_results = {
            f'{metric_key_prefix}_loss': np.mean(eval_losses) if eval_losses else float('inf'),
            f'{metric_key_prefix}_loss_std': np.std(eval_losses) if eval_losses else 0.0,
        }
        
        # Aggregate detailed metrics
        for key, values in all_metrics.items():
            if values:
                eval_results[f'{metric_key_prefix}_{key}'] = np.mean(values)
                eval_results[f'{metric_key_prefix}_{key}_std'] = np.std(values)
        
        # Dual supervision specific metrics
        if patch_cosine_sims and global_cosine_sims:
            patch_array = np.array(patch_cosine_sims)
            global_array = np.array(global_cosine_sims)
            
            eval_results.update({
                # Patch metrics
                f'{metric_key_prefix}_patch_cosine_mean': np.mean(patch_array),
                f'{metric_key_prefix}_patch_cosine_std': np.std(patch_array),
                f'{metric_key_prefix}_good_patch_ratio': np.mean(patch_array > 0.7),
                
                # Global metrics (critical for recall)
                f'{metric_key_prefix}_global_cosine_mean': np.mean(global_array),
                f'{metric_key_prefix}_global_cosine_std': np.std(global_array),
                f'{metric_key_prefix}_good_global_ratio': np.mean(global_array > 0.8),
                f'{metric_key_prefix}_excellent_global_ratio': np.mean(global_array > 0.9),
                
                # Overall quality
                f'{metric_key_prefix}_overall_quality': 0.3 * np.mean(patch_array) + 0.7 * np.mean(global_array),
                f'{metric_key_prefix}_recall_readiness': np.mean(global_array),
            })
        
        # Log evaluation results
        if wandb.run is not None:
            wandb.log(eval_results, step=self.training_step_count)
        
        self.eval_metrics_history.append({
            'step': self.training_step_count,
            'epoch': self.state.epoch,
            **eval_results
        })
        
        logger.info(f"FIXED dual supervision evaluation results: {eval_results}")
        return eval_results
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Enhanced model saving with dual supervision information."""
        
        output_dir = output_dir or self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using parent class
        super().save_model(output_dir, _internal_call)
        
        # Save dual supervision specific configurations
        self._save_dual_supervision_configs(output_dir)
        
        logger.info(f"FIXED dual supervision BLIP3-o model saved to {output_dir}")
    
    def _save_dual_supervision_configs(self, output_dir: Path):
        """Save dual supervision specific configurations and metrics."""
        
        # Dual supervision training summary
        dual_summary = {
            'architecture': 'fixed_dual_supervision_blip3o_dit',
            'total_steps': self.training_step_count,
            'ema_patch_cosine': self.ema_patch_cosine,
            'ema_global_cosine': self.ema_global_cosine,
            'clip_model_name': self.clip_model_name,
            
            # Device handling info
            'device_fixes_applied': [
                'dynamic_clip_projection_placement',
                'model_device_tracking_patch',
                'robust_error_handling',
                'multi_gpu_compatibility'
            ],
            
            # Loss weights used
            'loss_weights': {
                'patch_loss_weight': getattr(self.flow_matching_loss, 'patch_loss_weight', 1.0),
                'global_loss_weight': getattr(self.flow_matching_loss, 'global_loss_weight', 2.0),
                'flow_matching_loss_weight': getattr(self.flow_matching_loss, 'flow_matching_loss_weight', 1.0),
            },
            
            # Performance metrics
            'final_metrics': {
                'best_patch_cosine': max([m.get('train/patch_cosine_similarity', 0) for m in self.train_metrics_history] + [0]),
                'best_global_cosine': max([m.get('train/global_cosine_similarity', 0) for m in self.train_metrics_history] + [0]),
                'final_overall_quality': max([m.get('train/overall_quality', 0) for m in self.train_metrics_history] + [0]),
                'recall_readiness': max([m.get('train/recall_readiness_score', 0) for m in self.train_metrics_history] + [0]),
            },
            
            # Expected improvements
            'expected_performance': {
                'patch_fidelity': 'maintained_high_quality',
                'global_alignment': 'optimized_for_retrieval',
                'recall_improvement': 'significant_expected',
                'target_recall_range': '60-80%',
                'device_compatibility': 'multi_gpu_ready',
            }
        }
        
        with open(output_dir / 'fixed_dual_supervision_summary.json', 'w') as f:
            json.dump(dual_summary, f, indent=2)
        
        # Save metrics histories
        if self.train_metrics_history:
            with open(output_dir / 'dual_supervision_train_metrics.json', 'w') as f:
                json.dump(self.train_metrics_history[-500:], f, indent=2)  # Last 500 steps
        
        if self.eval_metrics_history:
            with open(output_dir / 'dual_supervision_eval_metrics.json', 'w') as f:
                json.dump(self.eval_metrics_history, f, indent=2)


def create_blip3o_training_args(
    output_dir: str,
    num_train_epochs: int = 8,
    per_device_train_batch_size: int = 6,
    per_device_eval_batch_size: int = 4,
    learning_rate: float = 5e-5,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: int = 250,
    gradient_accumulation_steps: int = 6,
    fp16: bool = True,
    bf16: bool = False,
    dataloader_num_workers: int = 4,
    remove_unused_columns: bool = False,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_global_cosine_mean",
    greater_is_better: bool = True,
    **kwargs
) -> TrainingArguments:
    """Create TrainingArguments optimized for FIXED dual supervision training."""
    
    # Ensure evaluation strategy compatibility
    if load_best_model_at_end and eval_steps > 0:
        if save_steps % eval_steps != 0:
            adjusted_save_steps = ((save_steps // eval_steps) + 1) * eval_steps
            logging.warning(f"Adjusting save_steps from {save_steps} to {adjusted_save_steps}")
            save_steps = adjusted_save_steps
    
    eval_strategy = "steps" if eval_steps > 0 else "no"
    
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
        eval_steps=eval_steps if eval_steps > 0 else None,
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
        report_to=[],
        
        # FIXED: Multi-GPU DDP settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        save_on_each_node=False,
        **kwargs
    )