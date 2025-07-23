"""
Enhanced BLIP3-o Trainer with CLS+Patch Support and Flexible Training
src/modules/trainers/blip3o_flexible_trainer.py

Supports:
1. Both patch-only (256 tokens) and CLS+patch (257 tokens) modes
2. Flexible shard selection for training
3. Training on same data evaluation (overfitting test)
4. Pure flow matching loss (BLIP3-o paper aligned)
5. Detailed progress tracking
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
    Enhanced BLIP3-o Trainer with CLS+Patch Support
    
    Features:
    - Support for both 256 (patch-only) and 257 (CLS+patch) token modes
    - Flexible shard selection for training
    - Evaluation on same training data (overfitting tests)
    - Pure flow matching loss (no contrastive loss)
    - Detailed training metrics and progress tracking
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
        training_mode: str = "cls_patch",  # "cls_patch" or "patch_only"
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
            logger.info("✅ Enhanced BLIP3-o Flexible Trainer initialized")
            logger.info(f"🎯 Training mode: {self.training_mode} ({self.expected_tokens} tokens)")
            logger.info(f"📊 Max training shards: {self.max_training_shards or 'All'}")
            logger.info(f"🔄 Same data evaluation: {self.enable_same_data_eval}")
            logger.info(f"📈 Detailed logging: {self.detailed_logging}")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Enhanced compute_loss with support for both training modes
        """
        # Ensure model is in training mode
        model.train()
        
        # Extract inputs - should come from flexible collate function
        eva_embeddings = inputs['encoder_hidden_states']      # [B, N, 4096] where N=256 or 257
        clip_embeddings = inputs['clip_embeddings']           # [B, N, 1024] where N=256 or 257
        timesteps = inputs['timestep']                        # [B] - Flow matching timesteps
        
        # Get noisy input with proper gradients
        if 'hidden_states' in inputs:
            noisy_clip = inputs['hidden_states']              # [B, N, 1024] - Noisy input with gradients
            noise = inputs.get('noise', torch.randn_like(clip_embeddings))
        else:
            # Fallback: create noisy input
            device = eva_embeddings.device
            base_noise = torch.randn_like(clip_embeddings, requires_grad=True)
            noise = torch.randn_like(clip_embeddings)
            alpha = timesteps.view(-1, 1, 1)
            noisy_clip = (1 - alpha) * base_noise + alpha * clip_embeddings.detach() + 0.1 * noise

        # Validate tensor properties and training mode compatibility
        batch_size, seq_len, eva_dim = eva_embeddings.shape
        _, clip_seq_len, clip_dim = clip_embeddings.shape
        
        # Mode validation
        if seq_len != self.expected_tokens:
            logger.warning(f"Token count mismatch: expected {self.expected_tokens}, got {seq_len}")
            
        if seq_len != clip_seq_len:
            raise ValueError(f"EVA and CLIP token count mismatch: {seq_len} vs {clip_seq_len}")
        
        # Shape validation
        assert eva_dim == 4096, f"Expected EVA 4096-dim, got {eva_dim}"
        assert clip_dim == 1024, f"Expected CLIP 1024-dim, got {clip_dim}"
        assert noisy_clip.shape == clip_embeddings.shape, f"Noisy input shape mismatch"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        # Critical: Verify noisy input has gradients
        if not noisy_clip.requires_grad:
            logger.error("CRITICAL: Noisy input doesn't have gradients!")
            # Emergency fix
            device = noisy_clip.device
            emergency_noise = torch.randn_like(clip_embeddings, requires_grad=True, device=device)
            alpha = timesteps.view(-1, 1, 1)
            noisy_clip = (1 - alpha) * emergency_noise + alpha * clip_embeddings.detach()
            
            if not noisy_clip.requires_grad:
                raise RuntimeError("Failed to create tensor with gradients!")
        
        # Forward pass through model
        model_outputs = model(
            hidden_states=noisy_clip,              # [B, N, 1024] - Noisy input with gradients
            timestep=timesteps,                    # [B] - Timesteps
            encoder_hidden_states=eva_embeddings,  # [B, N, 4096] - EVA conditioning
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
        
        if velocity_pred.shape != clip_embeddings.shape:
            raise ValueError(f"Output shape mismatch: {velocity_pred.shape} vs {clip_embeddings.shape}")
        
        if not velocity_pred.requires_grad:
            raise RuntimeError("Model output doesn't require gradients!")
        
        # Compute pure flow matching loss (BLIP3-o paper aligned)
        loss, metrics = self.flow_matching_loss(
            model_output=velocity_pred,           # [B, N, 1024] - Predicted velocity
            target_samples=clip_embeddings,       # [B, N, 1024] - Target CLIP embeddings
            timesteps=timesteps,                  # [B] - Timesteps
            eva_conditioning=eva_embeddings,      # [B, N, 4096] - EVA conditioning
            noise=noise,                         # [B, N, 1024] - Noise for flow matching
            return_metrics=True
        )
        
        # Verify loss requires gradients
        if not loss.requires_grad:
            raise RuntimeError("Loss doesn't require gradients!")
        
        if not torch.isfinite(loss):
            raise ValueError(f"Loss is not finite: {loss.item()}")
        
        # Store metrics for detailed logging
        if metrics and self.is_main_process:
            metrics.update({
                'step': self.training_step_count,
                'timestamp': time.time(),
                'training_mode': self.training_mode,
                'num_tokens': seq_len,
                'batch_size': batch_size,
            })
            self.metric_history.append(metrics)
            self.loss_history.append(loss.item())
        
        # Detailed progress logging
        if (self.is_main_process and self.detailed_logging and 
            self.training_step_count % self.args.logging_steps == 0):
            self._log_detailed_progress(loss, metrics, velocity_pred, clip_embeddings)
        
        # Same data evaluation
        if (self.enable_same_data_eval and 
            self.training_step_count % self.eval_frequency == 0 and
            self.training_step_count > 0):
            self._run_same_data_evaluation(model, inputs)
        
        self.training_step_count += 1
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare outputs
        outputs = {
            'velocity_prediction': velocity_pred,
            'target_samples': clip_embeddings,
            'training_metrics': metrics,
            'training_mode': self.training_mode,
            'num_tokens': seq_len,
        } if return_outputs else None
        
        return (loss, outputs) if return_outputs else loss

    def _log_detailed_progress(
        self,
        loss: torch.Tensor,
        metrics: Optional[Dict[str, float]],
        velocity_pred: torch.Tensor,
        target_samples: torch.Tensor
    ):
        """Log detailed training progress with mode-specific information"""
        if not self.is_main_process:
            return
        
        loss_value = loss.item()
        seq_len = target_samples.shape[1]
        
        # Mode-specific information
        mode_info = f"Mode: {self.training_mode} ({seq_len} tokens)"
        if seq_len == 257:
            mode_info += " [CLS+Patches]"
        else:
            mode_info += " [Patches Only]"
        
        progress_msg = f"Step {self.training_step_count}: Loss={loss_value:.4f}, {mode_info}"
        
        # Add key metrics
        if metrics:
            # Flow matching quality
            if 'velocity_cosine_sim' in metrics:
                progress_msg += f", VelCos={metrics['velocity_cosine_sim']:.3f}"
            
            # Token-specific metrics
            if 'token_cosine_sim' in metrics:
                progress_msg += f", TokenCos={metrics['token_cosine_sim']:.3f}"
            
            # Mode-specific metrics
            if seq_len == 257 and 'cls_cosine_sim' in metrics:
                progress_msg += f", CLSCos={metrics['cls_cosine_sim']:.3f}"
                progress_msg += f", PatchCos={metrics['patch_cosine_sim']:.3f}"
            
            # Global coherence
            if 'global_cosine_sim' in metrics:
                progress_msg += f", GlobalCos={metrics['global_cosine_sim']:.3f}"
            
            # Training quality
            if 'training_quality' in metrics:
                progress_msg += f", Quality={metrics['training_quality']}"
        
        logger.info(progress_msg)
        
        # Quality assessment
        if metrics and 'global_cosine_sim' in metrics:
            global_cos = metrics['global_cosine_sim']
            if global_cos > 0.8:
                logger.info("🎉 EXCELLENT: Very strong alignment!")
            elif global_cos > 0.6:
                logger.info("✅ GOOD: Strong training progress")
            elif global_cos > 0.4:
                logger.info("🔄 FAIR: Making progress")

    def _run_same_data_evaluation(self, model, inputs):
        """Run evaluation on the same training data"""
        if not self.is_main_process:
            return
        
        model.eval()
        
        with torch.no_grad():
            try:
                # Generate from current inputs
                eva_embeddings = inputs['encoder_hidden_states']
                clip_targets = inputs['clip_embeddings']
                
                # Generate embeddings
                generated = model.generate(
                    eva_features=eva_embeddings,
                    num_inference_steps=20,  # Fast evaluation
                    return_intermediate=False
                )
                
                # Compute evaluation metrics
                generated_norm = F.normalize(generated, p=2, dim=-1)
                target_norm = F.normalize(clip_targets, p=2, dim=-1)
                
                # Per-patch cosine similarity
                patch_cosine_sim = F.cosine_similarity(
                    generated_norm.view(-1, 1024),
                    target_norm.view(-1, 1024),
                    dim=-1
                )
                
                # Per-image average
                batch_size, seq_len, _ = generated.shape
                patch_cosine_sim = patch_cosine_sim.view(batch_size, seq_len)
                per_image_avg = patch_cosine_sim.mean(dim=1)
                
                eval_metrics = {
                    'step': self.training_step_count,
                    'same_data_eval': True,
                    'training_mode': self.training_mode,
                    'num_tokens': seq_len,
                    'per_patch_mean_cosine': patch_cosine_sim.mean().item(),
                    'per_image_mean_cosine': per_image_avg.mean().item(),
                    'per_image_std_cosine': per_image_avg.std().item(),
                    'min_image_cosine': per_image_avg.min().item(),
                    'max_image_cosine': per_image_avg.max().item(),
                    'timestamp': time.time(),
                }
                
                self.eval_history.append(eval_metrics)
                
                logger.info(f"📊 Same-data eval: Patch={eval_metrics['per_patch_mean_cosine']:.3f}, "
                          f"Image={eval_metrics['per_image_mean_cosine']:.3f} "
                          f"(±{eval_metrics['per_image_std_cosine']:.3f})")
                
            except Exception as e:
                logger.warning(f"Same-data evaluation failed: {e}")
            finally:
                model.train()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model with enhanced training information"""
        if not self.is_main_process:
            return
        
        output_dir = output_dir or self.args.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model using parent class
            super().save_model(output_dir, _internal_call)
            
            # Save enhanced training info
            self._save_enhanced_training_info(output_path)
            
            logger.info(f"✅ Enhanced model and training info saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            # Fallback save
            try:
                torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
                logger.info("Fallback model save completed")
            except Exception:
                raise e

    def _save_enhanced_training_info(self, output_path: Path):
        """Save comprehensive training information"""
        # Enhanced training summary
        summary = {
            'training_completed': True,
            'training_mode': self.training_mode,
            'expected_tokens': self.expected_tokens,
            'max_training_shards': self.max_training_shards,
            'total_steps': self.training_step_count,
            'same_data_evaluation_enabled': self.enable_same_data_eval,
            'architecture': f'BLIP3-o DiT ({self.training_mode} mode)',
            'paper_alignment': 'Pure flow matching loss (BLIP3-o paper)',
            'token_configuration': f'{self.expected_tokens} tokens per image',
            'training_strategy': 'Flexible shard training with detailed evaluation',
            'timestamp': time.time(),
        }
        
        # Add final metrics
        if self.metric_history:
            latest_metrics = self.metric_history[-1]
            summary.update({
                'final_loss': self.loss_history[-1] if self.loss_history else None,
                'final_metrics': latest_metrics,
                'training_progression': {
                    'total_steps': len(self.loss_history),
                    'loss_trend': 'decreasing' if len(self.loss_history) > 10 and 
                                 self.loss_history[-1] < self.loss_history[10] else 'unknown',
                    'best_global_cosine': max([m.get('global_cosine_sim', 0) for m in self.metric_history]),
                    'best_velocity_cosine': max([m.get('velocity_cosine_sim', 0) for m in self.metric_history]),
                }
            })
        
        # Add evaluation history
        if self.eval_history:
            summary['same_data_evaluation'] = {
                'num_evaluations': len(self.eval_history),
                'best_per_patch_cosine': max([e.get('per_patch_mean_cosine', 0) for e in self.eval_history]),
                'best_per_image_cosine': max([e.get('per_image_mean_cosine', 0) for e in self.eval_history]),
                'latest_evaluation': self.eval_history[-1],
                'overfitting_indicator': self.eval_history[-1].get('per_image_mean_cosine', 0) > 0.8,
            }
        
        # Save files
        with open(output_path / 'enhanced_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed metrics (last 200 steps)
        if self.metric_history:
            with open(output_path / 'detailed_training_metrics.json', 'w') as f:
                json.dump(self.metric_history[-200:], f, indent=2)
        
        # Save evaluation history
        if self.eval_history:
            with open(output_path / 'same_data_evaluation_history.json', 'w') as f:
                json.dump(self.eval_history, f, indent=2)
        
        logger.info("Enhanced training information saved successfully")

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            'training_mode': self.training_mode,
            'expected_tokens': self.expected_tokens,
            'total_steps': self.training_step_count,
            'loss_history_length': len(self.loss_history),
            'metric_history_length': len(self.metric_history),
            'eval_history_length': len(self.eval_history),
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
        
        if self.eval_history:
            latest_eval = self.eval_history[-1]
            stats['latest_evaluation_metrics'] = latest_eval
            stats['overfitting_detected'] = latest_eval.get('per_image_mean_cosine', 0) > 0.8
        
        return stats


def create_blip3o_flexible_training_args(
    output_dir: str,
    training_mode: str = "cls_patch",
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 2,
    learning_rate: float = 1e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    warmup_steps: int = 50,
    logging_steps: int = 10,
    save_steps: int = 100,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    dataloader_num_workers: int = 2,
    enable_evaluation: bool = True,
    eval_steps: int = 50,
    **kwargs
) -> TrainingArguments:
    """Create flexible training arguments for different modes"""
    
    # Adjust defaults based on training mode
    if training_mode == "cls_patch":
        # CLS+Patch mode might need slightly different settings
        per_device_train_batch_size = max(2, per_device_train_batch_size - 1)  # Slightly larger tensors
    
    eval_strategy = "steps" if enable_evaluation else "no"
    
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
        metric_for_best_model="eval_loss" if enable_evaluation else None,
        greater_is_better=False,
        save_total_limit=3,
        prediction_loss_only=not enable_evaluation,
        report_to=[],
        dataloader_pin_memory=torch.cuda.is_available(),
        
        # Multi-GPU optimizations
        ddp_find_unused_parameters=False,
        dataloader_persistent_workers=True,
        
        # Stability settings
        ignore_data_skip=True,
        
        **kwargs
    )