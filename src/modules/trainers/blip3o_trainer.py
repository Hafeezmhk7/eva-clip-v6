"""
Enhanced BLIP3-o Trainer with comprehensive evaluation metrics including cosine similarity,
recall-oriented metrics, and better monitoring for alignment performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import wandb
import numpy as np
import math
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedBLIP3oTrainer(Trainer):
    """
    Enhanced trainer with comprehensive evaluation metrics for alignment and recall performance.
    
    Key features:
    - Cosine similarity tracking during training
    - CLIP alignment evaluation
    - Recall-oriented metrics
    - Progressive training support
    - Enhanced logging and monitoring
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
        
        # Enhanced evaluation parameters
        eval_alignment_frequency: int = 100,  # Evaluate alignment every N steps
        eval_generation_steps: int = 20,      # Steps for generation evaluation
        cosine_similarity_threshold: float = 0.7,  # Threshold for good alignment
        track_token_diversity: bool = True,   # Track feature diversity
        save_embeddings_samples: bool = False,  # Save embedding samples for analysis
        
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
        
        # Enhanced evaluation parameters
        self.eval_alignment_frequency = eval_alignment_frequency
        self.eval_generation_steps = eval_generation_steps
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.track_token_diversity = track_token_diversity
        self.save_embeddings_samples = save_embeddings_samples
        
        # Metrics tracking
        self.train_metrics_history = []
        self.eval_metrics_history = []
        self.alignment_metrics_history = []
        self.loss_components = defaultdict(list)
        
        # For EMA tracking
        self.ema_cosine_sim = 0.0
        self.ema_alignment_score = 0.0
        self.ema_decay = 0.99
        
        logger.info("Enhanced BLIP3-o trainer initialized with alignment evaluation")
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Enhanced loss computation with detailed metrics tracking."""
        
        # Extract inputs
        eva_embeddings = inputs['eva_embeddings']      # [B, 256, 4096]
        clip_embeddings = inputs['clip_embeddings']    # [B, 256, 1024]
        
        batch_size = eva_embeddings.shape[0]
        device = eva_embeddings.device
        
        # Sample timesteps
        timesteps = self.flow_matching_loss.sample_timesteps(batch_size, device)
        
        # Sample noise
        noise = torch.randn_like(clip_embeddings)
        
        # Create noisy samples
        x_0 = torch.randn_like(clip_embeddings)
        noisy_clip = self.flow_matching_loss.interpolate_data(
            x_0=x_0,
            x_1=clip_embeddings,
            t=timesteps,
            noise=noise
        )
        
        # Forward pass
        model_output = model(
            hidden_states=noisy_clip,
            timestep=timesteps,
            encoder_hidden_states=eva_embeddings,
            return_dict=False
        )
        
        # Compute enhanced loss with metrics
        loss, metrics = self.flow_matching_loss(
            model_output=model_output,
            target_samples=clip_embeddings,
            timesteps=timesteps,
            eva_conditioning=eva_embeddings,
            noise=noise,
            return_metrics=True
        )
        
        # Store metrics
        if metrics is not None:
            for key, value in metrics.items():
                self.loss_components[key].append(value)
        
        # Enhanced logging and evaluation
        if self.training_step_count % self.args.logging_steps == 0:
            self._log_enhanced_training_metrics(
                metrics, timesteps, model_output, clip_embeddings, eva_embeddings
            )
        
        # Periodic alignment evaluation
        if (self.training_step_count % self.eval_alignment_frequency == 0 and 
            self.training_step_count > 0):
            self._evaluate_alignment_quality(model, eva_embeddings, clip_embeddings)
        
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
    
    def _log_enhanced_training_metrics(
        self,
        metrics: Optional[Dict[str, float]],
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
        target_clip: torch.Tensor,
        eva_embeddings: torch.Tensor,
    ):
        """Enhanced logging with alignment and diversity metrics."""
        
        if metrics is None:
            return
        
        # Compute additional real-time metrics
        with torch.no_grad():
            # CLIP-style alignment (key for recall performance)
            pred_global = F.normalize(model_output.mean(dim=1), dim=-1)
            target_global = F.normalize(target_clip.mean(dim=1), dim=-1)
            global_cosine_sim = F.cosine_similarity(pred_global, target_global, dim=1).mean().item()
            
            # Update EMA
            self.ema_cosine_sim = self.ema_decay * self.ema_cosine_sim + (1 - self.ema_decay) * global_cosine_sim
            
            # Token-level alignment distribution
            pred_tokens_norm = F.normalize(model_output, dim=-1)
            target_tokens_norm = F.normalize(target_clip, dim=-1)
            token_cosine_sims = F.cosine_similarity(pred_tokens_norm, target_tokens_norm, dim=-1)
            
            # Feature diversity metrics
            if self.track_token_diversity:
                pred_diversity = torch.var(model_output.flatten(1), dim=0).mean().item()
                target_diversity = torch.var(target_clip.flatten(1), dim=0).mean().item()
                eva_diversity = torch.var(eva_embeddings.flatten(1), dim=0).mean().item()
            else:
                pred_diversity = target_diversity = eva_diversity = 0.0
            
            # Cross-modal alignment (EVA -> Generated CLIP similarity)
            eva_global = F.normalize(eva_embeddings.mean(dim=1), dim=-1)
            # Project EVA to CLIP space (approximate)
            cross_modal_sim = F.cosine_similarity(
                eva_global[:, :min(1024, eva_global.shape[-1])], 
                pred_global, 
                dim=1
            ).mean().item()
        
        # Create comprehensive logging dictionary
        log_dict = {}
        
        # Core flow matching metrics
        for key, value in metrics.items():
            log_dict[f"train/{key}"] = value
        
        # Enhanced alignment metrics
        log_dict.update({
            "train/global_cosine_similarity": global_cosine_sim,
            "train/ema_cosine_similarity": self.ema_cosine_sim,
            "train/token_cosine_mean": token_cosine_sims.mean().item(),
            "train/token_cosine_std": token_cosine_sims.std().item(),
            "train/token_cosine_min": token_cosine_sims.min().item(),
            "train/token_cosine_max": token_cosine_sims.max().item(),
            "train/cross_modal_similarity": cross_modal_sim,
            
            # Alignment quality indicators
            "train/good_alignment_ratio": (token_cosine_sims > self.cosine_similarity_threshold).float().mean().item(),
            "train/alignment_quality_score": torch.clamp(global_cosine_sim * 2 - 1, 0, 1),  # 0-1 scale
        })
        
        # Diversity metrics
        if self.track_token_diversity:
            log_dict.update({
                "train/pred_diversity": pred_diversity,
                "train/target_diversity": target_diversity,
                "train/eva_diversity": eva_diversity,
                "train/diversity_ratio": pred_diversity / (target_diversity + 1e-8),
            })
        
        # Timestep and training diagnostics
        log_dict.update({
            "train/timestep_mean": timesteps.mean().item(),
            "train/timestep_std": timesteps.std().item(),
            "train/training_step": self.training_step_count,
            "train/epoch": self.state.epoch,
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
                f"Loss={metrics.get('total_loss', 0):.4f}, "
                f"FM_Loss={metrics.get('flow_matching_loss', 0):.4f}, "
                f"Align_Loss={metrics.get('alignment_loss', 0):.4f}, "
                f"Cosine_Sim={global_cosine_sim:.4f}, "
                f"EMA_Cosine={self.ema_cosine_sim:.4f}, "
                f"Good_Align%={log_dict['train/good_alignment_ratio']*100:.1f}%, "
                f"Quality={log_dict['train/alignment_quality_score']:.3f}"
            )
    
    def _evaluate_alignment_quality(
        self,
        model,
        eva_embeddings: torch.Tensor,
        target_clip: torch.Tensor,
        num_generation_samples: int = 4,
    ):
        """Evaluate alignment quality through generation and similarity analysis."""
        
        if eva_embeddings.shape[0] < num_generation_samples:
            return
        
        model.eval()
        with torch.no_grad():
            # Sample a subset for evaluation
            eval_eva = eva_embeddings[:num_generation_samples]
            eval_target = target_clip[:num_generation_samples]
            
            # Generate CLIP embeddings using the model
            try:
                generated_clip = model.generate(
                    encoder_hidden_states=eval_eva,
                    num_inference_steps=self.eval_generation_steps,
                )
                
                # Compute alignment metrics
                gen_global = F.normalize(generated_clip.mean(dim=1), dim=-1)
                target_global = F.normalize(eval_target.mean(dim=1), dim=-1)
                
                # Generation quality metrics
                generation_cosine_sim = F.cosine_similarity(gen_global, target_global, dim=1).mean().item()
                generation_l2_dist = torch.norm(generated_clip - eval_target, dim=-1).mean().item()
                
                # Feature magnitude analysis
                gen_norm = torch.norm(generated_clip, dim=-1).mean().item()
                target_norm = torch.norm(eval_target, dim=-1).mean().item()
                
                # Token-wise alignment analysis
                gen_tokens_norm = F.normalize(generated_clip, dim=-1)
                target_tokens_norm = F.normalize(eval_target, dim=-1)
                token_alignment = F.cosine_similarity(gen_tokens_norm, target_tokens_norm, dim=-1)
                
                alignment_metrics = {
                    "eval_align/generation_cosine_similarity": generation_cosine_sim,
                    "eval_align/generation_l2_distance": generation_l2_dist,
                    "eval_align/generated_norm": gen_norm,
                    "eval_align/target_norm": target_norm,
                    "eval_align/norm_ratio": gen_norm / (target_norm + 1e-8),
                    "eval_align/token_alignment_mean": token_alignment.mean().item(),
                    "eval_align/token_alignment_std": token_alignment.std().item(),
                    "eval_align/good_tokens_ratio": (token_alignment > self.cosine_similarity_threshold).float().mean().item(),
                    "eval_align/step": self.training_step_count,
                }
                
                # Log alignment evaluation
                if wandb.run is not None:
                    wandb.log(alignment_metrics, step=self.training_step_count)
                
                self.alignment_metrics_history.append(alignment_metrics)
                
                logger.info(
                    f"Alignment Eval Step {self.training_step_count}: "
                    f"Gen_Cosine={generation_cosine_sim:.4f}, "
                    f"L2_Dist={generation_l2_dist:.4f}, "
                    f"Token_Align={token_alignment.mean().item():.4f}, "
                    f"Good_Tokens%={alignment_metrics['eval_align/good_tokens_ratio']*100:.1f}%"
                )
                
            except Exception as e:
                logger.warning(f"Failed to evaluate alignment quality: {e}")
        
        model.train()
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Enhanced evaluation with comprehensive alignment and recall metrics."""
        
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
        cosine_similarities = []
        alignment_scores = []
        generation_metrics = []
        
        logger.info(f"Running enhanced evaluation on {len(eval_dataloader)} batches")
        
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                inputs = self._prepare_inputs(inputs)
                
                # Compute loss and metrics
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                eval_losses.append(loss.item())
                
                # Collect detailed metrics
                if outputs and outputs.get('metrics'):
                    for key, value in outputs['metrics'].items():
                        all_metrics[key].append(value)
                
                # Additional alignment analysis
                eva_emb = outputs['eva_embeddings']
                clip_emb = outputs['clip_embeddings']
                model_out = outputs['model_output']
                
                # Compute real-time cosine similarities
                pred_global = F.normalize(model_out.mean(dim=1), dim=-1)
                target_global = F.normalize(clip_emb.mean(dim=1), dim=-1)
                batch_cosine_sim = F.cosine_similarity(pred_global, target_global, dim=1)
                cosine_similarities.extend(batch_cosine_sim.cpu().numpy())
                
                # Alignment quality score
                alignment_score = torch.clamp(batch_cosine_sim * 2 - 1, 0, 1)
                alignment_scores.extend(alignment_score.cpu().numpy())
                
                # Generation evaluation (subset of batches)
                if step % 5 == 0 and step < 20:  # Evaluate generation every 5th batch, up to 20 batches
                    try:
                        gen_metrics = self._evaluate_generation_subset(model, eva_emb, clip_emb)
                        if gen_metrics:
                            generation_metrics.append(gen_metrics)
                    except Exception as e:
                        logger.warning(f"Generation evaluation failed at step {step}: {e}")
                
                if step % max(1, len(eval_dataloader) // 10) == 0:
                    logger.info(f"Evaluation step {step}/{len(eval_dataloader)}")
        
        # Aggregate results
        eval_results = {
            f'{metric_key_prefix}_loss': np.mean(eval_losses),
            f'{metric_key_prefix}_loss_std': np.std(eval_losses),
        }
        
        # Aggregate detailed metrics
        for key, values in all_metrics.items():
            eval_results[f'{metric_key_prefix}_{key}'] = np.mean(values)
            eval_results[f'{metric_key_prefix}_{key}_std'] = np.std(values)
        
        # Cosine similarity analysis
        if cosine_similarities:
            cosine_array = np.array(cosine_similarities)
            eval_results.update({
                f'{metric_key_prefix}_cosine_similarity_mean': np.mean(cosine_array),
                f'{metric_key_prefix}_cosine_similarity_std': np.std(cosine_array),
                f'{metric_key_prefix}_cosine_similarity_median': np.median(cosine_array),
                f'{metric_key_prefix}_cosine_similarity_min': np.min(cosine_array),
                f'{metric_key_prefix}_cosine_similarity_max': np.max(cosine_array),
                f'{metric_key_prefix}_good_alignment_ratio': np.mean(cosine_array > self.cosine_similarity_threshold),
            })
        
        # Alignment quality analysis
        if alignment_scores:
            alignment_array = np.array(alignment_scores)
            eval_results.update({
                f'{metric_key_prefix}_alignment_quality_mean': np.mean(alignment_array),
                f'{metric_key_prefix}_alignment_quality_std': np.std(alignment_array),
                f'{metric_key_prefix}_high_quality_ratio': np.mean(alignment_array > 0.7),
            })
        
        # Generation metrics aggregation
        if generation_metrics:
            for key in generation_metrics[0].keys():
                values = [m[key] for m in generation_metrics if key in m]
                if values:
                    eval_results[f'{metric_key_prefix}_gen_{key}'] = np.mean(values)
        
        # Overall quality score (weighted combination)
        if cosine_similarities and alignment_scores:
            quality_score = (
                0.4 * eval_results[f'{metric_key_prefix}_cosine_similarity_mean'] +
                0.3 * eval_results[f'{metric_key_prefix}_alignment_quality_mean'] +
                0.3 * eval_results[f'{metric_key_prefix}_good_alignment_ratio']
            )
            eval_results[f'{metric_key_prefix}_overall_quality_score'] = quality_score
        
        # Log evaluation results
        if wandb.run is not None:
            wandb.log(eval_results, step=self.training_step_count)
        
        self.eval_metrics_history.append({
            'step': self.training_step_count,
            'epoch': self.state.epoch,
            **eval_results
        })
        
        logger.info(f"Enhanced evaluation results: {eval_results}")
        return eval_results
    
    def _evaluate_generation_subset(
        self,
        model,
        eva_embeddings: torch.Tensor,
        target_clip: torch.Tensor,
        num_samples: int = 2,
    ) -> Dict[str, float]:
        """Evaluate generation quality on a small subset."""
        
        if eva_embeddings.shape[0] < num_samples:
            return {}
        
        try:
            # Sample subset
            subset_eva = eva_embeddings[:num_samples]
            subset_target = target_clip[:num_samples]
            
            # Generate
            generated = model.generate(
                encoder_hidden_states=subset_eva,
                num_inference_steps=self.eval_generation_steps,
            )
            
            # Compute metrics
            gen_flat = generated.flatten(1)
            target_flat = subset_target.flatten(1)
            
            cosine_sim = F.cosine_similarity(gen_flat, target_flat, dim=1).mean().item()
            l2_dist = torch.norm(generated - subset_target, dim=-1).mean().item()
            
            return {
                'cosine_similarity': cosine_sim,
                'l2_distance': l2_dist,
                'norm_generated': torch.norm(generated, dim=-1).mean().item(),
                'norm_target': torch.norm(subset_target, dim=-1).mean().item(),
            }
        except Exception:
            return {}
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Enhanced model saving with comprehensive metrics."""
        
        output_dir = output_dir or self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using parent class
        super().save_model(output_dir, _internal_call)
        
        # Save enhanced metrics and configs
        self._save_enhanced_configs(output_dir)
        
        logger.info(f"Enhanced BLIP3-o model and metrics saved to {output_dir}")
    
    def _save_enhanced_configs(self, output_dir: Path):
        """Save enhanced configurations and metrics."""
        
        # Save enhanced training summary
        enhanced_summary = {
            'total_steps': self.training_step_count,
            'ema_cosine_similarity': self.ema_cosine_sim,
            'ema_alignment_score': self.ema_alignment_score,
            'eval_alignment_frequency': self.eval_alignment_frequency,
            'cosine_similarity_threshold': self.cosine_similarity_threshold,
            'best_alignment_score': max([m.get('train/global_cosine_similarity', 0) for m in self.train_metrics_history] + [0]),
            'final_quality_metrics': self._compute_final_quality_metrics(),
        }
        
        with open(output_dir / 'enhanced_training_summary.json', 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
        
        # Save all metrics histories
        histories = {
            'train_metrics': self.train_metrics_history[-1000:],  # Last 1000 for space
            'eval_metrics': self.eval_metrics_history,
            'alignment_metrics': self.alignment_metrics_history,
        }
        
        with open(output_dir / 'enhanced_metrics_histories.json', 'w') as f:
            json.dump(histories, f, indent=2)
        
        # Save loss components analysis
        loss_analysis = {}
        for key, values in self.loss_components.items():
            if values:
                loss_analysis[key] = {
                    'mean': np.mean(values[-100:]),  # Last 100 steps
                    'trend': np.mean(values[-50:]) - np.mean(values[-100:-50]) if len(values) >= 100 else 0,
                    'stability': np.std(values[-50:]) if len(values) >= 50 else 0,
                }
        
        with open(output_dir / 'loss_components_analysis.json', 'w') as f:
            json.dump(loss_analysis, f, indent=2)
    
    def _compute_final_quality_metrics(self) -> Dict[str, float]:
        """Compute final quality assessment metrics."""
        if not self.train_metrics_history:
            return {}
        
        # Get recent metrics (last 10% of training)
        recent_count = max(10, len(self.train_metrics_history) // 10)
        recent_metrics = self.train_metrics_history[-recent_count:]
        
        quality_metrics = {}
        
        # Cosine similarity trends
        cosine_sims = [m.get('train/global_cosine_similarity', 0) for m in recent_metrics]
        if cosine_sims:
            quality_metrics.update({
                'final_cosine_similarity_mean': np.mean(cosine_sims),
                'cosine_similarity_stability': 1.0 - np.std(cosine_sims),  # Higher = more stable
                'cosine_similarity_trend': np.polyfit(range(len(cosine_sims)), cosine_sims, 1)[0],  # Positive = improving
            })
        
        # Alignment quality trends
        align_scores = [m.get('train/alignment_quality_score', 0) for m in recent_metrics]
        if align_scores:
            quality_metrics.update({
                'final_alignment_quality': np.mean(align_scores),
                'alignment_consistency': 1.0 - np.std(align_scores),
            })
        
        # Loss convergence
        losses = [m.get('train/total_loss', float('inf')) for m in recent_metrics]
        if losses:
            quality_metrics.update({
                'final_loss': np.mean(losses),
                'loss_stability': 1.0 / (1.0 + np.std(losses)),  # Higher = more stable
            })
        
        return quality_metrics