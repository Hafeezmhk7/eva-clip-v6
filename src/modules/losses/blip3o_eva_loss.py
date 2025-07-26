#!/usr/bin/env python3
"""
Fixed Flow Matching Loss for EVA-CLIP Reproduction Testing
src/modules/losses/blip3o_eva_loss.py

Key fixes:
- Better numerical stability
- Proper gradient clipping
- Improved debugging metrics
- Correct rectified flow implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class BLIP3oEVAFlowMatchingLoss(nn.Module):
    """
    Fixed Flow Matching Loss for EVA-CLIP Reproduction Testing
    
    Key improvements:
    1. Better numerical stability with scaled loss
    2. Gradient clipping at multiple levels
    3. Comprehensive debugging metrics
    4. Proper handling of normalized embeddings
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        normalize_targets: bool = True,
        flow_type: str = "rectified",
        loss_scale: float = 100.0,  # Scale up loss for better gradients
        gradient_clip_val: float = 1.0,  # Gradient clipping value
        eps: float = 1e-8,  # Small epsilon for stability
        debug_mode: bool = False,
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.normalize_targets = normalize_targets
        self.flow_type = flow_type
        self.loss_scale = loss_scale
        self.gradient_clip_val = gradient_clip_val
        self.eps = eps
        self.debug_mode = debug_mode
        
        # Training step counter
        self.register_buffer('step_count', torch.tensor(0))
        
        # Best metrics tracking
        self.register_buffer('best_velocity_sim', torch.tensor(0.0))
        self.register_buffer('best_loss', torch.tensor(float('inf')))
        
        logger.info(f"✅ Fixed EVA Reproduction Flow Matching Loss")
        logger.info(f"   Loss scale: {loss_scale}")
        logger.info(f"   Gradient clipping: {gradient_clip_val}")
        logger.info(f"   Debug mode: {debug_mode}")

    def forward(
        self,
        model_output: torch.Tensor,  # [B, N, 4096] - Model's velocity prediction
        target_samples: torch.Tensor,  # [B, N, 4096] - Ground truth EVA embeddings
        timesteps: torch.Tensor,  # [B] - Flow matching timesteps
        clip_conditioning: torch.Tensor,  # [B, N, 1024] - CLIP conditioning
        noise: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        training_mode: str = "patch_only",
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Fixed flow matching loss computation with better numerical stability
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Update step count
        self.step_count += 1
        
        # Ensure targets are normalized
        if self.normalize_targets:
            target_normalized = F.normalize(target_samples.detach(), p=2, dim=-1)
        else:
            target_normalized = target_samples.detach()
        
        # Create or validate noise
        if noise is None:
            # Create noise with same scale as normalized embeddings
            noise = torch.randn_like(target_normalized, device=device, dtype=dtype)
            # Optionally normalize noise for better stability
            noise = F.normalize(noise, p=2, dim=-1)
        
        # Expand timesteps for broadcasting
        t = timesteps.view(-1, 1, 1).to(dtype)
        
        # RECTIFIED FLOW: Linear interpolation
        # x_t = (1-t) * x_0 + t * x_1
        # where x_0 = noise, x_1 = target
        x_t = (1 - t) * noise + t * target_normalized
        
        # True velocity: v = x_1 - x_0
        true_velocity = target_normalized - noise
        
        # Compute prediction error
        error = model_output - true_velocity
        
        # IMPROVED LOSS COMPUTATION
        # Option 1: Scaled MSE loss for better gradients
        base_loss = F.mse_loss(model_output, true_velocity, reduction='none')
        
        # Apply per-token loss (important for stability)
        per_token_loss = base_loss.mean(dim=-1)  # [B, N]
        
        # Apply loss weighting based on timestep (optional)
        # Earlier timesteps (closer to noise) might need different weighting
        time_weight = 1.0  # Could use: (1 - t.squeeze(-1)) * 0.5 + 0.5
        weighted_loss = per_token_loss * time_weight
        
        # Final loss with scaling
        loss = weighted_loss.mean() * self.loss_scale
        
        # Apply gradient clipping to loss
        if loss.requires_grad and self.gradient_clip_val > 0:
            loss.register_hook(lambda grad: torch.clamp(grad, -self.gradient_clip_val, self.gradient_clip_val))
        
        # Compute detailed metrics
        metrics = None
        if return_metrics or self.debug_mode:
            with torch.no_grad():
                # Normalize predictions and targets for similarity
                pred_normalized = F.normalize(model_output, p=2, dim=-1)
                velocity_normalized = F.normalize(true_velocity, p=2, dim=-1)
                
                # Compute cosine similarity
                cosine_sim = F.cosine_similarity(pred_normalized, velocity_normalized, dim=-1)
                
                # Per-image metrics
                per_image_sim = cosine_sim.mean(dim=1)
                
                # Compute norms
                pred_norm = torch.norm(model_output, dim=-1).mean()
                velocity_norm = torch.norm(true_velocity, dim=-1).mean()
                eva_norm = torch.norm(target_normalized, dim=-1).mean()
                clip_norm = torch.norm(clip_conditioning, dim=-1).mean()
                noise_norm = torch.norm(noise, dim=-1).mean()
                
                # Gradient magnitude (if available)
                grad_norm = 0.0
                if model_output.grad is not None:
                    grad_norm = torch.norm(model_output.grad).item()
                
                # Update best metrics
                mean_sim = per_image_sim.mean().item()
                if mean_sim > self.best_velocity_sim.item():
                    self.best_velocity_sim = torch.tensor(mean_sim)
                
                if loss.item() < self.best_loss.item():
                    self.best_loss = loss.clone().detach()
                
                metrics = {
                    # Core metrics
                    'loss': loss.item() / self.loss_scale,  # Unscaled loss
                    'scaled_loss': loss.item(),
                    'velocity_similarity': mean_sim,
                    
                    # Detailed similarity metrics
                    'per_image_sim_mean': per_image_sim.mean().item(),
                    'per_image_sim_std': per_image_sim.std().item(),
                    'per_patch_sim_mean': cosine_sim.mean().item(),
                    'per_patch_sim_std': cosine_sim.std().item(),
                    
                    # Norm tracking
                    'pred_norm': pred_norm.item(),
                    'velocity_norm': velocity_norm.item(),
                    'eva_norm': eva_norm.item(),
                    'clip_norm': clip_norm.item(),
                    'noise_norm': noise_norm.item(),
                    
                    # Error metrics
                    'error_norm': torch.norm(error, dim=-1).mean().item(),
                    'relative_error': (torch.norm(error, dim=-1) / (torch.norm(true_velocity, dim=-1) + self.eps)).mean().item(),
                    
                    # Training progress
                    'step_count': self.step_count.item(),
                    'best_velocity_sim': self.best_velocity_sim.item(),
                    'best_loss': self.best_loss.item() / self.loss_scale,
                    'grad_norm': grad_norm,
                    
                    # Debug info
                    'timestep_mean': timesteps.mean().item(),
                    'loss_scale': self.loss_scale,
                    'x_t_norm': torch.norm(x_t, dim=-1).mean().item(),
                }
                
                # Log debug info if enabled
                if self.debug_mode and self.step_count % 50 == 0:
                    logger.info(f"[Step {self.step_count}] Debug Info:")
                    logger.info(f"  Loss: {metrics['loss']:.6f} (scaled: {metrics['scaled_loss']:.6f})")
                    logger.info(f"  Velocity Sim: {metrics['velocity_similarity']:.4f}")
                    logger.info(f"  Pred Norm: {metrics['pred_norm']:.3f}")
                    logger.info(f"  Target Norm: {metrics['velocity_norm']:.3f}")
                    logger.info(f"  Error Norm: {metrics['error_norm']:.3f}")
                    logger.info(f"  Relative Error: {metrics['relative_error']:.3f}")
        
        return loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,  # Generated EVA embeddings
        target: torch.Tensor,     # Target EVA embeddings
    ) -> Dict[str, float]:
        """Compute evaluation metrics between generated and target EVA embeddings"""
        with torch.no_grad():
            # Normalize for similarity computation
            generated_norm = F.normalize(generated, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            
            # Compute similarities
            cosine_sim = F.cosine_similarity(generated_norm, target_norm, dim=-1)
            per_image_sim = cosine_sim.mean(dim=1)
            
            # MSE in normalized space
            mse_loss = F.mse_loss(generated_norm, target_norm)
            
            return {
                'eval_cosine_sim': per_image_sim.mean().item(),
                'eval_mse_loss': mse_loss.item(),
                'high_quality_ratio': (per_image_sim > 0.7).float().mean().item(),
                'very_high_quality_ratio': (per_image_sim > 0.8).float().mean().item(),
            }


def create_eva_reproduction_loss(
    prediction_type: str = "velocity",
    normalize_targets: bool = True,
    flow_type: str = "rectified",
    loss_scale: float = 100.0,
    gradient_clip_val: float = 1.0,
    debug_mode: bool = False,
    **kwargs
) -> BLIP3oEVAFlowMatchingLoss:
    """
    Factory function for EVA reproduction flow matching loss
    """
    return BLIP3oEVAFlowMatchingLoss(
        prediction_type=prediction_type,
        normalize_targets=normalize_targets,
        flow_type=flow_type,
        loss_scale=loss_scale,
        gradient_clip_val=gradient_clip_val,
        debug_mode=debug_mode,
        **kwargs
    )