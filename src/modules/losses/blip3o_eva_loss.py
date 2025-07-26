#!/usr/bin/env python3
"""
Fixed Flow Matching Loss for EVA-CLIP Reproduction Testing
src/modules/losses/blip3o_eva_loss.py

MAJOR FIXES:
1. Better numerical stability with proper scaling
2. Fixed gradient flow issues
3. Improved debugging metrics
4. Proper handling of normalized embeddings
5. Better velocity field learning according to feedback
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
    
    Key improvements based on feedback:
    1. Better numerical stability without excessive scaling
    2. Proper boundary condition handling
    3. Comprehensive debugging metrics
    4. Fixed velocity field learning
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        normalize_targets: bool = True,
        flow_type: str = "rectified",
        loss_scale: float = 1.0,  # Reduced from 100.0 based on feedback
        gradient_clip_val: float = 1.0,
        eps: float = 1e-8,
        debug_mode: bool = False,
        # Improved stability parameters
        min_timestep: float = 1e-3,  # Avoid t=0 numerical issues
        max_timestep: float = 1.0 - 1e-3,  # Avoid t=1 numerical issues
        velocity_norm_weight: float = 0.01,  # Weight for velocity norm regularization
        boundary_loss_weight: float = 0.1,  # Weight for boundary conditions
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.normalize_targets = normalize_targets
        self.flow_type = flow_type
        self.loss_scale = loss_scale
        self.gradient_clip_val = gradient_clip_val
        self.eps = eps
        self.debug_mode = debug_mode
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.velocity_norm_weight = velocity_norm_weight
        self.boundary_loss_weight = boundary_loss_weight
        
        # Training step counter
        self.register_buffer('step_count', torch.tensor(0))
        
        # Best metrics tracking
        self.register_buffer('best_velocity_sim', torch.tensor(0.0))
        self.register_buffer('best_loss', torch.tensor(float('inf')))
        
        # Running statistics for stability
        self.register_buffer('velocity_norm_ema', torch.tensor(1.0))
        self.register_buffer('error_norm_ema', torch.tensor(1.0))
        
        logger.info(f"✅ Fixed EVA Reproduction Flow Matching Loss")
        logger.info(f"   Loss scale: {loss_scale}")
        logger.info(f"   Gradient clipping: {gradient_clip_val}")
        logger.info(f"   Timestep range: [{min_timestep}, {max_timestep}]")
        logger.info(f"   Velocity norm weight: {velocity_norm_weight}")
        logger.info(f"   Boundary loss weight: {boundary_loss_weight}")
        logger.info(f"   Debug mode: {debug_mode}")

    def _clamp_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Clamp timesteps to avoid numerical issues at boundaries"""
        return torch.clamp(timesteps, min=self.min_timestep, max=self.max_timestep)

    def _update_ema(self, ema_tensor: torch.Tensor, new_value: float, alpha: float = 0.01):
        """Update exponential moving average"""
        ema_tensor.data = alpha * new_value + (1 - alpha) * ema_tensor.data

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
        Fixed flow matching loss computation with improved stability
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Update step count
        self.step_count += 1
        
        # Clamp timesteps to avoid boundary issues
        timesteps = self._clamp_timesteps(timesteps)
        
        # Ensure targets are properly normalized
        if self.normalize_targets:
            target_normalized = F.normalize(target_samples.detach(), p=2, dim=-1)
        else:
            target_normalized = target_samples.detach()
        
        # Create or validate noise
        if noise is None:
            # Create noise with same scale as normalized embeddings
            noise = torch.randn_like(target_normalized, device=device, dtype=dtype)
            # Normalize noise for consistency
            noise = F.normalize(noise, p=2, dim=-1)
        
        # Expand timesteps for broadcasting
        t = timesteps.view(-1, 1, 1).to(dtype)
        
        # RECTIFIED FLOW: Linear interpolation
        # x_t = (1-t) * x_0 + t * x_1
        # where x_0 = noise, x_1 = target
        x_t = (1 - t) * noise + t * target_normalized
        
        # True velocity for rectified flow: v = x_1 - x_0
        true_velocity = target_normalized - noise
        
        # IMPROVED LOSS COMPUTATION
        # 1. Base MSE loss
        velocity_error = model_output - true_velocity
        base_loss = F.mse_loss(model_output, true_velocity, reduction='none')
        
        # 2. Per-token and per-sample loss
        per_token_loss = base_loss.mean(dim=-1)  # [B, N]
        per_sample_loss = per_token_loss.mean(dim=1)  # [B]
        
        # 3. Timestep weighting (optional - can help with boundary conditions)
        # Weight early timesteps (near noise) slightly more
        time_weight = 1.0 + 0.1 * (1 - t.squeeze(-1))  # [B, 1]
        weighted_sample_loss = per_sample_loss * time_weight.squeeze(-1)
        
        # 4. Main loss
        main_loss = weighted_sample_loss.mean()
        
        # 5. Velocity norm regularization (helps with stability)
        pred_norm = torch.norm(model_output, dim=-1).mean()
        target_norm = torch.norm(true_velocity, dim=-1).mean()
        norm_loss = F.mse_loss(pred_norm, target_norm) * self.velocity_norm_weight
        
        # 6. Boundary condition loss (ensure model learns correct boundary behavior)
        boundary_mask_0 = timesteps < 0.1
        boundary_mask_1 = timesteps > 0.9
        
        boundary_loss = 0.0
        if boundary_mask_0.any():
            # Near t=0, velocity should point from noise to target
            boundary_loss_0 = F.mse_loss(
                model_output[boundary_mask_0], 
                true_velocity[boundary_mask_0]
            )
            boundary_loss += boundary_loss_0 * self.boundary_loss_weight
        
        if boundary_mask_1.any():
            # Near t=1, velocity should still be consistent
            boundary_loss_1 = F.mse_loss(
                model_output[boundary_mask_1], 
                true_velocity[boundary_mask_1]
            )
            boundary_loss += boundary_loss_1 * self.boundary_loss_weight
        
        # 7. Total loss with scaling
        total_loss = (main_loss + norm_loss + boundary_loss) * self.loss_scale
        
        # Apply gradient clipping to loss
        if total_loss.requires_grad and self.gradient_clip_val > 0:
            total_loss.register_hook(
                lambda grad: torch.clamp(grad, -self.gradient_clip_val, self.gradient_clip_val)
            )
        
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
                pred_norm_mean = torch.norm(model_output, dim=-1).mean()
                velocity_norm_mean = torch.norm(true_velocity, dim=-1).mean()
                eva_norm = torch.norm(target_normalized, dim=-1).mean()
                clip_norm = torch.norm(clip_conditioning, dim=-1).mean()
                noise_norm = torch.norm(noise, dim=-1).mean()
                error_norm = torch.norm(velocity_error, dim=-1).mean()
                
                # Update EMAs
                self._update_ema(self.velocity_norm_ema, velocity_norm_mean.item())
                self._update_ema(self.error_norm_ema, error_norm.item())
                
                # Gradient magnitude (if available)
                grad_norm = 0.0
                if model_output.grad is not None:
                    grad_norm = torch.norm(model_output.grad).item()
                
                # Update best metrics
                mean_sim = per_image_sim.mean().item()
                if mean_sim > self.best_velocity_sim.item():
                    self.best_velocity_sim = torch.tensor(mean_sim)
                
                unscaled_loss = total_loss.item() / self.loss_scale
                if unscaled_loss < self.best_loss.item():
                    self.best_loss = torch.tensor(unscaled_loss)
                
                # Training health indicators
                is_learning = mean_sim > 0.01
                is_converging = error_norm.item() < self.error_norm_ema.item() * 1.1
                is_stable = not (torch.isnan(total_loss) or torch.isinf(total_loss))
                
                training_health = "good" if (is_learning and is_converging and is_stable) else "needs_attention"
                
                metrics = {
                    # Core metrics
                    'loss': unscaled_loss,
                    'scaled_loss': total_loss.item(),
                    'main_loss': main_loss.item(),
                    'norm_loss': norm_loss.item(),
                    'boundary_loss': boundary_loss if isinstance(boundary_loss, float) else boundary_loss.item(),
                    'velocity_similarity': mean_sim,
                    
                    # Detailed similarity metrics
                    'per_image_sim_mean': per_image_sim.mean().item(),
                    'per_image_sim_std': per_image_sim.std().item(),
                    'per_patch_sim_mean': cosine_sim.mean().item(),
                    'per_patch_sim_std': cosine_sim.std().item(),
                    
                    # Norm tracking (should be ~1.0 for normalized embeddings)
                    'pred_norm': pred_norm_mean.item(),
                    'velocity_norm': velocity_norm_mean.item(),
                    'eva_norm': eva_norm.item(),
                    'clip_norm': clip_norm.item(),
                    'noise_norm': noise_norm.item(),
                    
                    # Error analysis
                    'error_norm': error_norm.item(),
                    'relative_error': (error_norm / (velocity_norm_mean + self.eps)).item(),
                    
                    # Training progress
                    'step_count': self.step_count.item(),
                    'best_velocity_sim': self.best_velocity_sim.item(),
                    'best_loss': self.best_loss.item(),
                    'grad_norm': grad_norm,
                    'training_health': training_health,
                    
                    # Stability indicators
                    'velocity_norm_ema': self.velocity_norm_ema.item(),
                    'error_norm_ema': self.error_norm_ema.item(),
                    'timestep_mean': timesteps.mean().item(),
                    'timestep_std': timesteps.std().item(),
                    'loss_scale': self.loss_scale,
                    
                    # Flow matching specific
                    'x_t_norm': torch.norm(x_t, dim=-1).mean().item(),
                    'interpolation_factor': t.mean().item(),
                }
                
                # Quality assessment
                if mean_sim > 0.7:
                    quality = "excellent"
                elif mean_sim > 0.3:
                    quality = "good"
                elif mean_sim > 0.1:
                    quality = "fair"
                else:
                    quality = "poor"
                
                metrics['quality_assessment'] = quality
                
                # Log debug info if enabled
                if self.debug_mode and self.step_count % 50 == 0:
                    logger.info(f"[Step {self.step_count}] EVA Flow Matching Debug:")
                    logger.info(f"  Loss: {unscaled_loss:.6f} (quality: {quality})")
                    logger.info(f"  Velocity Sim: {mean_sim:.4f} (best: {self.best_velocity_sim.item():.4f})")
                    logger.info(f"  Norms - Pred: {pred_norm_mean.item():.3f}, Target: {velocity_norm_mean.item():.3f}")
                    logger.info(f"  Error: {error_norm.item():.3f} (relative: {metrics['relative_error']:.3f})")
                    logger.info(f"  Training Health: {training_health}")
                    
                    if not is_learning:
                        logger.warning(f"  ⚠️ Model not learning! Similarity very low.")
                    if not is_stable:
                        logger.warning(f"  ⚠️ Training unstable! NaN/Inf detected.")
        
        return total_loss, metrics

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
            
            # Quality thresholds
            high_quality = (per_image_sim > 0.7).float().mean().item()
            very_high_quality = (per_image_sim > 0.8).float().mean().item()
            excellent_quality = (per_image_sim > 0.9).float().mean().item()
            
            return {
                'eval_cosine_sim': per_image_sim.mean().item(),
                'eval_mse_loss': mse_loss.item(),
                'high_quality_ratio': high_quality,
                'very_high_quality_ratio': very_high_quality,
                'excellent_quality_ratio': excellent_quality,
                'eval_similarity_std': per_image_sim.std().item(),
            }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'total_steps': self.step_count.item(),
            'best_velocity_sim': self.best_velocity_sim.item(),
            'best_loss': self.best_loss.item(),
            'current_velocity_norm_ema': self.velocity_norm_ema.item(),
            'current_error_norm_ema': self.error_norm_ema.item(),
            'loss_scale': self.loss_scale,
            'training_healthy': self.best_velocity_sim.item() > 0.01,
            'converged': self.best_velocity_sim.item() > 0.3,
        }


def create_eva_reproduction_loss(
    prediction_type: str = "velocity",
    normalize_targets: bool = True,
    flow_type: str = "rectified",
    loss_scale: float = 1.0,
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