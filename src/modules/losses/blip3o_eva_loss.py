#!/usr/bin/env python3
"""
Fixed Flow Matching Loss for EVA-CLIP Reproduction
Key fixes:
1. Proper loss computation and scaling
2. Better numerical stability
3. Correct velocity field learning
4. Fixed target computation
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
    Fixed Flow Matching Loss for EVA reproduction
    
    This implements rectified flow matching where:
    - x_0 = noise (source)
    - x_1 = clean EVA embeddings (target)
    - x_t = (1-t) * x_0 + t * x_1 (linear interpolation)
    - v_t = x_1 - x_0 (velocity field)
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        flow_type: str = "rectified",
        loss_weight: float = 1.0,
        eps: float = 1e-8,
        # Boundary handling
        min_timestep: float = 1e-3,
        max_timestep: float = 1.0 - 1e-3,
        # Regularization
        velocity_reg_weight: float = 0.0,
        consistency_weight: float = 0.0,
        debug_mode: bool = False,
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.flow_type = flow_type
        self.loss_weight = loss_weight
        self.eps = eps
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.velocity_reg_weight = velocity_reg_weight
        self.consistency_weight = consistency_weight
        self.debug_mode = debug_mode
        
        # Validate inputs
        assert prediction_type in ["velocity", "noise", "sample"]
        assert flow_type in ["rectified", "reflow"]
        
        # Running statistics
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('loss_ema', torch.tensor(0.0))
        self.register_buffer('similarity_ema', torch.tensor(0.0))
        
        logger.info(f"Flow Matching Loss initialized:")
        logger.info(f"  Prediction type: {prediction_type}")
        logger.info(f"  Flow type: {flow_type}")
        logger.info(f"  Loss weight: {loss_weight}")
        logger.info(f"  Debug mode: {debug_mode}")

    def _clamp_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Clamp timesteps to avoid numerical issues"""
        return torch.clamp(timesteps, min=self.min_timestep, max=self.max_timestep)

    def _update_ema(self, tensor_buffer: torch.Tensor, new_value: float, alpha: float = 0.01):
        """Update exponential moving average"""
        if tensor_buffer.item() == 0.0:
            tensor_buffer.data = torch.tensor(new_value)
        else:
            tensor_buffer.data = alpha * new_value + (1 - alpha) * tensor_buffer.data

    def forward(
        self,
        model_output: torch.Tensor,  # [B, N, 4096] - Model's prediction
        target_samples: torch.Tensor,  # [B, N, 4096] - Clean EVA embeddings
        timesteps: torch.Tensor,  # [B] - Flow matching timesteps
        clip_conditioning: torch.Tensor,  # [B, N, 1024] - CLIP features (for logging)
        noise: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute flow matching loss
        
        Args:
            model_output: Model prediction [B, N, 4096]
            target_samples: Clean EVA embeddings [B, N, 4096]
            timesteps: Timesteps [B]
            clip_conditioning: CLIP conditioning [B, N, 1024]
            noise: Noise tensor [B, N, 4096] (optional)
            return_metrics: Whether to return detailed metrics
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Update step count
        self.step_count += 1
        
        # Clamp timesteps
        timesteps = self._clamp_timesteps(timesteps)
        
        # Ensure targets are normalized (EVA embeddings should be L2 normalized)
        target_normalized = F.normalize(target_samples.detach(), p=2, dim=-1)
        
        # Create noise if not provided
        if noise is None:
            noise = torch.randn_like(target_normalized, device=device, dtype=dtype)
            noise = F.normalize(noise, p=2, dim=-1)  # Normalize noise too
        
        # Expand timesteps for broadcasting [B, 1, 1]
        t = timesteps.view(batch_size, 1, 1).to(dtype)
        
        # RECTIFIED FLOW COMPUTATION
        if self.flow_type == "rectified":
            # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
            # where x_0 = noise, x_1 = target
            # Velocity field: v = x_1 - x_0 = target - noise
            true_velocity = target_normalized - noise
            
            # The model should predict this velocity
            target_for_loss = true_velocity
            
        else:  # reflow
            raise NotImplementedError("Reflow not implemented yet")
        
        # LOSS COMPUTATION
        if self.prediction_type == "velocity":
            # Direct velocity prediction loss
            prediction_loss = F.mse_loss(model_output, target_for_loss, reduction='none')
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented")
        
        # Reduce loss: mean over tokens and embedding dimensions, then over batch
        prediction_loss = prediction_loss.mean(dim=(1, 2))  # [B]
        
        # Optional: timestep weighting (can help with training dynamics)
        # Weight early timesteps slightly more to ensure good noise handling
        time_weights = 1.0 + 0.1 * (1 - t.squeeze(-1))  # [B, 1] -> [B]
        weighted_loss = prediction_loss * time_weights.squeeze(-1)
        
        # Main loss
        main_loss = weighted_loss.mean()
        
        # REGULARIZATION TERMS
        reg_loss = 0.0
        
        # Velocity magnitude regularization
        if self.velocity_reg_weight > 0:
            pred_magnitude = torch.norm(model_output, dim=-1).mean()
            target_magnitude = torch.norm(target_for_loss, dim=-1).mean()
            velocity_reg = F.mse_loss(pred_magnitude, target_magnitude.detach())
            reg_loss += self.velocity_reg_weight * velocity_reg
        
        # Total loss
        total_loss = main_loss + reg_loss
        total_loss = total_loss * self.loss_weight
        
        # METRICS COMPUTATION
        metrics = {}
        if return_metrics:
            with torch.no_grad():
                # Normalize predictions for similarity computation
                pred_normalized = F.normalize(model_output + self.eps, p=2, dim=-1)
                target_norm = F.normalize(target_for_loss + self.eps, p=2, dim=-1)
                
                # Cosine similarity
                cosine_sim = F.cosine_similarity(pred_normalized, target_norm, dim=-1)
                per_image_sim = cosine_sim.mean(dim=1)  # [B]
                mean_similarity = per_image_sim.mean().item()
                
                # Update EMAs
                self._update_ema(self.loss_ema, main_loss.item())
                self._update_ema(self.similarity_ema, mean_similarity)
                
                # Compute norms for monitoring
                pred_norm = torch.norm(model_output, dim=-1).mean().item()
                target_norm_val = torch.norm(target_for_loss, dim=-1).mean().item()
                eva_norm = torch.norm(target_normalized, dim=-1).mean().item()
                noise_norm = torch.norm(noise, dim=-1).mean().item()
                
                # Error analysis
                error = model_output - target_for_loss
                error_norm = torch.norm(error, dim=-1).mean().item()
                relative_error = error_norm / (target_norm_val + self.eps)
                
                # Quality assessment
                if mean_similarity > 0.8:
                    quality = "excellent"
                elif mean_similarity > 0.5:
                    quality = "good"
                elif mean_similarity > 0.2:
                    quality = "fair"
                else:
                    quality = "poor"
                
                metrics = {
                    # Core metrics
                    'loss': main_loss.item(),
                    'total_loss': total_loss.item(),
                    'reg_loss': reg_loss if isinstance(reg_loss, float) else reg_loss.item(),
                    'velocity_similarity': mean_similarity,
                    'velocity_similarity_std': per_image_sim.std().item(),
                    
                    # Norm tracking
                    'pred_norm': pred_norm,
                    'target_norm': target_norm_val,
                    'eva_norm': eva_norm,
                    'noise_norm': noise_norm,
                    'clip_norm': torch.norm(clip_conditioning, dim=-1).mean().item(),
                    
                    # Error analysis
                    'error_norm': error_norm,
                    'relative_error': relative_error,
                    
                    # Training progress
                    'step_count': self.step_count.item(),
                    'loss_ema': self.loss_ema.item(),
                    'similarity_ema': self.similarity_ema.item(),
                    'quality_assessment': quality,
                    
                    # Flow matching specific
                    'timestep_mean': timesteps.mean().item(),
                    'timestep_std': timesteps.std().item(),
                    'interpolation_weight': t.mean().item(),
                    
                    # Debugging info
                    'pred_min': model_output.min().item(),
                    'pred_max': model_output.max().item(),
                    'target_min': target_for_loss.min().item(),
                    'target_max': target_for_loss.max().item(),
                }
                
                # Check for numerical issues
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    metrics['numerical_issue'] = True
                    logger.error(f"[Step {self.step_count}] Numerical issue detected!")
                    logger.error(f"  Loss: {total_loss.item()}")
                    logger.error(f"  Pred norm: {pred_norm}")
                    logger.error(f"  Target norm: {target_norm_val}")
                
                # Debug logging
                if self.debug_mode and self.step_count % 50 == 0:
                    logger.info(f"[Step {self.step_count}] Flow Matching Debug:")
                    logger.info(f"  Loss: {main_loss.item():.6f} (quality: {quality})")
                    logger.info(f"  Velocity Sim: {mean_similarity:.4f}")
                    logger.info(f"  Norms - Pred: {pred_norm:.3f}, Target: {target_norm_val:.3f}")
                    logger.info(f"  Error: {error_norm:.3f} (relative: {relative_error:.3f})")
                    logger.info(f"  Timesteps: {timesteps.mean().item():.3f} ± {timesteps.std().item():.3f}")
        
        return total_loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,  # Generated EVA embeddings
        target: torch.Tensor,     # Target EVA embeddings
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        with torch.no_grad():
            # Normalize both
            generated_norm = F.normalize(generated, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            
            # Cosine similarity
            cosine_sim = F.cosine_similarity(generated_norm, target_norm, dim=-1)
            per_image_sim = cosine_sim.mean(dim=1)
            
            # MSE in normalized space
            mse_loss = F.mse_loss(generated_norm, target_norm)
            
            # Quality metrics
            high_quality = (per_image_sim > 0.7).float().mean().item()
            very_high_quality = (per_image_sim > 0.8).float().mean().item()
            excellent_quality = (per_image_sim > 0.9).float().mean().item()
            
            return {
                'eval_eva_similarity': per_image_sim.mean().item(),
                'eval_mse_loss': mse_loss.item(),
                'eval_high_quality_ratio': high_quality,
                'eval_very_high_quality_ratio': very_high_quality,
                'eval_excellent_quality_ratio': excellent_quality,
                'eval_similarity_std': per_image_sim.std().item(),
            }


def create_eva_reproduction_loss(
    prediction_type: str = "velocity",
    flow_type: str = "rectified", 
    loss_weight: float = 1.0,
    debug_mode: bool = False,
    **kwargs
) -> BLIP3oEVAFlowMatchingLoss:
    """Factory function for EVA reproduction loss"""
    return BLIP3oEVAFlowMatchingLoss(
        prediction_type=prediction_type,
        flow_type=flow_type,
        loss_weight=loss_weight,
        debug_mode=debug_mode,
        **kwargs
    )