#!/usr/bin/env python3
"""
FIXED: Flow Matching Loss for CLIP Reproduction with Proper Noise Scaling
Key fixes:
1. Data-adaptive noise scaling to match CLIP embedding distribution
2. Consistent scaling between training and inference
3. Proper velocity field computation with matched scales
4. Based on BLIP3-o and Stable Diffusion 3 best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class BLIP3oCLIPFlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss for CLIP reproduction with proper noise scaling
    
    Key improvements:
    - Data-adaptive noise scaling to match CLIP embedding statistics
    - Consistent noise distribution between training and inference
    - Proper rectified flow formulation with matched scales
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
        # NEW: Noise scaling parameters
        use_adaptive_noise_scaling: bool = True,
        noise_scale_momentum: float = 0.99,
        initial_noise_scale: float = 1.0,
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
        
        # NEW: Adaptive noise scaling
        self.use_adaptive_noise_scaling = use_adaptive_noise_scaling
        self.noise_scale_momentum = noise_scale_momentum
        
        # Validate inputs
        assert prediction_type in ["velocity", "noise", "sample"]
        assert flow_type in ["rectified", "reflow"]
        
        # Running statistics
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('loss_ema', torch.tensor(0.0))
        self.register_buffer('similarity_ema', torch.tensor(0.0))
        
        # NEW: Noise scaling statistics (adaptive to data)
        if self.use_adaptive_noise_scaling:
            self.register_buffer('target_norm_ema', torch.tensor(initial_noise_scale))
            self.register_buffer('target_std_ema', torch.tensor(initial_noise_scale))
            self.register_buffer('noise_scale', torch.tensor(initial_noise_scale))
            self.register_buffer('statistics_initialized', torch.tensor(False))
        
        logger.info(f"CLIP Flow Matching Loss initialized with noise scaling:")
        logger.info(f"  Prediction type: {prediction_type}")
        logger.info(f"  Flow type: {flow_type}")
        logger.info(f"  Adaptive noise scaling: {use_adaptive_noise_scaling}")
        logger.info(f"  Debug mode: {debug_mode}")

    def _update_noise_statistics(self, target_samples: torch.Tensor):
        """Update noise scaling statistics based on target distribution"""
        if not self.use_adaptive_noise_scaling:
            return
        
        with torch.no_grad():
            # Compute current batch statistics
            target_norm = torch.norm(target_samples, dim=-1).mean()
            target_std = target_samples.std()
            
            if not self.statistics_initialized:
                # Initialize on first batch
                self.target_norm_ema.data = target_norm
                self.target_std_ema.data = target_std
                self.noise_scale.data = target_std  # Use std as initial scale
                self.statistics_initialized.data = torch.tensor(True)
                
                if self.debug_mode:
                    logger.info(f"Initialized noise statistics:")
                    logger.info(f"  Target norm: {target_norm:.3f}")
                    logger.info(f"  Target std: {target_std:.3f}")
                    logger.info(f"  Initial noise scale: {self.noise_scale.item():.3f}")
            else:
                # Update with momentum
                self.target_norm_ema.data = (
                    self.noise_scale_momentum * self.target_norm_ema + 
                    (1 - self.noise_scale_momentum) * target_norm
                )
                self.target_std_ema.data = (
                    self.noise_scale_momentum * self.target_std_ema + 
                    (1 - self.noise_scale_momentum) * target_std
                )
                
                # Update noise scale based on target statistics
                # Use a combination of norm and std for robust scaling
                scale_from_norm = self.target_norm_ema / math.sqrt(target_samples.shape[-1])
                scale_from_std = self.target_std_ema
                
                # Weighted combination (more weight on std for stability)
                self.noise_scale.data = 0.3 * scale_from_norm + 0.7 * scale_from_std

    def _create_scaled_noise(self, target_samples: torch.Tensor) -> torch.Tensor:
        """Create noise scaled to match target distribution"""
        device = target_samples.device
        dtype = target_samples.dtype
        
        # Create standard Gaussian noise
        noise = torch.randn_like(target_samples, device=device, dtype=dtype)
        
        if self.use_adaptive_noise_scaling and self.statistics_initialized:
            # Scale noise to match target distribution
            noise = noise * self.noise_scale
            
            if self.debug_mode and self.step_count % 100 == 0:
                actual_noise_std = noise.std().item()
                target_std = target_samples.std().item()
                logger.debug(f"Noise scaling - Scale: {self.noise_scale.item():.3f}, "
                           f"Noise std: {actual_noise_std:.3f}, Target std: {target_std:.3f}")
        
        return noise

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
        model_output: torch.Tensor,  # [B, N, 1024] - Model's prediction
        target_samples: torch.Tensor,  # [B, N, 1024] - Clean CLIP embeddings
        timesteps: torch.Tensor,  # [B] - Flow matching timesteps
        eva_conditioning: torch.Tensor,  # [B, N, 4096] - EVA features (for logging)
        noise: Optional[torch.Tensor] = None,
        return_metrics: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute flow matching loss with proper noise scaling
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Update step count
        self.step_count += 1
        
        # Clamp timesteps
        timesteps = self._clamp_timesteps(timesteps)
        
        # Keep targets as-is (no normalization during training)
        target_clean = target_samples.detach()
        
        # Update noise scaling statistics
        self._update_noise_statistics(target_clean)
        
        # Create properly scaled noise
        if noise is None:
            noise = self._create_scaled_noise(target_clean)
        else:
            # If noise provided, ensure it's properly scaled
            if self.use_adaptive_noise_scaling and self.statistics_initialized:
                current_noise_std = noise.std()
                expected_noise_std = self.noise_scale
                if abs(current_noise_std - expected_noise_std) > 0.1 * expected_noise_std:
                    # Rescale provided noise if it doesn't match expected scale
                    noise = noise * (expected_noise_std / (current_noise_std + self.eps))
        
        # Expand timesteps for broadcasting [B, 1, 1]
        t = timesteps.view(batch_size, 1, 1).to(dtype)
        
        # RECTIFIED FLOW COMPUTATION with proper scaling
        if self.flow_type == "rectified":
            # Linear interpolation: x_t = (1-t) * noise + t * target
            # With proper scaling, both noise and target are on similar scales
            true_velocity = target_clean - noise
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
        
        # Optional: timestep weighting
        time_weights = 1.0 + 0.1 * (1 - t.squeeze(-1))  # [B]
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
                # ONLY normalize for cosine similarity computation
                pred_normalized = F.normalize(model_output + self.eps, p=2, dim=-1)
                target_norm = F.normalize(target_for_loss + self.eps, p=2, dim=-1)
                
                # Cosine similarity (requires normalization)
                cosine_sim = F.cosine_similarity(pred_normalized, target_norm, dim=-1)
                per_image_sim = cosine_sim.mean(dim=1)  # [B]
                mean_similarity = per_image_sim.mean().item()
                
                # Update EMAs
                self._update_ema(self.loss_ema, main_loss.item())
                self._update_ema(self.similarity_ema, mean_similarity)
                
                # Compute norms for monitoring (raw, unnormalized)
                pred_norm = torch.norm(model_output, dim=-1).mean().item()
                target_norm_val = torch.norm(target_for_loss, dim=-1).mean().item()
                clip_norm = torch.norm(target_clean, dim=-1).mean().item()
                noise_norm = torch.norm(noise, dim=-1).mean().item()
                
                # Error analysis (raw space)
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
                    'clip_norm': clip_norm,
                    'noise_norm': noise_norm,
                    'eva_norm': torch.norm(eva_conditioning, dim=-1).mean().item(),
                    
                    # NEW: Noise scaling metrics
                    'noise_scale': self.noise_scale.item() if self.use_adaptive_noise_scaling else 1.0,
                    'target_norm_ema': self.target_norm_ema.item() if self.use_adaptive_noise_scaling else 0.0,
                    'target_std_ema': self.target_std_ema.item() if self.use_adaptive_noise_scaling else 0.0,
                    'noise_target_ratio': noise_norm / (clip_norm + self.eps),
                    
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
                    logger.error(f"  Noise scale: {self.noise_scale.item() if self.use_adaptive_noise_scaling else 'N/A'}")
                
                # Debug logging
                if self.debug_mode and self.step_count % 50 == 0:
                    logger.info(f"[Step {self.step_count}] CLIP Flow Matching Debug:")
                    logger.info(f"  Loss: {main_loss.item():.6f} (quality: {quality})")
                    logger.info(f"  Velocity Sim: {mean_similarity:.4f}")
                    logger.info(f"  Norms - Pred: {pred_norm:.3f}, Target: {target_norm_val:.3f}, Noise: {noise_norm:.3f}")
                    if self.use_adaptive_noise_scaling:
                        logger.info(f"  Noise Scale: {self.noise_scale.item():.3f} (ratio: {metrics['noise_target_ratio']:.3f})")
                    logger.info(f"  Error: {error_norm:.3f} (relative: {relative_error:.3f})")
        
        return total_loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,  # Generated CLIP embeddings
        target: torch.Tensor,     # Target CLIP embeddings
    ) -> Dict[str, float]:
        """Compute evaluation metrics (normalize only for similarity)"""
        with torch.no_grad():
            # Normalize ONLY for cosine similarity computation
            generated_norm = F.normalize(generated, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            
            # Cosine similarity (requires normalization)
            cosine_sim = F.cosine_similarity(generated_norm, target_norm, dim=-1)
            per_image_sim = cosine_sim.mean(dim=1)
            
            # MSE in raw space (no normalization)
            mse_loss = F.mse_loss(generated, target)
            
            # Quality metrics (based on cosine similarity)
            high_quality = (per_image_sim > 0.7).float().mean().item()
            very_high_quality = (per_image_sim > 0.8).float().mean().item()
            excellent_quality = (per_image_sim > 0.9).float().mean().item()
            
            # Scale analysis
            generated_norm_val = torch.norm(generated, dim=-1).mean().item()
            target_norm_val = torch.norm(target, dim=-1).mean().item()
            
            return {
                'eval_clip_similarity': per_image_sim.mean().item(),
                'eval_mse_loss': mse_loss.item(),
                'eval_high_quality_ratio': high_quality,
                'eval_very_high_quality_ratio': very_high_quality,
                'eval_excellent_quality_ratio': excellent_quality,
                'eval_similarity_std': per_image_sim.std().item(),
                'eval_generated_norm': generated_norm_val,
                'eval_target_norm': target_norm_val,
                'eval_norm_ratio': generated_norm_val / (target_norm_val + 1e-8),
            }

    def get_noise_scale(self) -> float:
        """Get current noise scale for inference"""
        if self.use_adaptive_noise_scaling and self.statistics_initialized:
            return self.noise_scale.item()
        else:
            return 1.0


def create_clip_reproduction_loss(
    prediction_type: str = "velocity",
    flow_type: str = "rectified", 
    loss_weight: float = 1.0,
    use_adaptive_noise_scaling: bool = True,
    debug_mode: bool = False,
    **kwargs
) -> BLIP3oCLIPFlowMatchingLoss:
    """Factory function for CLIP reproduction loss with noise scaling"""
    return BLIP3oCLIPFlowMatchingLoss(
        prediction_type=prediction_type,
        flow_type=flow_type,
        loss_weight=loss_weight,
        use_adaptive_noise_scaling=use_adaptive_noise_scaling,
        debug_mode=debug_mode,
        **kwargs
    )