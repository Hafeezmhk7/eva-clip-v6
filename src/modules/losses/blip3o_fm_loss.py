#!/usr/bin/env python3
"""
FIXED: Flow Matching Loss for CLIP Reproduction with NO Noise Scaling
Key fixes:
1. NO noise scaling anywhere - always use standard Gaussian noise
2. Consistent noise distribution for training/inference consistency
3. Enhanced debugging for norm analysis
4. Clear separation between raw embeddings and normalized embeddings
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
    FIXED: Flow Matching Loss for CLIP reproduction with NO noise scaling
    
    Key improvements:
    - NO noise scaling anywhere (always standard Gaussian)
    - Normalization ONLY for cosine similarity computation
    - Enhanced debugging and monitoring
    - Consistent noise distribution between training and inference
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
        # FIXED: Completely disable noise scaling
        use_adaptive_noise_scaling: bool = False,  # Always False
        fixed_noise_scale: float = 1.0,           # Always 1.0 (no scaling)
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
        
        # FIXED: Force no noise scaling
        self.use_adaptive_noise_scaling = False  # Always disabled
        self.fixed_noise_scale = 1.0             # Always 1.0 (no scaling)
        
        # Validate inputs
        assert prediction_type in ["velocity", "noise", "sample"]
        assert flow_type in ["rectified", "reflow"]
        
        # Running statistics for monitoring only
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('loss_ema', torch.tensor(0.0))
        self.register_buffer('similarity_ema', torch.tensor(0.0))
        
        # FIXED: Always use standard noise (no scaling)
        self.register_buffer('noise_scale', torch.tensor(1.0))
        
        logger.info(f"FIXED CLIP Flow Matching Loss initialized (NO NOISE SCALING):")
        logger.info(f"  Prediction type: {prediction_type}")
        logger.info(f"  Flow type: {flow_type}")
        logger.info(f"  Noise scaling: DISABLED (always standard Gaussian)")
        logger.info(f"  Debug mode: {debug_mode}")
        logger.info(f"  🎲 Noise: Standard Gaussian (mean=0, std=1) everywhere")

    def _create_standard_gaussian_noise(self, target_samples: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Create standard Gaussian noise WITHOUT any scaling
        """
        device = target_samples.device
        dtype = target_samples.dtype
        
        # Create standard Gaussian noise (mean=0, std=1)
        noise = torch.randn_like(target_samples, device=device, dtype=dtype)
        
        if self.debug_mode:
            noise_std = noise.std().item()
            noise_mean = noise.mean().item()
            logger.debug(f"Created standard Gaussian noise: mean={noise_mean:.6f}, std={noise_std:.6f}")
        
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
        FIXED: Compute flow matching loss with standard Gaussian noise (NO SCALING)
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        dtype = model_output.dtype
        
        # Update step count
        self.step_count += 1
        
        # Clamp timesteps
        timesteps = self._clamp_timesteps(timesteps)
        
        # Keep targets as-is (NO normalization during training)
        target_clean = target_samples.detach()
        
        # FIXED: Use standard Gaussian noise (NO SCALING)
        if noise is None:
            noise = self._create_standard_gaussian_noise(target_clean)
        else:
            # If noise provided, verify it's approximately standard Gaussian
            if self.debug_mode:
                provided_mean = noise.mean().item()
                provided_std = noise.std().item()
                logger.debug(f"Using provided noise: mean={provided_mean:.6f}, std={provided_std:.6f}")
        
        # Expand timesteps for broadcasting [B, 1, 1]
        t = timesteps.view(batch_size, 1, 1).to(dtype)
        
        # RECTIFIED FLOW COMPUTATION with standard noise
        if self.flow_type == "rectified":
            # Linear interpolation: x_t = (1-t) * noise + t * target (standard noise)
            true_velocity = target_clean - noise
            target_for_loss = true_velocity
        else:  # reflow
            raise NotImplementedError("Reflow not implemented yet")
        
        # LOSS COMPUTATION in raw space (NO normalization)
        if self.prediction_type == "velocity":
            # Direct velocity prediction loss in raw embedding space
            prediction_loss = F.mse_loss(model_output, target_for_loss, reduction='none')
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented")
        
        # Reduce loss: mean over tokens and embedding dimensions, then over batch
        prediction_loss = prediction_loss.mean(dim=(1, 2))  # [B]
        
        # Optional: timestep weighting
        time_weights = 1.0  # Simplified: no timestep weighting for now
        weighted_loss = prediction_loss * time_weights
        
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
                eva_norm = torch.norm(eva_conditioning, dim=-1).mean().item()
                
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
                
                # Noise statistics (should be close to standard Gaussian)
                noise_stats = {
                    'mean': noise.mean().item(),
                    'std': noise.std().item(),
                    'norm': noise_norm,
                }
                
                # Detailed embedding statistics
                clip_stats = {
                    'mean': target_clean.mean().item(),
                    'std': target_clean.std().item(),
                    'min': target_clean.min().item(),
                    'max': target_clean.max().item(),
                    'norm': clip_norm,
                }
                
                pred_stats = {
                    'mean': model_output.mean().item(),
                    'std': model_output.std().item(),
                    'min': model_output.min().item(),
                    'max': model_output.max().item(),
                    'norm': pred_norm,
                }
                
                metrics = {
                    # Core metrics
                    'loss': main_loss.item(),
                    'total_loss': total_loss.item(),
                    'reg_loss': reg_loss if isinstance(reg_loss, float) else reg_loss.item(),
                    'velocity_similarity': mean_similarity,
                    'velocity_similarity_std': per_image_sim.std().item(),
                    
                    # Raw norm tracking (NO normalization applied)
                    'pred_norm': pred_norm,
                    'target_norm': target_norm_val,
                    'clip_norm': clip_norm,
                    'noise_norm': noise_norm,
                    'eva_norm': eva_norm,
                    
                    # Detailed embedding statistics
                    'clip_mean': clip_stats['mean'],
                    'clip_std': clip_stats['std'],
                    'clip_min': clip_stats['min'],
                    'clip_max': clip_stats['max'],
                    'pred_mean': pred_stats['mean'],
                    'pred_std': pred_stats['std'],
                    'pred_min': pred_stats['min'],
                    'pred_max': pred_stats['max'],
                    
                    # FIXED: Standard Gaussian noise statistics
                    'noise_mean': noise_stats['mean'],  # Should be ~0
                    'noise_std': noise_stats['std'],   # Should be ~1
                    'noise_is_standard_gaussian': abs(noise_stats['std'] - 1.0) < 0.2 and abs(noise_stats['mean']) < 0.2,
                    
                    # NO noise scaling metrics
                    'noise_scaling_disabled': True,
                    'standard_gaussian_noise': True,
                    'adaptive_scaling': False,
                    'fixed_noise_scale': 1.0,
                    'noise_scale_used': 1.0,  # Always 1.0
                    
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
                    
                    # Normalization tracking
                    'normalization_applied_during_training': False,  # Always False
                    'normalization_for_cosine_similarity_only': True,  # Always True
                    'raw_embedding_space_training': True,  # Always True
                    
                    # Scale consistency checks
                    'clip_eva_norm_ratio': clip_norm / (eva_norm + self.eps),
                    'pred_clip_norm_ratio': pred_norm / (clip_norm + self.eps),
                    'noise_is_unit_variance': abs(noise_stats['std'] - 1.0) < 0.1,
                    'noise_is_zero_mean': abs(noise_stats['mean']) < 0.1,
                }
                
                # Check for numerical issues
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    metrics['numerical_issue'] = True
                    logger.error(f"[Step {self.step_count}] Numerical issue detected!")
                    logger.error(f"  Loss: {total_loss.item()}")
                    logger.error(f"  Pred norm: {pred_norm}")
                    logger.error(f"  Target norm: {target_norm_val}")
                
                # Enhanced debug logging
                if self.debug_mode and self.step_count % 50 == 0:
                    logger.info(f"[Step {self.step_count}] FIXED CLIP Flow Matching Debug (NO NOISE SCALING):")
                    logger.info(f"  Loss: {main_loss.item():.6f} (quality: {quality})")
                    logger.info(f"  Velocity Sim: {mean_similarity:.4f}")
                    logger.info(f"  🎲 Standard Gaussian noise: mean={noise_stats['mean']:.6f}, std={noise_stats['std']:.6f}")
                    logger.info(f"  Noise is standard: {'✅' if metrics['noise_is_standard_gaussian'] else '❌'}")
                    logger.info(f"  Raw Norms - Pred: {pred_norm:.3f}, Target: {target_norm_val:.3f}, CLIP: {clip_norm:.3f}")
                    logger.info(f"  Error: {error_norm:.3f} (relative: {relative_error:.3f})")
                    logger.info(f"  CLIP stats: mean={clip_stats['mean']:.3f}, std={clip_stats['std']:.3f}")
                    logger.info(f"  Pred stats: mean={pred_stats['mean']:.3f}, std={pred_stats['std']:.3f}")
        
        return total_loss, metrics

    def compute_eval_loss(
        self,
        generated: torch.Tensor,  # Generated CLIP embeddings
        target: torch.Tensor,     # Target CLIP embeddings
    ) -> Dict[str, float]:
        """FIXED: Compute evaluation metrics with NO unwanted normalization"""
        with torch.no_grad():
            # FIXED: Normalize ONLY for cosine similarity computation
            generated_norm = F.normalize(generated, p=2, dim=-1)
            target_norm = F.normalize(target, p=2, dim=-1)
            
            # Cosine similarity (requires normalization)
            cosine_sim = F.cosine_similarity(generated_norm, target_norm, dim=-1)
            per_image_sim = cosine_sim.mean(dim=1)
            
            # FIXED: MSE in raw space (NO normalization)
            mse_loss = F.mse_loss(generated, target)
            
            # Quality metrics (based on cosine similarity)
            high_quality = (per_image_sim > 0.7).float().mean().item()
            very_high_quality = (per_image_sim > 0.8).float().mean().item()
            excellent_quality = (per_image_sim > 0.9).float().mean().item()
            
            # FIXED: Scale analysis (raw space, NO normalization)
            generated_norm_val = torch.norm(generated, dim=-1).mean().item()
            target_norm_val = torch.norm(target, dim=-1).mean().item()
            
            # Detailed embedding statistics
            generated_stats = {
                'mean': generated.mean().item(),
                'std': generated.std().item(),
                'min': generated.min().item(),
                'max': generated.max().item(),
            }
            
            target_stats = {
                'mean': target.mean().item(),
                'std': target.std().item(),
                'min': target.min().item(),
                'max': target.max().item(),
            }
            
            return {
                'eval_clip_similarity': per_image_sim.mean().item(),
                'eval_mse_loss': mse_loss.item(),
                'eval_high_quality_ratio': high_quality,
                'eval_very_high_quality_ratio': very_high_quality,
                'eval_excellent_quality_ratio': excellent_quality,
                'eval_similarity_std': per_image_sim.std().item(),
                
                # Raw embedding norms (NO normalization)
                'eval_generated_norm': generated_norm_val,
                'eval_target_norm': target_norm_val,
                'eval_norm_ratio': generated_norm_val / (target_norm_val + 1e-8),
                'eval_norm_consistency': 1.0 - abs(1.0 - generated_norm_val / (target_norm_val + 1e-8)),
                
                # Detailed embedding statistics
                'eval_generated_mean': generated_stats['mean'],
                'eval_generated_std': generated_stats['std'],
                'eval_generated_min': generated_stats['min'],
                'eval_generated_max': generated_stats['max'],
                'eval_target_mean': target_stats['mean'],
                'eval_target_std': target_stats['std'],
                'eval_target_min': target_stats['min'],
                'eval_target_max': target_stats['max'],
                
                # Normalization tracking
                'eval_normalization_applied': False,  # NO normalization except for similarity
                'eval_raw_embedding_space': True,    # Always in raw space
                'eval_noise_scaling_disabled': True,  # NO noise scaling anywhere
            }

    def get_noise_scale(self) -> float:
        """Get current noise scale (always 1.0 - no scaling)"""
        return 1.0
    
    def set_noise_scale(self, scale: float):
        """Ignore noise scale setting - always use 1.0"""
        if scale != 1.0:
            logger.warning(f"Attempted to set noise scale to {scale}, but noise scaling is disabled. Using 1.0.")
        self.noise_scale.data = torch.tensor(1.0)


def create_clip_reproduction_loss(
    prediction_type: str = "velocity",
    flow_type: str = "rectified", 
    loss_weight: float = 1.0,
    use_adaptive_noise_scaling: bool = False,  # FIXED: Always False
    fixed_noise_scale: float = 1.0,           # FIXED: Always 1.0
    debug_mode: bool = False,
    **kwargs
) -> BLIP3oCLIPFlowMatchingLoss:
    """FIXED: Factory function for CLIP reproduction loss with NO noise scaling"""
    
    # Force no noise scaling regardless of arguments
    if use_adaptive_noise_scaling:
        logger.warning("🚫 Adaptive noise scaling was requested but DISABLED - using standard Gaussian noise")
    if fixed_noise_scale != 1.0:
        logger.warning(f"🚫 Fixed noise scale {fixed_noise_scale} was requested but DISABLED - using 1.0 (no scaling)")
    
    return BLIP3oCLIPFlowMatchingLoss(
        prediction_type=prediction_type,
        flow_type=flow_type,
        loss_weight=loss_weight,
        use_adaptive_noise_scaling=False,  # Always False
        fixed_noise_scale=1.0,             # Always 1.0
        debug_mode=debug_mode,
        **kwargs
    )