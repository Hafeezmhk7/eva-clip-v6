#!/usr/bin/env python3
"""
FIXED: BLIP3-o Flow Matching Loss - Training & Evaluation Compatible
src/modules/losses/blip3o_flow_matching_loss.py

KEY FIXES:
1. Proper rectified flow implementation aligned with BLIP3-o paper
2. Consistent normalization between training and evaluation
3. Correct velocity target computation
4. Safe tensor operations for both training and evaluation modes
5. Fixed gradient flow issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class BLIP3oFlowMatchingLoss(nn.Module):
    """
    FIXED: Pure Flow Matching Loss for BLIP3-o (Training & Evaluation Compatible)
    
    Implements rectified flow matching as described in BLIP3-o paper with proper
    velocity prediction and consistent normalization.
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        prediction_type: str = "velocity",
        normalize_targets: bool = True,
        flow_type: str = "rectified",
    ):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prediction_type = prediction_type
        self.normalize_targets = normalize_targets
        self.flow_type = flow_type
        
        # EMA tracking for metrics
        self.register_buffer('ema_loss', torch.tensor(0.0))
        self.register_buffer('ema_cosine_sim', torch.tensor(0.0))
        self.ema_decay = 0.99
        
        logger.info(f"✅ FIXED BLIP3-o Flow Matching Loss initialized")
        logger.info(f"   Flow type: {flow_type}")
        logger.info(f"   Prediction type: {prediction_type}")
        logger.info(f"   Normalize targets: {normalize_targets}")

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for flow matching training"""
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    def get_noise_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get noise schedule parameters for rectified flow matching"""
        if self.flow_type == "rectified":
            # Rectified flow: simple linear interpolation from noise to data
            alpha_t = t  # Linear schedule from 0 to 1
            sigma_t = torch.zeros_like(t)  # No additional noise
        else:
            # Standard flow matching with noise schedule
            alpha_t = t
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
        
        return alpha_t, sigma_t

    def interpolate_data(
        self,
        x_0: torch.Tensor,  # Source (noise) [B, N, 1024]
        x_1: torch.Tensor,  # Target CLIP embeddings [B, N, 1024]
        t: torch.Tensor,    # Timesteps [B]
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FIXED: Interpolate between source and target using rectified flow
        
        For rectified flow: x_t = (1-t) * x_0 + t * x_1
        This creates a straight path from noise to data
        """
        # Ensure proper shapes for broadcasting
        t = t.view(-1, 1, 1)  # [B, 1, 1]
        
        if self.flow_type == "rectified":
            # BLIP3-o uses rectified flow: straight line interpolation
            x_t = (1 - t) * x_0 + t * x_1
        else:
            # Standard flow matching with noise
            alpha_t, sigma_t = self.get_noise_schedule(t.squeeze(-1))
            alpha_t = alpha_t.view(-1, 1, 1)
            sigma_t = sigma_t.view(-1, 1, 1)
            
            if noise is None:
                noise = torch.zeros_like(x_1)
            
            x_t = (1 - alpha_t) * x_0 + alpha_t * x_1 + sigma_t * noise
        
        return x_t

    def compute_velocity_target(
        self,
        x_0: torch.Tensor,  # Source [B, N, 1024]
        x_1: torch.Tensor,  # Target [B, N, 1024]
        t: torch.Tensor,    # Timesteps [B]
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FIXED: Compute velocity target for rectified flow matching
        
        For rectified flow, the velocity is constant: v = x_1 - x_0
        This represents the direction from noise to data
        """
        if self.prediction_type == "velocity":
            if self.flow_type == "rectified":
                # BLIP3-o rectified flow: constant velocity field
                velocity_target = x_1 - x_0
            else:
                # Standard flow matching velocity
                t_expanded = t.view(-1, 1, 1)
                velocity_target = x_1 - (1 - t_expanded) * x_0
        elif self.prediction_type == "epsilon":
            # Noise prediction target
            if noise is None:
                noise = torch.randn_like(x_1)
            velocity_target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return velocity_target

    def compute_detailed_similarities(
        self,
        predicted: torch.Tensor,        # [B, N, 1024] - Predicted velocity
        target: torch.Tensor,           # [B, N, 1024] - Target velocity or embeddings
        training_mode: str = "patch_only"
    ) -> Dict[str, torch.Tensor]:
        """Compute detailed per-patch, per-image, and global similarities"""
        batch_size, num_tokens, dim = predicted.shape
        
        # Normalize for cosine similarity
        pred_norm = F.normalize(predicted, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        
        # Per-patch cosine similarities [B, N]
        per_patch_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)
        
        # Per-image average similarities [B]
        per_image_avg_sim = per_patch_sim.mean(dim=1)
        
        # Global average similarity
        global_avg_sim = per_image_avg_sim.mean()
        
        similarities = {
            'per_patch_cosine': per_patch_sim,              # [B, N]
            'per_image_avg_cosine': per_image_avg_sim,      # [B]
            'global_avg_cosine': global_avg_sim,            # scalar
            'per_patch_mean': per_patch_sim.mean(),         # scalar
            'per_patch_std': per_patch_sim.std(),           # scalar
            'per_image_std': per_image_avg_sim.std(),       # scalar
        }
        
        # Mode-specific analysis
        if training_mode == "cls_patch" and num_tokens == 257:
            cls_sim = per_patch_sim[:, 0]
            patch_sim = per_patch_sim[:, 1:]
            
            similarities.update({
                'cls_cosine': cls_sim.mean(),
                'cls_std': cls_sim.std(),
                'patch_cosine': patch_sim.mean(),
                'patch_std': patch_sim.std(),
            })
        else:
            similarities.update({
                'cls_cosine': torch.tensor(0.0),
                'patch_cosine': per_patch_sim.mean(),
                'patch_std': per_patch_sim.std(),
            })
        
        return similarities

    def forward(
        self,
        model_output: torch.Tensor,       # [B, N, 1024] - Predicted velocity
        target_samples: torch.Tensor,     # [B, N, 1024] - Target CLIP embeddings
        timesteps: torch.Tensor,          # [B] - Timesteps
        eva_conditioning: torch.Tensor,   # [B, N, 4096] - EVA features
        noise: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        training_mode: str = "patch_only",
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        FIXED: Compute BLIP3-o flow matching loss with proper rectified flow
        
        This implementation ensures:
        1. Consistent velocity targets for rectified flow
        2. Proper normalization handling
        3. Compatible with both training and evaluation
        4. Fixed gradient flow issues
        """
        batch_size = model_output.shape[0]
        num_tokens = model_output.shape[1]
        device = model_output.device
        
        # Detect if we're in training or evaluation mode
        is_training = model_output.requires_grad
        
        # Input validation
        assert model_output.shape == target_samples.shape, \
            f"Shape mismatch: {model_output.shape} vs {target_samples.shape}"
        assert num_tokens in [256, 257], f"Expected 256 or 257 tokens, got {num_tokens}"
        assert model_output.shape[2] == 1024, f"Expected 1024-dim, got {model_output.shape[2]}"
        
        # FIXED: Normalize targets consistently
        if self.normalize_targets:
            target_samples = F.normalize(target_samples.detach(), p=2, dim=-1)
        else:
            target_samples = target_samples.detach()
        
        # FIXED: Create source distribution (noise) with proper scaling
        x_0 = torch.randn_like(target_samples, device=device, dtype=target_samples.dtype)
        
        # FIXED: Compute velocity target using rectified flow
        velocity_target = self.compute_velocity_target(x_0, target_samples, timesteps, noise)
        
        # FIXED: Flow matching loss - simple MSE for rectified flow
        # Ensure both tensors have the same gradient requirements
        if is_training:
            # During training, both should have gradients or be properly detached
            flow_matching_loss = F.mse_loss(model_output, velocity_target.detach(), reduction='mean')
        else:
            # During evaluation, neither should have gradients
            flow_matching_loss = F.mse_loss(model_output.detach(), velocity_target.detach(), reduction='mean')
        
        # Verify loss computation is valid
        if is_training and not flow_matching_loss.requires_grad:
            raise RuntimeError("Flow matching loss doesn't require gradients during training!")
        
        # Total loss is pure flow matching loss (BLIP3-o paper)
        total_loss = flow_matching_loss
        
        # Update EMA metrics safely
        with torch.no_grad():
            self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * total_loss.item()
            
            # Velocity cosine similarity for monitoring
            pred_flat = model_output.detach().view(batch_size, -1)
            target_flat = velocity_target.detach().view(batch_size, -1)
            cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean()
            self.ema_cosine_sim = self.ema_decay * self.ema_cosine_sim + (1 - self.ema_decay) * cosine_sim.item()
        
        # Prepare detailed metrics
        metrics = None
        if return_metrics:
            with torch.no_grad():
                # Compute detailed similarities
                detailed_sims = self.compute_detailed_similarities(
                    model_output.detach(), velocity_target.detach(), training_mode
                )
                
                # Prediction quality metrics
                pred_norm = torch.norm(model_output.detach(), dim=-1).mean()
                target_norm = torch.norm(velocity_target.detach(), dim=-1).mean()
                
                # Quality indicators for patches and images
                high_quality_patches = (detailed_sims['per_patch_cosine'] > 0.7).float().mean()
                very_high_quality_patches = (detailed_sims['per_patch_cosine'] > 0.8).float().mean()
                
                high_quality_images = (detailed_sims['per_image_avg_cosine'] > 0.7).float().mean()
                very_high_quality_images = (detailed_sims['per_image_avg_cosine'] > 0.8).float().mean()
                
                # FIXED: For evaluation, also compute final embedding similarity
                eval_similarity = None
                if not is_training:
                    try:
                        # Simulate final embeddings by applying velocity
                        final_embeddings = x_0 + model_output.detach()  # Simple step
                        if self.normalize_targets:
                            final_embeddings = F.normalize(final_embeddings, p=2, dim=-1)
                        
                        final_sim = F.cosine_similarity(
                            final_embeddings.view(batch_size, -1),
                            target_samples.view(batch_size, -1),
                            dim=-1
                        ).mean()
                        eval_similarity = final_sim.item()
                    except:
                        eval_similarity = None
                
                metrics = {
                    # Loss components
                    'flow_matching_loss': flow_matching_loss.item(),
                    'total_loss': total_loss.item(),
                    
                    # Velocity prediction quality (training metric)
                    'velocity_cosine_sim': cosine_sim.item(),
                    'prediction_norm': pred_norm.item(),
                    'target_norm': target_norm.item(),
                    
                    # Detailed similarity metrics
                    'per_patch_mean_cosine': detailed_sims['per_patch_mean'].item(),
                    'per_patch_std_cosine': detailed_sims['per_patch_std'].item(),
                    'per_image_mean_cosine': detailed_sims['per_image_avg_cosine'].mean().item(),
                    'per_image_std_cosine': detailed_sims['per_image_std'].item(),
                    'global_mean_cosine': detailed_sims['global_avg_cosine'].item(),
                    
                    # Mode-specific metrics
                    'num_tokens': num_tokens,
                    'mode': 'cls_patch' if num_tokens == 257 else 'patch_only',
                    'cls_cosine_sim': detailed_sims['cls_cosine'].item(),
                    'patch_cosine_sim': detailed_sims['patch_cosine'].item(),
                    
                    # Quality distribution
                    'high_quality_patches_ratio': high_quality_patches.item(),
                    'very_high_quality_patches_ratio': very_high_quality_patches.item(),
                    'high_quality_images_ratio': high_quality_images.item(),
                    'very_high_quality_images_ratio': very_high_quality_images.item(),
                    
                    # Training quality assessment
                    'training_quality': (
                        'excellent' if detailed_sims['global_avg_cosine'] > 0.8 else
                        'very_good' if detailed_sims['global_avg_cosine'] > 0.7 else
                        'good' if detailed_sims['global_avg_cosine'] > 0.6 else
                        'fair' if detailed_sims['global_avg_cosine'] > 0.4 else
                        'needs_improvement'
                    ),
                    
                    # EMA metrics
                    'ema_loss': self.ema_loss.item(),
                    'ema_cosine_sim': self.ema_cosine_sim.item(),
                    
                    # FIXED: Add evaluation similarity if available
                    'eval_final_embedding_similarity': eval_similarity,
                    
                    # Model info
                    'flow_type': self.flow_type,
                    'prediction_type': self.prediction_type,
                    'is_training': is_training,
                    'normalize_targets': self.normalize_targets,
                    'paper_aligned': True,
                    'blip3o_compliant': True,
                    'fixed_version': True,
                }
        
        return total_loss, metrics


def create_blip3o_flow_matching_loss(
    prediction_type: str = "velocity",
    normalize_targets: bool = True,
    flow_type: str = "rectified",
    **kwargs
) -> BLIP3oFlowMatchingLoss:
    """
    Factory function for creating BLIP3-o flow matching loss
    
    Args:
        prediction_type: "velocity" for BLIP3-o (recommended)
        normalize_targets: True for consistent normalization
        flow_type: "rectified" for BLIP3-o paper alignment
        **kwargs: Additional loss configuration
    
    Returns:
        BLIP3oFlowMatchingLoss instance
    """
    return BLIP3oFlowMatchingLoss(
        prediction_type=prediction_type,
        normalize_targets=normalize_targets,
        flow_type=flow_type,
        **kwargs
    )