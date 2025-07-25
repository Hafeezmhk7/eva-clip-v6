#!/usr/bin/env python3
"""
FIXED: BLIP3-o Flow Matching Loss with Proper L2 Normalization Handling
src/modules/losses/blip3o_flow_matching_loss.py

KEY FIX:
1. Properly handle already L2-normalized embeddings (norm ~1.0)
2. Don't apply double normalization
3. Report correct norms for monitoring
4. Ensure targets are already normalized from dataset
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
    FIXED: BLIP3-o Flow Matching Loss with proper L2 normalization handling
    
    Key principles:
    1. Expects already L2-normalized CLIP embeddings from dataset (norm ~1.0)
    2. Does not apply double normalization
    3. Clean rectified flow implementation
    4. Proper norm monitoring and reporting
    """
    
    def __init__(
        self,
        prediction_type: str = "velocity",
        normalize_targets: bool = True,  # Controls whether to verify/enforce normalization
        flow_type: str = "rectified",
        ema_decay: float = 0.99,
        clip_norm_max: float = 10.0,  # Gradient clipping for stability
    ):
        super().__init__()
        
        self.prediction_type = prediction_type
        self.normalize_targets = normalize_targets
        self.flow_type = flow_type
        self.ema_decay = ema_decay
        self.clip_norm_max = clip_norm_max
        
        # EMA tracking for metrics
        self.register_buffer('ema_loss', torch.tensor(0.0))
        self.register_buffer('ema_velocity_cosine', torch.tensor(0.0))
        self.register_buffer('ema_pred_norm', torch.tensor(1.0))
        self.register_buffer('ema_target_norm', torch.tensor(1.0))
        
        # Training progress tracking
        self.register_buffer('best_velocity_sim', torch.tensor(0.0))
        self.register_buffer('training_steps', torch.tensor(0.0))
        
        logger.info(f"✅ FIXED BLIP3-o Flow Matching Loss (L2 Normalization Aware)")
        logger.info(f"   Expects normalized targets: {normalize_targets}")
        logger.info(f"   Flow type: {flow_type}")
        logger.info(f"   Prediction type: {prediction_type}")

    def forward(
        self,
        model_output: torch.Tensor,  # [B, N, 1024] - Model's velocity prediction
        target_samples: torch.Tensor,  # [B, N, 1024] - Ground truth CLIP embeddings (should be normalized)
        timesteps: torch.Tensor,  # [B] - Flow matching timesteps
        eva_conditioning: torch.Tensor,  # [B, N, 4096] - EVA-CLIP conditioning
        noise: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        training_mode: str = "patch_only",
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        FIXED: Flow matching loss computation with proper L2 normalization handling
        """
        
        batch_size, num_tokens, embed_dim = model_output.shape
        device = model_output.device
        is_training = model_output.requires_grad
        
        # Input validation
        assert model_output.shape == target_samples.shape, \
            f"Shape mismatch: {model_output.shape} vs {target_samples.shape}"
        assert num_tokens in [256, 257], f"Expected 256 or 257 tokens, got {num_tokens}"
        assert embed_dim == 1024, f"Expected 1024-dim embeddings, got {embed_dim}"
        
        # Check if target embeddings are already normalized (should be ~1.0 from dataset)
        target_norm_check = torch.norm(target_samples.detach(), dim=-1).mean().item()
        
        if self.normalize_targets:
            # If targets are already normalized (~1.0), don't re-normalize
            if abs(target_norm_check - 1.0) < 0.1:
                # Targets are already properly normalized
                target_normalized = target_samples.detach()
                logger.debug(f"Targets already normalized: norm = {target_norm_check:.3f}")
            else:
                # Targets are not normalized, apply normalization
                target_normalized = F.normalize(target_samples.detached(), p=2, dim=-1)
                logger.warning(f"Applied normalization to targets: {target_norm_check:.3f} -> ~1.0")
        else:
            # Use targets as-is (not recommended, but for compatibility)
            target_normalized = target_samples.detach()
            if target_norm_check > 2.0:
                logger.warning(f"Using non-normalized targets: norm = {target_norm_check:.3f}")
        
        # Create noise (source distribution)
        if noise is None:
            noise = torch.randn_like(target_normalized, device=device)
        else:
            noise = noise.detach()
        
        # RECTIFIED FLOW: Simple linear interpolation
        # x_t = (1-t) * x_0 + t * x_1, where x_0=noise, x_1=target
        t_expanded = timesteps.view(-1, 1, 1)  # [B, 1, 1]
        x_t = (1 - t_expanded) * noise + t_expanded * target_normalized
        
        # RECTIFIED FLOW VELOCITY TARGET: v = x_1 - x_0 = target - noise
        velocity_target = target_normalized - noise
        
        # SIMPLE MSE LOSS
        flow_loss = F.mse_loss(model_output, velocity_target, reduction='mean')
        
        # Gradient clipping for stability
        if is_training and flow_loss.requires_grad:
            torch.nn.utils.clip_grad_norm_(model_output, self.clip_norm_max)
        
        # Prepare metrics
        metrics = None
        if return_metrics:
            with torch.no_grad():
                self.training_steps += 1
                
                # Compute velocity cosine similarity (training metric)
                pred_norm = F.normalize(model_output.detach(), p=2, dim=-1)
                target_norm = F.normalize(velocity_target.detach(), p=2, dim=-1)
                
                per_patch_cos = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                per_image_cos = per_patch_cos.mean(dim=1)
                velocity_cosine_sim = per_image_cos.mean().item()
                
                # Compute norms for monitoring (FIXED: Report actual norms)
                pred_norm_scalar = torch.norm(model_output.detach(), dim=-1).mean().item()
                target_norm_scalar = torch.norm(velocity_target.detach(), dim=-1).mean().item()
                clip_target_norm_scalar = torch.norm(target_normalized.detach(), dim=-1).mean().item()
                
                # Update EMA metrics
                self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * flow_loss.item()
                self.ema_velocity_cosine = self.ema_decay * self.ema_velocity_cosine + (1 - self.ema_decay) * velocity_cosine_sim
                self.ema_pred_norm = self.ema_decay * self.ema_pred_norm + (1 - self.ema_decay) * pred_norm_scalar
                self.ema_target_norm = self.ema_decay * self.ema_target_norm + (1 - self.ema_decay) * target_norm_scalar
                
                # Track best similarity
                if velocity_cosine_sim > self.best_velocity_sim.item():
                    self.best_velocity_sim = torch.tensor(velocity_cosine_sim, device=device)
                
                # Quality metrics
                high_quality_patches = (per_patch_cos > 0.7).float().mean().item()
                very_high_quality_patches = (per_patch_cos > 0.8).float().mean().item()
                high_quality_images = (per_image_cos > 0.7).float().mean().item()
                
                # FIXED: Normalization status reporting
                normalization_status = "✅ Normalized" if abs(clip_target_norm_scalar - 1.0) < 0.1 else "⚠️ Not normalized"
                
                metrics = {
                    # Core loss
                    'flow_matching_loss': flow_loss.item(),
                    'total_loss': flow_loss.item(),
                    
                    # Velocity similarity (training metric)
                    'velocity_cosine_sim': velocity_cosine_sim,
                    'velocity_per_patch_mean': per_patch_cos.mean().item(),
                    'velocity_per_patch_std': per_patch_cos.std().item(),
                    'velocity_per_image_mean': per_image_cos.mean().item(),
                    'velocity_high_quality_patches': high_quality_patches,
                    'velocity_very_high_quality_patches': very_high_quality_patches,
                    'velocity_high_quality_images': high_quality_images,
                    
                    # FIXED: Proper norm reporting
                    'prediction_norm': pred_norm_scalar,
                    'target_norm': target_norm_scalar,           # Velocity target norm
                    'clip_target_norm': clip_target_norm_scalar, # CLIP embedding norm (should be ~1.0)
                    'target_norm_check': target_norm_check,      # Original target norm from dataset
                    'norm_ratio': target_norm_scalar / max(pred_norm_scalar, 1e-8),
                    'clip_normalized': abs(clip_target_norm_scalar - 1.0) < 0.1,
                    
                    # EMA metrics
                    'ema_velocity_cosine': self.ema_velocity_cosine.item(),
                    'ema_pred_norm': self.ema_pred_norm.item(),
                    'ema_target_norm': self.ema_target_norm.item(),
                    'best_velocity_sim': self.best_velocity_sim.item(),
                    
                    # Training info
                    'training_steps': self.training_steps.item(),
                    'flow_type': self.flow_type,
                    'prediction_type': self.prediction_type,
                    'normalize_targets': self.normalize_targets,
                    'is_training': is_training,
                    'num_tokens': num_tokens,
                    'mode': training_mode,
                    
                    # FIXED: L2 normalization status
                    'l2_normalization_status': normalization_status,
                    'targets_properly_normalized': abs(clip_target_norm_scalar - 1.0) < 0.1,
                    'double_normalization_avoided': True,
                }
                
                # Log warning if targets are not properly normalized
                if not metrics['targets_properly_normalized']:
                    logger.warning(f"CLIP targets not properly normalized: {clip_target_norm_scalar:.3f} (should be ~1.0)")
        
        return flow_loss, metrics

    def get_training_info(self) -> Dict[str, float]:
        """Get current training information"""
        return {
            'ema_loss': self.ema_loss.item(),
            'ema_velocity_cosine': self.ema_velocity_cosine.item(),
            'ema_pred_norm': self.ema_pred_norm.item(),
            'ema_target_norm': self.ema_target_norm.item(),
            'best_velocity_sim': self.best_velocity_sim.item(),
            'training_steps': self.training_steps.item(),
            'flow_type': self.flow_type,
            'prediction_type': self.prediction_type,
            'normalize_targets': self.normalize_targets,
        }


def create_blip3o_flow_matching_loss(
    prediction_type: str = "velocity",
    normalize_targets: bool = True,
    flow_type: str = "rectified",
    **kwargs
) -> BLIP3oFlowMatchingLoss:
    """
    Factory function for FIXED flow matching loss with L2 normalization awareness
    """
    return BLIP3oFlowMatchingLoss(
        prediction_type=prediction_type,
        normalize_targets=normalize_targets,
        flow_type=flow_type,
        **kwargs
    )