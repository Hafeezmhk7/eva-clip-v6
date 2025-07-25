#!/usr/bin/env python3
"""
COMPLETE BLIP3-o Flow Matching Loss - FIXED VERSION
src/modules/losses/blip3o_flow_matching_loss.py

Fixed buffer assignment issues for torch tensors
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
    COMPLETE FIXED: BLIP3-o Flow Matching Loss with proper tensor buffer handling
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        prediction_type: str = "velocity",
        normalize_targets: bool = True,
        flow_type: str = "rectified",
        velocity_scale: float = 0.1,
        target_norm_scale: float = 1.0,
        adaptive_scaling: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prediction_type = prediction_type
        self.normalize_targets = normalize_targets
        self.flow_type = flow_type
        self.velocity_scale = velocity_scale
        self.target_norm_scale = target_norm_scale
        self.adaptive_scaling = adaptive_scaling
        self.ema_decay = ema_decay
        
        # EMA tracking for metrics and adaptive scaling
        self.register_buffer('ema_loss', torch.tensor(0.0))
        self.register_buffer('ema_cosine_sim', torch.tensor(0.0))
        self.register_buffer('ema_target_norm', torch.tensor(1.0))
        self.register_buffer('ema_pred_norm', torch.tensor(1.0))
        
        # Adaptive scaling buffers
        self.register_buffer('adaptive_scale', torch.tensor(1.0))
        self.register_buffer('scale_update_count', torch.tensor(0.0))
        
        # Training progress tracking
        self.register_buffer('best_cosine_sim', torch.tensor(0.0))
        self.register_buffer('steps_since_improvement', torch.tensor(0.0))
        
        logger.info(f"✅ FIXED BLIP3-o Flow Matching Loss initialized")
        logger.info(f"   Velocity scale: {velocity_scale}")
        logger.info(f"   Adaptive scaling: {adaptive_scaling}")

    def update_adaptive_scaling(self, pred_norm: float, target_norm: float, current_cosine: float):
        """
        FIXED: Update adaptive scaling factor with proper scalar handling
        """
        if not self.adaptive_scaling:
            return
            
        with torch.no_grad():
            device = self.adaptive_scale.device
            
            # Ensure inputs are scalar floats
            if torch.is_tensor(pred_norm):
                pred_norm = pred_norm.item()
            if torch.is_tensor(target_norm):
                target_norm = target_norm.item()
            if torch.is_tensor(current_cosine):
                current_cosine = current_cosine.item()
            
            # Convert inputs to tensors on correct device
            pred_norm_tensor = torch.tensor(pred_norm, device=device, dtype=torch.float32)
            target_norm_tensor = torch.tensor(target_norm, device=device, dtype=torch.float32)
            current_cosine_tensor = torch.tensor(current_cosine, device=device, dtype=torch.float32)
            
            # Compute ratio of target to prediction norms
            if pred_norm > 1e-8:
                norm_ratio = target_norm_tensor / pred_norm_tensor
                # Clamp to reasonable range
                norm_ratio = torch.clamp(norm_ratio, 0.1, 10.0)
                
                # Update adaptive scale with EMA
                self.adaptive_scale = self.ema_decay * self.adaptive_scale + (1 - self.ema_decay) * norm_ratio
                self.scale_update_count += 1
                
                # Track training progress - FIXED: proper tensor assignment
                if current_cosine_tensor > self.best_cosine_sim:
                    self.best_cosine_sim.copy_(current_cosine_tensor)
                    self.steps_since_improvement.zero_()
                else:
                    self.steps_since_improvement += 1

    def forward(
        self,
        model_output: torch.Tensor,
        target_samples: torch.Tensor,
        timesteps: torch.Tensor,
        eva_conditioning: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        training_mode: str = "patch_only",
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        SIMPLE WORKING: Flow matching loss with patch-wise cosine similarity
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
        
        # Normalize targets consistently
        if self.normalize_targets:
            target_samples_normalized = F.normalize(target_samples.detach(), p=2, dim=-1) * self.target_norm_scale
        else:
            target_samples_normalized = target_samples.detach()
        
        # Create source distribution (noise) with matching scale
        if noise is None:
            x_0 = torch.randn_like(target_samples_normalized, device=device) * 0.1
        else:
            x_0 = noise * 0.1
        
        # Compute velocity target using rectified flow
        if self.prediction_type == "velocity":
            if self.flow_type == "rectified":
                velocity_target = (target_samples_normalized - x_0) * self.velocity_scale
            else:
                t_expanded = timesteps.view(-1, 1, 1)
                velocity_target = (target_samples_normalized - (1 - t_expanded) * x_0) * self.velocity_scale
        else:
            velocity_target = noise * self.velocity_scale if noise is not None else torch.randn_like(target_samples_normalized) * self.velocity_scale
        
        # Apply adaptive scaling to model output if enabled (currently disabled)
        if is_training and self.adaptive_scaling:
            scaled_model_output = model_output * self.adaptive_scale
        else:
            scaled_model_output = model_output
        
        # Flow matching loss
        if is_training:
            flow_matching_loss = F.mse_loss(scaled_model_output, velocity_target.detach(), reduction='mean')
        else:
            flow_matching_loss = F.mse_loss(scaled_model_output.detach(), velocity_target.detach(), reduction='mean')
        
        # Verify loss computation is valid
        if is_training and not flow_matching_loss.requires_grad:
            raise RuntimeError("Flow matching loss doesn't require gradients during training!")
        
        total_loss = flow_matching_loss
        
        # Prepare metrics if requested
        metrics = None
        if return_metrics:
            with torch.no_grad():
                # Compute scalar norms for tracking
                pred_norm = torch.norm(scaled_model_output.detach(), dim=-1).mean().item()
                target_norm = torch.norm(velocity_target.detach(), dim=-1).mean().item()
                
                # PATCH-WISE cosine similarity computation (matches evaluation methodology)
                pred_patches = scaled_model_output.detach()  # [B, N, 1024]
                target_patches = velocity_target.detach()    # [B, N, 1024]
                
                # Normalize for cosine similarity computation
                pred_norm_tensor = F.normalize(pred_patches, p=2, dim=-1)
                target_norm_tensor = F.normalize(target_patches, p=2, dim=-1)
                
                # Per-patch cosine similarities [B, N]
                per_patch_cosine = F.cosine_similarity(pred_norm_tensor, target_norm_tensor, dim=-1)
                
                # Per-image average similarities [B]
                per_image_cosine = per_patch_cosine.mean(dim=1)
                
                # Overall batch cosine similarity
                patch_wise_cosine_sim = per_image_cosine.mean().item()
                
                # Quality metrics (same as evaluation)
                high_quality_patches = (per_patch_cosine > 0.7).float().mean().item()
                very_high_quality_patches = (per_patch_cosine > 0.8).float().mean().item()
                high_quality_images = (per_image_cosine > 0.7).float().mean().item()
                very_high_quality_images = (per_image_cosine > 0.8).float().mean().item()
                
                # Update EMA metrics
                self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * total_loss.item()
                self.ema_pred_norm = self.ema_decay * self.ema_pred_norm + (1 - self.ema_decay) * pred_norm
                self.ema_target_norm = self.ema_decay * self.ema_target_norm + (1 - self.ema_decay) * target_norm
                self.ema_cosine_sim = self.ema_decay * self.ema_cosine_sim + (1 - self.ema_decay) * patch_wise_cosine_sim
                
                # Create metrics dictionary
                metrics = {
                    # Core loss components
                    'flow_matching_loss': flow_matching_loss.item(),
                    'total_loss': total_loss.item(),
                    
                    # Norm tracking
                    'prediction_norm': pred_norm,
                    'target_norm': target_norm,
                    'norm_ratio': target_norm / max(pred_norm, 1e-8),
                    'adaptive_scale': self.adaptive_scale.item(),
                    
                    # PATCH-WISE cosine similarity (matches evaluation methodology)
                    'patch_wise_cosine_sim': patch_wise_cosine_sim,
                    'per_patch_mean_cosine': per_patch_cosine.mean().item(),
                    'per_patch_std_cosine': per_patch_cosine.std().item(),
                    'per_image_mean_cosine': per_image_cosine.mean().item(),
                    'per_image_std_cosine': per_image_cosine.std().item(),
                    
                    # Quality distribution (same as evaluation)
                    'high_quality_patches_ratio': high_quality_patches,
                    'very_high_quality_patches_ratio': very_high_quality_patches,
                    'high_quality_images_ratio': high_quality_images,
                    'very_high_quality_images_ratio': very_high_quality_images,
                    
                    # Min/max for detailed analysis
                    'min_patch_cosine': per_patch_cosine.min().item(),
                    'max_patch_cosine': per_patch_cosine.max().item(),
                    'min_image_cosine': per_image_cosine.min().item(),
                    'max_image_cosine': per_image_cosine.max().item(),
                    
                    # Model configuration info
                    'flow_type': self.flow_type,
                    'prediction_type': self.prediction_type,
                    'velocity_scale': self.velocity_scale,
                    'is_training': is_training,
                    'num_tokens': num_tokens,
                    'mode': training_mode,
                    'patches_per_image': num_tokens,
                    'total_patches_in_batch': batch_size * num_tokens,
                }
        
        return total_loss, metrics

    def get_scaling_info(self) -> Dict[str, float]:
        """Get current scaling information for debugging"""
        return {
            'adaptive_scale': self.adaptive_scale.item(),
            'ema_pred_norm': self.ema_pred_norm.item(),
            'ema_target_norm': self.ema_target_norm.item(),
            'velocity_scale': self.velocity_scale,
            'target_norm_scale': self.target_norm_scale,
            'scale_update_count': self.scale_update_count.item(),
            'best_cosine_sim': self.best_cosine_sim.item(),
            'steps_since_improvement': self.steps_since_improvement.item(),
        }


def create_blip3o_flow_matching_loss(
    prediction_type: str = "velocity",
    normalize_targets: bool = True,
    flow_type: str = "rectified",
    velocity_scale: float = 0.1,
    target_norm_scale: float = 1.0,
    adaptive_scaling: bool = True,
    ema_decay: float = 0.99,
    **kwargs
) -> BLIP3oFlowMatchingLoss:
    """
    FIXED: Factory function with proper scaling parameters
    """
    return BLIP3oFlowMatchingLoss(
        prediction_type=prediction_type,
        normalize_targets=normalize_targets,
        flow_type=flow_type,
        velocity_scale=velocity_scale,
        target_norm_scale=target_norm_scale,
        adaptive_scaling=adaptive_scaling,
        ema_decay=ema_decay,
        **kwargs
    )