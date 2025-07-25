#!/usr/bin/env python3
"""
COMPLETELY FIXED: BLIP3-o Flow Matching Loss with Both Velocity and Embedding Tracking
src/modules/losses/blip3o_flow_matching_loss.py
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
    FIXED: BLIP3-o Flow Matching Loss that tracks BOTH velocity and embedding similarity
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
        adaptive_scaling: bool = False,  # Disabled by default for stability
        ema_decay: float = 0.99,
        track_embeddings: bool = True,  # NEW: Track embedding similarity
        embedding_test_steps: int = 10,  # NEW: Steps for quick embedding test
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
        self.track_embeddings = track_embeddings
        self.embedding_test_steps = embedding_test_steps
        
        # EMA tracking for metrics
        self.register_buffer('ema_loss', torch.tensor(0.0))
        self.register_buffer('ema_velocity_cosine', torch.tensor(0.0))
        self.register_buffer('ema_embedding_cosine', torch.tensor(0.0))  # NEW
        self.register_buffer('ema_target_norm', torch.tensor(1.0))
        self.register_buffer('ema_pred_norm', torch.tensor(1.0))
        
        # Adaptive scaling buffers
        self.register_buffer('adaptive_scale', torch.tensor(1.0))
        self.register_buffer('scale_update_count', torch.tensor(0.0))
        
        # Training progress tracking
        self.register_buffer('best_velocity_sim', torch.tensor(0.0))
        self.register_buffer('best_embedding_sim', torch.tensor(0.0))  # NEW
        self.register_buffer('steps_since_improvement', torch.tensor(0.0))
        
        # Step counter for embedding testing
        self.step_count = 0
        
        logger.info(f"✅ FIXED BLIP3-o Flow Matching Loss initialized")
        logger.info(f"   Velocity scale: {velocity_scale}")
        logger.info(f"   Track embeddings: {track_embeddings}")
        logger.info(f"   Embedding test steps: {embedding_test_steps}")

    def update_adaptive_scaling(self, pred_norm: float, target_norm: float, current_cosine: float):
        """Update adaptive scaling factor"""
        if not self.adaptive_scaling:
            return
            
        with torch.no_grad():
            device = self.adaptive_scale.device
            
            if torch.is_tensor(pred_norm):
                pred_norm = pred_norm.item()
            if torch.is_tensor(target_norm):
                target_norm = target_norm.item()
            if torch.is_tensor(current_cosine):
                current_cosine = current_cosine.item()
            
            pred_norm_tensor = torch.tensor(pred_norm, device=device, dtype=torch.float32)
            target_norm_tensor = torch.tensor(target_norm, device=device, dtype=torch.float32)
            current_cosine_tensor = torch.tensor(current_cosine, device=device, dtype=torch.float32)
            
            if pred_norm > 1e-8:
                norm_ratio = target_norm_tensor / pred_norm_tensor
                norm_ratio = torch.clamp(norm_ratio, 0.1, 10.0)
                
                self.adaptive_scale = self.ema_decay * self.adaptive_scale + (1 - self.ema_decay) * norm_ratio
                self.scale_update_count += 1
                
                if current_cosine_tensor > self.best_velocity_sim:
                    self.best_velocity_sim.copy_(current_cosine_tensor)
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
        model_ref: Optional[nn.Module] = None,  # NEW: Model reference for embedding testing
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        FIXED: Flow matching loss with BOTH velocity and embedding similarity tracking
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
        
        # Create source distribution (noise)
        if noise is None:
            x_0 = torch.randn_like(target_samples_normalized, device=device)
        else:
            x_0 = noise
        
        # Compute velocity target using rectified flow
        if self.prediction_type == "velocity":
            if self.flow_type == "rectified":
                # For rectified flow: velocity = (target - source)
                velocity_target = target_samples_normalized - x_0
            else:
                t_expanded = timesteps.view(-1, 1, 1)
                velocity_target = target_samples_normalized - (1 - t_expanded) * x_0
        else:
            velocity_target = noise if noise is not None else torch.randn_like(target_samples_normalized)
        
        # Apply velocity scaling to targets (make them smaller for stability)
        velocity_target_scaled = velocity_target * self.velocity_scale
        
        # Apply adaptive scaling to model output if enabled
        if is_training and self.adaptive_scaling:
            scaled_model_output = model_output * self.adaptive_scale
        else:
            scaled_model_output = model_output
        
        # Flow matching loss (both velocity and target are scaled consistently)
        flow_matching_loss = F.mse_loss(scaled_model_output, velocity_target_scaled.detach(), reduction='mean')
        
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
                target_norm = torch.norm(velocity_target_scaled.detach(), dim=-1).mean().item()
                
                # VELOCITY SIMILARITY (what we're training on)
                pred_norm_tensor = F.normalize(scaled_model_output.detach(), p=2, dim=-1)
                target_norm_tensor = F.normalize(velocity_target_scaled.detach(), p=2, dim=-1)
                
                per_patch_velocity_cosine = F.cosine_similarity(pred_norm_tensor, target_norm_tensor, dim=-1)
                per_image_velocity_cosine = per_patch_velocity_cosine.mean(dim=1)
                velocity_cosine_sim = per_image_velocity_cosine.mean().item()
                
                # EMBEDDING SIMILARITY (what we actually care about)
                embedding_cosine_sim = 0.0
                if self.track_embeddings and model_ref is not None and is_training:
                    # Every few steps, test actual embedding generation
                    self.step_count += 1
                    if self.step_count % 10 == 0:  # Test every 10 steps
                        try:
                            model_ref.eval()
                            embedding_test_result = model_ref.quick_generate_test(
                                eva_features=eva_conditioning,
                                target_embeddings=target_samples,
                                velocity_scale=self.velocity_scale,
                                num_steps=self.embedding_test_steps
                            )
                            embedding_cosine_sim = embedding_test_result['embedding_similarity']
                            model_ref.train()  # Return to training mode
                        except Exception as e:
                            logger.debug(f"Embedding test failed: {e}")
                            embedding_cosine_sim = 0.0
                    else:
                        embedding_cosine_sim = self.ema_embedding_cosine.item()
                
                # Update EMA metrics
                self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * total_loss.item()
                self.ema_pred_norm = self.ema_decay * self.ema_pred_norm + (1 - self.ema_decay) * pred_norm
                self.ema_target_norm = self.ema_decay * self.ema_target_norm + (1 - self.ema_decay) * target_norm
                self.ema_velocity_cosine = self.ema_decay * self.ema_velocity_cosine + (1 - self.ema_decay) * velocity_cosine_sim
                
                if embedding_cosine_sim > 0:
                    self.ema_embedding_cosine = self.ema_decay * self.ema_embedding_cosine + (1 - self.ema_decay) * embedding_cosine_sim
                
                # Track best similarities
                if velocity_cosine_sim > self.best_velocity_sim.item():
                    self.best_velocity_sim = torch.tensor(velocity_cosine_sim, device=device)
                
                if embedding_cosine_sim > self.best_embedding_sim.item():
                    self.best_embedding_sim = torch.tensor(embedding_cosine_sim, device=device)
                
                # Quality metrics for velocity predictions
                high_quality_velocity_patches = (per_patch_velocity_cosine > 0.7).float().mean().item()
                very_high_quality_velocity_patches = (per_patch_velocity_cosine > 0.8).float().mean().item()
                high_quality_velocity_images = (per_image_velocity_cosine > 0.7).float().mean().item()
                
                # Create comprehensive metrics dictionary
                metrics = {
                    # Core loss components
                    'flow_matching_loss': flow_matching_loss.item(),
                    'total_loss': total_loss.item(),
                    
                    # Norm tracking
                    'prediction_norm': pred_norm,
                    'target_norm': target_norm,
                    'norm_ratio': target_norm / max(pred_norm, 1e-8),
                    'adaptive_scale': self.adaptive_scale.item(),
                    
                    # VELOCITY SIMILARITY (training metric)
                    'velocity_cosine_sim': velocity_cosine_sim,
                    'velocity_per_patch_mean': per_patch_velocity_cosine.mean().item(),
                    'velocity_per_patch_std': per_patch_velocity_cosine.std().item(),
                    'velocity_per_image_mean': per_image_velocity_cosine.mean().item(),
                    'velocity_high_quality_patches': high_quality_velocity_patches,
                    'velocity_very_high_quality_patches': very_high_quality_velocity_patches,
                    'velocity_high_quality_images': high_quality_velocity_images,
                    
                    # EMBEDDING SIMILARITY (evaluation metric)
                    'embedding_cosine_sim': embedding_cosine_sim,
                    'embedding_test_steps': self.embedding_test_steps,
                    'embedding_test_frequency': 10,
                    
                    # EMA tracking
                    'ema_velocity_cosine': self.ema_velocity_cosine.item(),
                    'ema_embedding_cosine': self.ema_embedding_cosine.item(),
                    'best_velocity_sim': self.best_velocity_sim.item(),
                    'best_embedding_sim': self.best_embedding_sim.item(),
                    
                    # Model configuration info
                    'flow_type': self.flow_type,
                    'prediction_type': self.prediction_type,
                    'velocity_scale': self.velocity_scale,
                    'is_training': is_training,
                    'num_tokens': num_tokens,
                    'mode': training_mode,
                    'step_count': self.step_count,
                    
                    # Status flags
                    'tracking_embeddings': self.track_embeddings,
                    'embedding_test_available': model_ref is not None,
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
            'best_velocity_sim': self.best_velocity_sim.item(),
            'best_embedding_sim': self.best_embedding_sim.item(),
            'ema_velocity_cosine': self.ema_velocity_cosine.item(),
            'ema_embedding_cosine': self.ema_embedding_cosine.item(),
            'steps_since_improvement': self.steps_since_improvement.item(),
        }


def create_blip3o_flow_matching_loss(
    prediction_type: str = "velocity",
    normalize_targets: bool = True,
    flow_type: str = "rectified",
    velocity_scale: float = 0.1,
    target_norm_scale: float = 1.0,
    adaptive_scaling: bool = False,
    ema_decay: float = 0.99,
    track_embeddings: bool = True,
    embedding_test_steps: int = 10,
    **kwargs
) -> BLIP3oFlowMatchingLoss:
    """
    Factory function for FIXED flow matching loss with embedding tracking
    """
    return BLIP3oFlowMatchingLoss(
        prediction_type=prediction_type,
        normalize_targets=normalize_targets,
        flow_type=flow_type,
        velocity_scale=velocity_scale,
        target_norm_scale=target_norm_scale,
        adaptive_scaling=adaptive_scaling,
        ema_decay=ema_decay,
        track_embeddings=track_embeddings,
        embedding_test_steps=embedding_test_steps,
        **kwargs
    )