"""
Simplified Global Flow Matching Loss - TRAINS DIRECTLY ON GLOBAL FEATURES
Place this file as: src/modules/losses/global_flow_matching_loss.py

KEY FIX: Single flow matching loss on [B, 768] global features.
No more complex dual supervision - just one clean objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
from transformers import CLIPModel


class GlobalFlowMatchingLoss(nn.Module):
    """
    Simplified Global Flow Matching Loss
    
    KEY FIX: Trains directly on global [B, 768] embeddings that will be used for evaluation.
    This eliminates the training-inference mismatch.
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        prediction_type: str = "v_prediction",
        schedule_type: str = "linear",
        clip_model_name: str = "openai/clip-vit-large-patch14",
    ):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prediction_type = prediction_type
        self.schedule_type = schedule_type
        
        # Load CLIP for target computation
        self._load_clip_model(clip_model_name)
        
        # EMA metrics for monitoring
        self.register_buffer('ema_cosine', torch.tensor(0.0))
        self.register_buffer('ema_l2', torch.tensor(0.0))
        self.ema_decay = 0.99
        
        print(f"✅ Global flow matching loss initialized")
        print(f"   Training target: [B, 768] global embeddings")
        print(f"   Single objective: global flow matching")
    
    def _load_clip_model(self, clip_model_name):
        """Load CLIP model for target computation"""
        try:
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
            self.clip_visual_proj = self.clip_model.visual_projection
            print(f"✅ CLIP model loaded: {clip_model_name}")
        except Exception as e:
            print(f"⚠️ Failed to load CLIP: {e}")
            self.clip_visual_proj = nn.Linear(1024, 768, bias=False)
            self.clip_visual_proj.requires_grad_(False)
    
    def compute_target_global_features(self, clip_patches):
        """
        Compute target global features from CLIP patches.
        
        Args:
            clip_patches: [B, 256, 1024] CLIP patch embeddings
            
        Returns:
            target_global: [B, 768] target global embeddings
        """
        with torch.no_grad():
            # Pool patches to global
            pooled = clip_patches.mean(dim=1)  # [B, 1024]
            
            # Ensure same device
            if self.clip_visual_proj.weight.device != pooled.device:
                self.clip_visual_proj = self.clip_visual_proj.to(pooled.device)
            
            # Apply CLIP projection
            target_global = self.clip_visual_proj(pooled)  # [B, 768]
            
            # Normalize like CLIP
            target_global = F.normalize(target_global, p=2, dim=-1)
            
            return target_global
    
    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for flow matching"""
        return torch.rand(batch_size, device=device)
    
    def get_noise_schedule(self, t):
        """Get noise schedule parameters"""
        if self.schedule_type == "linear":
            alpha_t = t
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
        elif self.schedule_type == "cosine":
            alpha_t = 0.5 * (1 - torch.cos(math.pi * t))
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.cos(math.pi * t / 2)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")
        
        return alpha_t, sigma_t
    
    def interpolate_global_data(self, x_0, x_1, t, noise=None):
        """
        Flow matching interpolation for global features.
        
        Args:
            x_0: [B, 768] source (noise) 
            x_1: [B, 768] target (global CLIP features)
            t: [B] timesteps
            noise: [B, 768] additional noise
            
        Returns:
            x_t: [B, 768] interpolated features
        """
        if noise is None:
            noise = torch.randn_like(x_1)
        
        alpha_t, sigma_t = self.get_noise_schedule(t)
        
        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1)
        sigma_t = sigma_t.view(-1, 1)
        
        # Flow matching interpolation
        x_t = (1 - alpha_t) * x_0 + alpha_t * x_1 + sigma_t * noise
        
        return x_t
    
    def compute_velocity_target(self, x_0, x_1, t, noise=None):
        """Compute velocity target for flow matching"""
        if noise is None:
            noise = torch.randn_like(x_1)
        
        if self.prediction_type == "v_prediction":
            if self.schedule_type == "linear":
                dsigma_dt = -(self.sigma_max - self.sigma_min)
                dsigma_dt = torch.full_like(t, dsigma_dt).view(-1, 1)
            elif self.schedule_type == "cosine":
                dsigma_dt = (self.sigma_max - self.sigma_min) * (math.pi / 2) * torch.sin(math.pi * t / 2)
                dsigma_dt = dsigma_dt.view(-1, 1)
            
            velocity_target = x_1 - x_0 - dsigma_dt * noise
        elif self.prediction_type == "epsilon":
            velocity_target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return velocity_target
    
    def forward(
        self,
        predicted_global,    # [B, 768] - Model predictions
        clip_patches,        # [B, 256, 1024] - CLIP patch targets
        timesteps,          # [B] - Timesteps
        noise=None,         # [B, 768] - Noise
        return_metrics=False,
    ):
        """
        Compute global flow matching loss.
        
        Args:
            predicted_global: [B, 768] model predictions
            clip_patches: [B, 256, 1024] CLIP patch targets
            timesteps: [B] timesteps
            noise: [B, 768] optional noise
            return_metrics: whether to return detailed metrics
            
        Returns:
            loss: scalar loss
            metrics: optional dict of metrics
        """
        # Compute target global features
        target_global = self.compute_target_global_features(clip_patches)  # [B, 768]
        
        # Sample source distribution
        x_0 = torch.randn_like(target_global)
        
        # Sample noise for interpolation
        if noise is None:
            noise = torch.randn_like(target_global)
        
        # Compute velocity target
        velocity_target = self.compute_velocity_target(x_0, target_global, timesteps, noise)
        
        # Flow matching loss
        loss = F.mse_loss(predicted_global, velocity_target)
        
        # Compute metrics
        metrics = None
        if return_metrics:
            with torch.no_grad():
                # Cosine similarity
                cosine_sim = F.cosine_similarity(
                    F.normalize(predicted_global, dim=-1),
                    F.normalize(velocity_target, dim=-1),
                    dim=-1
                ).mean().item()
                
                # L2 distance
                l2_dist = torch.norm(predicted_global - velocity_target, dim=-1).mean().item()
                
                # Update EMA
                self.ema_cosine = self.ema_decay * self.ema_cosine + (1 - self.ema_decay) * cosine_sim
                self.ema_l2 = self.ema_decay * self.ema_l2 + (1 - self.ema_decay) * l2_dist
                
                # Target quality metrics
                target_norm = torch.norm(target_global, dim=-1).mean().item()
                pred_norm = torch.norm(predicted_global, dim=-1).mean().item()
                
                # Direct comparison metrics (most important)
                direct_cosine = F.cosine_similarity(
                    F.normalize(predicted_global, dim=-1),
                    F.normalize(target_global, dim=-1),
                    dim=-1
                ).mean().item()
                
                metrics = {
                    'global_flow_loss': loss.item(),
                    'global_cosine_similarity': cosine_sim,
                    'global_l2_distance': l2_dist,
                    'ema_global_cosine': self.ema_cosine.item(),
                    'ema_global_l2': self.ema_l2.item(),
                    'target_norm': target_norm,
                    'prediction_norm': pred_norm,
                    'timestep_mean': timesteps.mean().item(),
                    'timestep_std': timesteps.std().item(),
                    
                    # Most important: direct target comparison
                    'direct_global_cosine': direct_cosine,
                    'expected_recall_percent': min(direct_cosine * 70, 70),  # Rough estimate
                    
                    # Training quality indicators
                    'training_quality': 'excellent' if direct_cosine > 0.8 else 'good' if direct_cosine > 0.6 else 'needs_improvement',
                    'convergence_indicator': direct_cosine,
                }
        
        return loss, metrics


def create_global_flow_matching_loss(
    sigma_min: float = 1e-4,
    sigma_max: float = 1.0,
    prediction_type: str = "v_prediction",
    schedule_type: str = "linear",
    clip_model_name: str = "openai/clip-vit-large-patch14",
    **kwargs
):
    """Factory function for global flow matching loss"""
    return GlobalFlowMatchingLoss(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        prediction_type=prediction_type,
        schedule_type=schedule_type,
        clip_model_name=clip_model_name,
    )