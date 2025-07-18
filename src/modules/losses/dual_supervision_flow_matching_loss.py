"""
Dual Supervision Flow Matching Loss for BLIP3-o DiT with Enhanced Recall Performance
Place this file at: src/modules/losses/dual_supervision_flow_matching_loss.py

Architecture:
EVA [B,256,4096] → DiT → [B,256,1024] → {
    Patch Output: [B,256,1024] (patch loss)
    Global Path: Avg Pool → MLP → Frozen CLIP Proj → [B,768]
}

Dual Loss:
L1: MSE(dit_patches, clip_patches) - patch fidelity
L2: MSE(dit_global, clip_global) - retrieval capability  
L3: Flow Matching Loss - velocity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from transformers import CLIPModel, CLIPProcessor

from ..config.blip3o_config import FlowMatchingConfig


class DualSupervisionFlowMatchingLoss(nn.Module):
    """
    Dual Supervision Flow Matching Loss for enhanced recall performance.
    
    Combines:
    1. Patch-level reconstruction loss for fine-grained quality
    2. Global alignment loss for retrieval performance  
    3. Flow matching loss for generative capability
    """
    
    def __init__(
        self,
        config: Optional[FlowMatchingConfig] = None,
        
        # Flow matching parameters
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        prediction_type: str = "v_prediction",
        schedule_type: str = "linear",
        clip_dim: int = 1024,
        eva_dim: int = 4096,
        
        # Dual supervision loss weights
        patch_loss_weight: float = 1.0,
        global_loss_weight: float = 2.0,  # Higher for retrieval focus
        flow_matching_loss_weight: float = 1.0,
        
        # Loss type configuration
        use_cosine_similarity: bool = False,
        
        # CLIP model for frozen projection
        clip_model_name: str = "openai/clip-vit-large-patch14",
        
        # Progressive training
        use_progressive_training: bool = True,
        min_timestep: float = 0.001,
        max_timestep: float = 0.999,
    ):
        super().__init__()
        
        # Use config if provided
        if config is not None:
            self.sigma_min = config.sigma_min
            self.sigma_max = config.sigma_max
            self.prediction_type = config.prediction_type
            self.schedule_type = config.schedule_type
            self.clip_dim = config.clip_dim
            self.eva_dim = config.eva_dim
        else:
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max
            self.prediction_type = prediction_type
            self.schedule_type = schedule_type
            self.clip_dim = clip_dim
            self.eva_dim = eva_dim
        
        # Dual supervision loss weights
        self.patch_loss_weight = patch_loss_weight
        self.global_loss_weight = global_loss_weight
        self.flow_matching_loss_weight = flow_matching_loss_weight
        self.use_cosine_similarity = use_cosine_similarity
        
        # Progressive training
        self.use_progressive_training = use_progressive_training
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.training_step = 0
        
        # Load frozen CLIP model for global loss computation
        self.clip_model_name = clip_model_name
        self._load_clip_model()
        
        # For tracking metrics
        self.register_buffer('ema_patch_cosine', torch.tensor(0.0))
        self.register_buffer('ema_global_cosine', torch.tensor(0.0))
        self.ema_decay = 0.999
        
        print(f"✅ Dual Supervision Loss initialized:")
        print(f"   Patch weight: {patch_loss_weight}")
        print(f"   Global weight: {global_loss_weight}")
        print(f"   Flow matching weight: {flow_matching_loss_weight}")
        print(f"   Use cosine similarity: {use_cosine_similarity}")
        print(f"   CLIP model: {clip_model_name}")
    
    def _load_clip_model(self):
        """Load frozen CLIP model for global loss computation."""
        try:
            print(f"🔄 Loading CLIP model: {self.clip_model_name}")
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            
            # Freeze all CLIP parameters
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            self.clip_model.eval()
            
            # Get the visual projection layer
            self.clip_visual_projection = self.clip_model.visual_projection  # [1024 → 768]
            
            print(f"✅ CLIP model loaded and frozen")
            print(f"   Visual projection: {self.clip_visual_projection.weight.shape}")
            
        except Exception as e:
            print(f"⚠️ Failed to load CLIP model: {e}")
            print("   Creating dummy projection layer")
            self.clip_visual_projection = nn.Linear(1024, 768, bias=False)
            self.clip_visual_projection.requires_grad_(False)
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Enhanced timestep sampling with progressive training support."""
        if self.use_progressive_training and self.training:
            progress = min(1.0, self.training_step / 10000)
            t_min = self.min_timestep + (0.3 - self.min_timestep) * (1 - progress)
            t_max = 0.7 + (self.max_timestep - 0.7) * progress
            timesteps = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
        else:
            timesteps = torch.rand(batch_size, device=device)
        
        return timesteps
    
    def get_noise_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get noise schedule parameters for flow matching interpolation."""
        if self.schedule_type == "linear":
            alpha_t = t
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
        elif self.schedule_type == "cosine":
            alpha_t = 0.5 * (1 - torch.cos(math.pi * t))
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.cos(math.pi * t / 2)
        elif self.schedule_type == "sigmoid":
            alpha_t = torch.sigmoid(6 * (t - 0.5))
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(-6 * (t - 0.5))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return alpha_t, sigma_t
    
    def interpolate_data(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Interpolate between source and target distributions for flow matching."""
        batch_size = x_1.shape[0]
        
        if noise is None:
            noise = torch.randn_like(x_1)
        
        alpha_t, sigma_t = self.get_noise_schedule(t)
        alpha_t = alpha_t.view(batch_size, 1, 1)
        sigma_t = sigma_t.view(batch_size, 1, 1)
        
        x_t = (1 - alpha_t) * x_0 + alpha_t * x_1 + sigma_t * noise
        return x_t
    
    def compute_velocity_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity target for flow matching."""
        if noise is None:
            noise = torch.randn_like(x_1)
        
        if self.prediction_type == "v_prediction":
            if self.schedule_type == "linear":
                dsigma_dt = -(self.sigma_max - self.sigma_min)
                dsigma_dt = torch.full_like(t, dsigma_dt).view(-1, 1, 1)
            elif self.schedule_type == "cosine":
                dsigma_dt = (self.sigma_max - self.sigma_min) * (math.pi / 2) * torch.sin(math.pi * t / 2)
                dsigma_dt = dsigma_dt.view(-1, 1, 1)
            elif self.schedule_type == "sigmoid":
                sigmoid_term = torch.sigmoid(-6 * (t - 0.5))
                dsigma_dt = (self.sigma_max - self.sigma_min) * (-6) * sigmoid_term * (1 - sigmoid_term)
                dsigma_dt = dsigma_dt.view(-1, 1, 1)
            
            velocity_target = x_1 - x_0 - dsigma_dt * noise
        elif self.prediction_type == "epsilon":
            velocity_target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return velocity_target
    
    def compute_patch_loss(
        self,
        dit_output: torch.Tensor,      # [B, 256, 1024] - DiT patch outputs
        clip_patches: torch.Tensor,    # [B, 256, 1024] - Target CLIP patches
    ) -> torch.Tensor:
        """
        Compute patch-level reconstruction loss for fine-grained quality.
        L1: MSE(dit_patches, clip_patches)
        """
        if self.use_cosine_similarity:
            # Cosine similarity loss
            dit_norm = F.normalize(dit_output, dim=-1)
            clip_norm = F.normalize(clip_patches, dim=-1)
            cosine_sim = F.cosine_similarity(dit_norm, clip_norm, dim=-1)  # [B, 256]
            patch_loss = 1.0 - cosine_sim.mean()
        else:
            # MSE loss
            patch_loss = F.mse_loss(dit_output, clip_patches)
        
        return patch_loss
    
    def compute_global_loss(
        self,
        dit_global: torch.Tensor,      # [B, 768] - DiT global output (after MLP + CLIP proj)
        clip_global: torch.Tensor,     # [B, 768] - Target CLIP global (after CLIP proj)
    ) -> torch.Tensor:
        """
        Compute global alignment loss for retrieval performance.
        L2: MSE(dit_global, clip_global) - in CLIP's 768-dim aligned space
        """
        if self.use_cosine_similarity:
            # Cosine similarity loss
            dit_norm = F.normalize(dit_global, dim=-1)
            clip_norm = F.normalize(clip_global, dim=-1)
            cosine_sim = F.cosine_similarity(dit_norm, clip_norm, dim=-1)  # [B]
            global_loss = 1.0 - cosine_sim.mean()
        else:
            # MSE loss
            global_loss = F.mse_loss(dit_global, clip_global)
        
        return global_loss
    
    def apply_clip_visual_projection(self, clip_features: torch.Tensor) -> torch.Tensor:
        """Apply frozen CLIP visual projection to get aligned 768-dim features."""
        # Average pool: [B, 256, 1024] → [B, 1024]
        pooled_features = clip_features.mean(dim=1)
        
        # Apply frozen CLIP projection: [B, 1024] → [B, 768]
        with torch.no_grad():
            projected_features = self.clip_visual_projection(pooled_features)
        
        return projected_features
    
    def compute_detailed_metrics(
        self,
        dit_output: torch.Tensor,
        dit_global: torch.Tensor,
        clip_patches: torch.Tensor,
        clip_global: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute detailed metrics for dual supervision training."""
        with torch.no_grad():
            # Patch-level metrics
            patch_mse = F.mse_loss(dit_output, clip_patches).item()
            
            dit_patch_norm = F.normalize(dit_output, dim=-1)
            clip_patch_norm = F.normalize(clip_patches, dim=-1)
            patch_cosine = F.cosine_similarity(dit_patch_norm, clip_patch_norm, dim=-1).mean().item()
            
            # Global-level metrics
            global_mse = F.mse_loss(dit_global, clip_global).item()
            
            dit_global_norm = F.normalize(dit_global, dim=-1)
            clip_global_norm = F.normalize(clip_global, dim=-1)
            global_cosine = F.cosine_similarity(dit_global_norm, clip_global_norm, dim=-1).mean().item()
            
            # Update EMA metrics
            self.ema_patch_cosine = self.ema_decay * self.ema_patch_cosine + (1 - self.ema_decay) * patch_cosine
            self.ema_global_cosine = self.ema_decay * self.ema_global_cosine + (1 - self.ema_decay) * global_cosine
            
            # Feature magnitude analysis
            dit_patch_norm_mean = torch.norm(dit_output, dim=-1).mean().item()
            clip_patch_norm_mean = torch.norm(clip_patches, dim=-1).mean().item()
            dit_global_norm_mean = torch.norm(dit_global, dim=-1).mean().item()
            clip_global_norm_mean = torch.norm(clip_global, dim=-1).mean().item()
            
            # Quality indicators
            good_patch_alignment = (F.cosine_similarity(dit_patch_norm, clip_patch_norm, dim=-1) > 0.7).float().mean().item()
            good_global_alignment = (F.cosine_similarity(dit_global_norm, clip_global_norm, dim=-1) > 0.8).float().mean().item()
            
            return {
                # Patch metrics
                "patch_mse_loss": patch_mse,
                "patch_cosine_similarity": patch_cosine,
                "ema_patch_cosine": self.ema_patch_cosine.item(),
                "patch_norm_dit": dit_patch_norm_mean,
                "patch_norm_clip": clip_patch_norm_mean,
                "good_patch_alignment_ratio": good_patch_alignment,
                
                # Global metrics (key for recall)
                "global_mse_loss": global_mse,
                "global_cosine_similarity": global_cosine,
                "ema_global_cosine": self.ema_global_cosine.item(),
                "global_norm_dit": dit_global_norm_mean,
                "global_norm_clip": clip_global_norm_mean,
                "good_global_alignment_ratio": good_global_alignment,
                
                # Training diagnostics
                "timestep_mean": timesteps.mean().item(),
                "timestep_std": timesteps.std().item(),
                "training_step": self.training_step,
                
                # Overall quality (weighted for retrieval)
                "overall_quality_score": 0.3 * patch_cosine + 0.7 * global_cosine,
            }
    
    def forward(
        self,
        # DiT outputs
        dit_output: torch.Tensor,          # [B, 256, 1024] - DiT patch outputs
        dit_global: torch.Tensor,          # [B, 768] - DiT global output (after MLP + CLIP proj)
        
        # Targets
        clip_patches: torch.Tensor,        # [B, 256, 1024] - Target CLIP patches
        clip_global: torch.Tensor,         # [B, 768] - Target CLIP global (after CLIP proj)
        
        # Flow matching inputs
        timesteps: torch.Tensor,           # [B] - Timesteps
        eva_conditioning: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute dual supervision loss with three components:
        L1: Patch reconstruction loss
        L2: Global alignment loss  
        L3: Flow matching loss
        """
        batch_size = clip_patches.shape[0]
        device = clip_patches.device
        
        # 1. Patch reconstruction loss
        patch_loss = self.compute_patch_loss(dit_output, clip_patches)
        
        # 2. Global alignment loss (in CLIP's 768-dim space)
        global_loss = self.compute_global_loss(dit_global, clip_global)
        
        # 3. Flow matching loss (for generative capability)
        if noise is None:
            noise = torch.randn_like(clip_patches)
        
        x_0 = torch.randn_like(clip_patches)
        velocity_target = self.compute_velocity_target(x_0, clip_patches, timesteps, noise)
        flow_loss = F.mse_loss(dit_output, velocity_target)
        
        # Combined loss with weights
        total_loss = (
            self.patch_loss_weight * patch_loss +
            self.global_loss_weight * global_loss +
            self.flow_matching_loss_weight * flow_loss
        )
        
        # Update training step
        if self.training:
            self.training_step += 1
        
        # Compute metrics
        metrics = None
        if return_metrics:
            metrics = self.compute_detailed_metrics(
                dit_output, dit_global, clip_patches, clip_global, timesteps
            )
            metrics.update({
                "patch_loss": patch_loss.item(),
                "global_loss": global_loss.item(),
                "flow_matching_loss": flow_loss.item(),
                "total_loss": total_loss.item(),
                "patch_weight": self.patch_loss_weight,
                "global_weight": self.global_loss_weight,
                "flow_weight": self.flow_matching_loss_weight,
            })
        
        return total_loss, metrics


def create_dual_supervision_loss(
    config: Optional[FlowMatchingConfig] = None,
    patch_loss_weight: float = 1.0,
    global_loss_weight: float = 2.0,
    flow_matching_loss_weight: float = 1.0,
    use_cosine_similarity: bool = False,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    **kwargs
) -> DualSupervisionFlowMatchingLoss:
    """Factory function for dual supervision flow matching loss."""
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return DualSupervisionFlowMatchingLoss(
        config=config,
        patch_loss_weight=patch_loss_weight,
        global_loss_weight=global_loss_weight,
        flow_matching_loss_weight=flow_matching_loss_weight,
        use_cosine_similarity=use_cosine_similarity,
        clip_model_name=clip_model_name,
        **kwargs
    )