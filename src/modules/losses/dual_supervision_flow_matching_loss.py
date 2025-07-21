"""
FIXED: Dual Supervision Flow Matching Loss with Global Generation Training
KEY FIX: Adds global flow matching to train the model to generate directly in global space,
not just post-process from patches. This resolves the training-inference mismatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from transformers import CLIPModel

# Import from the correct module
from .flow_matching_loss import BLIP3oFlowMatchingLoss
from ..config.blip3o_config import FlowMatchingConfig


class DualSupervisionFlowMatchingLoss(nn.Module):
    """
    FIXED: Dual Supervision Flow Matching Loss with Global Generation Training
    
    Combines THREE loss components:
    1. Patch-level flow matching (for fine details)
    2. Global-level flow matching (for recall performance) ← KEY FIX
    3. Dual supervision losses (patch + global alignment)
    
    This trains the model to generate BOTH patch and global representations,
    resolving the training-inference mismatch that caused poor recall.
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
        
        # FIXED: Enhanced loss weights for global generation
        patch_loss_weight: float = 1.0,
        global_loss_weight: float = 2.0,
        patch_flow_weight: float = 1.0,      # Patch flow matching weight
        global_flow_weight: float = 3.0,     # Global flow matching weight (higher!)
        
        # Loss configuration
        use_cosine_similarity: bool = False,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        
        # Progressive training
        use_progressive_training: bool = True,
        min_timestep: float = 0.001,
        max_timestep: float = 0.999,
    ):
        super().__init__()
        
        # Store configuration
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
        
        # FIXED: Enhanced loss weights
        self.patch_loss_weight = patch_loss_weight
        self.global_loss_weight = global_loss_weight
        self.patch_flow_weight = patch_flow_weight
        self.global_flow_weight = global_flow_weight  # KEY: Higher weight for global generation
        
        self.use_cosine_similarity = use_cosine_similarity
        
        # Progressive training
        self.use_progressive_training = use_progressive_training
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.training_step = 0
        
        # Load frozen CLIP model for global loss computation
        self.clip_model_name = clip_model_name
        self._load_clip_model()
        
        # Tracking metrics
        self.register_buffer('ema_patch_cosine', torch.tensor(0.0))
        self.register_buffer('ema_global_cosine', torch.tensor(0.0))
        self.register_buffer('ema_global_generation_cosine', torch.tensor(0.0))  # NEW
        self.ema_decay = 0.999
        
        print(f"✅ FIXED Dual Supervision Loss with Global Generation:")
        print(f"   Patch flow weight: {patch_flow_weight}")
        print(f"   Global flow weight: {global_flow_weight} (KEY FIX)")
        print(f"   Global supervision weight: {global_loss_weight}")
        print(f"   Trains both patch AND global generation")
    
    def _load_clip_model(self):
        """Load frozen CLIP model for global loss computation."""
        try:
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
            self.clip_visual_projection = self.clip_model.visual_projection
            print(f"✅ CLIP model loaded for global target computation")
        except Exception as e:
            print(f"⚠️ Failed to load CLIP model: {e}")
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
        
        # Reshape for broadcasting
        while alpha_t.dim() < x_1.dim():
            alpha_t = alpha_t.unsqueeze(-1)
        while sigma_t.dim() < x_1.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        
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
                while t.dim() < x_1.dim():
                    t = t.unsqueeze(-1)
                dsigma_dt = torch.full_like(t, dsigma_dt)
            elif self.schedule_type == "cosine":
                dsigma_dt = (self.sigma_max - self.sigma_min) * (math.pi / 2) * torch.sin(math.pi * t / 2)
                while dsigma_dt.dim() < x_1.dim():
                    dsigma_dt = dsigma_dt.unsqueeze(-1)
            
            velocity_target = x_1 - x_0 - dsigma_dt * noise
        elif self.prediction_type == "epsilon":
            velocity_target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return velocity_target
    
    def compute_patch_flow_loss(
        self,
        dit_patch_output: torch.Tensor,    # [B, 256, 1024] - DiT patch velocity predictions
        target_patch_velocity: torch.Tensor,  # [B, 256, 1024] - Target patch velocity
    ) -> torch.Tensor:
        """Compute patch-level flow matching loss."""
        return F.mse_loss(dit_patch_output, target_patch_velocity)
    
    def compute_global_flow_loss(
        self,
        dit_global_output: torch.Tensor,   # [B, 768] - DiT global velocity predictions
        target_global_velocity: torch.Tensor,  # [B, 768] - Target global velocity
    ) -> torch.Tensor:
        """
        FIXED: Compute global-level flow matching loss.
        This is the KEY FIX that trains the model to generate in global space.
        """
        return F.mse_loss(dit_global_output, target_global_velocity)
    
    def compute_patch_supervision_loss(
        self,
        dit_output: torch.Tensor,      # [B, 256, 1024] - DiT patch outputs
        clip_patches: torch.Tensor,    # [B, 256, 1024] - Target CLIP patches
    ) -> torch.Tensor:
        """Compute patch-level supervision loss."""
        if self.use_cosine_similarity:
            dit_norm = F.normalize(dit_output, dim=-1)
            clip_norm = F.normalize(clip_patches, dim=-1)
            cosine_sim = F.cosine_similarity(dit_norm, clip_norm, dim=-1)
            return 1.0 - cosine_sim.mean()
        else:
            return F.mse_loss(dit_output, clip_patches)
    
    def compute_global_supervision_loss(
        self,
        dit_global: torch.Tensor,      # [B, 768] - DiT global output
        clip_global: torch.Tensor,     # [B, 768] - Target CLIP global
    ) -> torch.Tensor:
        """Compute global-level supervision loss."""
        if self.use_cosine_similarity:
            dit_norm = F.normalize(dit_global, dim=-1)
            clip_norm = F.normalize(clip_global, dim=-1)
            cosine_sim = F.cosine_similarity(dit_norm, clip_norm, dim=-1)
            return 1.0 - cosine_sim.mean()
        else:
            return F.mse_loss(dit_global, clip_global)
    
    def compute_target_global_features(self, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute target global features from CLIP patches."""
        with torch.no_grad():
            # Average pool: [B, 256, 1024] → [B, 1024]
            pooled_clip = clip_embeddings.mean(dim=1)
            
            # Ensure CLIP projection is on correct device
            target_device = pooled_clip.device
            if self.clip_visual_projection.weight.device != target_device:
                self.clip_visual_projection = self.clip_visual_projection.to(target_device)
            
            # Apply CLIP visual projection: [B, 1024] → [B, 768]
            target_global = self.clip_visual_projection(pooled_clip)
        
        return target_global
    
    def compute_detailed_metrics(
        self,
        dit_patch_output: torch.Tensor,
        dit_global_output: torch.Tensor,
        clip_patches: torch.Tensor,
        clip_global: torch.Tensor,
        timesteps: torch.Tensor,
        patch_flow_loss: float,
        global_flow_loss: float,
    ) -> Dict[str, float]:
        """Compute detailed metrics for training monitoring."""
        with torch.no_grad():
            # Patch-level metrics
            patch_cosine = F.cosine_similarity(
                F.normalize(dit_patch_output, dim=-1),
                F.normalize(clip_patches, dim=-1),
                dim=-1
            ).mean().item()
            
            # Global-level metrics
            if dit_global_output is not None and clip_global is not None:
                global_cosine = F.cosine_similarity(
                    F.normalize(dit_global_output, dim=-1),
                    F.normalize(clip_global, dim=-1),
                    dim=-1
                ).mean().item()
            else:
                global_cosine = 0.0
            
            # Update EMA metrics
            self.ema_patch_cosine = self.ema_decay * self.ema_patch_cosine + (1 - self.ema_decay) * patch_cosine
            self.ema_global_cosine = self.ema_decay * self.ema_global_cosine + (1 - self.ema_decay) * global_cosine
            
            # NEW: Global generation quality metric
            global_generation_cosine = global_cosine  # Since we're training global generation
            self.ema_global_generation_cosine = self.ema_decay * self.ema_global_generation_cosine + (1 - self.ema_decay) * global_generation_cosine
            
            return {
                # Flow matching losses
                "patch_flow_loss": patch_flow_loss,
                "global_flow_loss": global_flow_loss,
                
                # Alignment metrics
                "patch_cosine_similarity": patch_cosine,
                "global_cosine_similarity": global_cosine,
                "global_generation_cosine": global_generation_cosine,  # NEW: Key for recall
                
                # EMA metrics
                "ema_patch_cosine": self.ema_patch_cosine.item(),
                "ema_global_cosine": self.ema_global_cosine.item(),
                "ema_global_generation_cosine": self.ema_global_generation_cosine.item(),  # NEW
                
                # Training diagnostics
                "timestep_mean": timesteps.mean().item(),
                "training_step": self.training_step,
                
                # Quality indicators (weighted for recall)
                "overall_quality_score": 0.3 * patch_cosine + 0.7 * global_cosine,
                "recall_readiness_score": global_generation_cosine,  # NEW: Key metric
            }
    
    def forward(
        self,
        # DiT outputs
        dit_patch_output: torch.Tensor,     # [B, 256, 1024] - DiT patch outputs
        dit_global_output: torch.Tensor,    # [B, 768] - DiT global outputs
        
        # Targets
        clip_patches: torch.Tensor,         # [B, 256, 1024] - Target CLIP patches
        clip_global: torch.Tensor,          # [B, 768] - Target CLIP global
        
        # Flow matching inputs
        timesteps: torch.Tensor,            # [B] - Timesteps
        eva_conditioning: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        FIXED: Compute dual supervision loss with BOTH patch and global flow matching.
        
        This is the KEY FIX that resolves the training-inference mismatch by training
        the model to generate directly in both patch and global spaces.
        """
        batch_size = clip_patches.shape[0]
        device = clip_patches.device
        
        # 1. Patch-level supervision loss
        patch_supervision_loss = self.compute_patch_supervision_loss(dit_patch_output, clip_patches)
        
        # 2. Global-level supervision loss
        global_supervision_loss = self.compute_global_supervision_loss(dit_global_output, clip_global)
        
        # FIXED: 3. Patch-level flow matching loss
        if noise is None:
            noise = torch.randn_like(clip_patches)
        
        # Create patch flow matching targets
        x_0_patch = torch.randn_like(clip_patches)
        patch_velocity_target = self.compute_velocity_target(x_0_patch, clip_patches, timesteps, noise)
        patch_flow_loss = self.compute_patch_flow_loss(dit_patch_output, patch_velocity_target)
        
        # FIXED: 4. Global-level flow matching loss (KEY FIX)
        # Create global flow matching targets
        x_0_global = torch.randn_like(clip_global)
        global_noise = torch.randn_like(clip_global)
        global_velocity_target = self.compute_velocity_target(x_0_global, clip_global, timesteps, global_noise)
        global_flow_loss = self.compute_global_flow_loss(dit_global_output, global_velocity_target)
        
        # FIXED: Combined loss with proper weighting
        total_loss = (
            self.patch_loss_weight * patch_supervision_loss +
            self.global_loss_weight * global_supervision_loss +
            self.patch_flow_weight * patch_flow_loss +
            self.global_flow_weight * global_flow_loss  # KEY: Train global generation
        )
        
        # Update training step
        if self.training:
            self.training_step += 1
        
        # Compute metrics
        metrics = None
        if return_metrics:
            metrics = self.compute_detailed_metrics(
                dit_patch_output, dit_global_output, clip_patches, clip_global, timesteps,
                patch_flow_loss.item(), global_flow_loss.item()
            )
            metrics.update({
                "patch_supervision_loss": patch_supervision_loss.item(),
                "global_supervision_loss": global_supervision_loss.item(),
                "patch_flow_loss": patch_flow_loss.item(),
                "global_flow_loss": global_flow_loss.item(),  # NEW
                "total_loss": total_loss.item(),
                
                # Loss weights for monitoring
                "patch_supervision_weight": self.patch_loss_weight,
                "global_supervision_weight": self.global_loss_weight,
                "patch_flow_weight": self.patch_flow_weight,
                "global_flow_weight": self.global_flow_weight,  # NEW
            })
        
        return total_loss, metrics


def create_dual_supervision_loss(
    config: Optional[FlowMatchingConfig] = None,
    patch_loss_weight: float = 1.0,
    global_loss_weight: float = 2.0,
    patch_flow_weight: float = 1.0,
    global_flow_weight: float = 3.0,  # Higher for global generation
    use_cosine_similarity: bool = False,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    **kwargs
) -> DualSupervisionFlowMatchingLoss:
    """Factory function for FIXED dual supervision flow matching loss."""
    
    return DualSupervisionFlowMatchingLoss(
        config=config,
        patch_loss_weight=patch_loss_weight,
        global_loss_weight=global_loss_weight,
        patch_flow_weight=patch_flow_weight,
        global_flow_weight=global_flow_weight,
        use_cosine_similarity=use_cosine_similarity,
        clip_model_name=clip_model_name,
        **kwargs
    )


# Alias for backward compatibility
create_dual_supervision_loss = create_dual_supervision_loss