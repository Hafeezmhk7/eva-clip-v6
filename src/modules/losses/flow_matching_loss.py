"""
UPDATED: Dual Supervision Flow Matching Loss for BLIP3-o DiT Training
Implements both patch-level and global-level supervision for improved recall performance.

Loss Components:
1. Patch Loss: MSE(dit_patches, clip_patches) - maintains local detail
2. Global Loss: MSE(dit_global, clip_global) - ensures retrieval capability
3. Flow Matching Loss: Standard velocity prediction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from typing import Optional, Tuple, Dict, Any
import math

from ..config.blip3o_config import FlowMatchingConfig


class FlowMatchingLoss(nn.Module):
    """Base Flow Matching Loss for BLIP3-o DiT training."""
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        prediction_type: str = "v_prediction",
        schedule_type: str = "linear",
    ):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prediction_type = prediction_type
        self.schedule_type = schedule_type
        
        assert prediction_type in ["v_prediction", "epsilon"], f"Invalid prediction type: {prediction_type}"
        assert 0 <= sigma_min < sigma_max <= 10.0, f"Invalid sigma range: [{sigma_min}, {sigma_max}]"
        assert schedule_type in ["linear", "cosine"], f"Invalid schedule type: {schedule_type}"
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps uniformly from [0, 1] for flow matching."""
        return torch.rand(batch_size, device=device, dtype=torch.float32)
    
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
        """Interpolate between source and target distributions."""
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
        """Compute the target velocity field for flow matching."""
        if self.prediction_type == "v_prediction":
            if noise is None:
                noise = torch.randn_like(x_1)
            
            if self.schedule_type == "linear":
                dsigma_dt = -(self.sigma_max - self.sigma_min)
                dsigma_dt = torch.full_like(t, dsigma_dt).view(-1, 1, 1)
            elif self.schedule_type == "cosine":
                dsigma_dt = (self.sigma_max - self.sigma_min) * (math.pi / 2) * torch.sin(math.pi * t / 2)
                dsigma_dt = dsigma_dt.view(-1, 1, 1)
            
            velocity_target = x_1 - x_0 - dsigma_dt * noise
            
        elif self.prediction_type == "epsilon":
            if noise is None:
                noise = torch.randn_like(x_1)
            velocity_target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return velocity_target
    
    def forward(
        self,
        model_output: torch.Tensor,
        target_samples: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute flow matching loss."""
        batch_size = target_samples.shape[0]
        
        if noise is None:
            noise = torch.randn_like(target_samples)
        
        x_0 = torch.randn_like(target_samples)
        
        velocity_target = self.compute_velocity_target(
            x_0=x_0,
            x_1=target_samples,
            t=timesteps,
            noise=noise
        )
        
        if reduction == "mean":
            loss = F.mse_loss(model_output, velocity_target, reduction="mean")
        elif reduction == "sum":
            loss = F.mse_loss(model_output, velocity_target, reduction="sum")
        elif reduction == "none":
            loss = F.mse_loss(model_output, velocity_target, reduction="none")
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return loss


class DualSupervisionFlowMatchingLoss(FlowMatchingLoss):
    """
    UPDATED: Dual Supervision Flow Matching Loss for BLIP3-o training.
    
    Combines three loss components:
    1. Flow Matching Loss: Standard velocity prediction for patch outputs
    2. Patch Reconstruction Loss: MSE between DiT patches and CLIP patches  
    3. Global Alignment Loss: MSE between DiT global output and CLIP global output
    """
    
    def __init__(
        self,
        config: Optional[FlowMatchingConfig] = None,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        prediction_type: str = "v_prediction",
        clip_dim: int = 1024,
        eva_dim: int = 4096,
        clip_global_dim: int = 768,
        schedule_type: str = "linear",
        # NEW: Loss weighting parameters
        patch_loss_weight: float = 1.0,
        global_loss_weight: float = 1.0,
        flow_matching_loss_weight: float = 1.0,
        use_cosine_similarity: bool = False,
        clip_model_name: str = "openai/clip-vit-large-patch14",
    ):
        if config is not None:
            super().__init__(
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                prediction_type=config.prediction_type,
                schedule_type=config.schedule_type,
            )
            self.clip_dim = config.clip_dim
            self.eva_dim = config.eva_dim
        else:
            super().__init__(sigma_min, sigma_max, prediction_type, schedule_type)
            self.clip_dim = clip_dim
            self.eva_dim = eva_dim
        
        self.clip_global_dim = clip_global_dim
        
        # Loss weights
        self.patch_loss_weight = patch_loss_weight
        self.global_loss_weight = global_loss_weight
        self.flow_matching_loss_weight = flow_matching_loss_weight
        self.use_cosine_similarity = use_cosine_similarity
        
        # Load CLIP model for extracting ground truth targets
        self.clip_model_name = clip_model_name
        self.clip_model = None
        self.clip_processor = None
        self._load_clip_model()
        
        print(f"✅ Dual Supervision Loss initialized:")
        print(f"   Patch loss weight: {patch_loss_weight}")
        print(f"   Global loss weight: {global_loss_weight}")
        print(f"   Flow matching weight: {flow_matching_loss_weight}")
        print(f"   Use cosine similarity: {use_cosine_similarity}")
    
    def _load_clip_model(self):
        """Load CLIP model for extracting ground truth targets."""
        print(f"📦 Loading CLIP model: {self.clip_model_name}")
        
        try:
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            
            # Set to eval mode and freeze
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            print(f"✅ CLIP model loaded and frozen")
            
        except Exception as e:
            print(f"⚠️  Failed to load CLIP model: {e}")
            print("Will use provided targets instead of extracting from CLIP")
    
    def extract_clip_targets(self, images):
        """
        Extract both patch and global CLIP targets from images.
        
        Args:
            images: List of PIL Images or tensor
            
        Returns:
            Dict containing:
            - patch_targets: [B, 256, 1024] CLIP patch embeddings
            - global_targets: [B, 768] CLIP global embeddings
        """
        if self.clip_model is None:
            raise ValueError("CLIP model not available for target extraction")
        
        batch_size = len(images)
        device = next(self.clip_model.parameters()).device
        
        patch_targets = []
        global_targets = []
        
        with torch.no_grad():
            for img in images:
                # Process image
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get vision model outputs
                vision_outputs = self.clip_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract patch embeddings (remove CLS token)
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 1024]
                patch_targets.append(patch_embeddings.squeeze(0))
                
                # Extract global embeddings (CLS token + visual projection)
                cls_token = vision_outputs.last_hidden_state[:, 0, :]  # [1, 1024]
                global_embedding = self.clip_model.visual_projection(cls_token)  # [1, 768]
                global_embedding = F.normalize(global_embedding, p=2, dim=-1)
                global_targets.append(global_embedding.squeeze(0))
        
        return {
            'patch_targets': torch.stack(patch_targets),    # [B, 256, 1024]
            'global_targets': torch.stack(global_targets),  # [B, 768]
        }
    
    def compute_patch_reconstruction_loss(
        self,
        dit_patches: torch.Tensor,      # [B, 256, 1024]
        clip_patches: torch.Tensor,    # [B, 256, 1024]
    ) -> torch.Tensor:
        """
        Compute patch-level reconstruction loss.
        
        Args:
            dit_patches: DiT output patches [B, 256, 1024]
            clip_patches: CLIP target patches [B, 256, 1024]
            
        Returns:
            Patch reconstruction loss
        """
        if self.use_cosine_similarity:
            # Cosine similarity loss (better for normalized embeddings)
            dit_norm = F.normalize(dit_patches, p=2, dim=-1)
            clip_norm = F.normalize(clip_patches, p=2, dim=-1)
            cosine_sim = (dit_norm * clip_norm).sum(dim=-1).mean()
            loss = 1.0 - cosine_sim
        else:
            # MSE loss
            loss = F.mse_loss(dit_patches, clip_patches, reduction="mean")
        
        return loss
    
    def compute_global_alignment_loss(
        self,
        dit_global: torch.Tensor,       # [B, 768]
        clip_global: torch.Tensor,     # [B, 768]
    ) -> torch.Tensor:
        """
        Compute global-level alignment loss.
        
        Args:
            dit_global: DiT global output [B, 768]
            clip_global: CLIP target global [B, 768]
            
        Returns:
            Global alignment loss
        """
        if dit_global is None:
            return torch.tensor(0.0, device=clip_global.device, dtype=clip_global.dtype)
        
        if self.use_cosine_similarity:
            # Cosine similarity loss (better for retrieval)
            dit_norm = F.normalize(dit_global, p=2, dim=-1)
            clip_norm = F.normalize(clip_global, p=2, dim=-1)
            cosine_sim = (dit_norm * clip_norm).sum(dim=-1).mean()
            loss = 1.0 - cosine_sim
        else:
            # MSE loss
            loss = F.mse_loss(dit_global, clip_global, reduction="mean")
        
        return loss
    
    def compute_detailed_metrics(
        self,
        dit_patches: torch.Tensor,
        dit_global: Optional[torch.Tensor],
        clip_patches: torch.Tensor,
        clip_global: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute detailed metrics for monitoring training."""
        
        with torch.no_grad():
            metrics = {}
            
            # Patch-level metrics
            patch_mse = F.mse_loss(dit_patches, clip_patches).item()
            patch_cosine = F.cosine_similarity(
                dit_patches.flatten(1), clip_patches.flatten(1), dim=1
            ).mean().item()
            
            metrics.update({
                "patch_mse": patch_mse,
                "patch_cosine_similarity": patch_cosine,
                "patch_l2_norm": torch.norm(dit_patches, dim=-1).mean().item(),
                "clip_patch_l2_norm": torch.norm(clip_patches, dim=-1).mean().item(),
            })
            
            # Global-level metrics
            if dit_global is not None:
                global_mse = F.mse_loss(dit_global, clip_global).item()
                global_cosine = F.cosine_similarity(dit_global, clip_global, dim=1).mean().item()
                
                metrics.update({
                    "global_mse": global_mse,
                    "global_cosine_similarity": global_cosine,
                    "global_l2_norm": torch.norm(dit_global, dim=-1).mean().item(),
                    "clip_global_l2_norm": torch.norm(clip_global, dim=-1).mean().item(),
                })
            
            # Timestep statistics
            metrics.update({
                "timestep_mean": timesteps.mean().item(),
                "timestep_std": timesteps.std().item(),
            })
            
            return metrics
    
    def forward(
        self,
        # DiT model outputs
        dit_patch_output: torch.Tensor,         # [B, 256, 1024] from DiT
        dit_global_output: Optional[torch.Tensor],  # [B, 768] from global pipeline
        
        # Ground truth targets
        clip_patch_targets: torch.Tensor,      # [B, 256, 1024] CLIP patches
        clip_global_targets: torch.Tensor,     # [B, 768] CLIP global
        
        # Flow matching parameters
        timesteps: torch.Tensor,               # [B] timesteps
        eva_conditioning: Optional[torch.Tensor] = None,  # [B, 256, 4096]
        noise: Optional[torch.Tensor] = None,  # [B, 256, 1024]
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute dual supervision loss.
        
        Args:
            dit_patch_output: DiT patch output [B, 256, 1024]
            dit_global_output: DiT global output [B, 768] (can be None)
            clip_patch_targets: CLIP patch targets [B, 256, 1024]
            clip_global_targets: CLIP global targets [B, 768]
            timesteps: Flow matching timesteps [B]
            eva_conditioning: EVA-CLIP conditioning (optional)
            noise: Noise for flow matching (optional)
            return_metrics: Whether to return detailed metrics
            
        Returns:
            Tuple of (total_loss, optional_metrics)
        """
        # Validate inputs
        assert dit_patch_output.shape == clip_patch_targets.shape, \
            f"Patch shape mismatch: {dit_patch_output.shape} vs {clip_patch_targets.shape}"
        
        if dit_global_output is not None:
            assert dit_global_output.shape == clip_global_targets.shape, \
                f"Global shape mismatch: {dit_global_output.shape} vs {clip_global_targets.shape}"
        
        # 1. Flow Matching Loss (patch-level velocity prediction)
        flow_loss = super().forward(
            model_output=dit_patch_output,
            target_samples=clip_patch_targets,
            timesteps=timesteps,
            noise=noise,
            reduction="mean"
        )
        
        # 2. Patch Reconstruction Loss
        patch_loss = self.compute_patch_reconstruction_loss(
            dit_patches=dit_patch_output,
            clip_patches=clip_patch_targets
        )
        
        # 3. Global Alignment Loss
        global_loss = self.compute_global_alignment_loss(
            dit_global=dit_global_output,
            clip_global=clip_global_targets
        )
        
        # Combine losses with weights
        total_loss = (
            self.flow_matching_loss_weight * flow_loss +
            self.patch_loss_weight * patch_loss +
            self.global_loss_weight * global_loss
        )
        
        # Compute detailed metrics if requested
        metrics = None
        if return_metrics:
            metrics = self.compute_detailed_metrics(
                dit_patches=dit_patch_output,
                dit_global=dit_global_output,
                clip_patches=clip_patch_targets,
                clip_global=clip_global_targets,
                timesteps=timesteps,
            )
            
            # Add loss components to metrics
            metrics.update({
                "flow_matching_loss": flow_loss.item(),
                "patch_reconstruction_loss": patch_loss.item(),
                "global_alignment_loss": global_loss.item(),
                "total_loss": total_loss.item(),
                "weighted_flow_loss": (self.flow_matching_loss_weight * flow_loss).item(),
                "weighted_patch_loss": (self.patch_loss_weight * patch_loss).item(),
                "weighted_global_loss": (self.global_loss_weight * global_loss).item(),
            })
        
        return total_loss, metrics


def create_dual_supervision_loss(
    config: Optional[FlowMatchingConfig] = None,
    patch_loss_weight: float = 1.0,
    global_loss_weight: float = 2.0,  # Higher weight for retrieval
    flow_matching_loss_weight: float = 1.0,
    use_cosine_similarity: bool = True,  # Better for retrieval tasks
    clip_model_name: str = "openai/clip-vit-large-patch14",
    **kwargs
) -> DualSupervisionFlowMatchingLoss:
    """
    Factory function to create dual supervision flow matching loss.
    
    Args:
        config: Flow matching configuration
        patch_loss_weight: Weight for patch reconstruction loss
        global_loss_weight: Weight for global alignment loss (higher for retrieval)
        flow_matching_loss_weight: Weight for flow matching loss
        use_cosine_similarity: Use cosine similarity instead of MSE
        clip_model_name: CLIP model name for target extraction
        **kwargs: Additional parameters
        
    Returns:
        DualSupervisionFlowMatchingLoss instance
    """
    if config is None:
        from ..config.blip3o_config import get_default_flow_matching_config
        config = get_default_flow_matching_config()
    
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
    )


# Legacy compatibility
BLIP3oFlowMatchingLoss = DualSupervisionFlowMatchingLoss
create_blip3o_flow_matching_loss = create_dual_supervision_loss