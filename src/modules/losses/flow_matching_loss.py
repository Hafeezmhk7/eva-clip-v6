"""
Flow Matching Loss Implementation for BLIP3-o DiT Training.
Exact implementation following the flow matching methodology used in BLIP3-o.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from ..config.blip3o_config import FlowMatchingConfig


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss for BLIP3-o DiT training.
    
    Flow matching trains models to predict velocity fields that transport samples
    from a source distribution (Gaussian noise) to a target distribution (CLIP features).
    
    This implementation follows the exact methodology from BLIP3-o paper:
    - Linear interpolation paths between noise and data
    - Velocity field prediction (v-parameterization)
    - Optimal transport-inspired training objective
    """
    
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
        
        # Validate parameters
        assert prediction_type in ["v_prediction", "epsilon"], f"Invalid prediction type: {prediction_type}"
        assert 0 <= sigma_min < sigma_max <= 10.0, f"Invalid sigma range: [{sigma_min}, {sigma_max}]"
        assert schedule_type in ["linear", "cosine"], f"Invalid schedule type: {schedule_type}"
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample random timesteps uniformly from [0, 1] for flow matching.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensors on
            
        Returns:
            Random timesteps in [0, 1] with shape [batch_size]
        """
        return torch.rand(batch_size, device=device, dtype=torch.float32)
    
    def get_noise_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get noise schedule parameters for flow matching interpolation.
        
        In BLIP3-o flow matching:
        x_t = (1-t) * x_0 + t * x_1 + sigma_t * epsilon
        where x_0 ~ N(0,I), x_1 is target data, and sigma_t provides additional noise
        
        Args:
            t: Time values in [0, 1]
            
        Returns:
            alpha_t: Weight for data interpolation
            sigma_t: Additional noise level
        """
        if self.schedule_type == "linear":
            # Linear interpolation: alpha increases linearly from 0 to 1
            alpha_t = t
            # Noise decreases linearly
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - t)
        
        elif self.schedule_type == "cosine":
            # Cosine schedule for smoother interpolation
            alpha_t = 0.5 * (1 - torch.cos(math.pi * t))
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.cos(math.pi * t / 2)
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return alpha_t, sigma_t
    
    def interpolate_data(
        self,
        x_0: torch.Tensor,              # [B, 64, 768] - Source (noise)
        x_1: torch.Tensor,              # [B, 64, 768] - Target (CLIP data)
        t: torch.Tensor,                # [B] - Timesteps
        noise: Optional[torch.Tensor] = None,  # [B, 64, 768] - Additional noise
    ) -> torch.Tensor:
        """
        Interpolate between source and target distributions.
        
        Flow matching interpolation:
        x_t = (1-alpha_t) * x_0 + alpha_t * x_1 + sigma_t * epsilon
        
        Args:
            x_0: Source distribution (typically Gaussian noise)
            x_1: Target distribution (CLIP embeddings)
            t: Interpolation times [0, 1]
            noise: Additional noise term
            
        Returns:
            Interpolated samples x_t
        """
        batch_size = x_1.shape[0]
        
        # Sample additional noise if not provided
        if noise is None:
            noise = torch.randn_like(x_1)
        
        # Get interpolation weights
        alpha_t, sigma_t = self.get_noise_schedule(t)
        
        # Reshape for broadcasting: [B] -> [B, 1, 1]
        alpha_t = alpha_t.view(batch_size, 1, 1)
        sigma_t = sigma_t.view(batch_size, 1, 1)
        
        # Flow matching interpolation
        x_t = (1 - alpha_t) * x_0 + alpha_t * x_1 + sigma_t * noise
        
        return x_t
    
    def compute_velocity_target(
        self,
        x_0: torch.Tensor,              # [B, 64, 768] - Source
        x_1: torch.Tensor,              # [B, 64, 768] - Target
        t: torch.Tensor,                # [B] - Timesteps
        noise: Optional[torch.Tensor] = None,  # [B, 64, 768] - Additional noise
    ) -> torch.Tensor:
        """
        Compute the target velocity field for flow matching.
        
        The velocity field is the time derivative of the interpolation path:
        v_t = d/dt[x_t] = x_1 - x_0 - d(sigma_t)/dt * epsilon
        
        Args:
            x_0: Source samples
            x_1: Target samples  
            t: Timesteps
            noise: Additional noise
            
        Returns:
            Target velocity field
        """
        if self.prediction_type == "v_prediction":
            # For v-prediction, compute the analytical velocity
            if noise is None:
                noise = torch.randn_like(x_1)
            
            # Get noise schedule derivatives
            if self.schedule_type == "linear":
                # d(sigma_t)/dt = -(sigma_max - sigma_min)
                dsigma_dt = -(self.sigma_max - self.sigma_min)
                dsigma_dt = torch.full_like(t, dsigma_dt).view(-1, 1, 1)
            
            elif self.schedule_type == "cosine":
                # d(sigma_t)/dt = -(sigma_max - sigma_min) * (-pi/2) * sin(pi*t/2)
                dsigma_dt = (self.sigma_max - self.sigma_min) * (math.pi / 2) * torch.sin(math.pi * t / 2)
                dsigma_dt = dsigma_dt.view(-1, 1, 1)
            
            # Velocity: v_t = x_1 - x_0 - d(sigma_t)/dt * epsilon
            velocity_target = x_1 - x_0 - dsigma_dt * noise
            
        elif self.prediction_type == "epsilon":
            # For epsilon prediction, target is the noise
            if noise is None:
                noise = torch.randn_like(x_1)
            velocity_target = noise
        
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return velocity_target
    
    def forward(
        self,
        model_output: torch.Tensor,      # [B, 64, 768] - Model predictions
        target_samples: torch.Tensor,    # [B, 64, 768] - CLIP targets
        timesteps: torch.Tensor,         # [B] - Timesteps
        noise: Optional[torch.Tensor] = None,  # [B, 64, 768] - Noise
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            model_output: Predicted velocity/noise from the model
            target_samples: Target CLIP embeddings
            timesteps: Sampled timesteps
            noise: Random noise (sampled if None)
            reduction: Loss reduction method
            
        Returns:
            Flow matching loss
        """
        batch_size = target_samples.shape[0]
        
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(target_samples)
        
        # Sample source distribution (Gaussian noise)
        x_0 = torch.randn_like(target_samples)
        
        # Compute target velocity field
        velocity_target = self.compute_velocity_target(
            x_0=x_0,
            x_1=target_samples,
            t=timesteps,
            noise=noise
        )
        
        # Compute MSE loss between predicted and target velocity
        if reduction == "mean":
            loss = F.mse_loss(model_output, velocity_target, reduction="mean")
        elif reduction == "sum":
            loss = F.mse_loss(model_output, velocity_target, reduction="sum")
        elif reduction == "none":
            loss = F.mse_loss(model_output, velocity_target, reduction="none")
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return loss


class BLIP3oFlowMatchingLoss(FlowMatchingLoss):
    """
    Specialized Flow Matching Loss for BLIP3-o training.
    
    Extends the base flow matching loss with BLIP3-o specific features:
    - CLIP/EVA-CLIP dimension handling
    - Optional regularization terms
    - Detailed metrics computation
    - Integration with EVA-CLIP conditioning
    """
    
    def __init__(
        self,
        config: Optional[FlowMatchingConfig] = None,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        prediction_type: str = "v_prediction",
        clip_dim: int = 768,
        eva_dim: int = 1280,
        regularization_weight: float = 0.0,
        schedule_type: str = "linear",
    ):
        # Use config if provided, otherwise use individual parameters
        if config is not None:
            super().__init__(
                sigma_min=config.sigma_min,
                sigma_max=config.sigma_max,
                prediction_type=config.prediction_type,
                schedule_type=config.schedule_type,
            )
            self.clip_dim = config.clip_dim
            self.eva_dim = config.eva_dim
            self.regularization_weight = config.regularization_weight
        else:
            super().__init__(sigma_min, sigma_max, prediction_type, schedule_type)
            self.clip_dim = clip_dim
            self.eva_dim = eva_dim
            self.regularization_weight = regularization_weight
    
    def compute_regularization_loss(
        self,
        model_output: torch.Tensor,      # [B, 64, 768] - Model predictions
        eva_conditioning: torch.Tensor,  # [B, 64, 1280] - EVA conditioning
    ) -> torch.Tensor:
        """
        Compute optional regularization loss.
        
        This can help with training stability and encourage certain properties
        in the generated embeddings.
        """
        if self.regularization_weight <= 0:
            return torch.tensor(0.0, device=model_output.device, dtype=model_output.dtype)
        
        # L2 regularization on model output magnitude
        output_norm_loss = torch.mean(torch.norm(model_output, dim=-1))
        
        # Optional: Encourage similarity with EVA conditioning in some space
        # (This is a simple example - more sophisticated regularization can be added)
        
        return self.regularization_weight * output_norm_loss
    
    def compute_detailed_metrics(
        self,
        model_output: torch.Tensor,      # [B, 64, 768] - Model predictions
        target_samples: torch.Tensor,    # [B, 64, 768] - CLIP targets
        velocity_target: torch.Tensor,   # [B, 64, 768] - Target velocity
        timesteps: torch.Tensor,         # [B] - Timesteps
    ) -> Dict[str, float]:
        """
        Compute detailed metrics for monitoring training.
        
        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            # Basic loss components
            mse_loss = F.mse_loss(model_output, velocity_target).item()
            
            # Cosine similarity between prediction and target
            pred_flat = model_output.flatten(1)
            target_flat = velocity_target.flatten(1)
            cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item()
            
            # L2 norms
            pred_norm = torch.norm(model_output, dim=-1).mean().item()
            target_norm = torch.norm(velocity_target, dim=-1).mean().item()
            data_norm = torch.norm(target_samples, dim=-1).mean().item()
            
            # Timestep statistics
            t_mean = timesteps.mean().item()
            t_std = timesteps.std().item()
            
            # Signal-to-noise ratio approximation
            signal_power = torch.var(target_samples).item()
            noise_power = torch.var(model_output - velocity_target).item()
            snr = 10 * math.log10(signal_power / (noise_power + 1e-8))
            
            return {
                "mse_loss": mse_loss,
                "cosine_similarity": cosine_sim,
                "pred_norm": pred_norm,
                "target_norm": target_norm,
                "data_norm": data_norm,
                "timestep_mean": t_mean,
                "timestep_std": t_std,
                "snr_db": snr,
            }
    
    def forward(
        self,
        model_output: torch.Tensor,           # [B, 64, 768] - Model predictions
        target_samples: torch.Tensor,         # [B, 64, 768] - CLIP targets
        timesteps: torch.Tensor,              # [B] - Timesteps
        eva_conditioning: Optional[torch.Tensor] = None,  # [B, 64, 1280] - EVA conditioning
        noise: Optional[torch.Tensor] = None, # [B, 64, 768] - Noise
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute BLIP3-o flow matching loss with optional detailed metrics.
        
        Returns:
            loss: Total loss tensor
            metrics: Optional dictionary of detailed metrics
        """
        # Validate input dimensions
        assert target_samples.shape[-1] == self.clip_dim, f"Expected CLIP dim {self.clip_dim}, got {target_samples.shape[-1]}"
        if eva_conditioning is not None:
            assert eva_conditioning.shape[-1] == self.eva_dim, f"Expected EVA dim {self.eva_dim}, got {eva_conditioning.shape[-1]}"
        
        # Compute main flow matching loss
        flow_loss = super().forward(
            model_output=model_output,
            target_samples=target_samples,
            timesteps=timesteps,
            noise=noise,
            reduction="mean"
        )
        
        # Compute regularization loss
        reg_loss = torch.tensor(0.0, device=model_output.device, dtype=model_output.dtype)
        if eva_conditioning is not None:
            reg_loss = self.compute_regularization_loss(model_output, eva_conditioning)
        
        # Total loss
        total_loss = flow_loss + reg_loss
        
        # Compute detailed metrics if requested
        metrics = None
        if return_metrics:
            # Recompute velocity target for metrics
            x_0 = torch.randn_like(target_samples)
            if noise is None:
                noise = torch.randn_like(target_samples)
            velocity_target = self.compute_velocity_target(x_0, target_samples, timesteps, noise)
            
            metrics = self.compute_detailed_metrics(model_output, target_samples, velocity_target, timesteps)
            metrics.update({
                "flow_matching_loss": flow_loss.item(),
                "regularization_loss": reg_loss.item(),
                "total_loss": total_loss.item(),
            })
        
        return total_loss, metrics


def create_blip3o_flow_matching_loss(
    config: Optional[FlowMatchingConfig] = None,
    **kwargs
) -> BLIP3oFlowMatchingLoss:
    """
    Factory function to create BLIP3-o flow matching loss.
    
    Args:
        config: Flow matching configuration
        **kwargs: Additional parameters to override config
        
    Returns:
        BLIP3oFlowMatchingLoss instance
    """
    if config is None:
        from ..config.blip3o_config import get_default_flow_matching_config
        config = get_default_flow_matching_config()
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return BLIP3oFlowMatchingLoss(config=config)