"""
FIXED Global Flow Matching Loss - Direct Global Training
Place this file as: src/modules/losses/global_flow_matching_loss.py

KEY FIXES:
1. Simplified to single global objective
2. Proper target computation with frozen CLIP projection
3. Enhanced metrics for monitoring
4. Contrastive loss option for better alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
from transformers import CLIPModel


class GlobalFlowMatchingLoss(nn.Module):
    """
    FIXED Global Flow Matching Loss for Direct Global Training
    
    Trains directly on [B, 768] global embeddings to eliminate training-inference mismatch
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
        prediction_type: str = "v_prediction",
        schedule_type: str = "linear",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        use_contrastive_loss: bool = True,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prediction_type = prediction_type
        self.schedule_type = schedule_type
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        # Load CLIP for target computation
        self._load_clip_model(clip_model_name)
        
        # EMA metrics for monitoring
        self.register_buffer('ema_cosine', torch.tensor(0.0))
        self.register_buffer('ema_l2', torch.tensor(0.0))
        self.register_buffer('ema_contrastive', torch.tensor(0.0))
        self.ema_decay = 0.99
        
        print(f"✅ FIXED Global Flow Matching Loss initialized")
        print(f"   Target: [B, 768] global embeddings")
        print(f"   Objective: Single global flow matching")
        print(f"   Contrastive loss: {use_contrastive_loss}")
    
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
            # Create fallback projection
            self.clip_visual_proj = nn.Linear(1024, 768, bias=False)
            self.clip_visual_proj.requires_grad_(False)
            nn.init.xavier_uniform_(self.clip_visual_proj.weight)
    
    def compute_target_global_features(self, clip_patches):
        """
        Compute target global features from CLIP patches
        
        Args:
            clip_patches: [B, 256, 1024] CLIP patch embeddings
            
        Returns:
            target_global: [B, 768] target global embeddings
        """
        with torch.no_grad():
            # Pool patches to global representation
            pooled = clip_patches.mean(dim=1)  # [B, 1024]
            
            # Ensure same device
            if self.clip_visual_proj.weight.device != pooled.device:
                self.clip_visual_proj = self.clip_visual_proj.to(pooled.device)
            
            # Apply CLIP projection to get [B, 768]
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
        elif self.schedule_type == "sigmoid":
            alpha_t = torch.sigmoid(10 * (t - 0.5))
            sigma_t = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(-10 * (t - 0.5))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")
        
        return alpha_t, sigma_t
    
    def interpolate_global_data(self, x_0, x_1, t, noise=None):
        """
        Flow matching interpolation for global features
        
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
            elif self.schedule_type == "sigmoid":
                dsigma_dt = -(self.sigma_max - self.sigma_min) * 10 * torch.sigmoid(-10 * (t - 0.5)) * (1 - torch.sigmoid(-10 * (t - 0.5)))
                dsigma_dt = dsigma_dt.view(-1, 1)
            
            velocity_target = x_1 - x_0 - dsigma_dt * noise
        elif self.prediction_type == "epsilon":
            velocity_target = noise
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        return velocity_target
    
    def compute_contrastive_loss(self, predicted_global, target_global):
        """
        Compute contrastive loss for better alignment
        
        Args:
            predicted_global: [B, 768] predicted features
            target_global: [B, 768] target features
            
        Returns:
            contrastive_loss: scalar loss
        """
        # Normalize features
        pred_norm = F.normalize(predicted_global, p=2, dim=-1)
        target_norm = F.normalize(target_global, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(pred_norm, target_norm.t()) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = predicted_global.shape[0]
        labels = torch.arange(batch_size, device=predicted_global.device)
        
        # Symmetric contrastive loss
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.t(), labels)
        contrastive_loss = (loss_i + loss_j) / 2
        
        return contrastive_loss
    
    def forward(
        self,
        predicted_global,    # [B, 768] - Model predictions
        clip_patches,        # [B, 256, 1024] - CLIP patch targets
        timesteps,          # [B] - Timesteps
        noise=None,         # [B, 768] - Noise
        return_metrics=False,
    ):
        """
        Compute global flow matching loss
        
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
        
        # Primary flow matching loss
        flow_loss = F.mse_loss(predicted_global, velocity_target)
        
        # Add contrastive loss if enabled
        contrastive_loss = torch.tensor(0.0, device=predicted_global.device)
        if self.use_contrastive_loss:
            contrastive_loss = self.compute_contrastive_loss(predicted_global, target_global)
        
        # Combined loss
        total_loss = flow_loss + self.contrastive_weight * contrastive_loss
        
        # Compute metrics
        metrics = None
        if return_metrics:
            with torch.no_grad():
                # Flow matching metrics
                cosine_sim = F.cosine_similarity(
                    F.normalize(predicted_global, dim=-1),
                    F.normalize(velocity_target, dim=-1),
                    dim=-1
                ).mean().item()
                
                l2_dist = torch.norm(predicted_global - velocity_target, dim=-1).mean().item()
                
                # Direct target comparison (most important)
                direct_cosine = F.cosine_similarity(
                    F.normalize(predicted_global, dim=-1),
                    F.normalize(target_global, dim=-1),
                    dim=-1
                ).mean().item()
                
                # Update EMA metrics
                self.ema_cosine = self.ema_decay * self.ema_cosine + (1 - self.ema_decay) * direct_cosine
                self.ema_l2 = self.ema_decay * self.ema_l2 + (1 - self.ema_decay) * l2_dist
                if self.use_contrastive_loss:
                    self.ema_contrastive = self.ema_decay * self.ema_contrastive + (1 - self.ema_decay) * contrastive_loss.item()
                
                # Quality metrics
                high_quality_ratio = (F.cosine_similarity(
                    F.normalize(predicted_global, dim=-1),
                    F.normalize(target_global, dim=-1),
                    dim=-1
                ) > 0.8).float().mean().item()
                
                excellent_quality_ratio = (F.cosine_similarity(
                    F.normalize(predicted_global, dim=-1),
                    F.normalize(target_global, dim=-1),
                    dim=-1
                ) > 0.9).float().mean().item()
                
                # Diversity metrics
                pred_std = torch.std(predicted_global, dim=0).mean().item()
                target_std = torch.std(target_global, dim=0).mean().item()
                
                metrics = {
                    # Primary metrics
                    'global_flow_loss': flow_loss.item(),
                    'contrastive_loss': contrastive_loss.item(),
                    'total_loss': total_loss.item(),
                    
                    # Flow matching metrics
                    'flow_cosine_similarity': cosine_sim,
                    'flow_l2_distance': l2_dist,
                    
                    # Target alignment metrics (most important)
                    'direct_global_cosine': direct_cosine,
                    'high_quality_ratio': high_quality_ratio,
                    'excellent_quality_ratio': excellent_quality_ratio,
                    
                    # EMA metrics
                    'ema_global_cosine': self.ema_cosine.item(),
                    'ema_global_l2': self.ema_l2.item(),
                    'ema_contrastive': self.ema_contrastive.item() if self.use_contrastive_loss else 0.0,
                    
                    # Performance predictions
                    'expected_recall_percent': min(direct_cosine * 70, 70),
                    'convergence_indicator': direct_cosine,
                    'training_success_indicator': direct_cosine > 0.7,
                    
                    # Normalization and diversity
                    'pred_norm_mean': torch.norm(predicted_global, dim=-1).mean().item(),
                    'target_norm_mean': torch.norm(target_global, dim=-1).mean().item(),
                    'prediction_diversity': pred_std,
                    'target_diversity': target_std,
                    
                    # Training diagnostics
                    'timestep_mean': timesteps.mean().item(),
                    'timestep_std': timesteps.std().item(),
                    'noise_level': torch.norm(noise, dim=-1).mean().item() if noise is not None else 0.0,
                    
                    # Quality assessment
                    'training_quality': (
                        'excellent' if direct_cosine > 0.85 else
                        'good' if direct_cosine > 0.7 else
                        'fair' if direct_cosine > 0.5 else
                        'needs_improvement'
                    ),
                }
        
        return total_loss, metrics


class EnhancedGlobalFlowMatchingLoss(GlobalFlowMatchingLoss):
    """Enhanced version with additional regularization and stability improvements"""
    
    def __init__(self, *args, **kwargs):
        # Extract additional parameters
        self.gradient_penalty_weight = kwargs.pop('gradient_penalty_weight', 0.01)
        self.feature_matching_weight = kwargs.pop('feature_matching_weight', 0.05)
        
        super().__init__(*args, **kwargs)
        
        print(f"✅ Enhanced Global Flow Matching Loss initialized")
        print(f"   Gradient penalty: {self.gradient_penalty_weight}")
        print(f"   Feature matching: {self.feature_matching_weight}")
    
    def compute_gradient_penalty(self, predicted_global, target_global):
        """Compute gradient penalty for stability"""
        # Create interpolation
        alpha = torch.rand(predicted_global.shape[0], 1, device=predicted_global.device)
        interpolated = alpha * target_global + (1 - alpha) * predicted_global
        interpolated.requires_grad_(True)
        
        # Compute gradients
        grad_outputs = torch.ones_like(interpolated)
        gradients = torch.autograd.grad(
            outputs=interpolated.sum(),
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute penalty
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def compute_feature_matching_loss(self, predicted_global, target_global):
        """Compute feature matching loss for better distribution alignment"""
        # Compute feature statistics
        pred_mean = predicted_global.mean(dim=0)
        target_mean = target_global.mean(dim=0)
        
        pred_std = predicted_global.std(dim=0)
        target_std = target_global.std(dim=0)
        
        # Feature matching loss
        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)
        
        return mean_loss + std_loss
    
    def forward(self, predicted_global, clip_patches, timesteps, noise=None, return_metrics=False):
        """Enhanced forward pass with additional regularization"""
        # Get base loss and metrics
        base_loss, metrics = super().forward(
            predicted_global, clip_patches, timesteps, noise, return_metrics
        )
        
        # Compute target for additional losses
        target_global = self.compute_target_global_features(clip_patches)
        
        # Additional regularization losses
        gradient_penalty = torch.tensor(0.0, device=predicted_global.device)
        feature_matching_loss = torch.tensor(0.0, device=predicted_global.device)
        
        if self.gradient_penalty_weight > 0:
            gradient_penalty = self.compute_gradient_penalty(predicted_global, target_global)
        
        if self.feature_matching_weight > 0:
            feature_matching_loss = self.compute_feature_matching_loss(predicted_global, target_global)
        
        # Total enhanced loss
        enhanced_loss = (
            base_loss + 
            self.gradient_penalty_weight * gradient_penalty +
            self.feature_matching_weight * feature_matching_loss
        )
        
        # Add to metrics if requested
        if return_metrics and metrics is not None:
            metrics.update({
                'gradient_penalty': gradient_penalty.item(),
                'feature_matching_loss': feature_matching_loss.item(),
                'base_loss': base_loss.item(),
                'enhanced_total_loss': enhanced_loss.item(),
            })
        
        return enhanced_loss, metrics


def create_global_flow_matching_loss(
    enhanced: bool = False,
    sigma_min: float = 1e-4,
    sigma_max: float = 1.0,
    prediction_type: str = "v_prediction",
    schedule_type: str = "linear",
    clip_model_name: str = "openai/clip-vit-large-patch14",
    use_contrastive_loss: bool = True,
    contrastive_weight: float = 0.1,
    temperature: float = 0.07,
    **kwargs
):
    """Factory function for global flow matching loss"""
    if enhanced:
        return EnhancedGlobalFlowMatchingLoss(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            prediction_type=prediction_type,
            schedule_type=schedule_type,
            clip_model_name=clip_model_name,
            use_contrastive_loss=use_contrastive_loss,
            contrastive_weight=contrastive_weight,
            temperature=temperature,
            **kwargs
        )
    else:
        return GlobalFlowMatchingLoss(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            prediction_type=prediction_type,
            schedule_type=schedule_type,
            clip_model_name=clip_model_name,
            use_contrastive_loss=use_contrastive_loss,
            contrastive_weight=contrastive_weight,
            temperature=temperature,
        )