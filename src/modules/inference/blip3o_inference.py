"""
FIXED: Dual Supervision BLIP3-o Inference Module
Updated for the new dual supervision architecture with global flow matching
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DualSupervisionBLIP3oInference:
    """
    FIXED: Inference pipeline for dual supervision BLIP3-o DiT model.
    
    Supports the new architecture with:
    - Dual supervision (patch + global)
    - Global flow matching
    - Multiple generation modes
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ):
        """Initialize dual supervision inference pipeline."""
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        self.compile_model = compile_model
        
        # Load model and configuration
        self.model, self.config = self._load_model()
        self.flow_matching_loss = self._load_flow_matching_config()
        
        # Check model capabilities
        self.model_capabilities = self._check_model_capabilities()
        
        logger.info(f"Dual supervision BLIP3-o inference pipeline initialized")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model capabilities: {self.model_capabilities}")
        
        if hasattr(self.model, 'get_num_parameters'):
            logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
    
    def _setup_device(self, device_arg: str) -> torch.device:
        """Setup computation device."""
        if device_arg == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(device_arg)
            logger.info(f"Using device: {device}")
        
        return device
    
    def _load_model(self) -> Tuple[nn.Module, Any]:
        """Load trained dual supervision BLIP3-o model."""
        # Load model configuration
        config_path = self.model_path / "config.json"
        blip3o_config_path = self.model_path / "blip3o_model_config.json"
        
        if blip3o_config_path.exists():
            config_file = blip3o_config_path
        elif config_path.exists():
            config_file = config_path
        else:
            raise FileNotFoundError(f"No config file found in {self.model_path}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Import config class
        from ..config.blip3o_config import BLIP3oDiTConfig
        config = BLIP3oDiTConfig(**config_dict)
        
        # Try to import and create dual supervision model
        try:
            from ..models.dual_supervision_blip3o_dit import create_blip3o_dit_model
            model = create_blip3o_dit_model(
                config=config,
                load_clip_projection=True,
                enable_dual_supervision=True,
            )
            logger.info("✅ Using dual supervision model")
        except ImportError as e:
            logger.error(f"Failed to import dual supervision model: {e}")
            raise ImportError("Dual supervision model required but not available")
        
        # Load model weights
        model_files = [
            self.model_path / "model.safetensors",
            self.model_path / "pytorch_model.bin", 
            self.model_path / "pytorch_model.safetensors"
        ]
        
        model_file = None
        for file_path in model_files:
            if file_path.exists():
                model_file = file_path
                break
        
        if model_file is None:
            raise FileNotFoundError(f"No model weights found in {self.model_path}")
        
        logger.info(f"Loading weights from: {model_file}")
        
        # Load weights
        if model_file.suffix == ".bin":
            state_dict = torch.load(model_file, map_location=self.device)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_file))
        
        # Check for dual supervision keys
        dual_keys = [k for k in state_dict.keys() if 'global_velocity_proj' in k]
        if dual_keys:
            logger.info(f"✅ Found dual supervision keys: {dual_keys}")
        else:
            logger.warning("❌ No dual supervision keys found - model may not be trained with dual supervision")
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
        
        # Move to device and set data type
        model = model.to(device=self.device, dtype=self.torch_dtype)
        model.eval()
        
        # Compile model if requested
        if self.compile_model:
            try:
                model = torch.compile(model)
                logger.info("Model compiled for optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        logger.info(f"✅ Dual supervision model loaded successfully")
        
        return model, config
    
    def _load_flow_matching_config(self):
        """Load dual supervision flow matching configuration."""
        fm_config_path = self.model_path / "flow_matching_config.json"
        
        if fm_config_path.exists():
            with open(fm_config_path, 'r') as f:
                fm_config_dict = json.load(f)
            logger.info("Flow matching configuration loaded")
        else:
            logger.warning("Flow matching config not found, using default")
            fm_config_dict = {}
        
        try:
            from ..losses.dual_supervision_flow_matching_loss import create_dual_supervision_loss
            flow_matching_loss = create_dual_supervision_loss(**fm_config_dict)
            logger.info("✅ Using dual supervision flow matching loss")
        except ImportError:
            from ..losses.flow_matching_loss import create_blip3o_flow_matching_loss
            flow_matching_loss = create_blip3o_flow_matching_loss(**fm_config_dict)
            logger.warning("⚠️ Using standard flow matching loss (fallback)")
        
        return flow_matching_loss
    
    def _check_model_capabilities(self) -> Dict[str, bool]:
        """Check dual supervision model capabilities."""
        capabilities = {
            'has_frozen_clip_proj': hasattr(self.model, 'frozen_clip_visual_proj') and self.model.frozen_clip_visual_proj is not None,
            'has_global_adaptation_mlp': hasattr(self.model, 'global_adaptation_mlp'),
            'has_global_velocity_proj': hasattr(self.model, 'global_velocity_proj'),
            'supports_generation_modes': hasattr(self.model, 'generate'),
            'supports_training_modes': hasattr(self.model, '__call__') and 'training_mode' in str(self.model.forward.__code__.co_varnames),
            'is_dual_supervision_model': hasattr(self.model, 'global_velocity_proj'),
            'supports_global_generation': hasattr(self.model, 'generate') and hasattr(self.model, 'global_velocity_proj'),
        }
        
        return capabilities
    
    @torch.no_grad()
    def generate(
        self,
        eva_embeddings: torch.Tensor,
        num_inference_steps: int = 50,
        generation_mode: str = "auto",
        return_global_only: bool = True,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """
        Generate CLIP embeddings using dual supervision model.
        
        Args:
            eva_embeddings: EVA-CLIP conditioning [batch_size, 256, 4096]
            num_inference_steps: Number of sampling steps
            generation_mode: "auto", "global", "patch", or "dual"
            return_global_only: Whether to return global embeddings
            guidance_scale: Guidance scale (not used in current implementation)
            generator: Random number generator
            return_intermediate: Whether to return intermediate states
            
        Returns:
            Generated CLIP embeddings
        """
        # Validate input
        self._validate_eva_input(eva_embeddings)
        
        # Move to correct device and dtype
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Determine generation mode
        if generation_mode == "auto":
            if self.model_capabilities.get('supports_global_generation', False):
                generation_mode = "global"
                logger.info("🎯 Auto-selected GLOBAL generation mode (dual supervision)")
            else:
                generation_mode = "patch"
                logger.info("🔄 Auto-selected PATCH generation mode (fallback)")
        
        logger.info(f"Using generation mode: {generation_mode}")
        
        # Generate using the appropriate method
        if (generation_mode == "global" and 
            self.model_capabilities.get('supports_global_generation', False)):
            
            # Use global generation (preferred for recall)
            try:
                generated = self.model.generate(
                    encoder_hidden_states=eva_embeddings,
                    num_inference_steps=num_inference_steps,
                    generation_mode="global",
                    return_global_only=True,
                    generator=generator,
                    return_intermediate=return_intermediate,
                )
                logger.debug("✅ Used global generation mode")
                
                if return_intermediate:
                    return generated[0], generated[1]
                else:
                    return generated
                    
            except Exception as e:
                logger.warning(f"Global generation failed: {e}, falling back to dual mode")
                generation_mode = "dual"
        
        if generation_mode == "dual" and hasattr(self.model, 'generate'):
            # Use dual generation mode
            try:
                generated = self.model.generate(
                    encoder_hidden_states=eva_embeddings,
                    num_inference_steps=num_inference_steps,
                    generation_mode="dual",
                    generator=generator,
                    return_intermediate=return_intermediate,
                )
                
                if isinstance(generated, dict):
                    # Extract the appropriate output
                    if return_global_only and 'global_generation' in generated:
                        result = generated['global_generation']
                    elif 'patch_generation' in generated:
                        result = generated['patch_generation']
                    else:
                        result = list(generated.values())[0]
                else:
                    result = generated
                
                logger.debug("✅ Used dual generation mode")
                return result
                
            except Exception as e:
                logger.warning(f"Dual generation failed: {e}, falling back to standard")
                generation_mode = "patch"
        
        # Fallback to standard generation
        if hasattr(self.model, 'generate'):
            try:
                generated = self.model.generate(
                    encoder_hidden_states=eva_embeddings,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    return_intermediate=return_intermediate,
                )
                
                # Handle return format
                if return_intermediate:
                    generated_clip, intermediate = generated
                else:
                    generated_clip = generated
                    intermediate = None
                
                # Convert to global if needed and possible
                if (return_global_only and 
                    generated_clip.dim() == 3 and 
                    generated_clip.shape[1] == 256):
                    
                    # Pool patch outputs to global
                    generated_clip = generated_clip.mean(dim=1)  # [B, 1024]
                    
                    # Apply CLIP projection if available
                    if self.model_capabilities.get('has_frozen_clip_proj', False):
                        generated_clip = self.model.frozen_clip_visual_proj(generated_clip)
                        logger.debug("Applied CLIP projection")
                
                logger.debug("✅ Used standard generation with post-processing")
                
                if return_intermediate:
                    return generated_clip, intermediate
                else:
                    return generated_clip
                    
            except Exception as e:
                logger.error(f"All generation methods failed: {e}")
                raise
        else:
            raise RuntimeError("Model does not support generation")
    
    def _validate_eva_input(self, eva_embeddings: torch.Tensor):
        """Validate EVA-CLIP input tensor."""
        if eva_embeddings.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, num_tokens, eva_dim], got {eva_embeddings.dim()}D")
        
        # Support 256 tokens (16x16)
        expected_tokens = self.config.input_size * self.config.input_size
        if eva_embeddings.shape[1] != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} tokens, got {eva_embeddings.shape[1]}")
        
        if eva_embeddings.shape[2] != self.config.eva_embedding_size:
            raise ValueError(f"Expected {self.config.eva_embedding_size}-dim EVA features, got {eva_embeddings.shape[2]}")
    
    def forward_with_loss(
        self,
        eva_embeddings: torch.Tensor,
        clip_targets: torch.Tensor,
        training_mode: str = "dual_supervision",
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with loss computation (for evaluation).
        
        Args:
            eva_embeddings: EVA-CLIP conditioning [B, 256, 4096]
            clip_targets: Target CLIP embeddings [B, 256, 1024]
            training_mode: "dual_supervision", "dual_flow", "global_generation"
            return_metrics: Whether to return detailed metrics
            
        Returns:
            loss, metrics
        """
        batch_size = eva_embeddings.shape[0]
        device = eva_embeddings.device
        
        # Sample timesteps and noise for flow matching
        if hasattr(self.flow_matching_loss, 'sample_timesteps'):
            timesteps = self.flow_matching_loss.sample_timesteps(batch_size, device)
        else:
            timesteps = torch.rand(batch_size, device=device)
        
        # Create noisy input for flow matching
        noise = torch.randn_like(clip_targets)
        x_0 = torch.randn_like(clip_targets)
        
        if hasattr(self.flow_matching_loss, 'interpolate_data'):
            noisy_clip = self.flow_matching_loss.interpolate_data(
                x_0=x_0, x_1=clip_targets, t=timesteps, noise=noise
            )
        else:
            # Simple linear interpolation fallback
            alpha = timesteps.view(-1, 1, 1)
            noisy_clip = (1 - alpha) * x_0 + alpha * clip_targets + 0.1 * noise
        
        # Forward pass
        if self.model_capabilities.get('supports_training_modes', False):
            outputs = self.model(
                hidden_states=noisy_clip,
                timestep=timesteps,
                encoder_hidden_states=eva_embeddings,
                training_mode=training_mode,
                return_dict=True
            )
        else:
            outputs = self.model(
                hidden_states=noisy_clip,
                timestep=timesteps,
                encoder_hidden_states=eva_embeddings,
                return_dict=True
            )
        
        # Compute loss using dual supervision loss if available
        if hasattr(self.flow_matching_loss, '__call__') and len(self.flow_matching_loss.__class__.__name__) > 20:
            # This is likely the dual supervision loss
            
            # Compute target global features
            if hasattr(self.model, 'frozen_clip_visual_proj') and self.model.frozen_clip_visual_proj is not None:
                with torch.no_grad():
                    pooled_clip = clip_targets.mean(dim=1)  # [B, 1024]
                    target_global = self.model.frozen_clip_visual_proj(pooled_clip)  # [B, 768]
            else:
                target_global = clip_targets.mean(dim=1)  # Fallback
            
            # Use dual supervision loss
            loss, metrics = self.flow_matching_loss(
                dit_patch_output=outputs.get('patch_output', outputs.get('patch_velocity', outputs.get('last_hidden_state'))),
                dit_global_output=outputs.get('global_output', outputs.get('global_velocity')),
                clip_patches=clip_targets,
                clip_global=target_global,
                timesteps=timesteps,
                eva_conditioning=eva_embeddings,
                noise=noise,
                return_metrics=return_metrics,
            )
        else:
            # Standard flow matching loss
            model_output = outputs.get('patch_output', outputs.get('last_hidden_state'))
            loss, metrics = self.flow_matching_loss(
                model_output=model_output,
                target_samples=clip_targets,
                timesteps=timesteps,
                eva_conditioning=eva_embeddings,
                noise=noise,
                return_metrics=return_metrics,
            )
        
        return loss, metrics if return_metrics else None
    
    def evaluate_on_batch(
        self,
        eva_embeddings: torch.Tensor,
        clip_targets: torch.Tensor,
        num_inference_steps: int = 50,
        generation_mode: str = "auto",
    ) -> Dict[str, float]:
        """
        Evaluate model on a batch of samples.
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # 1. Compute loss
        try:
            loss, loss_metrics = self.forward_with_loss(
                eva_embeddings, clip_targets, return_metrics=True
            )
            metrics['loss'] = loss.item()
            if loss_metrics:
                for key, value in loss_metrics.items():
                    metrics[f'loss_{key}'] = value
        except Exception as e:
            logger.warning(f"Loss computation failed: {e}")
            metrics['loss'] = float('inf')
        
        # 2. Generate embeddings
        try:
            generated = self.generate(
                eva_embeddings=eva_embeddings,
                num_inference_steps=num_inference_steps,
                generation_mode=generation_mode,
                return_global_only=True,
            )
            
            # Compute generation quality metrics
            with torch.no_grad():
                # Handle different target formats
                if clip_targets.dim() == 3 and clip_targets.shape[1] == 256:
                    # Convert patch targets to global for comparison
                    clip_global = clip_targets.mean(dim=1)  # [B, 1024]
                    if hasattr(self.model, 'frozen_clip_visual_proj') and self.model.frozen_clip_visual_proj is not None:
                        clip_global = self.model.frozen_clip_visual_proj(clip_global)  # [B, 768]
                else:
                    clip_global = clip_targets
                
                # Ensure same dimensions
                if generated.shape != clip_global.shape:
                    logger.warning(f"Shape mismatch: generated {generated.shape} vs target {clip_global.shape}")
                    if generated.shape[-1] != clip_global.shape[-1]:
                        # Skip comparison if dimensions don't match
                        metrics['generation_error'] = 'dimension_mismatch'
                        return metrics
                
                # Normalize for comparison
                generated_norm = torch.nn.functional.normalize(generated, p=2, dim=-1)
                target_norm = torch.nn.functional.normalize(clip_global, p=2, dim=-1)
                
                # Compute metrics
                cosine_sim = torch.nn.functional.cosine_similarity(generated_norm, target_norm, dim=-1)
                l2_dist = torch.norm(generated - clip_global, dim=-1)
                
                metrics.update({
                    'generation_cosine_similarity': cosine_sim.mean().item(),
                    'generation_cosine_std': cosine_sim.std().item(),
                    'generation_l2_distance': l2_dist.mean().item(),
                    'generation_norm': torch.norm(generated, dim=-1).mean().item(),
                    'target_norm': torch.norm(clip_global, dim=-1).mean().item(),
                })
                
        except Exception as e:
            logger.warning(f"Generation evaluation failed: {e}")
            metrics['generation_error'] = str(e)
        
        return metrics


def load_dual_supervision_blip3o_inference(
    model_path: Union[str, Path],
    device: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> DualSupervisionBLIP3oInference:
    """
    Load dual supervision BLIP3-o inference pipeline.
    
    Args:
        model_path: Path to trained dual supervision model
        device: Device to use
        torch_dtype: Data type for model
        
    Returns:
        DualSupervisionBLIP3oInference instance
    """
    return DualSupervisionBLIP3oInference(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )