"""
Inference utilities for BLIP3-o DiT model.
Handles loading trained models and generating CLIP embeddings from EVA-CLIP conditioning.
FIXED: Parameter handling in generate method for better compatibility
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

from ..models.blip3o_dit import BLIP3oDiTModel
from ..config.blip3o_config import BLIP3oDiTConfig
from ..losses.dual_supervision_flow_matching_loss import BLIP3oFlowMatchingLoss, create_blip3o_flow_matching_loss
from ..datasets.blip3o_dataset import BLIP3oEmbeddingDataset, create_blip3o_dataloader

logger = logging.getLogger(__name__)


class BLIP3oInference:
    """
    Inference pipeline for BLIP3-o DiT model.
    
    Provides methods for:
    - Loading trained BLIP3-o models
    - Generating CLIP embeddings from EVA-CLIP conditioning
    - Batch inference and evaluation
    - Quality metrics computation
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        compile_model: bool = False,
    ):
        """
        Initialize BLIP3-o inference pipeline.
        
        Args:
            model_path: Path to trained model directory
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Data type for model (None for auto)
            compile_model: Whether to compile model for optimization
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        self.compile_model = compile_model
        
        # Load model and configuration
        self.model, self.config = self._load_model()
        self.flow_matching_loss = self._load_flow_matching_config()
        
        logger.info(f"BLIP3-o inference pipeline initialized")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Device: {self.device}")
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
    
    def _load_model(self) -> Tuple[BLIP3oDiTModel, BLIP3oDiTConfig]:
        """Load trained BLIP3-o model and configuration."""
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
        
        # Create configuration
        config = BLIP3oDiTConfig(**config_dict)
        
        # Create model
        model = BLIP3oDiTModel(config)
        
        # Load model weights
        model_files = [
            self.model_path / "pytorch_model.bin",
            self.model_path / "model.safetensors",
            self.model_path / "pytorch_model.safetensors"
        ]
        
        model_file = None
        for file_path in model_files:
            if file_path.exists():
                model_file = file_path
                break
        
        if model_file is None:
            raise FileNotFoundError(f"No model weights found in {self.model_path}")
        
        # Load weights
        if model_file.suffix == ".bin":
            state_dict = torch.load(model_file, map_location=self.device)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_file))
        
        # Handle potential key mismatches
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    logger.warning(f"Shape mismatch for {key}: {model_state_dict[key].shape} vs {value.shape}")
            else:
                logger.warning(f"Unexpected key in state dict: {key}")
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
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
        
        logger.info(f"Model loaded successfully from {model_file}")
        
        return model, config
    
    def _load_flow_matching_config(self) -> Optional[BLIP3oFlowMatchingLoss]:
        """Load flow matching configuration if available."""
        fm_config_path = self.model_path / "flow_matching_config.json"
        
        if not fm_config_path.exists():
            logger.warning("Flow matching config not found, using default")
            return create_blip3o_flow_matching_loss()
        
        with open(fm_config_path, 'r') as f:
            fm_config_dict = json.load(f)
        
        flow_matching_loss = create_blip3o_flow_matching_loss(**fm_config_dict)
        logger.info("Flow matching configuration loaded")
        
        return flow_matching_loss
    
    @torch.no_grad()
    def generate(
        self,
        eva_embeddings: torch.Tensor,  # FIXED: Clear parameter name
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        eta: float = 0.0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate CLIP embeddings from EVA-CLIP conditioning.
        FIXED: Clear parameter naming to avoid confusion with model.generate()
        
        Args:
            eva_embeddings: EVA-CLIP conditioning [batch_size, num_tokens, eva_dim]
            num_inference_steps: Number of sampling steps
            guidance_scale: Guidance scale (currently not used)
            generator: Random number generator for reproducibility
            return_intermediate: Whether to return intermediate states
            eta: DDIM parameter for stochasticity
            
        Returns:
            Generated CLIP embeddings [batch_size, num_tokens, clip_dim]
            Optionally with intermediate states if return_intermediate=True
        """
        # Validate input
        self._validate_eva_input(eva_embeddings)
        
        # Move to correct device and dtype
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate using model's built-in generation method
        # FIXED: Use correct parameter name for the underlying model
        if return_intermediate:
            generated_clip, intermediate_states = self.model.generate(
                encoder_hidden_states=eva_embeddings,  # Model expects this parameter name
                num_inference_steps=num_inference_steps,
                generator=generator,
                return_intermediate=True,
            )
            return generated_clip, intermediate_states
        else:
            generated_clip = self.model.generate(
                encoder_hidden_states=eva_embeddings,  # Model expects this parameter name
                num_inference_steps=num_inference_steps,
                generator=generator,
                return_intermediate=False,
            )
            return generated_clip
    
    def _validate_eva_input(self, eva_embeddings: torch.Tensor):
        """Validate EVA-CLIP input tensor - FIXED: Support 256 tokens."""
        if eva_embeddings.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, num_tokens, eva_dim], got {eva_embeddings.dim()}D")
        
        # FIXED: Support 256 tokens instead of hardcoded 64
        expected_tokens = self.config.input_size * self.config.input_size  # Should be 256
        if eva_embeddings.shape[1] != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} tokens, got {eva_embeddings.shape[1]}")
        
        if eva_embeddings.shape[2] != self.config.eva_embedding_size:
            raise ValueError(f"Expected {self.config.eva_embedding_size}-dim EVA features, got {eva_embeddings.shape[2]}")
    
    def generate_from_dataset(
        self,
        dataset_path: Union[str, Path],
        num_samples: Optional[int] = None,
        batch_size: int = 8,
        num_inference_steps: int = 50,
        output_path: Optional[Union[str, Path]] = None,
        compute_metrics: bool = True,
        save_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate CLIP embeddings for samples from a dataset.
        
        Args:
            dataset_path: Path to embeddings dataset
            num_samples: Number of samples to generate (None for all)
            batch_size: Batch size for generation
            num_inference_steps: Number of sampling steps
            output_path: Path to save results
            compute_metrics: Whether to compute quality metrics
            save_intermediate: Whether to save intermediate states
            
        Returns:
            Dictionary with generation results and metrics
        """
        # Load dataset
        dataset = BLIP3oEmbeddingDataset(
            embeddings_path=dataset_path,
            subset_size=num_samples,
            normalize_embeddings=True,
            split="all"
        )
        
        # Create dataloader
        dataloader = create_blip3o_dataloader(
            embeddings_path=dataset_path,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing in inference
            split="all",
            subset_size=num_samples,
        )
        
        logger.info(f"Generating samples for {len(dataset)} items in {len(dataloader)} batches")
        
        # Storage for results
        results = {
            'eva_embeddings': [],
            'clip_targets': [],
            'generated_clip': [],
            'captions': [],
            'keys': [],
            'generation_metrics': {},
            'intermediate_states': [] if save_intermediate else None,
        }
        
        # Generate samples
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
            eva_emb = batch['eva_embeddings'].to(device=self.device, dtype=self.torch_dtype)
            clip_targets = batch['clip_embeddings']
            
            # Generate CLIP embeddings - FIXED: Use correct method signature
            if save_intermediate:
                generated_clip, intermediate = self.generate(
                    eva_emb,  # FIXED: Use positional argument
                    num_inference_steps=num_inference_steps,
                    return_intermediate=True,
                )
                results['intermediate_states'].append([state.cpu() for state in intermediate])
            else:
                generated_clip = self.generate(
                    eva_emb,  # FIXED: Use positional argument
                    num_inference_steps=num_inference_steps,
                )
            
            # Store results
            results['eva_embeddings'].append(eva_emb.cpu())
            results['clip_targets'].append(clip_targets)
            results['generated_clip'].append(generated_clip.cpu())
            results['captions'].extend(batch['captions'])
            results['keys'].extend(batch['keys'])
        
        # Concatenate results
        results['eva_embeddings'] = torch.cat(results['eva_embeddings'], dim=0)
        results['clip_targets'] = torch.cat(results['clip_targets'], dim=0)
        results['generated_clip'] = torch.cat(results['generated_clip'], dim=0)
        
        logger.info(f"Generated {results['generated_clip'].shape[0]} samples")
        
        # Compute metrics
        if compute_metrics:
            results['generation_metrics'] = self._compute_generation_metrics(
                generated=results['generated_clip'],
                targets=results['clip_targets'],
                eva_conditioning=results['eva_embeddings'],
            )
        
        # Save results
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"Results saved to {output_path}")
        
        return results
    
    def _compute_generation_metrics(
        self,
        generated: torch.Tensor,      # [N, 64, 768]
        targets: torch.Tensor,        # [N, 64, 768]
        eva_conditioning: torch.Tensor,  # [N, 64, 1280]
    ) -> Dict[str, float]:
        """Compute comprehensive generation quality metrics."""
        
        with torch.no_grad():
            # Flatten for similarity computation
            generated_flat = generated.flatten(1)  # [N, 64*768]
            targets_flat = targets.flatten(1)      # [N, 64*768]
            
            # Cosine similarity
            cosine_similarities = nn.functional.cosine_similarity(
                generated_flat, targets_flat, dim=1
            )
            
            # L2 distances
            l2_distances = torch.norm(generated - targets, dim=-1).mean(dim=1)  # [N]
            
            # Embedding norms
            generated_norms = torch.norm(generated, dim=-1).mean(dim=1)  # [N]
            target_norms = torch.norm(targets, dim=-1).mean(dim=1)       # [N]
            
            # Token-wise statistics
            token_cosine_sims = nn.functional.cosine_similarity(
                generated, targets, dim=-1
            ).mean(dim=1)  # [N]
            
            # Variance analysis
            generated_var = generated.var(dim=-1).mean()
            target_var = targets.var(dim=-1).mean()
            
            # Distribution comparison
            generated_mean = generated.mean()
            target_mean = targets.mean()
            generated_std = generated.std()
            target_std = targets.std()
            
            metrics = {
                # Primary similarity metrics
                'cosine_similarity_mean': cosine_similarities.mean().item(),
                'cosine_similarity_std': cosine_similarities.std().item(),
                'cosine_similarity_min': cosine_similarities.min().item(),
                'cosine_similarity_max': cosine_similarities.max().item(),
                
                # Distance metrics
                'l2_distance_mean': l2_distances.mean().item(),
                'l2_distance_std': l2_distances.std().item(),
                
                # Norm metrics
                'generated_norm_mean': generated_norms.mean().item(),
                'target_norm_mean': target_norms.mean().item(),
                'norm_ratio_mean': (generated_norms / target_norms).mean().item(),
                
                # Token-wise metrics
                'token_cosine_sim_mean': token_cosine_sims.mean().item(),
                'token_cosine_sim_std': token_cosine_sims.std().item(),
                
                # Distribution metrics
                'generated_mean': generated_mean.item(),
                'target_mean': target_mean.item(),
                'generated_std': generated_std.item(),
                'target_std': target_std.item(),
                'mean_difference': (generated_mean - target_mean).abs().item(),
                'std_ratio': (generated_std / target_std).item(),
                
                # Variance metrics
                'generated_variance': generated_var.item(),
                'target_variance': target_var.item(),
                'variance_ratio': (generated_var / target_var).item(),
            }
        
        # Log metrics
        logger.info("Generation Quality Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def interpolate_embeddings(
        self,
        eva_start: torch.Tensor,      # [num_tokens, eva_dim]
        eva_end: torch.Tensor,        # [num_tokens, eva_dim]
        num_steps: int = 10,
        num_inference_steps: int = 50,
    ) -> List[torch.Tensor]:
        """
        Generate interpolation between two EVA embeddings.
        
        Args:
            eva_start: Starting EVA embedding [num_tokens, eva_dim]
            eva_end: Ending EVA embedding [num_tokens, eva_dim]
            num_steps: Number of interpolation steps
            num_inference_steps: Steps for each generation
            
        Returns:
            List of generated CLIP embeddings
        """
        interpolations = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            eva_interp = (1 - alpha) * eva_start + alpha * eva_end
            eva_interp = eva_interp.unsqueeze(0)  # Add batch dimension
            
            # FIXED: Use correct method signature
            generated_clip = self.generate(
                eva_interp,  # FIXED: Use positional argument
                num_inference_steps=num_inference_steps,
            ).squeeze(0)  # Remove batch dimension
            
            interpolations.append(generated_clip.cpu())
        
        return interpolations
    
    def evaluate_model(
        self,
        dataset_path: Union[str, Path],
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        split: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset using flow matching loss.
        
        Args:
            dataset_path: Path to evaluation dataset
            batch_size: Batch size for evaluation
            num_samples: Number of samples to evaluate (None for all)
            split: Dataset split to use
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create evaluation dataloader
        eval_dataloader = create_blip3o_dataloader(
            embeddings_path=dataset_path,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            split=split,
            subset_size=num_samples,
        )
        
        logger.info(f"Evaluating on {len(eval_dataloader)} batches")
        
        eval_losses = []
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                eva_emb = batch['eva_embeddings'].to(device=self.device, dtype=self.torch_dtype)
                clip_targets = batch['clip_embeddings'].to(device=self.device, dtype=self.torch_dtype)
                
                # Sample timesteps and noise
                batch_size = eva_emb.shape[0]
                timesteps = self.flow_matching_loss.sample_timesteps(batch_size, self.device)
                noise = torch.randn_like(clip_targets)
                
                # Create noisy samples
                x_0 = torch.randn_like(clip_targets)
                noisy_clip = self.flow_matching_loss.interpolate_data(
                    x_0=x_0, x_1=clip_targets, t=timesteps, noise=noise
                )
                
                # Forward pass
                model_output = self.model(
                    hidden_states=noisy_clip,
                    timestep=timesteps,
                    encoder_hidden_states=eva_emb,
                    return_dict=False
                )
                
                # Compute loss and metrics
                loss, metrics = self.flow_matching_loss(
                    model_output=model_output,
                    target_samples=clip_targets,
                    timesteps=timesteps,
                    eva_conditioning=eva_emb,
                    noise=noise,
                    return_metrics=True
                )
                
                eval_losses.append(loss.item())
                if metrics:
                    all_metrics.append(metrics)
        
        # Aggregate results
        results = {
            'eval_loss': np.mean(eval_losses),
            'eval_loss_std': np.std(eval_losses),
        }
        
        # Aggregate detailed metrics
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                results[f'eval_{key}'] = np.mean(values)
                results[f'eval_{key}_std'] = np.std(values)
        
        logger.info(f"Evaluation results: {results}")
        
        return results


def load_blip3o_inference(
    model_path: Union[str, Path],
    device: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> BLIP3oInference:
    """
    Convenience function to load BLIP3-o inference pipeline.
    
    Args:
        model_path: Path to trained model
        device: Device to use
        torch_dtype: Data type for model
        
    Returns:
        BLIP3oInference instance
    """
    return BLIP3oInference(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )