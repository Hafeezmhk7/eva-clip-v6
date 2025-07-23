"""
FIXED Data Collator for BLIP3-o Training - Proper Gradient Setup
src/modules/datasets/blip3o_data_collator.py

CRITICAL FIXES:
1. Proper tensor creation with gradient requirements
2. Correct handling of noisy input generation
3. Ensures tensors are connected to computation graph from start
4. BLIP3-o paper aligned data handling
"""

import torch
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BLIP3oPatchDataCollator:
    """
    FIXED Data collator for BLIP3-o patch-level training
    
    Creates properly connected tensors with gradients for flow matching training
    """
    
    def __init__(
        self,
        normalize_embeddings: bool = True,
        max_batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.normalize_embeddings = normalize_embeddings
        self.max_batch_size = max_batch_size
        self.device = device
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        FIXED collate function with proper gradient flow setup
        
        Args:
            batch: List of samples with 'eva_embeddings', 'clip_embeddings', 'captions'
            
        Returns:
            Batch dictionary with proper tensor setup for gradient flow
        """
        
        # Limit batch size if specified
        if self.max_batch_size and len(batch) > self.max_batch_size:
            batch = batch[:self.max_batch_size]
        
        batch_size = len(batch)
        
        # Extract embeddings and captions
        eva_embeddings = []
        clip_embeddings = []
        captions = []
        
        for sample in batch:
            eva_emb = sample['eva_embeddings']  # [256, 4096]
            clip_emb = sample['clip_embeddings']  # [256, 1024]
            caption = sample.get('caption', f'Sample caption {len(captions)}')
            
            # Validate shapes
            if eva_emb.shape != (256, 4096):
                logger.warning(f"Invalid EVA shape: {eva_emb.shape}, expected [256, 4096]")
                continue
            if clip_emb.shape != (256, 1024):
                logger.warning(f"Invalid CLIP shape: {clip_emb.shape}, expected [256, 1024]")
                continue
            
            eva_embeddings.append(eva_emb)
            clip_embeddings.append(clip_emb)
            captions.append(caption)
        
        if not eva_embeddings:
            raise ValueError("No valid samples in batch after filtering")
        
        # Stack into batch tensors
        try:
            eva_batch = torch.stack(eva_embeddings, dim=0)  # [B, 256, 4096]
            clip_batch = torch.stack(clip_embeddings, dim=0)  # [B, 256, 1024]
        except Exception as e:
            logger.error(f"Failed to stack embeddings: {e}")
            raise
        
        # Move to device if specified
        if self.device is not None:
            eva_batch = eva_batch.to(self.device)
            clip_batch = clip_batch.to(self.device)
        
        # Ensure proper dtype
        eva_batch = eva_batch.float()
        clip_batch = clip_batch.float()
        
        # Normalize if requested
        if self.normalize_embeddings:
            eva_batch = torch.nn.functional.normalize(eva_batch, p=2, dim=-1)
            clip_batch = torch.nn.functional.normalize(clip_batch, p=2, dim=-1)
        
        # CRITICAL FIX: Ensure EVA embeddings are detached (conditioning, no gradients needed)
        eva_batch = eva_batch.detach()
        
        # CRITICAL FIX: For CLIP embeddings (targets), detach them too
        # The gradients should flow through the model inputs, not the targets
        clip_batch = clip_batch.detach()
        
        # CRITICAL FIX: Create noisy input for flow matching training
        # This tensor MUST have gradients and be properly connected
        batch_size, seq_len, clip_dim = clip_batch.shape
        device = clip_batch.device
        dtype = clip_batch.dtype
        
        # Generate noise for flow matching
        noise = torch.randn_like(clip_batch, device=device, dtype=dtype)
        
        # Sample timesteps for flow matching
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        
        # Create interpolated noisy input (this will be the model input)
        # This is where gradients need to flow from
        alpha = timesteps.view(-1, 1, 1)  # [B, 1, 1]
        
        # CRITICAL: Create base noise with proper gradient requirement
        # This tensor must be connected to the computation graph
        base_noise = torch.randn_like(clip_batch, device=device, dtype=dtype, requires_grad=True)
        
        # Linear interpolation for flow matching (rectified flow style)
        # noisy_input = (1 - alpha) * noise + alpha * clip_targets + small_noise
        noisy_input = (1 - alpha) * base_noise + alpha * clip_batch.detach() + 0.1 * noise
        
        # CRITICAL: Ensure the noisy input requires gradients
        if not noisy_input.requires_grad:
            logger.warning("Noisy input doesn't require gradients, fixing...")
            noisy_input = noisy_input.requires_grad_(True)
        
        # Validate tensor properties
        assert eva_batch.shape == (batch_size, 256, 4096), f"EVA batch shape: {eva_batch.shape}"
        assert clip_batch.shape == (batch_size, 256, 1024), f"CLIP batch shape: {clip_batch.shape}"
        assert noisy_input.shape == (batch_size, 256, 1024), f"Noisy input shape: {noisy_input.shape}"
        assert timesteps.shape == (batch_size,), f"Timesteps shape: {timesteps.shape}"
        
        # Log gradient status for debugging
        logger.debug(f"Batch collated: eva_batch requires_grad={eva_batch.requires_grad}")
        logger.debug(f"Batch collated: clip_batch requires_grad={clip_batch.requires_grad}")
        logger.debug(f"Batch collated: noisy_input requires_grad={noisy_input.requires_grad}")
        logger.debug(f"Batch collated: base_noise requires_grad={base_noise.requires_grad}")
        
        return {
            'eva_embeddings': eva_batch,        # [B, 256, 4096] - EVA conditioning (detached)
            'clip_embeddings': clip_batch,      # [B, 256, 1024] - Target CLIP patches (detached)
            'hidden_states': noisy_input,       # [B, 256, 1024] - Noisy input for model (with gradients)
            'timesteps': timesteps,             # [B] - Flow matching timesteps
            'noise': noise,                     # [B, 256, 1024] - Original noise
            'base_noise': base_noise,           # [B, 256, 1024] - Base noise with gradients
            'captions': captions,               # List[str] - Text captions
            'batch_size': batch_size,           # int
        }


def create_blip3o_data_collator(
    normalize_embeddings: bool = True,
    max_batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> BLIP3oPatchDataCollator:
    """
    Create BLIP3-o data collator with proper gradient flow setup
    
    Args:
        normalize_embeddings: Whether to normalize embeddings
        max_batch_size: Maximum batch size to prevent OOM
        device: Device to move tensors to
        
    Returns:
        BLIP3oPatchDataCollator instance
    """
    
    return BLIP3oPatchDataCollator(
        normalize_embeddings=normalize_embeddings,
        max_batch_size=max_batch_size,
        device=device,
    )


class BLIP3oEvaluationDataCollator:
    """
    Data collator for BLIP3-o evaluation (no gradient requirements)
    """
    
    def __init__(
        self,
        normalize_embeddings: bool = True,
        max_batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.normalize_embeddings = normalize_embeddings
        self.max_batch_size = max_batch_size
        self.device = device
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for evaluation (simpler, no gradient requirements)
        """
        
        # Limit batch size if specified
        if self.max_batch_size and len(batch) > self.max_batch_size:
            batch = batch[:self.max_batch_size]
        
        batch_size = len(batch)
        
        # Extract embeddings and captions
        eva_embeddings = []
        clip_embeddings = []
        captions = []
        
        for sample in batch:
            eva_emb = sample['eva_embeddings']  # [256, 4096]
            clip_emb = sample['clip_embeddings']  # [256, 1024]
            caption = sample.get('caption', f'Eval caption {len(captions)}')
            
            eva_embeddings.append(eva_emb)
            clip_embeddings.append(clip_emb)
            captions.append(caption)
        
        # Stack into batch tensors
        eva_batch = torch.stack(eva_embeddings, dim=0)  # [B, 256, 4096]
        clip_batch = torch.stack(clip_embeddings, dim=0)  # [B, 256, 1024]
        
        # Move to device if specified
        if self.device is not None:
            eva_batch = eva_batch.to(self.device)
            clip_batch = clip_batch.to(self.device)
        
        # Ensure proper dtype
        eva_batch = eva_batch.float()
        clip_batch = clip_batch.float()
        
        # Normalize if requested
        if self.normalize_embeddings:
            eva_batch = torch.nn.functional.normalize(eva_batch, p=2, dim=-1)
            clip_batch = torch.nn.functional.normalize(clip_batch, p=2, dim=-1)
        
        # For evaluation, all tensors can be detached
        eva_batch = eva_batch.detach()
        clip_batch = clip_batch.detach()
        
        return {
            'eva_embeddings': eva_batch,        # [B, 256, 4096] - EVA conditioning
            'clip_embeddings': clip_batch,      # [B, 256, 1024] - Target CLIP patches
            'captions': captions,               # List[str] - Text captions
            'batch_size': batch_size,           # int
        }


def create_blip3o_eval_data_collator(
    normalize_embeddings: bool = True,
    max_batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> BLIP3oEvaluationDataCollator:
    """
    Create BLIP3-o evaluation data collator
    """
    
    return BLIP3oEvaluationDataCollator(
        normalize_embeddings=normalize_embeddings,
        max_batch_size=max_batch_size,
        device=device,
    )