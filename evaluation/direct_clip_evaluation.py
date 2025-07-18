#!/usr/bin/env python3
"""
Comprehensive CLIP and BLIP3o Recall Evaluation
Tests image-to-text recall using different vision encoding methods.

This script supports:
1. CLIP Global Token: CLS token + visual projection → 768-dim
2. CLIP Patch-based: Patch embeddings → average → visual projection → 768-dim  
3. BLIP3o Model: EVA → BLIP3o DiT → average → visual projection → 768-dim

Usage examples:
- Global token: --method global
- Patch-based: --method patch  
- BLIP3o model: --method blip3o --blip3o_model_path /path/to/model
- Compare all: --method all --blip3o_model_path /path/to/model
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, AutoModel
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
import json
import argparse
import time
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import BLIP3o modules (graceful failure if not available)
try:
    from src.modules.inference.blip3o_inference import BLIP3oInference
    from src.modules.models.blip3o_dit import BLIP3oDiTModel
    from src.modules.config.blip3o_config import BLIP3oDiTConfig
    BLIP3O_AVAILABLE = True
    logger.info("BLIP3o inference module found")
except ImportError as e:
    BLIP3O_AVAILABLE = False
    logger.warning(f"BLIP3o inference not available: {e}")
    logger.warning("BLIP3o evaluation will be skipped")
    
    # Fallback: Try to import just the necessary components
    try:
        import sys
        import os
        # Add the project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(project_root)
        
        from src.modules.inference.blip3o_inference import BLIP3oInference
        BLIP3O_AVAILABLE = True
        logger.info("BLIP3o inference module found via fallback import")
    except ImportError as e2:
        logger.warning(f"Fallback import also failed: {e2}")
        BLIP3O_AVAILABLE = False


class ComprehensiveRecallEvaluator:
    """
    Comprehensive evaluator for CLIP and BLIP3o recall evaluation.
    Supports multiple vision encoding methods:
    - Global token (CLS token)
    - Patch-based (averaged patches)
    - BLIP3o model (EVA → BLIP3o DiT)
    """
    
    def __init__(self, 
                 device: str = "auto", 
                 torch_dtype: Optional[torch.dtype] = None,
                 blip3o_model_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Data type for models
            blip3o_model_path: Path to BLIP3o model (for BLIP3o evaluation)
        """
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        self.blip3o_model_path = blip3o_model_path
        
        # Load models
        self._load_clip_model()
        self._load_eva_model()
        
        # Load BLIP3o model if path provided and available
        self.blip3o_inference = None
        if blip3o_model_path and BLIP3O_AVAILABLE:
            self._load_blip3o_model()
        elif blip3o_model_path and not BLIP3O_AVAILABLE:
            logger.warning("BLIP3o model path provided but inference module not available")
        
        logger.info("Comprehensive Recall Evaluator initialized")
        logger.info(f"Using device: {self.device}")
        logger.info(f"CLIP model: openai/clip-vit-large-patch14")
        logger.info(f"EVA model: BAAI/EVA-CLIP-8B")
        logger.info(f"BLIP3o available: {self.blip3o_inference is not None}")
    
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
    
    def _load_clip_model(self):
        """Load CLIP ViT-L/14 model (same as used in main codebase)."""
        logger.info("Loading CLIP ViT-L/14...")
        
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.clip_model.eval()
        
        # Log model dimensions for verification
        vision_proj_shape = self.clip_model.visual_projection.weight.shape
        text_proj_shape = self.clip_model.text_projection.weight.shape
        logger.info(f"CLIP visual projection: {vision_proj_shape} (1024 → 768)")
        logger.info(f"CLIP text projection: {text_proj_shape} (768 → 768)")
        
        logger.info("CLIP model loaded successfully")
    
    def _load_eva_model(self):
        """Load EVA-CLIP-8B model (same as used in main codebase)."""
        logger.info("Loading EVA-CLIP-8B...")
        
        self.eva_model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.eva_model.eval()
        
        logger.info("EVA-CLIP model loaded successfully")
    
    def _load_blip3o_model(self):
        """Load BLIP3o DiT model if available."""
        try:
            logger.info(f"Loading BLIP3o model from {self.blip3o_model_path}...")
            
            # Check if path exists
            blip3o_path = Path(self.blip3o_model_path)
            if not blip3o_path.exists():
                raise FileNotFoundError(f"BLIP3o model path does not exist: {blip3o_path}")
            
            # Load the BLIP3o inference module
            self.blip3o_inference = BLIP3oInference(
                model_path=str(blip3o_path),
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            
            logger.info("BLIP3o model loaded successfully")
            logger.info(f"BLIP3o device: {self.blip3o_inference.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP3o model: {e}")
            logger.error(f"Model path checked: {self.blip3o_model_path}")
            logger.error("Make sure the model path is correct and the model files exist")
            self.blip3o_inference = None
            raise
    
    def extract_clip_text_embeddings(self, captions: List[str]) -> torch.Tensor:
        """
        Extract CLIP text embeddings (same as main codebase).
        
        Args:
            captions: List of text captions
            
        Returns:
            Text embeddings [B, 768] (normalized)
        """
        with torch.no_grad():
            inputs = self.clip_processor(
                text=captions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get text features (includes text projection and normalization)
            text_embeddings = self.clip_model.get_text_features(**inputs)
            # get_text_features already applies normalization, but let's be explicit
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings.cpu().float()
    
    def extract_clip_vision_global_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract CLIP vision global tokens (CLS token + visual projection).
        
        This extracts the CLS token from CLIP vision encoder and applies visual projection
        to get 768-dim embeddings that match text embedding space.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Global vision embeddings [B, 768] (normalized)
        """
        global_embeddings = []
        
        with torch.no_grad():
            for img in images:
                # Process image (same as in main codebase)
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                         for k, v in inputs.items()}
                
                # Get vision model outputs
                vision_outputs = self.clip_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract CLS token (global representation)
                # vision_outputs.last_hidden_state: [1, 257, 1024] 
                # Index 0 is CLS token, indices 1-256 are patch tokens
                cls_token = vision_outputs.last_hidden_state[:, 0, :]  # [1, 1024]
                
                # Apply CLIP's visual projection to align with text space
                # This converts 1024-dim vision features → 768-dim aligned features
                vision_projected = self.clip_model.visual_projection(cls_token)  # [1, 768]
                
                # Normalize to unit norm (same as text embeddings)
                vision_projected = F.normalize(vision_projected, p=2, dim=-1)
                
                global_embeddings.append(vision_projected.squeeze(0).cpu().float())  # [768]
        
        return torch.stack(global_embeddings)  # [B, 768]
    
    def extract_clip_vision_patch_averaged(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract CLIP vision embeddings using patch averaging + visual projection.
        
        This extracts patch embeddings, averages them, then applies visual projection
        to get 768-dim embeddings that match text embedding space.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Patch-averaged vision embeddings [B, 768] (normalized)
        """
        patch_averaged_embeddings = []
        
        with torch.no_grad():
            for img in images:
                # Process image (same as in main codebase)
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                         for k, v in inputs.items()}
                
                # Get vision model outputs
                vision_outputs = self.clip_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract patch embeddings (remove CLS token)
                # vision_outputs.last_hidden_state: [1, 257, 1024]
                # Indices 1-256 are patch tokens, index 0 is CLS token
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 1024]
                
                # Average patch embeddings to get global representation
                patch_averaged = patch_embeddings.mean(dim=1)  # [1, 1024]
                
                # Apply CLIP's visual projection to align with text space
                vision_projected = self.clip_model.visual_projection(patch_averaged)  # [1, 768]
                
                # Normalize to unit norm (same as text embeddings)
                vision_projected = F.normalize(vision_projected, p=2, dim=-1)
                
                patch_averaged_embeddings.append(vision_projected.squeeze(0).cpu().float())  # [768]
        
        return torch.stack(patch_averaged_embeddings)  # [B, 768]
    
    def extract_eva_vision_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract EVA-CLIP vision embeddings (same method as training).
        
        This extracts patch embeddings from EVA-CLIP vision model in the exact
        format expected by BLIP3o DiT model: [B, 256, 4096]
        
        Args:
            images: List of PIL Images
            
        Returns:
            EVA vision embeddings [B, 256, 4096]
        """
        eva_embeddings = []
        
        logger.info(f"Extracting EVA embeddings for {len(images)} images...")
        
        with torch.no_grad():
            for i, img in enumerate(images):
                if i % 100 == 0:
                    logger.debug(f"Processing EVA image {i}/{len(images)}")
                
                # Use same processing as in main codebase (extract_embeddings_g.py)
                inputs = self.eva_processor(images=img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device=self.device, dtype=self.torch_dtype)
                
                # Get vision model outputs
                vision_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token) → [1, 256, hidden_dim]
                # vision_outputs.last_hidden_state: [1, 257, hidden_dim] where 257 = 1 CLS + 256 patches
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
                batch_size, num_patches, hidden_dim = patch_embeddings.shape
                
                # Verify we have the expected dimensions
                if num_patches != 256:
                    logger.warning(f"Expected 256 patches, got {num_patches}")
                if hidden_dim != 4096:
                    logger.warning(f"Expected 4096 dimensions, got {hidden_dim}")
                
                # Reshape to 16x16 grid → [1, 16, 16, hidden_dim]
                grid_size = int(np.sqrt(num_patches))  # Should be 16
                spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
                
                # Convert to tokens format → [256, hidden_dim] and then back to [1, 256, hidden_dim]
                tokens = spatial_grid.reshape(1, num_patches, hidden_dim)  # [1, 256, 4096]
                
                eva_embeddings.append(tokens.squeeze(0).cpu().float())  # [256, 4096]
        
        result = torch.stack(eva_embeddings)  # [B, 256, 4096]
        logger.info(f"EVA embeddings extracted: {result.shape}")
        logger.info(f"EVA embedding range: [{result.min():.4f}, {result.max():.4f}]")
        
        return result
    
    def extract_blip3o_generated_embeddings(self, eva_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate CLIP embeddings from EVA-CLIP using BLIP3o DiT, then apply visual projection.
        
        Flow: EVA [B, 256, 4096] → BLIP3o DiT → CLIP-like [B, 256, 1024] → Average → Visual Projection → [B, 768]
        
        Args:
            eva_embeddings: EVA-CLIP vision embeddings [B, 256, 4096]
            
        Returns:
            Generated CLIP embeddings [B, 768] (normalized)
        """
        if self.blip3o_inference is None:
            raise ValueError("BLIP3o model not loaded. Provide blip3o_model_path during initialization.")
        
        logger.info(f"Generating CLIP embeddings from EVA using BLIP3o DiT...")
        logger.info(f"Input EVA embeddings shape: {eva_embeddings.shape}")
        
        # Move to correct device
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        generated_embeddings = []
        
        with torch.no_grad():
            # Process in batches to manage memory
            batch_size = 8  # Adjust based on GPU memory
            num_samples = eva_embeddings.shape[0]
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_eva = eva_embeddings[i:end_idx]  # [batch_size, 256, 4096]
                
                logger.debug(f"Processing BLIP3o batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
                logger.debug(f"Batch EVA shape: {batch_eva.shape}")
                
                try:
                    # Generate CLIP-like embeddings using BLIP3o DiT
                    # This should output [batch_size, 256, 1024] to match CLIP patch dimensions
                    generated_clip = self.blip3o_inference.generate(
                        batch_eva,  # [batch_size, 256, 4096]
                        num_inference_steps=50,  # You can adjust this
                    )  # → [batch_size, 256, 1024]
                    
                    logger.debug(f"Generated CLIP shape: {generated_clip.shape}")
                    
                    if generated_clip.shape[-1] != 1024:
                        logger.warning(f"Expected 1024-dim CLIP features, got {generated_clip.shape[-1]}")
                    
                    if generated_clip.shape[1] != 256:
                        logger.warning(f"Expected 256 tokens, got {generated_clip.shape[1]}")
                    
                    # Average pool across the 256 tokens to get global representation
                    # [batch_size, 256, 1024] → [batch_size, 1024]
                    generated_global = generated_clip.mean(dim=1)  # [batch_size, 1024]
                    
                    logger.debug(f"Generated global shape: {generated_global.shape}")
                    
                    # Apply CLIP's visual projection to align with text space
                    # [batch_size, 1024] → [batch_size, 768]
                    generated_global = generated_global.to(device=self.device, dtype=self.torch_dtype)
                    generated_projected = self.clip_model.visual_projection(generated_global)  # [batch_size, 768]
                    
                    logger.debug(f"Generated projected shape: {generated_projected.shape}")
                    
                    # Normalize to unit norm (same as text embeddings)
                    generated_projected = F.normalize(generated_projected, p=2, dim=-1)
                    
                    # Verify normalization
                    norms = torch.norm(generated_projected, p=2, dim=-1)
                    logger.debug(f"Generated embedding norms: mean={norms.mean():.6f}, std={norms.std():.6f}")
                    
                    generated_embeddings.append(generated_projected.cpu().float())
                    
                except Exception as e:
                    logger.error(f"Error in BLIP3o generation for batch {i//batch_size + 1}: {e}")
                    logger.error(f"Batch EVA shape: {batch_eva.shape}")
                    raise
        
        # Concatenate all batches
        result = torch.cat(generated_embeddings, dim=0)  # [B, 768]
        
        logger.info(f"BLIP3o generation completed: {result.shape}")
        logger.info(f"Generated embedding range: [{result.min():.4f}, {result.max():.4f}]")
        
        # Final verification
        final_norms = torch.norm(result, p=2, dim=-1)
        logger.info(f"Final embedding norms: mean={final_norms.mean():.6f}, std={final_norms.std():.6f}")
        
        return result  # [B, 768] - normalized
    
    def compute_pairwise_cosine_similarity(self,
                                         embeddings_a: torch.Tensor,
                                         embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings_a: First set of embeddings [N, D]
            embeddings_b: Second set of embeddings [M, D]
            
        Returns:
            Similarity matrix [N, M]
        """
        # Ensure both tensors are on the same device
        if embeddings_a.device != embeddings_b.device:
            embeddings_a = embeddings_a.cpu()
            embeddings_b = embeddings_b.cpu()
        
        # Normalize embeddings (should already be normalized, but ensure it)
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)
        
        # Compute pairwise cosine similarity via matrix multiplication
        similarity_matrix = torch.mm(embeddings_a, embeddings_b.t())
        
        return similarity_matrix
    
    def compute_image_to_text_recall(self,
                                   image_embeddings: torch.Tensor,
                                   text_embeddings: torch.Tensor,
                                   image_to_text_mapping: List[List[int]],
                                   k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Compute image-to-text recall metrics.
        
        Args:
            image_embeddings: Image embeddings [N_images, D]
            text_embeddings: Text embeddings [N_texts, D]
            image_to_text_mapping: Maps each image index to list of corresponding text indices
            k_values: K values for Recall@K
            
        Returns:
            Dictionary of recall metrics
        """
        # Ensure embeddings are normalized
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix: [N_images, N_texts]
        similarity_matrix = self.compute_pairwise_cosine_similarity(
            image_embeddings, text_embeddings
        )
        
        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        logger.info(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
        logger.info(f"Similarity mean: {similarity_matrix.mean():.4f} ± {similarity_matrix.std():.4f}")
        
        # Compute recall for each K value
        recall_results = {}
        
        for k in k_values:
            correct_retrievals = 0
            total_queries = len(image_to_text_mapping)
            
            # Get top-k text indices for each image
            _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)  # [N_images, k]
            
            # Debug first few retrievals for k=1
            if k == 1:
                logger.info("Sample image-to-text retrievals (first 5):")
                for i in range(min(5, len(image_to_text_mapping))):
                    correct_texts = image_to_text_mapping[i]
                    retrieved_text = top_k_indices[i, 0].item()
                    similarity_score = similarity_matrix[i, retrieved_text].item()
                    is_correct = retrieved_text in correct_texts
                    logger.info(f"  Image {i}: retrieved text {retrieved_text} (sim={similarity_score:.4f}), "
                               f"correct={is_correct}, target_texts={correct_texts}")
            
            # Check if any of the top-k retrieved texts belong to the query image
            for img_idx, correct_text_indices in enumerate(image_to_text_mapping):
                retrieved_indices = top_k_indices[img_idx].cpu().numpy()
                
                # Check if any retrieved index is in the correct text indices
                if any(ret_idx in correct_text_indices for ret_idx in retrieved_indices):
                    correct_retrievals += 1
            
            recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0
            recall_results[f'recall@{k}'] = recall_at_k
            
            logger.info(f"Recall@{k}: {correct_retrievals}/{total_queries} = {recall_at_k:.4f} ({recall_at_k*100:.2f}%)")
        
        # Additional metrics
        recall_results['num_queries'] = len(image_to_text_mapping)
        recall_results['num_gallery'] = len(text_embeddings)
        recall_results['avg_texts_per_image'] = np.mean([len(texts) for texts in image_to_text_mapping])
        recall_results['embedding_dim'] = image_embeddings.shape[1]
        
        return recall_results
    
    def evaluate_recall_by_method(self,
                                images: List[Image.Image],
                                captions_per_image: List[List[str]],
                                method: str = "global",
                                k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Evaluate recall using specified method.
        
        Args:
            images: List of PIL Images
            captions_per_image: List of caption lists for each image
            method: Evaluation method ("global", "patch", "blip3o")
            k_values: K values for Recall@K
            
        Returns:
            Dictionary with recall metrics
        """
        logger.info(f"Evaluating recall using method: {method}")
        
        # Extract text embeddings (same for all methods)
        logger.info("Extracting text embeddings...")
        all_text_embeddings = []
        image_to_text_mapping = []
        text_idx = 0
        
        # Flatten captions and create mapping
        for img_idx, caption_list in enumerate(captions_per_image):
            current_text_indices = []
            for caption in caption_list:
                current_text_indices.append(text_idx)
                text_idx += 1
            image_to_text_mapping.append(current_text_indices)
        
        # Extract all captions
        all_captions = [caption for caption_list in captions_per_image for caption in caption_list]
        text_embeddings = self.extract_clip_text_embeddings(all_captions)
        
        # Extract image embeddings based on method
        logger.info(f"Extracting image embeddings using {method} method...")
        
        if method == "global":
            image_embeddings = self.extract_clip_vision_global_tokens(images)
            method_description = "CLS token + visual projection"
            
        elif method == "patch":
            image_embeddings = self.extract_clip_vision_patch_averaged(images)
            method_description = "Patch averaging + visual projection"
            
        elif method == "blip3o":
            if self.blip3o_inference is None:
                raise ValueError("BLIP3o model not available. Check model path and installation.")
            
            logger.info("=== BLIP3o Evaluation Pipeline ===")
            logger.info("Step 1: Extracting EVA-CLIP embeddings...")
            
            # Extract EVA embeddings first
            eva_embeddings = self.extract_eva_vision_embeddings(images)
            logger.info(f"EVA embeddings extracted: {eva_embeddings.shape}")
            
            logger.info("Step 2: Generating CLIP embeddings using BLIP3o DiT...")
            
            # Generate CLIP embeddings using BLIP3o
            image_embeddings = self.extract_blip3o_generated_embeddings(eva_embeddings)
            logger.info(f"BLIP3o embeddings generated: {image_embeddings.shape}")
            
            method_description = "EVA → BLIP3o DiT → visual projection"
            
            logger.info("=== BLIP3o Evaluation Complete ===")
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'global', 'patch', or 'blip3o'.")
        
        logger.info(f"Image embeddings shape: {image_embeddings.shape}")
        logger.info(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Verify embedding normalization
        img_norms = torch.norm(image_embeddings, p=2, dim=-1)
        text_norms = torch.norm(text_embeddings, p=2, dim=-1)
        logger.info(f"Image embedding norms - mean: {img_norms.mean():.6f}, std: {img_norms.std():.6f}")
        logger.info(f"Text embedding norms - mean: {text_norms.mean():.6f}, std: {text_norms.std():.6f}")
        
        # Compute recall metrics
        recall_results = self.compute_image_to_text_recall(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            image_to_text_mapping=image_to_text_mapping,
            k_values=k_values
        )
        
        # Add method info
        recall_results.update({
            'method': method,
            'method_description': method_description,
            'model': 'clip-vit-large-patch14',
            'embedding_dim': image_embeddings.shape[-1]
        })
        
        return recall_results
    
    def validate_blip3o_pipeline(self, test_images: List[Image.Image]) -> bool:
        """
        Validate that the BLIP3o pipeline is working correctly with a small test.
        
        Args:
            test_images: Small list of test images (1-3 images)
            
        Returns:
            True if validation passes, False otherwise
        """
        if self.blip3o_inference is None:
            logger.error("BLIP3o model not loaded - cannot validate pipeline")
            return False
        
        try:
            logger.info("🧪 Validating BLIP3o pipeline with test images...")
            
            # Test with just 1-2 images
            test_imgs = test_images[:2]
            
            # Step 1: Extract EVA embeddings
            logger.info("  Step 1: Testing EVA embedding extraction...")
            eva_embeddings = self.extract_eva_vision_embeddings(test_imgs)
            logger.info(f"  ✅ EVA embeddings: {eva_embeddings.shape}")
            
            # Step 2: Test BLIP3o generation
            logger.info("  Step 2: Testing BLIP3o generation...")
            generated_embeddings = self.extract_blip3o_generated_embeddings(eva_embeddings)
            logger.info(f"  ✅ Generated embeddings: {generated_embeddings.shape}")
            
            # Step 3: Validate output dimensions and properties
            logger.info("  Step 3: Validating output properties...")
            
            expected_shape = (len(test_imgs), 768)
            if generated_embeddings.shape != expected_shape:
                logger.error(f"  ❌ Shape mismatch: expected {expected_shape}, got {generated_embeddings.shape}")
                return False
            
            # Check normalization
            norms = torch.norm(generated_embeddings, p=2, dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
                logger.warning(f"  ⚠️  Embeddings not properly normalized: norms {norms}")
            else:
                logger.info(f"  ✅ Embeddings properly normalized")
            
            # Check for NaN or infinite values
            if torch.isnan(generated_embeddings).any():
                logger.error("  ❌ NaN values detected in generated embeddings")
                return False
            
            if torch.isinf(generated_embeddings).any():
                logger.error("  ❌ Infinite values detected in generated embeddings")
                return False
            
            logger.info("  ✅ No NaN or infinite values detected")
            
            # Check reasonable value ranges
            emb_min, emb_max = generated_embeddings.min().item(), generated_embeddings.max().item()
            if emb_min < -2.0 or emb_max > 2.0:
                logger.warning(f"  ⚠️  Unusual embedding values: range [{emb_min:.4f}, {emb_max:.4f}]")
            else:
                logger.info(f"  ✅ Embedding values in reasonable range: [{emb_min:.4f}, {emb_max:.4f}]")
            
            logger.info("🎉 BLIP3o pipeline validation PASSED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ BLIP3o pipeline validation FAILED: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def validate_model_paths(blip3o_model_path: Optional[str] = None) -> bool:
    """
    Validate that all required model paths exist and are accessible.
    
    Args:
        blip3o_model_path: Path to BLIP3o model (optional)
        
    Returns:
        True if all paths are valid, False otherwise
    """
    all_valid = True
    
    # Check BLIP3o model path if provided
    if blip3o_model_path:
        blip3o_path = Path(blip3o_model_path)
        
        logger.info(f"🔍 Validating BLIP3o model path: {blip3o_path}")
        
        if not blip3o_path.exists():
            logger.error(f"❌ BLIP3o model path does not exist: {blip3o_path}")
            all_valid = False
        elif not blip3o_path.is_dir():
            logger.error(f"❌ BLIP3o model path is not a directory: {blip3o_path}")
            all_valid = False
        else:
            # Check for expected model files
            expected_files = ["config.json", "pytorch_model.bin"]  # Adjust based on your model structure
            missing_files = []
            
            for file_name in expected_files:
                file_path = blip3o_path / file_name
                if not file_path.exists():
                    # Try alternative names
                    alt_files = [f"model.{file_name.split('.')[-1]}", f"blip3o_{file_name}"]
                    found = False
                    for alt_name in alt_files:
                        if (blip3o_path / alt_name).exists():
                            found = True
                            break
                    
                    if not found:
                        missing_files.append(file_name)
            
            if missing_files:
                logger.warning(f"⚠️  Some expected files missing in BLIP3o model: {missing_files}")
                logger.warning("Model might still work if files are named differently")
            else:
                logger.info("✅ BLIP3o model path validation passed")
    
    return all_valid


def run_comprehensive_evaluation(evaluator: ComprehensiveRecallEvaluator,
                                images: List[Image.Image],
                                captions_per_image: List[List[str]],
                                methods: List[str],
                                k_values: List[int] = [1, 5, 10]) -> Dict[str, Dict[str, float]]:
    """
    Run comprehensive evaluation using multiple methods.
    
    Args:
        evaluator: The evaluator instance
        images: List of PIL Images
        captions_per_image: List of caption lists for each image
        methods: List of methods to evaluate
        k_values: K values for Recall@K
        
    Returns:
        Dictionary mapping method names to their results
    """
    all_results = {}
    
    for method in methods:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Evaluating method: {method.upper()}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            results = evaluator.evaluate_recall_by_method(
                images=images,
                captions_per_image=captions_per_image,
                method=method,
                k_values=k_values
            )
            evaluation_time = time.time() - start_time
            results['evaluation_time'] = evaluation_time
            
            all_results[method] = results
            
            # Print results for this method
            print(f"\n📊 {method.upper()} Results:")
            print(f"Method: {results['method_description']}")
            print(f"Time: {evaluation_time:.2f}s")
            for k in k_values:
                if f'recall@{k}' in results:
                    recall_k = results[f'recall@{k}']
                    print(f"  Recall@{k:2d}: {recall_k:.4f} ({recall_k*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to evaluate method {method}: {e}")
            all_results[method] = {'error': str(e)}
    
    return all_results


def load_coco_samples(coco_root: Path, num_samples: int = 1000) -> Tuple[List[Image.Image], List[List[str]], List[int]]:
    """
    Load COCO validation samples.
    
    Args:
        coco_root: Path to COCO dataset
        num_samples: Number of samples to load
        
    Returns:
        Tuple of (images, captions_per_image, image_ids)
    """
    logger.info(f"Loading {num_samples} COCO validation samples...")
    
    # Load COCO annotations
    annotations_file = coco_root / "annotations" / "captions_val2017.json"
    images_dir = coco_root / "images" / "val2017"
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO images not found: {images_dir}")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image info mapping
    images_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    image_captions = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(ann['caption'])
    
    # Load samples
    images = []
    captions_per_image = []
    image_ids = []
    
    loaded_count = 0
    for image_id, captions in image_captions.items():
        if loaded_count >= num_samples:
            break
        
        if image_id not in images_info:
            continue
        
        image_info = images_info[image_id]
        image_path = images_dir / image_info['file_name']
        
        # Check if image file exists
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            continue
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            images.append(image)
            captions_per_image.append(captions[:5])  # Max 5 captions per image
            image_ids.append(image_id)
            loaded_count += 1
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(images)} images with {sum(len(caps) for caps in captions_per_image)} captions")
    return images, captions_per_image, image_ids


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CLIP and BLIP3o Recall Evaluation")
    parser.add_argument("--coco_root", type=str, required=True,
                       help="Path to MS-COCO dataset root directory")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of COCO samples to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing (currently not used in batching)")
    parser.add_argument("--method", type=str, default="global", 
                       choices=["global", "patch", "blip3o", "all"],
                       help="Evaluation method: global (CLS token), patch (averaged patches), blip3o (BLIP3o model), or all")
    parser.add_argument("--blip3o_model_path", type=str, default=None,
                       help="Path to BLIP3o model (required for blip3o method)")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save results JSON file")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 5, 10],
                       help="K values for Recall@K computation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.method in ["blip3o", "all"] and not args.blip3o_model_path:
        if not BLIP3O_AVAILABLE:
            logger.error("BLIP3o method requested but inference module not available")
            logger.error("Make sure src.modules.inference.blip3o_inference is properly installed")
            sys.exit(1)
        else:
            logger.error("BLIP3o method requested but --blip3o_model_path not provided")
            sys.exit(1)
    
    # Validate model paths
    if args.blip3o_model_path:
        logger.info("🔍 Validating model paths...")
        if not validate_model_paths(args.blip3o_model_path):
            logger.error("❌ Model path validation failed")
            sys.exit(1)
        logger.info("✅ Model path validation passed")
    
    # Convert paths
    coco_root = Path(args.coco_root)
    
    # Determine methods to evaluate
    if args.method == "all":
        methods_to_evaluate = ["global", "patch"]
        if args.blip3o_model_path and BLIP3O_AVAILABLE:
            methods_to_evaluate.append("blip3o")
    else:
        methods_to_evaluate = [args.method]
    
    # Initialize evaluator
    logger.info("Initializing Comprehensive Recall Evaluator...")
    evaluator = ComprehensiveRecallEvaluator(
        device=args.device,
        blip3o_model_path=args.blip3o_model_path
    )
    
    # Load COCO samples
    logger.info(f"Loading {args.num_samples} COCO validation samples...")
    images, captions_per_image, image_ids = load_coco_samples(coco_root, args.num_samples)
    
    # Run evaluations
    logger.info(f"Running evaluation for methods: {methods_to_evaluate}")
    
    # Validate BLIP3o pipeline if needed
    if "blip3o" in methods_to_evaluate:
        logger.info("🧪 Validating BLIP3o pipeline before full evaluation...")
        validation_passed = evaluator.validate_blip3o_pipeline(images[:3])  # Test with first 3 images
        
        if not validation_passed:
            logger.error("❌ BLIP3o pipeline validation failed!")
            logger.error("Check model path, dependencies, and model compatibility")
            logger.error("Removing BLIP3o from evaluation methods")
            methods_to_evaluate = [m for m in methods_to_evaluate if m != "blip3o"]
            
            if not methods_to_evaluate:
                logger.error("No valid evaluation methods remaining. Exiting.")
                sys.exit(1)
        else:
            logger.info("✅ BLIP3o pipeline validation passed! Proceeding with full evaluation...")
    
    start_time = time.time()
    
    all_results = run_comprehensive_evaluation(
        evaluator=evaluator,
        images=images,
        captions_per_image=captions_per_image,
        methods=methods_to_evaluate,
        k_values=args.k_values
    )
    
    total_time = time.time() - start_time
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RECALL EVALUATION RESULTS")
    print("="*80)
    print(f"Dataset: MS-COCO 2017 Validation ({len(images)} images, {sum(len(caps) for caps in captions_per_image)} captions)")
    print(f"Total evaluation time: {total_time:.2f}s")
    
    # Results table
    print(f"\n📋 Results Summary:")
    print(f"{'Method':<15} {'Description':<35} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'Time':<8}")
    print("-" * 80)
    
    for method, results in all_results.items():
        if 'error' in results:
            print(f"{method:<15} {'ERROR: ' + results['error']:<35} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
        else:
            description = results.get('method_description', 'Unknown')[:34]
            r1 = f"{results.get('recall@1', 0)*100:.1f}%" if 'recall@1' in results else "N/A"
            r5 = f"{results.get('recall@5', 0)*100:.1f}%" if 'recall@5' in results else "N/A"
            r10 = f"{results.get('recall@10', 0)*100:.1f}%" if 'recall@10' in results else "N/A"
            eval_time = f"{results.get('evaluation_time', 0):.1f}s"
            print(f"{method:<15} {description:<35} {r1:<8} {r5:<8} {r10:<8} {eval_time:<8}")
    
    # Detailed results for each method
    for method, results in all_results.items():
        if 'error' not in results:
            print(f"\n🔍 Detailed Results - {method.upper()}:")
            print(f"   Method: {results['method_description']}")
            print(f"   Model: {results.get('model', 'Unknown')}")
            print(f"   Embedding dim: {results.get('embedding_dim', 'Unknown')}")
            print(f"   Evaluation time: {results.get('evaluation_time', 0):.2f}s")
            print(f"   Recall metrics:")
            for k in args.k_values:
                if f'recall@{k}' in results:
                    recall_k = results[f'recall@{k}']
                    print(f"     Recall@{k:2d}: {recall_k:.4f} ({recall_k*100:.2f}%)")
    
    # Literature comparison
    if 'global' in all_results and 'error' not in all_results['global']:
        global_r1 = all_results['global']['recall@1']
        print(f"\n🎯 Literature Comparison (Global Token Method):")
        print(f"   Expected CLIP ViT-L/14 recall@1: ~58-60%")
        print(f"   Your global token result: {global_r1*100:.2f}%")
        
        if global_r1 > 0.55:
            print(f"   ✅ Results look reasonable for CLIP ViT-L/14!")
        else:
            print(f"   ⚠️  Results seem low - check implementation")
    
    # Method comparison
    if len(all_results) > 1 and not any('error' in r for r in all_results.values()):
        print(f"\n📈 Method Comparison:")
        baseline_method = 'global' if 'global' in all_results else list(all_results.keys())[0]
        baseline_r1 = all_results[baseline_method]['recall@1']
        
        for method, results in all_results.items():
            if method != baseline_method and 'recall@1' in results:
                method_r1 = results['recall@1']
                diff = method_r1 - baseline_r1
                diff_pct = (diff / baseline_r1) * 100 if baseline_r1 > 0 else 0
                status = "✅" if diff >= 0 else "⚠️"
                print(f"   {method} vs {baseline_method}: {diff:+.4f} ({diff_pct:+.1f}%) {status}")
    
    print("="*80)
    
    # Save results if requested
    if args.save_results:
        results_to_save = {
            'evaluation_info': {
                'dataset': 'MS-COCO 2017 Validation',
                'num_images': len(images),
                'num_captions': sum(len(caps) for caps in captions_per_image),
                'samples_requested': args.num_samples,
                'methods_evaluated': list(all_results.keys()),
                'total_time': total_time,
                'device': str(evaluator.device),
                'k_values': args.k_values,
            },
            'method_results': all_results,
            'model_info': {
                'clip_model': 'openai/clip-vit-large-patch14',
                'eva_model': 'BAAI/EVA-CLIP-8B',
                'blip3o_model': args.blip3o_model_path,
                'blip3o_available': evaluator.blip3o_inference is not None,
            }
        }
        
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Results saved to: {save_path}")


if __name__ == "__main__":
    main()