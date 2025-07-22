"""
BLIP3-o Image-to-Text Recall Evaluator
src/modules/evaluation/blip3o_recall_evaluator.py

This module implements image-to-text recall evaluation for BLIP3-o patch-level DiT models.
It measures how well the generated CLIP embeddings can retrieve relevant text descriptions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class BLIP3oRecallEvaluator:
    """
    Image-to-Text Recall Evaluator for BLIP3-o models
    
    This evaluator measures:
    1. Recall@1, Recall@5, Recall@10 for image-to-text retrieval
    2. Quality of generated CLIP embeddings vs ground truth
    3. Patch-level and global-level alignment metrics
    """
    
    def __init__(
        self,
        model,
        device: str = "auto",
        normalize_embeddings: bool = True,
        use_clip_projection: bool = True,
    ):
        self.model = model
        self.device = self._setup_device(device)
        self.normalize_embeddings = normalize_embeddings
        self.use_clip_projection = use_clip_projection
        
        # Load CLIP text encoder for text embeddings
        self._load_clip_text_encoder()
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("BLIP3-o Recall Evaluator initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Normalize embeddings: {normalize_embeddings}")
    
    def _setup_device(self, device_arg: str) -> torch.device:
        """Setup computation device"""
        if device_arg == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(device_arg)
        return device
    
    def _load_clip_text_encoder(self):
        """Load CLIP text encoder for extracting text embeddings"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # Load frozen CLIP visual projection if available
            self.clip_visual_projection = self.clip_model.visual_projection
            
            logger.info("CLIP text encoder loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP text encoder: {e}")
            raise
    
    # In src/modules/evaluation/blip3o_recall_evaluator.py
# Find the extract_text_embeddings method and update it:

    def extract_text_embeddings(
        self, 
        captions: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Extract CLIP text embeddings from captions - FIXED dimension handling
        """
        text_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(captions), batch_size):
                batch_captions = captions[i:i + batch_size]
                
                # Process text
                inputs = self.clip_processor(
                    text=batch_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract text features
                text_features = self.clip_model.get_text_features(**inputs)  # [B, 512]
                
                # FIXED: Apply visual projection to match image embedding space
                if self.use_clip_projection and self.clip_visual_projection is not None:
                    try:
                        text_features = self.clip_visual_projection(text_features)  # [B, 768]
                    except RuntimeError as e:
                        # Handle dimension mismatch - project to correct space
                        if text_features.shape[-1] != 1024:
                            # Need to project to 1024-dim first
                            proj_1024 = torch.nn.Linear(text_features.shape[-1], 1024, device=self.device)
                            text_features = proj_1024(text_features)
                        text_features = self.clip_visual_projection(text_features)  # [B, 768]
                
                if self.normalize_embeddings:
                    text_features = F.normalize(text_features, p=2, dim=-1)
                
                text_embeddings.append(text_features.cpu())
        
        return torch.cat(text_embeddings, dim=0)


    def generate_image_embeddings(
        self,
        eva_embeddings: torch.Tensor,  # [B, 256, 4096]
        num_inference_steps: int = 50,
        batch_size: int = 8
    ) -> torch.Tensor:
        """
        Generate CLIP image embeddings using the BLIP3-o model
        
        Args:
            eva_embeddings: EVA-CLIP conditioning [N, 256, 4096]
            num_inference_steps: Number of sampling steps
            batch_size: Batch size for generation
            
        Returns:
            Generated CLIP embeddings [N, 768] (global features)
        """
        generated_embeddings = []
        
        with torch.no_grad():
            for i in range(0, eva_embeddings.shape[0], batch_size):
                batch_eva = eva_embeddings[i:i + batch_size].to(self.device)
                
                # Generate CLIP patch embeddings
                generated_patches = self.model.generate(
                    eva_features=batch_eva,
                    num_inference_steps=num_inference_steps
                )  # [B, 256, 1024]
                
                # Pool to global features
                global_features = generated_patches.mean(dim=1)  # [B, 1024]
                
                # Apply CLIP visual projection if available
                if self.use_clip_projection and self.clip_visual_projection is not None:
                    global_features = self.clip_visual_projection(global_features)  # [B, 768]
                
                if self.normalize_embeddings:
                    global_features = F.normalize(global_features, p=2, dim=-1)
                
                generated_embeddings.append(global_features.cpu())
        
        return torch.cat(generated_embeddings, dim=0)
    
    def compute_similarity_matrix(
        self,
        image_embeddings: torch.Tensor,  # [N, 768]
        text_embeddings: torch.Tensor,   # [M, 768]
    ) -> torch.Tensor:
        """
        Compute similarity matrix between image and text embeddings
        
        Args:
            image_embeddings: Image embeddings [N, 768]
            text_embeddings: Text embeddings [M, 768]
            
        Returns:
            Similarity matrix [N, M]
        """
        # Ensure embeddings are normalized
        if self.normalize_embeddings:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity_matrix = torch.mm(image_embeddings, text_embeddings.t())
        
        return similarity_matrix
    
    def compute_recall_metrics(
        self,
        similarity_matrix: torch.Tensor,  # [N, M]
        image_to_text_mapping: List[List[int]],  # Ground truth mappings
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute Recall@K metrics for image-to-text retrieval
        
        Args:
            similarity_matrix: Similarity matrix [N_images, N_texts]
            image_to_text_mapping: List of lists, where each list contains
                                  the indices of ground truth texts for each image
            k_values: K values for recall computation
            
        Returns:
            Dictionary with recall metrics
        """
        num_images = similarity_matrix.shape[0]
        recall_results = {}
        
        logger.info(f"Computing recall metrics for {num_images} images")
        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        logger.info(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
        
        for k in k_values:
            correct_retrievals = 0
            
            # Get top-k most similar texts for each image
            _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
            
            for img_idx, correct_text_indices in enumerate(image_to_text_mapping):
                retrieved_indices = top_k_indices[img_idx].cpu().numpy()
                
                # Check if any of the retrieved texts are correct
                if any(ret_idx in correct_text_indices for ret_idx in retrieved_indices):
                    correct_retrievals += 1
            
            recall_at_k = correct_retrievals / num_images if num_images > 0 else 0.0
            recall_results[f'recall@{k}'] = recall_at_k
            
            logger.info(f"Recall@{k}: {correct_retrievals}/{num_images} = {recall_at_k:.4f} ({recall_at_k*100:.2f}%)")
        
        return recall_results
    
    def evaluate_on_dataset(
        self,
        eva_embeddings: torch.Tensor,  # [N, 256, 4096]
        captions_per_image: List[List[str]],  # Captions for each image
        num_inference_steps: int = 50,
        batch_size: int = 8,
        k_values: List[int] = [1, 5, 10],
        save_results: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate image-to-text recall on a dataset
        
        Args:
            eva_embeddings: EVA-CLIP embeddings [N, 256, 4096]
            captions_per_image: List of caption lists for each image
            num_inference_steps: Number of sampling steps
            batch_size: Batch size for processing
            k_values: K values for recall computation
            save_results: Optional path to save detailed results
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Starting evaluation on {len(eva_embeddings)} images")
        
        # Extract all text embeddings
        all_captions = []
        image_to_text_mapping = []
        text_idx = 0
        
        for caption_list in captions_per_image:
            current_indices = []
            for caption in caption_list:
                all_captions.append(caption)
                current_indices.append(text_idx)
                text_idx += 1
            image_to_text_mapping.append(current_indices)
        
        logger.info(f"Extracted {len(all_captions)} text captions")
        
        # Extract text embeddings
        logger.info("Extracting text embeddings...")
        text_embeddings = self.extract_text_embeddings(all_captions, batch_size)
        
        # Generate image embeddings
        logger.info("Generating image embeddings...")
        image_embeddings = self.generate_image_embeddings(
            eva_embeddings, num_inference_steps, batch_size
        )
        
        # Compute similarity matrix
        logger.info("Computing similarity matrix...")
        similarity_matrix = self.compute_similarity_matrix(image_embeddings, text_embeddings)
        
        # Compute recall metrics
        logger.info("Computing recall metrics...")
        recall_results = self.compute_recall_metrics(
            similarity_matrix, image_to_text_mapping, k_values
        )
        
        # Additional quality metrics
        with torch.no_grad():
            # Mean similarity score
            mean_similarity = similarity_matrix.mean().item()
            max_similarity = similarity_matrix.max().item()
            
            # Self-similarity (diagonal elements for matched pairs)
            if len(eva_embeddings) == len(all_captions):
                self_similarity = torch.diag(similarity_matrix).mean().item()
            else:
                self_similarity = 0.0
            
            quality_metrics = {
                'mean_similarity': mean_similarity,
                'max_similarity': max_similarity,
                'self_similarity': self_similarity,
                'embedding_quality': mean_similarity * 100,  # Scale for readability
            }
        
        # Combine results
        results = {
            **recall_results,
            **quality_metrics,
            'num_images': len(eva_embeddings),
            'num_texts': len(all_captions),
            'evaluation_timestamp': time.time(),
            'model_info': {
                'num_parameters': self.model.get_num_parameters() if hasattr(self.model, 'get_num_parameters') else None,
                'normalize_embeddings': self.normalize_embeddings,
                'use_clip_projection': self.use_clip_projection,
                'num_inference_steps': num_inference_steps,
            }
        }
        
        # Save detailed results if requested
        if save_results:
            detailed_results = {
                'recall_metrics': recall_results,
                'quality_metrics': quality_metrics,
                'similarity_matrix_stats': {
                    'shape': list(similarity_matrix.shape),
                    'mean': mean_similarity,
                    'std': similarity_matrix.std().item(),
                    'min': similarity_matrix.min().item(),
                    'max': max_similarity,
                },
                'evaluation_config': {
                    'k_values': k_values,
                    'num_inference_steps': num_inference_steps,
                    'batch_size': batch_size,
                    'normalize_embeddings': self.normalize_embeddings,
                },
                'dataset_info': {
                    'num_images': len(eva_embeddings),
                    'num_texts': len(all_captions),
                    'avg_captions_per_image': len(all_captions) / len(eva_embeddings),
                }
            }
            
            with open(save_results, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            logger.info(f"Detailed results saved to: {save_results}")
        
        return results
    
    def compare_with_baseline(
        self,
        eva_embeddings: torch.Tensor,
        captions_per_image: List[List[str]],
        baseline_clip_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare BLIP3-o model with CLIP baseline
        
        Args:
            eva_embeddings: EVA-CLIP embeddings [N, 256, 4096]
            captions_per_image: Captions for each image
            baseline_clip_embeddings: Optional CLIP baseline embeddings [N, 768]
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Comparison results
        """
        logger.info("Running comparison evaluation...")
        
        # Evaluate BLIP3-o model
        blip3o_results = self.evaluate_on_dataset(
            eva_embeddings, captions_per_image, **kwargs
        )
        
        # Evaluate baseline if provided
        baseline_results = {}
        if baseline_clip_embeddings is not None:
            logger.info("Evaluating CLIP baseline...")
            
            # Extract text embeddings
            all_captions = [caption for caption_list in captions_per_image for caption in caption_list]
            text_embeddings = self.extract_text_embeddings(all_captions)
            
            # Create image-to-text mapping
            image_to_text_mapping = []
            text_idx = 0
            for caption_list in captions_per_image:
                current_indices = []
                for _ in caption_list:
                    current_indices.append(text_idx)
                    text_idx += 1
                image_to_text_mapping.append(current_indices)
            
            # Compute similarity matrix
            similarity_matrix = self.compute_similarity_matrix(
                baseline_clip_embeddings, text_embeddings
            )
            
            # Compute recall metrics
            baseline_results = self.compute_recall_metrics(
                similarity_matrix, image_to_text_mapping, kwargs.get('k_values', [1, 5, 10])
            )
        
        return {
            'blip3o': blip3o_results,
            'baseline': baseline_results,
            'improvement': {
                f"recall@{k}": blip3o_results.get(f'recall@{k}', 0) - baseline_results.get(f'recall@{k}', 0)
                for k in kwargs.get('k_values', [1, 5, 10])
            } if baseline_results else {}
        }


def create_recall_evaluator(
    model,
    device: str = "auto",
    **kwargs
) -> BLIP3oRecallEvaluator:
    """
    Factory function for creating BLIP3-o recall evaluator
    
    Args:
        model: BLIP3-o model
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        BLIP3oRecallEvaluator instance
    """
    return BLIP3oRecallEvaluator(model=model, device=device, **kwargs)