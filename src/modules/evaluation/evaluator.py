"""
Main evaluator class for BLIP3-o DiT model evaluation.
FIXED: Added proper normalization and debugging for recall evaluation.
UPDATED: Now uses CLIP's visual projection for both methods to ensure fair comparison
in the aligned 768-dimensional embedding space, as recommended by literature.
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, AutoModel
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
from tqdm import tqdm
import json
import datetime

from .metrics import (
    compute_cosine_similarity, 
    compute_recall_metrics, 
    compute_alignment_metrics,
    print_metrics,
    compare_metrics,
    compute_pairwise_cosine_similarity  # FIXED: Added missing import
)
from .coco_dataset import COCOEvaluationDataset, create_coco_dataloader
from ..models.blip3o_dit import BLIP3oDiTModel
from ..config.blip3o_config import BLIP3oDiTConfig
from ..inference.blip3o_inference import BLIP3oInference

from .distance_metrics import (
    compute_comprehensive_distance_metrics,
    compute_per_sample_distances,
    analyze_distance_distribution,
    print_distance_metrics
)

logger = logging.getLogger(__name__)


class BLIP3oEvaluator:
    """
    Main evaluator class for BLIP3-o DiT model evaluation.
    
    FIXED: Now properly normalizes embeddings and includes debugging output.
    UPDATED: Now properly uses CLIP's visual projection for both methods:
    - Method (a): CLIP vision → CLIP visual projection → 768-dim aligned space → NORMALIZED
    - Method (b): Generated CLIP → CLIP visual projection → 768-dim aligned space → NORMALIZED
    
    This ensures fair comparison in the same embedding space that CLIP was trained to align.
    """
    
    def __init__(
        self,
        blip3o_model_path: Union[str, Path],
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            blip3o_model_path: Path to trained BLIP3-o DiT model
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Data type for models
        """
        self.blip3o_model_path = Path(blip3o_model_path)
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        
        # Initialize models
        self.clip_processor = None
        self.clip_model = None
        self.eva_processor = None
        self.eva_model = None
        self.blip3o_inference = None
        
        # Load all models
        self._load_models()
        
        logger.info("BLIP3-o evaluator initialized with CLIP visual projection support and proper normalization")
    
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
    
    def _load_models(self):
        """Load CLIP, EVA-CLIP, and BLIP3-o models (same as used in training)."""
        logger.info("Loading models...")
        
        # Load CLIP ViT-L/14 (same as in extract_embeddings_g.py)
        logger.info("Loading CLIP ViT-L/14...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.clip_model.eval()
        
        # UPDATED: Log CLIP projection dimensions for verification
        vision_proj_shape = self.clip_model.visual_projection.weight.shape
        text_proj_shape = self.clip_model.text_projection.weight.shape
        logger.info(f"CLIP visual projection: {vision_proj_shape} (1024 → 768)")
        logger.info(f"CLIP text projection: {text_proj_shape} (768 → 768)")
        
        # Load EVA-CLIP-8B (same as in extract_embeddings_g.py)
        logger.info("Loading EVA-CLIP-8B...")
        self.eva_model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.eva_model.eval()
        
        # Load trained BLIP3-o DiT model
        logger.info(f"Loading BLIP3-o DiT model from {self.blip3o_model_path}...")
        self.blip3o_inference = BLIP3oInference(
            model_path=self.blip3o_model_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )
        
        logger.info("All models loaded successfully")
    
    def extract_clip_text_embeddings(self, captions: List[str]) -> torch.Tensor:
        """Extract CLIP text embeddings (already 768-dim, aligned space) - NORMALIZED."""
        with torch.no_grad():
            inputs = self.clip_processor(
                text=captions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            # FIXED: Ensure inputs are on correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get text features through the standard CLIP path (includes text projection)
            text_embeddings = self.clip_model.get_text_features(**inputs)
            # FIXED: Ensure normalization (get_text_features should already normalize, but let's be explicit)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings.cpu().float()  # Move to CPU for consistency
    
    def extract_clip_vision_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract CLIP vision embeddings using CLIP's visual projection.
        FIXED: Now properly normalizes the output embeddings.
        UPDATED: Now applies CLIP's visual projection to get 768-dim aligned features.
        FIXED: Proper device handling to avoid CUDA/CPU mismatch.
        """
        clip_embeddings = []
        
        with torch.no_grad():
            for img in images:
                # Use same processing as in extract_embeddings_g.py
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                         for k, v in inputs.items()}
                
                vision_outputs = self.clip_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token) -> [1, 256, 1024]
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
                batch_size, num_patches, hidden_dim = patch_embeddings.shape
                
                # Average pool to get global representation -> [1, 1024]
                vision_global = patch_embeddings.mean(dim=1)  # [1, 1024]
                
                # UPDATED: Apply CLIP's visual projection to get aligned 768-dim features
                # FIXED: Ensure tensor is on same device as model before projection
                vision_global = vision_global.to(device=self.device, dtype=self.torch_dtype)
                vision_projected = self.clip_model.visual_projection(vision_global)  # [1, 768]
                
                # FIXED: Add proper normalization - THIS WAS THE MAIN ISSUE!
                vision_projected = F.normalize(vision_projected, p=2, dim=-1)
                
                clip_embeddings.append(vision_projected.squeeze(0).cpu().float())  # [768]
        
        return torch.stack(clip_embeddings)  # [B, 768] - all normalized
    
    def extract_eva_vision_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract EVA-CLIP vision embeddings using same method as training."""
        eva_embeddings = []
        
        with torch.no_grad():
            for img in images:
                # Use same processing as in extract_embeddings_g.py
                inputs = self.eva_processor(images=img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device=self.device, dtype=self.torch_dtype)
                
                vision_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token) -> [1, 256, hidden_dim]
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
                batch_size, num_patches, hidden_dim = patch_embeddings.shape
                
                # Reshape to 16x16 grid -> [1, 16, 16, hidden_dim]
                grid_size = int(np.sqrt(num_patches))  # 16
                spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
                
                # Convert to 256 tokens format -> [256, hidden_dim]
                tokens = spatial_grid.reshape(num_patches, hidden_dim)
                eva_embeddings.append(tokens.cpu().float())
        
        return torch.stack(eva_embeddings)  # [B, 256, 4096]
    
    def generate_clip_from_eva(self, eva_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate CLIP embeddings from EVA-CLIP using trained BLIP3-o DiT.
        FIXED: Now properly normalizes the output embeddings.
        UPDATED: Now applies CLIP's visual projection to generated embeddings.
        FIXED: Proper device handling to avoid CUDA/CPU mismatch.
        """
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        with torch.no_grad():
            # Generate CLIP-like embeddings using BLIP3-o
            generated_clip = self.blip3o_inference.generate(
                eva_embeddings,  # [B, 256, 4096]
                num_inference_steps=50,
            )  # -> [B, 256, 1024]
            
            # Average pool to get global representation -> [B, 1024]
            generated_global = generated_clip.mean(dim=1)  # [B, 1024]
            
            # UPDATED: Apply CLIP's visual projection to align with text space
            # FIXED: Ensure tensor is on same device as model before projection
            generated_global = generated_global.to(device=self.device, dtype=self.torch_dtype)
            generated_projected = self.clip_model.visual_projection(generated_global)  # [B, 768]
            
            # FIXED: Add proper normalization - THIS WAS THE SECOND MAIN ISSUE!
            generated_projected = F.normalize(generated_projected, p=2, dim=-1)
        
        return generated_projected.cpu().float()  # [B, 768] - all normalized
    
    def evaluate_alignment(
        self,
        coco_root: Union[str, Path],
        max_samples: Optional[int] = None,
        batch_size: int = 16,
        save_results: bool = True,
        results_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate alignment using cosine similarity (Task 1).
        UPDATED: Both methods now use CLIP's visual projection for fair comparison.
        
        Args:
            coco_root: Path to MS-COCO dataset
            max_samples: Maximum samples to evaluate (None for all)
            batch_size: Batch size for processing
            save_results: Whether to save detailed results
            results_dir: Directory to save results
            
        Returns:
            Dictionary containing alignment metrics
        """
        logger.info("Starting alignment evaluation (Task 1) with CLIP visual projection...")
        
        # Create COCO dataloader
        dataloader = create_coco_dataloader(
            coco_root=coco_root,
            batch_size=batch_size,
            max_samples=max_samples,
            shuffle=False,
            num_workers=4,
        )
        
        all_results = {
            'clip_text_clip_vision_similarities': [],
            'clip_text_generated_similarities': [],
            'image_ids': [],
            'captions': [],
        }
        
        logger.info(f"Processing {len(dataloader)} batches...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating alignment")):
            images = batch['images']
            captions_batch = batch['captions']
            image_ids = batch['image_ids']
            
            # Flatten captions (each image can have multiple captions)
            flat_captions = []
            flat_image_indices = []
            for i, caption_list in enumerate(captions_batch):
                for caption in caption_list:
                    flat_captions.append(caption)
                    flat_image_indices.append(i)
            
            if not flat_captions:
                continue
            
            # Extract embeddings
            try:
                # (a) CLIP text embeddings (already 768-dim, aligned)
                clip_text_embeddings = self.extract_clip_text_embeddings(flat_captions)
                
                # CLIP vision embeddings for corresponding images (NOW with visual projection)
                selected_images = [images[i] for i in flat_image_indices]
                clip_vision_embeddings = self.extract_clip_vision_embeddings(selected_images)
                # clip_vision_embeddings is now [N, 768] - already projected and normalized
                
                # (b) EVA-CLIP -> BLIP3-o DiT -> Generated CLIP embeddings (with visual projection)
                eva_vision_embeddings = self.extract_eva_vision_embeddings(selected_images)
                generated_clip_embeddings = self.generate_clip_from_eva(eva_vision_embeddings)
                # generated_clip_embeddings is now [N, 768] - already projected and normalized
                
                # FIXED: Remove redundant normalization since embeddings are already normalized
                # But ensure they're properly normalized
                clip_vision_embeddings = F.normalize(clip_vision_embeddings, p=2, dim=-1)
                generated_clip_embeddings = F.normalize(generated_clip_embeddings, p=2, dim=-1)
                
                # Compute cosine similarities in the aligned 768-dim space
                clip_text_clip_vision_sim = compute_cosine_similarity(
                    clip_text_embeddings, clip_vision_embeddings
                )
                
                clip_text_generated_sim = compute_cosine_similarity(
                    clip_text_embeddings, generated_clip_embeddings
                )
                
                # Store results
                all_results['clip_text_clip_vision_similarities'].extend(
                    clip_text_clip_vision_sim.cpu().numpy().tolist()
                )
                all_results['clip_text_generated_similarities'].extend(
                    clip_text_generated_sim.cpu().numpy().tolist()
                )
                all_results['image_ids'].extend([image_ids[i] for i in flat_image_indices])
                all_results['captions'].extend(flat_captions)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Compute summary metrics
        if all_results['clip_text_clip_vision_similarities']:
            clip_vision_similarities = np.array(all_results['clip_text_clip_vision_similarities'])
            generated_similarities = np.array(all_results['clip_text_generated_similarities'])
            
            metrics = {
                # Method (a): CLIP text + CLIP vision (with visual projection)
                'clip_text_clip_vision_mean': float(np.mean(clip_vision_similarities)),
                'clip_text_clip_vision_std': float(np.std(clip_vision_similarities)),
                'clip_text_clip_vision_min': float(np.min(clip_vision_similarities)),
                'clip_text_clip_vision_max': float(np.max(clip_vision_similarities)),
                
                # Method (b): CLIP text + Generated CLIP (with visual projection)
                'clip_text_generated_mean': float(np.mean(generated_similarities)),
                'clip_text_generated_std': float(np.std(generated_similarities)),
                'clip_text_generated_min': float(np.min(generated_similarities)),
                'clip_text_generated_max': float(np.max(generated_similarities)),
                
                # Differences
                'difference_mean': float(np.mean(generated_similarities - clip_vision_similarities)),
                'difference_std': float(np.std(generated_similarities - clip_vision_similarities)),
                'difference_abs_mean': float(np.mean(np.abs(generated_similarities - clip_vision_similarities))),
                
                # Additional stats
                'num_samples': len(clip_vision_similarities),
                'correlation': float(np.corrcoef(clip_vision_similarities, generated_similarities)[0, 1]),
                
                # Embedding space info
                'embedding_space': 'clip_aligned_768dim',
                'uses_visual_projection': True,
                'normalization_applied': True,
            }
        else:
            metrics = {'error': 'No valid samples processed'}
        
        # Save results if requested
        if save_results and results_dir:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            with open(results_dir / 'alignment_detailed_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Save summary metrics
            with open(results_dir / 'alignment_summary_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Alignment results saved to {results_dir}")
        
        return metrics
    
    def evaluate_recall(
        self,
        coco_root: Union[str, Path],
        max_samples: Optional[int] = None,
        batch_size: int = 16,
        k_values: List[int] = [1, 5, 10],
        save_results: bool = True,
        results_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate recall metrics for image-to-text retrieval (Task 2).
        FIXED: Now properly normalizes embeddings and includes debugging output.
        UPDATED: Both methods now use CLIP's visual projection for fair comparison.
        
        (a) Image → CLIP ViT-L/14 → CLIP visual projection → retrieval against text embeddings
        (b) Image → EVA-CLIP → BLIP3-o DiT → CLIP visual projection → retrieval against text embeddings
        
        Args:
            coco_root: Path to MS-COCO dataset
            max_samples: Maximum samples to evaluate
            batch_size: Batch size for processing
            k_values: K values for Recall@K computation
            save_results: Whether to save detailed results
            results_dir: Directory to save results
            
        Returns:
            Dictionary containing recall metrics
        """
        logger.info("Starting FIXED image-to-text recall evaluation (Task 2) with CLIP visual projection...")
        
        # Create COCO dataloader
        dataloader = create_coco_dataloader(
            coco_root=coco_root,
            batch_size=batch_size,
            max_samples=max_samples,
            shuffle=False,
            num_workers=4,
        )
        
        # Collect all embeddings and create proper query-gallery structure
        all_image_clip_embeddings = []      # Image embeddings from CLIP (with visual projection)
        all_image_generated_embeddings = [] # Image embeddings from EVA->BLIP3o (with visual projection)
        all_text_embeddings = []            # Text embeddings (gallery)
        image_to_text_mapping = []          # Maps image indices to their text indices
        
        logger.info(f"Extracting embeddings from {len(dataloader)} batches...")
        
        text_idx = 0  # Global text index counter
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            images = batch['images']
            captions_batch = batch['captions']
            image_ids = batch['image_ids']
            
            # Process each image and its captions
            for i, (image, caption_list) in enumerate(zip(images, captions_batch)):
                image_id = image_ids[i]
                
                try:
                    # Extract image embeddings using both methods (both with visual projection)
                    
                    # Method (a): CLIP vision embeddings (with visual projection + normalization)
                    clip_vision_emb = self.extract_clip_vision_embeddings([image])
                    clip_vision_global = clip_vision_emb.squeeze(0).cpu()  # [768] - already projected & normalized
                    
                    # Method (b): EVA-CLIP -> Generated CLIP embeddings (with visual projection + normalization)
                    eva_vision_emb = self.extract_eva_vision_embeddings([image])
                    generated_clip_emb = self.generate_clip_from_eva(eva_vision_emb)
                    generated_clip_global = generated_clip_emb.squeeze(0).cpu()  # [768] - already projected & normalized
                    
                    # Extract text embeddings for all captions of this image
                    text_emb = self.extract_clip_text_embeddings(caption_list)  # [num_captions, 768] - already normalized
                    
                    # Store image embeddings (one per image)
                    all_image_clip_embeddings.append(clip_vision_global)
                    all_image_generated_embeddings.append(generated_clip_global)
                    
                    # Store text embeddings and mapping
                    current_text_indices = []
                    for j, caption in enumerate(caption_list):
                        all_text_embeddings.append(text_emb[j])
                        current_text_indices.append(text_idx)
                        text_idx += 1
                    
                    # Map this image to its corresponding text indices
                    image_to_text_mapping.append(current_text_indices)
                
                except Exception as e:
                    logger.error(f"Error processing image {image_id}: {e}")
                    continue
        
        if not all_image_clip_embeddings:
            return {'error': 'No valid embeddings extracted'}
        
        # Convert to tensors
        image_clip_embeddings = torch.stack(all_image_clip_embeddings)      # [N_images, 768]
        image_generated_embeddings = torch.stack(all_image_generated_embeddings)  # [N_images, 768]
        text_embeddings = torch.stack(all_text_embeddings)                 # [N_texts, 768]
        
        logger.info(f"Collected {len(image_clip_embeddings)} images and {len(text_embeddings)} texts")
        logger.info("All embeddings are in the same CLIP-aligned 768-dimensional space with proper normalization")
        
        # FIXED: Verify embeddings are normalized (debugging output)
        image_clip_norms = torch.norm(image_clip_embeddings, p=2, dim=-1)
        image_gen_norms = torch.norm(image_generated_embeddings, p=2, dim=-1)
        text_norms = torch.norm(text_embeddings, p=2, dim=-1)
        
        logger.info(f"CLIP image embedding norms - mean: {image_clip_norms.mean():.4f}, std: {image_clip_norms.std():.4f}")
        logger.info(f"Generated image embedding norms - mean: {image_gen_norms.mean():.4f}, std: {image_gen_norms.std():.4f}")
        logger.info(f"Text embedding norms - mean: {text_norms.mean():.4f}, std: {text_norms.std():.4f}")
        
        # Compute recall metrics for image-to-text retrieval
        logger.info("Computing image-to-text recall metrics...")
        
        # Method (a): Image (CLIP vision with projection) to text retrieval
        recall_metrics_clip = self._compute_image_to_text_recall_fixed(
            image_embeddings=image_clip_embeddings,
            text_embeddings=text_embeddings,
            image_to_text_mapping=image_to_text_mapping,
            k_values=k_values
        )
        
        # Method (b): Image (Generated CLIP with projection) to text retrieval
        recall_metrics_generated = self._compute_image_to_text_recall_fixed(
            image_embeddings=image_generated_embeddings,
            text_embeddings=text_embeddings,
            image_to_text_mapping=image_to_text_mapping,
            k_values=k_values
        )
        
        # Combine results with prefixes
        combined_metrics = {}
        
        # Add method (a) results
        for key, value in recall_metrics_clip.items():
            combined_metrics[f'clip_vision_{key}'] = value
        
        # Add method (b) results
        for key, value in recall_metrics_generated.items():
            combined_metrics[f'generated_{key}'] = value
        
        # Compute differences
        for k in k_values:
            recall_a = recall_metrics_clip[f'recall@{k}']
            recall_b = recall_metrics_generated[f'recall@{k}']
            combined_metrics[f'recall@{k}_difference'] = recall_b - recall_a
            combined_metrics[f'recall@{k}_relative_change'] = (recall_b - recall_a) / recall_a * 100 if recall_a > 0 else 0
        
        # Add embedding space info
        combined_metrics['embedding_space'] = 'clip_aligned_768dim'
        combined_metrics['uses_visual_projection'] = True
        combined_metrics['normalization_applied'] = True
        
        # Save results if requested
        if save_results and results_dir:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            detailed_results = {
                'image_clip_embeddings': image_clip_embeddings.cpu().numpy().tolist(),
                'image_generated_embeddings': image_generated_embeddings.cpu().numpy().tolist(),
                'text_embeddings': text_embeddings.cpu().numpy().tolist(),
                'image_to_text_mapping': image_to_text_mapping,
                'metrics': combined_metrics,
                'evaluation_info': {
                    'embedding_space': 'clip_aligned_768dim',
                    'uses_visual_projection': True,
                    'clip_vision_projection_applied': True,
                    'generated_embeddings_projection_applied': True,
                    'normalization_applied': True,
                }
            }
            
            with open(results_dir / 'recall_detailed_results.json', 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            # Save summary metrics
            with open(results_dir / 'recall_summary_metrics.json', 'w') as f:
                json.dump(combined_metrics, f, indent=2)
            
            logger.info(f"Recall results saved to {results_dir}")
        
        return combined_metrics
    
    def _compute_image_to_text_recall_fixed(
        self,
        image_embeddings: torch.Tensor,     # [N_images, 768] (already projected & normalized)
        text_embeddings: torch.Tensor,      # [N_texts, 768] (already aligned & normalized)
        image_to_text_mapping: List[List[int]],  # Maps image idx to list of text indices
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute recall metrics for image-to-text retrieval.
        FIXED: Now includes proper normalization checks and debugging output.
        UPDATED: Now works with projected 768-dim embeddings for both image and text.
        
        Args:
            image_embeddings: Image embeddings [N_images, 768] (projected to aligned space)
            text_embeddings: Text embeddings [N_texts, 768] (already in aligned space)
            image_to_text_mapping: Maps each image to its corresponding text indices
            k_values: K values for Recall@K
            
        Returns:
            Dictionary of recall metrics
        """
        # FIXED: Ensure both embeddings are normalized before similarity computation
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Verify normalization (debugging)
        img_norms = torch.norm(image_embeddings, p=2, dim=-1)
        text_norms = torch.norm(text_embeddings, p=2, dim=-1)
        logger.info(f"DEBUG: Image norm check - mean: {img_norms.mean():.6f}, std: {img_norms.std():.6f}")
        logger.info(f"DEBUG: Text norm check - mean: {text_norms.mean():.6f}, std: {text_norms.std():.6f}")
        
        # Compute similarity matrix: [N_images, N_texts]
        # Both embeddings are now in the same 768-dim aligned space
        similarity_matrix = compute_pairwise_cosine_similarity(
            image_embeddings, text_embeddings
        )
        
        logger.info(f"DEBUG: Similarity matrix shape: {similarity_matrix.shape}")
        logger.info(f"DEBUG: Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
        logger.info(f"DEBUG: Similarity mean: {similarity_matrix.mean():.4f}, std: {similarity_matrix.std():.4f}")
        
        # Compute recall for each K value
        recall_results = {}
        
        for k in k_values:
            correct_retrievals = 0
            total_queries = len(image_to_text_mapping)
            
            # Get top-k text indices for each image
            _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)  # [N_images, k]
            
            # Debug first few retrievals for k=1
            if k == 1:
                logger.info("DEBUG: First 5 image-to-text retrievals:")
                for i in range(min(5, len(image_to_text_mapping))):
                    correct_texts = image_to_text_mapping[i]
                    retrieved_text = top_k_indices[i, 0].item()
                    similarity_score = similarity_matrix[i, retrieved_text].item()
                    is_correct = retrieved_text in correct_texts
                    logger.info(f"  Image {i}: retrieved text {retrieved_text}, similarity {similarity_score:.4f}, "
                               f"correct: {is_correct}, correct_texts: {correct_texts}")
            
            # Check if any of the top-k retrieved texts belong to the query image
            for img_idx, correct_text_indices in enumerate(image_to_text_mapping):
                retrieved_indices = top_k_indices[img_idx].cpu().numpy()
                
                # Check if any retrieved index is in the correct text indices
                if any(ret_idx in correct_text_indices for ret_idx in retrieved_indices):
                    correct_retrievals += 1
            
            recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0
            recall_results[f'recall@{k}'] = recall_at_k
            
            logger.info(f"DEBUG: Recall@{k}: {correct_retrievals}/{total_queries} = {recall_at_k:.4f} ({recall_at_k*100:.2f}%)")
        
        # Additional metrics
        recall_results['num_queries'] = len(image_to_text_mapping)
        recall_results['num_gallery'] = len(text_embeddings)
        recall_results['avg_texts_per_image'] = np.mean([len(texts) for texts in image_to_text_mapping])
        recall_results['embedding_dim'] = image_embeddings.shape[1]  # Should be 768
        
        return recall_results

    # ==================== DISTANCE EVALUATION METHODS ====================
    
    def evaluate_distance(
        self,
        coco_root: Union[str, Path],
        max_samples: Optional[int] = None,
        batch_size: int = 16,
        use_visual_projection: bool = True,
        save_results: bool = True,
        results_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate distance metrics between target CLIP embeddings and predicted embeddings (Task 3).
        
        This method computes various distance metrics between:
        - Target: CLIP ViT-L/14 vision embeddings (ground truth)
        - Predicted: Generated CLIP embeddings from EVA-CLIP → BLIP3-o DiT
        
        Args:
            coco_root: Path to MS-COCO dataset
            max_samples: Maximum samples to evaluate (None for all)
            batch_size: Batch size for processing
            use_visual_projection: Whether to use CLIP's visual projection for fair comparison
            save_results: Whether to save detailed results
            results_dir: Directory to save results
            
        Returns:
            Dictionary containing distance metrics
        """
        logger.info("Starting distance evaluation (Task 3)...")
        logger.info(f"Using CLIP visual projection: {use_visual_projection}")
        
        # Create COCO dataloader
        dataloader = create_coco_dataloader(
            coco_root=coco_root,
            batch_size=batch_size,
            max_samples=max_samples,
            shuffle=False,
            num_workers=4,
        )
        
        all_target_embeddings = []
        all_predicted_embeddings = []
        all_image_ids = []
        all_captions = []
        
        logger.info(f"Processing {len(dataloader)} batches...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing distance metrics")):
            images = batch['images']
            captions_batch = batch['captions']
            image_ids = batch['image_ids']
            
            # Process each image separately to get individual embeddings
            for i, (image, caption_list) in enumerate(zip(images, captions_batch)):
                image_id = image_ids[i]
                
                try:
                    # Extract target CLIP vision embeddings
                    if use_visual_projection:
                        # Method: CLIP vision → visual projection → 768-dim aligned
                        target_clip_emb = self.extract_clip_vision_embeddings([image])  # [1, 768]
                        target_embedding = target_clip_emb.squeeze(0).cpu()  # [768]
                    else:
                        # Method: CLIP vision → raw 1024-dim (without projection)
                        target_embedding = self._extract_clip_vision_raw([image]).squeeze(0).cpu()  # [1024]
                    
                    # Generate predicted CLIP embeddings
                    eva_vision_emb = self.extract_eva_vision_embeddings([image])  # [1, 256, 4096]
                    
                    if use_visual_projection:
                        # Method: EVA → BLIP3-o → visual projection → 768-dim aligned
                        predicted_clip_emb = self.generate_clip_from_eva(eva_vision_emb)  # [1, 768]
                        predicted_embedding = predicted_clip_emb.squeeze(0).cpu()  # [768]
                    else:
                        # Method: EVA → BLIP3-o → raw 1024-dim (without projection)
                        predicted_embedding = self._generate_clip_from_eva_raw(eva_vision_emb).squeeze(0).cpu()  # [1024]
                    
                    # Store embeddings
                    all_target_embeddings.append(target_embedding)
                    all_predicted_embeddings.append(predicted_embedding)
                    all_image_ids.append(image_id)
                    all_captions.extend(caption_list)
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_id}: {e}")
                    continue
        
        if not all_target_embeddings:
            return {'error': 'No valid embeddings extracted'}
        
        # Convert to tensors
        target_embeddings = torch.stack(all_target_embeddings)  # [N, D]
        predicted_embeddings = torch.stack(all_predicted_embeddings)  # [N, D]
        
        logger.info(f"Collected {len(target_embeddings)} embeddings")
        embedding_dim = target_embeddings.shape[-1]
        logger.info(f"Embedding dimension: {embedding_dim} ({'aligned with visual projection' if use_visual_projection else 'raw CLIP features'})")
        
        # Compute comprehensive distance metrics
        logger.info("Computing distance metrics...")
        
        # Compute main distance metrics
        distance_metrics = compute_comprehensive_distance_metrics(
            target_embeddings=target_embeddings,
            predicted_embeddings=predicted_embeddings
        )
        
        # Add evaluation configuration info
        distance_metrics.update({
            'embedding_space': f'clip_aligned_{embedding_dim}dim' if use_visual_projection else f'clip_raw_{embedding_dim}dim',
            'uses_visual_projection': use_visual_projection,
            'evaluation_method': 'distance_metrics',
            'num_images_evaluated': len(target_embeddings),
            'coco_dataset': str(coco_root),
        })
        
        # Compute per-sample distances for detailed analysis
        per_sample_distances = compute_per_sample_distances(
            target_embeddings=target_embeddings,
            predicted_embeddings=predicted_embeddings
        )
        
        # Analyze distance distributions
        distribution_analysis = analyze_distance_distribution(
            target_embeddings=target_embeddings,
            predicted_embeddings=predicted_embeddings
        )
        
        # Save results if requested
        if save_results and results_dir:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary metrics
            summary_file = results_dir / 'distance_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(distance_metrics, f, indent=2)
            
            # Save detailed per-sample results
            detailed_results = {
                'target_embeddings': target_embeddings.cpu().numpy().tolist(),
                'predicted_embeddings': predicted_embeddings.cpu().numpy().tolist(),
                'image_ids': all_image_ids,
                'captions': all_captions[:len(all_image_ids)],  # Match length
                'per_sample_distances': {
                    k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in per_sample_distances.items()
                },
                'distance_metrics': distance_metrics,
                'distribution_analysis': distribution_analysis,
                'evaluation_info': {
                    'embedding_space': distance_metrics['embedding_space'],
                    'uses_visual_projection': use_visual_projection,
                    'evaluation_date': str(datetime.datetime.now()),
                    'coco_root': str(coco_root),
                    'max_samples': max_samples,
                    'batch_size': batch_size,
                }
            }
            
            with open(results_dir / 'distance_detailed_results.json', 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            # Save distribution analysis separately for easier access
            with open(results_dir / 'distance_distribution_analysis.json', 'w') as f:
                json.dump(distribution_analysis, f, indent=2)
            
            logger.info(f"Distance evaluation results saved to {results_dir}")
        
        return distance_metrics

    def _extract_clip_vision_raw(self, images: List) -> torch.Tensor:
        """
        Extract raw CLIP vision embeddings without visual projection.
        Helper method for distance evaluation without projection.
        """
        clip_embeddings = []
        
        with torch.no_grad():
            for img in images:
                # Use same processing as in extract_embeddings_g.py
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                         for k, v in inputs.items()}
                
                vision_outputs = self.clip_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token) -> [1, 256, 1024]
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
                
                # Average pool to get global representation -> [1, 1024]
                vision_global = patch_embeddings.mean(dim=1)  # [1, 1024]
                
                # NO visual projection - keep raw 1024-dim features
                clip_embeddings.append(vision_global.squeeze(0).cpu().float())  # [1024]
        
        return torch.stack(clip_embeddings)  # [B, 1024]

    def _generate_clip_from_eva_raw(self, eva_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate raw CLIP embeddings from EVA-CLIP without visual projection.
        Helper method for distance evaluation without projection.
        """
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        with torch.no_grad():
            # Generate CLIP-like embeddings using BLIP3-o
            generated_clip = self.blip3o_inference.generate(
                eva_embeddings,  # [B, 256, 4096]
                num_inference_steps=50,
            )  # -> [B, 256, 1024]
            
            # Average pool to get global representation -> [B, 1024]
            generated_global = generated_clip.mean(dim=1)  # [B, 1024]
            
            # NO visual projection - keep raw 1024-dim features
        
        return generated_global.cpu().float()


if __name__ == "__main__":
    # Test the evaluator
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <blip3o_model_path> <coco_root>")
        sys.exit(1)
    
    blip3o_model_path = sys.argv[1]
    coco_root = sys.argv[2]
    
    print("🧪 Testing BLIP3-o evaluator with FIXED normalization and CLIP visual projection...")
    
    try:
        # Initialize evaluator
        evaluator = BLIP3oEvaluator(
            blip3o_model_path=blip3o_model_path,
            device="auto",
        )
        
        print("✅ Evaluator initialized")
        
        # Test alignment evaluation with small sample
        print("Testing alignment evaluation...")
        alignment_metrics = evaluator.evaluate_alignment(
            coco_root=coco_root,
            max_samples=10,
            batch_size=2,
            save_results=False,
        )
        
        print_metrics(alignment_metrics, "Alignment Test Results")
        
        # Test recall evaluation with small sample
        print("Testing recall evaluation...")
        recall_metrics = evaluator.evaluate_recall(
            coco_root=coco_root,
            max_samples=10,
            batch_size=2,
            save_results=False,
        )
        
        print_metrics(recall_metrics, "Recall Test Results")
        
        # Test distance evaluation with small sample
        print("Testing distance evaluation...")
        distance_metrics = evaluator.evaluate_distance(
            coco_root=coco_root,
            max_samples=10,
            batch_size=2,
            save_results=False,
        )
        
        print_distance_metrics(distance_metrics, "Distance Test Results")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()