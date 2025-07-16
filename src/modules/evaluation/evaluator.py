"""
Main evaluator class for BLIP3-o DiT model evaluation.
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

from .metrics import (
    compute_cosine_similarity, 
    compute_recall_metrics, 
    compute_alignment_metrics,
    print_metrics,
    compare_metrics
)
from .coco_dataset import COCOEvaluationDataset, create_coco_dataloader
from ..models.blip3o_dit import BLIP3oDiTModel
from ..config.blip3o_config import BLIP3oDiTConfig
from ..inference.blip3o_inference import BLIP3oInference

logger = logging.getLogger(__name__)


class BLIP3oEvaluator:
    """
    Main evaluator class for BLIP3-o DiT model evaluation.
    
    Handles:
    - Loading CLIP and EVA-CLIP models (same as used in training)
    - Loading trained BLIP3-o DiT model
    - Alignment evaluation (Task 1)
    - Recall evaluation (Task 2)
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
        
        logger.info("BLIP3-o evaluator initialized")
    
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
        """Extract CLIP text embeddings."""
        with torch.no_grad():
            inputs = self.clip_processor(
                text=captions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_embeddings = self.clip_model.get_text_features(**inputs)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        return text_embeddings
    
    def extract_clip_vision_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract CLIP vision embeddings using same method as training."""
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
                
                # Reshape to 16x16 grid -> [1, 16, 16, 1024]
                grid_size = int(np.sqrt(num_patches))  # 16
                spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
                
                # Convert to 256 tokens format -> [256, 1024]
                tokens = spatial_grid.reshape(num_patches, hidden_dim)
                clip_embeddings.append(tokens.cpu().float())
        
        return torch.stack(clip_embeddings)  # [B, 256, 1024]
    
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
        """Generate CLIP embeddings from EVA-CLIP using trained BLIP3-o DiT."""
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        with torch.no_grad():
            generated_clip = self.blip3o_inference.generate(
                encoder_hidden_states=eva_embeddings,
                num_inference_steps=50,  # Can be adjusted
            )
        
        return generated_clip.cpu().float()
    
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
        
        Args:
            coco_root: Path to MS-COCO dataset
            max_samples: Maximum samples to evaluate (None for all)
            batch_size: Batch size for processing
            save_results: Whether to save detailed results
            results_dir: Directory to save results
            
        Returns:
            Dictionary containing alignment metrics
        """
        logger.info("Starting alignment evaluation (Task 1)...")
        
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
                # (a) CLIP text embeddings
                clip_text_embeddings = self.extract_clip_text_embeddings(flat_captions)
                
                # CLIP vision embeddings for corresponding images
                selected_images = [images[i] for i in flat_image_indices]
                clip_vision_embeddings = self.extract_clip_vision_embeddings(selected_images)
                
                # Average pool CLIP vision embeddings to get global representation
                clip_vision_global = clip_vision_embeddings.mean(dim=1)  # [N, 1024]
                clip_vision_global = F.normalize(clip_vision_global, p=2, dim=-1)
                
                # (b) EVA-CLIP -> BLIP3-o DiT -> Generated CLIP embeddings
                eva_vision_embeddings = self.extract_eva_vision_embeddings(selected_images)
                generated_clip_embeddings = self.generate_clip_from_eva(eva_vision_embeddings)
                
                # Average pool generated CLIP embeddings
                generated_clip_global = generated_clip_embeddings.mean(dim=1)  # [N, 1024]
                generated_clip_global = F.normalize(generated_clip_global, p=2, dim=-1)
                
                # Compute cosine similarities
                clip_text_clip_vision_sim = compute_cosine_similarity(
                    clip_text_embeddings, clip_vision_global
                )
                
                clip_text_generated_sim = compute_cosine_similarity(
                    clip_text_embeddings, generated_clip_global
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
                # Method (a): CLIP text + CLIP vision
                'clip_text_clip_vision_mean': float(np.mean(clip_vision_similarities)),
                'clip_text_clip_vision_std': float(np.std(clip_vision_similarities)),
                'clip_text_clip_vision_min': float(np.min(clip_vision_similarities)),
                'clip_text_clip_vision_max': float(np.max(clip_vision_similarities)),
                
                # Method (b): CLIP text + Generated CLIP (from EVA)
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
        
        (a) Image → CLIP ViT-L/14 → retrieval against text embeddings
        (b) Image → EVA-CLIP → BLIP3-o DiT → retrieval against text embeddings
        
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
        logger.info("Starting image-to-text recall evaluation (Task 2)...")
        
        # Create COCO dataloader
        dataloader = create_coco_dataloader(
            coco_root=coco_root,
            batch_size=batch_size,
            max_samples=max_samples,
            shuffle=False,
            num_workers=4,
        )
        
        # Collect all embeddings and create proper query-gallery structure
        all_image_clip_embeddings = []      # Image embeddings from CLIP
        all_image_generated_embeddings = [] # Image embeddings from EVA->BLIP3o
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
                    # Extract image embeddings using both methods
                    
                    # Method (a): CLIP vision embeddings
                    clip_vision_emb = self.extract_clip_vision_embeddings([image])
                    clip_vision_global = clip_vision_emb.mean(dim=1).squeeze(0)  # [1024]
                    
                    # Method (b): EVA-CLIP -> Generated CLIP embeddings
                    eva_vision_emb = self.extract_eva_vision_embeddings([image])
                    generated_clip_emb = self.generate_clip_from_eva(eva_vision_emb)
                    generated_clip_global = generated_clip_emb.mean(dim=1).squeeze(0)  # [1024]
                    
                    # Extract text embeddings for all captions of this image
                    text_emb = self.extract_clip_text_embeddings(caption_list)  # [num_captions, 1024]
                    
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
        image_clip_embeddings = torch.stack(all_image_clip_embeddings)      # [N_images, 1024]
        image_generated_embeddings = torch.stack(all_image_generated_embeddings)  # [N_images, 1024]
        text_embeddings = torch.stack(all_text_embeddings)                 # [N_texts, 1024]
        
        logger.info(f"Collected {len(image_clip_embeddings)} images and {len(text_embeddings)} texts")
        
        # Compute recall metrics for image-to-text retrieval
        logger.info("Computing image-to-text recall metrics...")
        
        # Method (a): Image (CLIP vision) to text retrieval
        recall_metrics_clip = self._compute_image_to_text_recall(
            image_embeddings=image_clip_embeddings,
            text_embeddings=text_embeddings,
            image_to_text_mapping=image_to_text_mapping,
            k_values=k_values
        )
        
        # Method (b): Image (Generated CLIP) to text retrieval
        recall_metrics_generated = self._compute_image_to_text_recall(
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
            }
            
            with open(results_dir / 'recall_detailed_results.json', 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            # Save summary metrics
            with open(results_dir / 'recall_summary_metrics.json', 'w') as f:
                json.dump(combined_metrics, f, indent=2)
            
            logger.info(f"Recall results saved to {results_dir}")
        
        return combined_metrics
    
    def _compute_image_to_text_recall(
        self,
        image_embeddings: torch.Tensor,     # [N_images, D]
        text_embeddings: torch.Tensor,      # [N_texts, D]
        image_to_text_mapping: List[List[int]],  # Maps image idx to list of text indices
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute recall metrics for image-to-text retrieval.
        
        Args:
            image_embeddings: Image embeddings [N_images, D]
            text_embeddings: Text embeddings [N_texts, D]
            image_to_text_mapping: Maps each image to its corresponding text indices
            k_values: K values for Recall@K
            
        Returns:
            Dictionary of recall metrics
        """
        from .metrics import compute_pairwise_cosine_similarity
        
        # Compute similarity matrix: [N_images, N_texts]
        similarity_matrix = compute_pairwise_cosine_similarity(
            image_embeddings, text_embeddings
        )
        
        # Compute recall for each K value
        recall_results = {}
        
        for k in k_values:
            correct_retrievals = 0
            total_queries = len(image_to_text_mapping)
            
            # Get top-k text indices for each image
            _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)  # [N_images, k]
            
            # Check if any of the top-k retrieved texts belong to the query image
            for img_idx, correct_text_indices in enumerate(image_to_text_mapping):
                retrieved_indices = top_k_indices[img_idx].cpu().numpy()
                
                # Check if any retrieved index is in the correct text indices
                if any(ret_idx in correct_text_indices for ret_idx in retrieved_indices):
                    correct_retrievals += 1
            
            recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0
            recall_results[f'recall@{k}'] = recall_at_k
        
        # Additional metrics
        recall_results['num_queries'] = len(image_to_text_mapping)
        recall_results['num_gallery'] = len(text_embeddings)
        recall_results['avg_texts_per_image'] = np.mean([len(texts) for texts in image_to_text_mapping])
        
        return recall_results


if __name__ == "__main__":
    # Test the evaluator
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python evaluator.py <blip3o_model_path> <coco_root>")
        sys.exit(1)
    
    blip3o_model_path = sys.argv[1]
    coco_root = sys.argv[2]
    
    print("🧪 Testing BLIP3-o evaluator...")
    
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
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()