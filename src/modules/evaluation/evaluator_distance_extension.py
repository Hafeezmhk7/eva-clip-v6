"""
Distance Evaluation Extension for BLIP3oEvaluator (Task 3)
This file contains the evaluate_distance method to be added to the BLIP3oEvaluator class.

Add this method to src/modules/evaluation/evaluator.py
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import logging
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


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
    from .coco_dataset import create_coco_dataloader
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
    
    # Import distance metrics functions
    from .distance_metrics import (
        compute_comprehensive_distance_metrics,
        compute_per_sample_distances,
        analyze_distance_distribution
    )
    
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
                'evaluation_date': str(torch.datetime.datetime.now()),
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


# Additional helper functions for the evaluator class

def add_distance_evaluation_methods():
    """
    Instructions for adding these methods to the BLIP3oEvaluator class:
    
    1. Add the following import at the top of evaluator.py:
       from .distance_metrics import (
           compute_comprehensive_distance_metrics,
           compute_per_sample_distances,
           analyze_distance_distribution,
           print_distance_metrics
       )
    
    2. Add these methods to the BLIP3oEvaluator class:
       - evaluate_distance (main method above)
       - _extract_clip_vision_raw (helper method above)
       - _generate_clip_from_eva_raw (helper method above)
    
    3. Update the __all__ list in src/modules/evaluation/__init__.py to include:
       "compute_comprehensive_distance_metrics",
       "compute_per_sample_distances", 
       "analyze_distance_distribution",
       "print_distance_metrics"
    """
    pass


if __name__ == "__main__":
    print("📏 Distance evaluation extension for BLIP3oEvaluator")
    print("Add the evaluate_distance method and helper methods to your evaluator.py file")
    print("Follow the instructions in add_distance_evaluation_methods() function")