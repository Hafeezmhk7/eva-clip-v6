#!/usr/bin/env python3
"""
CLIP vs BLIP3o Patch Reconstruction Evaluation

This script compares:
(a) CLIP ViT-L/14 patch embeddings [B, 256, 1024] (ground truth)
(b) EVA-CLIP → BLIP3o DiT → Generated patch embeddings [B, 256, 1024]

Key Metrics:
- Token-wise L2 distances (per patch position)
- Per-sample L2 distances (reconstruction quality)
- Spatial reconstruction analysis (16x16 grid)
- Distribution and correlation analysis

Usage:
python patch_reconstruction_evaluation.py \
    --coco_root ./data/coco \
    --blip3o_model_path /path/to/model \
    --num_samples 1000 \
    --save_results results/patch_reconstruction.json
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, AutoModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Try to import BLIP3o modules
try:
    from src.modules.inference.blip3o_inference import BLIP3oInference
    BLIP3O_AVAILABLE = True
    logger.info("BLIP3o inference module found")
except ImportError as e:
    BLIP3O_AVAILABLE = False
    logger.warning(f"BLIP3o inference not available: {e}")
    
    # Fallback import
    try:
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(project_root)
        from src.modules.inference.blip3o_inference import BLIP3oInference
        BLIP3O_AVAILABLE = True
        logger.info("BLIP3o inference module found via fallback")
    except ImportError:
        BLIP3O_AVAILABLE = False


class PatchReconstructionEvaluator:
    """
    Evaluator for comparing CLIP patch embeddings with BLIP3o-generated patch embeddings.
    
    This class extracts and compares 3D patch embeddings [B, 256, 1024] to evaluate
    how well BLIP3o DiT reconstructs the spatial structure of CLIP patch representations.
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
            blip3o_model_path: Path to BLIP3o model
        """
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        self.blip3o_model_path = blip3o_model_path
        
        # Load models
        self._load_clip_model()
        self._load_eva_model()
        
        # Load BLIP3o model
        self.blip3o_inference = None
        if blip3o_model_path and BLIP3O_AVAILABLE:
            self._load_blip3o_model()
        
        logger.info("Patch Reconstruction Evaluator initialized")
        logger.info(f"Using device: {self.device}")
    
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
        
        return device
    
    def _load_clip_model(self):
        """Load CLIP ViT-L/14 model."""
        logger.info("Loading CLIP ViT-L/14...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.clip_model.eval()
        logger.info("CLIP model loaded successfully")
    
    def _load_eva_model(self):
        """Load EVA-CLIP-8B model."""
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
        """Load BLIP3o DiT model."""
        try:
            logger.info(f"Loading BLIP3o model from {self.blip3o_model_path}...")
            self.blip3o_inference = BLIP3oInference(
                model_path=str(self.blip3o_model_path),
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            logger.info("BLIP3o model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP3o model: {e}")
            self.blip3o_inference = None
            raise
    
    def extract_clip_patch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract CLIP patch embeddings (without CLS token).
        
        Args:
            images: List of PIL Images
            
        Returns:
            Patch embeddings [B, 256, 1024]
        """
        patch_embeddings = []
        
        logger.info(f"Extracting CLIP patch embeddings for {len(images)} images...")
        
        with torch.no_grad():
            for i, img in enumerate(images):
                if i % 100 == 0:
                    logger.debug(f"Processing CLIP image {i}/{len(images)}")
                
                # Process image
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
                # Index 0 is CLS token, indices 1-256 are patch tokens
                patches = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 1024]
                
                patch_embeddings.append(patches.squeeze(0).cpu().float())  # [256, 1024]
        
        result = torch.stack(patch_embeddings)  # [B, 256, 1024]
        logger.info(f"CLIP patch embeddings extracted: {result.shape}")
        logger.info(f"CLIP patch embedding range: [{result.min():.4f}, {result.max():.4f}]")
        
        return result
    
    def extract_eva_patch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract EVA-CLIP patch embeddings.
        
        Args:
            images: List of PIL Images
            
        Returns:
            EVA patch embeddings [B, 256, 4096]
        """
        eva_embeddings = []
        
        logger.info(f"Extracting EVA patch embeddings for {len(images)} images...")
        
        with torch.no_grad():
            for i, img in enumerate(images):
                if i % 100 == 0:
                    logger.debug(f"Processing EVA image {i}/{len(images)}")
                
                # Process image
                inputs = self.eva_processor(images=img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device=self.device, dtype=self.torch_dtype)
                
                # Get vision model outputs
                vision_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract patch embeddings (remove CLS token)
                # vision_outputs.last_hidden_state: [1, 257, 4096]
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 4096]
                
                eva_embeddings.append(patch_embeddings.squeeze(0).cpu().float())  # [256, 4096]
        
        result = torch.stack(eva_embeddings)  # [B, 256, 4096]
        logger.info(f"EVA patch embeddings extracted: {result.shape}")
        logger.info(f"EVA patch embedding range: [{result.min():.4f}, {result.max():.4f}]")
        
        return result
    
    def generate_blip3o_patch_embeddings(self, eva_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate CLIP-like patch embeddings using BLIP3o DiT.
        
        Args:
            eva_embeddings: EVA patch embeddings [B, 256, 4096]
            
        Returns:
            Generated CLIP-like patch embeddings [B, 256, 1024]
        """
        if self.blip3o_inference is None:
            raise ValueError("BLIP3o model not loaded")
        
        logger.info(f"Generating CLIP patch embeddings using BLIP3o DiT...")
        logger.info(f"Input EVA embeddings shape: {eva_embeddings.shape}")
        
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        generated_embeddings = []
        
        with torch.no_grad():
            # Process in batches for memory management
            batch_size = 8
            num_samples = eva_embeddings.shape[0]
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_eva = eva_embeddings[i:end_idx]  # [batch_size, 256, 4096]
                
                logger.debug(f"Processing BLIP3o batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
                
                try:
                    # Generate CLIP-like patch embeddings
                    generated_patches = self.blip3o_inference.generate(
                        batch_eva,  # [batch_size, 256, 4096]
                        num_inference_steps=50,
                    )  # → [batch_size, 256, 1024]
                    
                    logger.debug(f"Generated patches shape: {generated_patches.shape}")
                    
                    if generated_patches.shape[-1] != 1024:
                        logger.warning(f"Expected 1024-dim features, got {generated_patches.shape[-1]}")
                    if generated_patches.shape[1] != 256:
                        logger.warning(f"Expected 256 tokens, got {generated_patches.shape[1]}")
                    
                    generated_embeddings.append(generated_patches.cpu().float())
                    
                except Exception as e:
                    logger.error(f"Error in BLIP3o generation for batch {i//batch_size + 1}: {e}")
                    raise
        
        # Concatenate all batches
        result = torch.cat(generated_embeddings, dim=0)  # [B, 256, 1024]
        
        logger.info(f"BLIP3o patch generation completed: {result.shape}")
        logger.info(f"Generated patch embedding range: [{result.min():.4f}, {result.max():.4f}]")
        
        return result
    
    def compute_3d_l2_distances(self, 
                               target_patches: torch.Tensor, 
                               predicted_patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute various L2 distance metrics for 3D patch tensors.
        
        Args:
            target_patches: Target CLIP patch embeddings [B, 256, 1024]
            predicted_patches: Predicted patch embeddings [B, 256, 1024]
            
        Returns:
            Dictionary containing various L2 distance metrics
        """
        logger.info("Computing 3D L2 distance metrics...")
        
        # Validate shapes
        assert target_patches.shape == predicted_patches.shape, \
            f"Shape mismatch: {target_patches.shape} vs {predicted_patches.shape}"
        
        B, num_tokens, embed_dim = target_patches.shape
        logger.info(f"Computing distances for {B} samples, {num_tokens} tokens, {embed_dim} dimensions")
        
        with torch.no_grad():
            distances = {}
            
            # 1. Token-wise L2 distances: [B, 256]
            # For each token position, compute L2 distance across the embedding dimension
            token_wise_distances = torch.norm(
                target_patches - predicted_patches, p=2, dim=-1
            )  # [B, 256]
            distances['token_wise'] = token_wise_distances
            
            # 2. Per-sample L2 distances: [B]
            # For each sample, compute L2 distance across all tokens (flattened)
            target_flat = target_patches.view(B, -1)  # [B, 256*1024]
            predicted_flat = predicted_patches.view(B, -1)  # [B, 256*1024]
            per_sample_distances = torch.norm(
                target_flat - predicted_flat, p=2, dim=-1
            )  # [B]
            distances['per_sample'] = per_sample_distances
            
            # 3. Per-token average L2 distances: [256]
            # Average L2 distance for each token position across all samples
            per_token_avg_distances = token_wise_distances.mean(dim=0)  # [256]
            distances['per_token_avg'] = per_token_avg_distances
            
            # 4. Global L2 distance: scalar
            # Overall L2 distance across all samples and tokens
            global_distance = torch.norm(target_flat - predicted_flat, p=2)
            distances['global'] = global_distance
            
            # 5. Normalized distances (divide by embedding norms)
            target_norms = torch.norm(target_patches, p=2, dim=-1)  # [B, 256]
            predicted_norms = torch.norm(predicted_patches, p=2, dim=-1)  # [B, 256]
            
            # Avoid division by zero
            safe_target_norms = torch.clamp(target_norms, min=1e-8)
            normalized_distances = token_wise_distances / safe_target_norms  # [B, 256]
            distances['normalized_token_wise'] = normalized_distances
            
            # 6. Cosine distances for each token: [B, 256]
            # 1 - cosine similarity for each token
            target_norm = F.normalize(target_patches, p=2, dim=-1)  # [B, 256, 1024]
            predicted_norm = F.normalize(predicted_patches, p=2, dim=-1)  # [B, 256, 1024]
            
            cosine_similarities = (target_norm * predicted_norm).sum(dim=-1)  # [B, 256]
            cosine_distances = 1.0 - cosine_similarities  # [B, 256]
            distances['cosine_token_wise'] = cosine_distances
            
            # 7. Spatial distances (16x16 grid analysis)
            # Reshape to spatial grid and compute spatial statistics
            spatial_target = target_patches.view(B, 16, 16, embed_dim)  # [B, 16, 16, 1024]
            spatial_predicted = predicted_patches.view(B, 16, 16, embed_dim)  # [B, 16, 16, 1024]
            
            spatial_distances = torch.norm(
                spatial_target - spatial_predicted, p=2, dim=-1
            )  # [B, 16, 16]
            distances['spatial'] = spatial_distances
            
            logger.info("3D L2 distance computation completed")
            
        return distances
    
    def compute_distance_statistics(self, distances: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical summaries of the distance metrics.
        
        Args:
            distances: Dictionary of distance tensors
            
        Returns:
            Dictionary of statistics for each distance type
        """
        logger.info("Computing distance statistics...")
        
        statistics = {}
        
        for distance_name, distance_tensor in distances.items():
            if distance_tensor.numel() == 1:  # Scalar values
                statistics[distance_name] = {
                    'value': distance_tensor.item(),
                    'type': 'scalar'
                }
            else:
                # Multi-dimensional tensors
                stats = {
                    'mean': distance_tensor.mean().item(),
                    'std': distance_tensor.std().item(),
                    'min': distance_tensor.min().item(),
                    'max': distance_tensor.max().item(),
                    'median': distance_tensor.median().item(),
                    'shape': list(distance_tensor.shape),
                    'type': 'tensor'
                }
                
                # Add percentiles for detailed analysis
                percentiles = [5, 25, 75, 95]
                for p in percentiles:
                    stats[f'p{p}'] = torch.quantile(distance_tensor.float(), p/100).item()
                
                statistics[distance_name] = stats
        
        return statistics
    
    def analyze_spatial_patterns(self, spatial_distances: torch.Tensor) -> Dict[str, any]:
        """
        Analyze spatial patterns in the 16x16 patch reconstruction errors.
        
        Args:
            spatial_distances: Spatial distance tensor [B, 16, 16]
            
        Returns:
            Dictionary with spatial analysis results
        """
        logger.info("Analyzing spatial reconstruction patterns...")
        
        # Average spatial error map across all samples
        avg_spatial_error = spatial_distances.mean(dim=0)  # [16, 16]
        
        # Find areas with highest and lowest reconstruction errors
        flat_errors = avg_spatial_error.flatten()
        worst_patches = torch.topk(flat_errors, k=5, largest=True)
        best_patches = torch.topk(flat_errors, k=5, largest=False)
        
        # Convert flat indices back to 2D coordinates
        worst_coords = [(idx.item() // 16, idx.item() % 16) for idx in worst_patches.indices]
        best_coords = [(idx.item() // 16, idx.item() % 16) for idx in best_patches.indices]
        
        # Analyze quadrants (divide 16x16 into 4 quadrants)
        quadrant_errors = {}
        quadrant_errors['top_left'] = spatial_distances[:, :8, :8].mean().item()
        quadrant_errors['top_right'] = spatial_distances[:, :8, 8:].mean().item()
        quadrant_errors['bottom_left'] = spatial_distances[:, 8:, :8].mean().item()
        quadrant_errors['bottom_right'] = spatial_distances[:, 8:, 8:].mean().item()
        
        # Analyze center vs edges
        center_errors = spatial_distances[:, 4:12, 4:12].mean().item()  # Center 8x8
        edge_mask = torch.ones(16, 16, dtype=torch.bool)
        edge_mask[4:12, 4:12] = False
        edge_errors = spatial_distances[:, edge_mask].mean().item()
        
        spatial_analysis = {
            'avg_spatial_error_map': avg_spatial_error.cpu().numpy().tolist(),
            'worst_patches': {
                'coordinates': worst_coords,
                'errors': worst_patches.values.cpu().numpy().tolist()
            },
            'best_patches': {
                'coordinates': best_coords,
                'errors': best_patches.values.cpu().numpy().tolist()
            },
            'quadrant_errors': quadrant_errors,
            'center_vs_edge': {
                'center_error': center_errors,
                'edge_error': edge_errors,
                'center_vs_edge_ratio': center_errors / edge_errors if edge_errors > 0 else 0
            },
            'global_spatial_stats': {
                'mean': spatial_distances.mean().item(),
                'std': spatial_distances.std().item(),
                'max_error_location': worst_coords[0],
                'min_error_location': best_coords[0]
            }
        }
        
        return spatial_analysis
    
    def evaluate_patch_reconstruction(self, 
                                    images: List[Image.Image]) -> Dict[str, any]:
        """
        Run complete patch reconstruction evaluation.
        
        Args:
            images: List of PIL Images to evaluate
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting patch reconstruction evaluation...")
        logger.info(f"Evaluating {len(images)} images")
        
        # Extract CLIP patch embeddings (target)
        logger.info("=== Extracting Target CLIP Patch Embeddings ===")
        target_patches = self.extract_clip_patch_embeddings(images)
        
        # Extract EVA patch embeddings
        logger.info("=== Extracting EVA Patch Embeddings ===")
        eva_patches = self.extract_eva_patch_embeddings(images)
        
        # Generate BLIP3o patch embeddings (predicted)
        logger.info("=== Generating BLIP3o Patch Embeddings ===")
        predicted_patches = self.generate_blip3o_patch_embeddings(eva_patches)
        
        # Compute L2 distances
        logger.info("=== Computing L2 Distance Metrics ===")
        distances = self.compute_3d_l2_distances(target_patches, predicted_patches)
        
        # Compute statistics
        logger.info("=== Computing Distance Statistics ===")
        statistics = self.compute_distance_statistics(distances)
        
        # Analyze spatial patterns
        logger.info("=== Analyzing Spatial Patterns ===")
        spatial_analysis = self.analyze_spatial_patterns(distances['spatial'])
        
        # Compile results
        results = {
            'evaluation_info': {
                'num_images': len(images),
                'target_shape': list(target_patches.shape),
                'predicted_shape': list(predicted_patches.shape),
                'eva_shape': list(eva_patches.shape),
            },
            'distance_statistics': statistics,
            'spatial_analysis': spatial_analysis,
            'raw_distances': {
                # Convert tensors to lists for JSON serialization
                k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in distances.items()
            }
        }
        
        logger.info("Patch reconstruction evaluation completed")
        
        return results


def load_coco_samples(coco_root: Path, num_samples: int = 1000) -> Tuple[List[Image.Image], List[int]]:
    """Load COCO validation samples for patch reconstruction evaluation."""
    logger.info(f"Loading {num_samples} COCO validation samples...")
    
    annotations_file = coco_root / "annotations" / "captions_val2017.json"
    images_dir = coco_root / "images" / "val2017"
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO images not found: {images_dir}")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get unique images
    images_info = {img['id']: img for img in coco_data['images']}
    
    images = []
    image_ids = []
    
    for image_id, image_info in list(images_info.items())[:num_samples]:
        image_path = images_dir / image_info['file_name']
        
        if not image_path.exists():
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            image_ids.append(image_id)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(images)} images")
    return images, image_ids


def print_results(results: Dict[str, any]):
    """Print formatted evaluation results."""
    print("\n" + "="*80)
    print("📊 PATCH RECONSTRUCTION EVALUATION RESULTS")
    print("="*80)
    
    eval_info = results['evaluation_info']
    print(f"Images evaluated: {eval_info['num_images']}")
    print(f"Target shape: {eval_info['target_shape']}")
    print(f"Predicted shape: {eval_info['predicted_shape']}")
    
    print(f"\n📏 Distance Statistics:")
    print("-" * 50)
    
    stats = results['distance_statistics']
    for distance_name, distance_stats in stats.items():
        if distance_stats['type'] == 'scalar':
            print(f"{distance_name:25s}: {distance_stats['value']:.6f}")
        else:
            mean_val = distance_stats['mean']
            std_val = distance_stats['std']
            print(f"{distance_name:25s}: {mean_val:.6f} ± {std_val:.6f}")
    
    print(f"\n🗺️  Spatial Analysis:")
    print("-" * 50)
    
    spatial = results['spatial_analysis']
    print(f"Center vs Edge error ratio: {spatial['center_vs_edge']['center_vs_edge_ratio']:.4f}")
    print(f"Worst patch location: {spatial['worst_patches']['coordinates'][0]}")
    print(f"Best patch location: {spatial['best_patches']['coordinates'][0]}")
    
    print(f"\nQuadrant errors:")
    for quadrant, error in spatial['quadrant_errors'].items():
        print(f"  {quadrant:15s}: {error:.6f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="CLIP vs BLIP3o Patch Reconstruction Evaluation")
    parser.add_argument("--coco_root", type=str, required=True,
                       help="Path to MS-COCO dataset root directory")
    parser.add_argument("--blip3o_model_path", type=str, required=True,
                       help="Path to BLIP3o model")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of COCO samples to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save results JSON file")
    parser.add_argument("--save_plots", type=str, default=None,
                       help="Directory to save visualization plots")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not BLIP3O_AVAILABLE:
        logger.error("BLIP3o inference module not available")
        sys.exit(1)
    
    # Convert paths
    coco_root = Path(args.coco_root)
    blip3o_model_path = Path(args.blip3o_model_path)
    
    if not blip3o_model_path.exists():
        logger.error(f"BLIP3o model path does not exist: {blip3o_model_path}")
        sys.exit(1)
    
    # Initialize evaluator
    logger.info("Initializing Patch Reconstruction Evaluator...")
    evaluator = PatchReconstructionEvaluator(
        device=args.device,
        blip3o_model_path=str(blip3o_model_path)
    )
    
    # Load COCO samples
    logger.info(f"Loading {args.num_samples} COCO validation samples...")
    images, image_ids = load_coco_samples(coco_root, args.num_samples)
    
    # Run evaluation
    logger.info("Running patch reconstruction evaluation...")
    start_time = time.time()
    
    results = evaluator.evaluate_patch_reconstruction(images)
    
    evaluation_time = time.time() - start_time
    results['evaluation_time'] = evaluation_time
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {save_path}")
    
    print(f"\n⏱️  Evaluation completed in {evaluation_time:.2f} seconds")
    print(f"📊 Key Insight: Average per-sample L2 distance = {results['distance_statistics']['per_sample']['mean']:.6f}")


if __name__ == "__main__":
    main()