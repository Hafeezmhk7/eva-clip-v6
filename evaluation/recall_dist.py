#!/usr/bin/env python3
"""
UPDATED: Dual Supervision BLIP3-o Recall Evaluation Script
Tests both patch-level and global-level performance to validate the dual supervision improvements.

Key Features:
1. Tests global embeddings for recall performance (primary goal)
2. Tests patch-level reconstruction quality
3. Compares with baseline CLIP recall performance
4. Detailed analysis of both supervision levels
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Try to import BLIP3o modules
try:
    from src.modules.inference.blip3o_inference import BLIP3oInference
    from src.modules.models.blip3o_dit import BLIP3oDiTModel, load_blip3o_dit_model
    from src.modules.config.blip3o_config import BLIP3oDiTConfig
    BLIP3O_AVAILABLE = True
    logger.info("Dual supervision BLIP3o modules found")
except ImportError as e:
    BLIP3O_AVAILABLE = False
    logger.warning(f"BLIP3o modules not available: {e}")


class DualSupervisionRecallEvaluator:
    """
    UPDATED: Comprehensive evaluator for dual supervision BLIP3-o recall evaluation.
    
    Tests both supervision levels:
    1. Global embeddings [B, 768] for image-to-text recall (primary goal)
    2. Patch embeddings [B, 256, 1024] for reconstruction quality
    3. Comparison with baseline CLIP performance
    """
    
    def __init__(self, 
                 device: str = "auto", 
                 torch_dtype: Optional[torch.dtype] = None,
                 blip3o_model_path: Optional[str] = None):
        """
        Initialize the dual supervision evaluator.
        
        Args:
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Data type for models
            blip3o_model_path: Path to trained dual supervision model
        """
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        self.blip3o_model_path = blip3o_model_path
        
        # Load baseline models for comparison
        self._load_clip_model()
        self._load_eva_model()
        
        # Load dual supervision BLIP3o model
        self.blip3o_model = None
        if blip3o_model_path and BLIP3O_AVAILABLE:
            self._load_dual_supervision_model()
        elif blip3o_model_path and not BLIP3O_AVAILABLE:
            logger.warning("BLIP3o model path provided but modules not available")
        
        logger.info("Dual Supervision Recall Evaluator initialized")
        logger.info(f"Using device: {self.device}")
        logger.info(f"CLIP model: openai/clip-vit-large-patch14")
        logger.info(f"EVA model: BAAI/EVA-CLIP-8B")
        logger.info(f"Dual supervision BLIP3o available: {self.blip3o_model is not None}")
    
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
        """Load CLIP ViT-L/14 model for baseline comparison."""
        logger.info("Loading CLIP ViT-L/14 for baseline...")
        
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.clip_model.eval()
        
        logger.info("CLIP baseline model loaded successfully")
    
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
    
    def _load_dual_supervision_model(self):
        """Load trained dual supervision BLIP3o model."""
        try:
            logger.info(f"Loading dual supervision BLIP3o model from {self.blip3o_model_path}...")
            
            # Check if path exists
            model_path = Path(self.blip3o_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            # Load the model
            self.blip3o_model = load_blip3o_dit_model(
                model_path=str(model_path),
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
            
            # Set to eval mode
            self.blip3o_model.eval()
            
            # Ensure frozen CLIP projection is loaded
            if not hasattr(self.blip3o_model, 'frozen_clip_visual_proj') or self.blip3o_model.frozen_clip_visual_proj is None:
                logger.info("Loading frozen CLIP projection...")
                self.blip3o_model.load_frozen_clip_projection()
            
            logger.info("Dual supervision BLIP3o model loaded successfully")
            logger.info(f"Model has frozen CLIP projection: {self.blip3o_model.frozen_clip_visual_proj is not None}")
            
        except Exception as e:
            logger.error(f"Failed to load dual supervision model: {e}")
            self.blip3o_model = None
            raise
    
    def extract_clip_text_embeddings(self, captions: List[str]) -> torch.Tensor:
        """Extract CLIP text embeddings for baseline comparison."""
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
        
        return text_embeddings.cpu().float()
    
    def extract_clip_global_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract CLIP global embeddings for baseline comparison."""
        global_embeddings = []
        
        with torch.no_grad():
            for img in images:
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                         for k, v in inputs.items()}
                
                # Get global features (CLS token + visual projection)
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                global_embeddings.append(image_features.squeeze(0).cpu().float())
        
        return torch.stack(global_embeddings)
    
    def extract_clip_patch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract CLIP patch embeddings for patch-level comparison."""
        patch_embeddings = []
        
        with torch.no_grad():
            for img in images:
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                         for k, v in inputs.items()}
                
                vision_outputs = self.clip_model.vision_model(
                    pixel_values=inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Extract patch embeddings (remove CLS token)
                patches = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 1024]
                patch_embeddings.append(patches.squeeze(0).cpu().float())
        
        return torch.stack(patch_embeddings)
    
    def extract_eva_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract EVA-CLIP embeddings for BLIP3o conditioning."""
        eva_embeddings = []
        
        with torch.no_grad():
            for img in images:
                inputs = self.eva_processor(images=img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device=self.device, dtype=self.torch_dtype)
                
                vision_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token)
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 4096]
                eva_embeddings.append(patch_embeddings.squeeze(0).cpu().float())
        
        return torch.stack(eva_embeddings)
    
    def generate_dual_supervision_embeddings(self, eva_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate both patch and global embeddings using dual supervision BLIP3o.
        
        Args:
            eva_embeddings: EVA-CLIP conditioning [B, 256, 4096]
            
        Returns:
            Dict containing:
            - global_embeddings: [B, 768] for recall evaluation
            - patch_embeddings: [B, 256, 1024] for patch quality
        """
        if self.blip3o_model is None:
            raise ValueError("Dual supervision BLIP3o model not loaded")
        
        logger.info(f"Generating dual supervision embeddings...")
        logger.info(f"Input EVA embeddings shape: {eva_embeddings.shape}")
        
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        with torch.no_grad():
            # Generate using the model's generate method
            # This should return global embeddings by default
            global_embeddings = self.blip3o_model.generate(
                encoder_hidden_states=eva_embeddings,
                num_inference_steps=50,
                return_global_only=True,  # Get [B, 768] for recall
            )
            
            # Also get patch embeddings for quality assessment
            patch_embeddings = self.blip3o_model.generate(
                encoder_hidden_states=eva_embeddings,
                num_inference_steps=50,
                return_global_only=False,  # Get [B, 256, 1024] patches
            )
        
        logger.info(f"Generated global embeddings: {global_embeddings.shape if global_embeddings is not None else 'None'}")
        logger.info(f"Generated patch embeddings: {patch_embeddings.shape}")
        
        return {
            'global_embeddings': global_embeddings.cpu() if global_embeddings is not None else None,
            'patch_embeddings': patch_embeddings.cpu(),
        }
    
    def compute_recall_metrics(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_to_text_mapping: List[List[int]],
        k_values: List[int] = [1, 5, 10],
        method_name: str = "unknown"
    ) -> Dict[str, float]:
        """Compute image-to-text recall metrics."""
        
        # Ensure embeddings are normalized
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(image_embeddings, text_embeddings.t())
        
        logger.info(f"[{method_name}] Similarity matrix shape: {similarity_matrix.shape}")
        logger.info(f"[{method_name}] Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
        
        # Compute recall for each K value
        recall_results = {}
        
        for k in k_values:
            correct_retrievals = 0
            total_queries = len(image_to_text_mapping)
            
            # Get top-k text indices for each image
            _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
            
            # Check if any of the top-k retrieved texts belong to the query image
            for img_idx, correct_text_indices in enumerate(image_to_text_mapping):
                retrieved_indices = top_k_indices[img_idx].cpu().numpy()
                
                if any(ret_idx in correct_text_indices for ret_idx in retrieved_indices):
                    correct_retrievals += 1
            
            recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0
            recall_results[f'recall@{k}'] = recall_at_k
            
            logger.info(f"[{method_name}] Recall@{k}: {correct_retrievals}/{total_queries} = {recall_at_k:.4f} ({recall_at_k*100:.2f}%)")
        
        # Additional metrics
        recall_results.update({
            'num_queries': len(image_to_text_mapping),
            'num_gallery': len(text_embeddings),
            'avg_texts_per_image': np.mean([len(texts) for texts in image_to_text_mapping]),
            'embedding_dim': image_embeddings.shape[1],
            'method': method_name,
        })
        
        return recall_results
    
    def compute_patch_quality_metrics(
        self,
        generated_patches: torch.Tensor,
        target_patches: torch.Tensor,
        method_name: str = "unknown"
    ) -> Dict[str, float]:
        """Compute patch-level reconstruction quality metrics."""
        
        with torch.no_grad():
            # Basic reconstruction metrics
            mse_loss = F.mse_loss(generated_patches, target_patches).item()
            
            # Cosine similarity (token-wise)
            gen_flat = generated_patches.flatten(1)
            target_flat = target_patches.flatten(1)
            cosine_sim = F.cosine_similarity(gen_flat, target_flat, dim=1).mean().item()
            
            # L2 distances per token
            l2_distances = torch.norm(generated_patches - target_patches, dim=-1).mean().item()
            
            # Embedding norms
            gen_norm = torch.norm(generated_patches, dim=-1).mean().item()
            target_norm = torch.norm(target_patches, dim=-1).mean().item()
            
            metrics = {
                'patch_mse': mse_loss,
                'patch_cosine_similarity': cosine_sim,
                'patch_l2_distance': l2_distances,
                'patch_gen_norm': gen_norm,
                'patch_target_norm': target_norm,
                'patch_norm_ratio': gen_norm / (target_norm + 1e-8),
                'method': method_name,
            }
        
        logger.info(f"[{method_name}] Patch quality metrics:")
        for key, value in metrics.items():
            if key != 'method':
                logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def run_comprehensive_evaluation(
        self,
        images: List[Image.Image],
        captions_per_image: List[List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive dual supervision evaluation.
        
        Args:
            images: List of PIL Images
            captions_per_image: List of caption lists for each image
            k_values: K values for Recall@K
            
        Returns:
            Dictionary mapping method names to their results
        """
        logger.info("Starting comprehensive dual supervision evaluation...")
        logger.info(f"Evaluating {len(images)} images")
        
        all_results = {}
        
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
        
        all_captions = [caption for caption_list in captions_per_image for caption in caption_list]
        text_embeddings = self.extract_clip_text_embeddings(all_captions)
        
        # 1. BASELINE: CLIP Global Embeddings
        logger.info("\n" + "="*60)
        logger.info("🔍 BASELINE: CLIP Global Embeddings")
        logger.info("="*60)
        
        clip_global_embeddings = self.extract_clip_global_embeddings(images)
        
        clip_results = self.compute_recall_metrics(
            image_embeddings=clip_global_embeddings,
            text_embeddings=text_embeddings,
            image_to_text_mapping=image_to_text_mapping,
            k_values=k_values,
            method_name="CLIP_Baseline"
        )
        all_results['clip_baseline'] = clip_results
        
        # 2. DUAL SUPERVISION: BLIP3o Global Embeddings (Primary Test)
        if self.blip3o_model is not None:
            logger.info("\n" + "="*60)
            logger.info("🎯 DUAL SUPERVISION: BLIP3o Global Embeddings")
            logger.info("="*60)
            
            # Extract EVA conditioning
            eva_embeddings = self.extract_eva_embeddings(images)
            
            # Generate dual supervision embeddings
            dual_outputs = self.generate_dual_supervision_embeddings(eva_embeddings)
            
            # Test global embeddings for recall (primary goal)
            if dual_outputs['global_embeddings'] is not None:
                blip3o_global_results = self.compute_recall_metrics(
                    image_embeddings=dual_outputs['global_embeddings'],
                    text_embeddings=text_embeddings,
                    image_to_text_mapping=image_to_text_mapping,
                    k_values=k_values,
                    method_name="BLIP3o_Global"
                )
                all_results['blip3o_global'] = blip3o_global_results
            else:
                logger.warning("No global embeddings generated - check model configuration")
            
            # 3. PATCH QUALITY: Compare patch reconstruction
            logger.info("\n" + "="*60)
            logger.info("📊 PATCH QUALITY: Reconstruction Analysis")
            logger.info("="*60)
            
            # Extract CLIP patch targets
            clip_patch_embeddings = self.extract_clip_patch_embeddings(images)
            
            # Compute patch quality metrics
            patch_quality = self.compute_patch_quality_metrics(
                generated_patches=dual_outputs['patch_embeddings'],
                target_patches=clip_patch_embeddings,
                method_name="BLIP3o_Patches"
            )
            all_results['patch_quality'] = patch_quality
        
        else:
            logger.warning("Dual supervision BLIP3o model not available - skipping BLIP3o evaluation")
        
        return all_results
    
    def print_comparison_results(self, results: Dict[str, Dict[str, float]]):
        """Print comprehensive comparison results."""
        
        print("\n" + "="*80)
        print("📊 DUAL SUPERVISION RECALL EVALUATION RESULTS")
        print("="*80)
        
        # Results summary table
        print(f"\n📋 Recall Performance Comparison:")
        print(f"{'Method':<25} {'R@1':<10} {'R@5':<10} {'R@10':<10} {'Improvement':<15}")
        print("-" * 75)
        
        baseline_r1 = 0
        if 'clip_baseline' in results:
            baseline = results['clip_baseline']
            r1 = baseline.get('recall@1', 0) * 100
            r5 = baseline.get('recall@5', 0) * 100
            r10 = baseline.get('recall@10', 0) * 100
            baseline_r1 = r1
            print(f"{'CLIP Baseline':<25} {r1:>6.1f}% {r5:>8.1f}% {r10:>9.1f}% {'(baseline)':<15}")
        
        if 'blip3o_global' in results:
            blip3o = results['blip3o_global']
            r1 = blip3o.get('recall@1', 0) * 100
            r5 = blip3o.get('recall@5', 0) * 100
            r10 = blip3o.get('recall@10', 0) * 100
            improvement = r1 - baseline_r1
            improvement_str = f"{improvement:+.1f}%" if baseline_r1 > 0 else "N/A"
            print(f"{'BLIP3o Dual Supervision':<25} {r1:>6.1f}% {r5:>8.1f}% {r10:>9.1f}% {improvement_str:<15}")
        
        # Patch quality results
        if 'patch_quality' in results:
            patch = results['patch_quality']
            print(f"\n📊 Patch Reconstruction Quality:")
            print(f"   Cosine Similarity: {patch.get('patch_cosine_similarity', 0):.4f}")
            print(f"   MSE Loss: {patch.get('patch_mse', 0):.6f}")
            print(f"   L2 Distance: {patch.get('patch_l2_distance', 0):.4f}")
        
        # Performance analysis
        print(f"\n🎯 Performance Analysis:")
        
        if 'blip3o_global' in results and 'clip_baseline' in results:
            blip3o_r1 = results['blip3o_global'].get('recall@1', 0) * 100
            baseline_r1 = results['clip_baseline'].get('recall@1', 0) * 100
            
            if blip3o_r1 > baseline_r1 + 5:  # Significant improvement
                print(f"   ✅ SIGNIFICANT IMPROVEMENT: {blip3o_r1:.1f}% vs {baseline_r1:.1f}%")
                print(f"   🎉 Dual supervision architecture successful!")
            elif blip3o_r1 > baseline_r1:
                print(f"   ✅ Improvement: {blip3o_r1:.1f}% vs {baseline_r1:.1f}%")
                print(f"   📈 Dual supervision showing positive results")
            else:
                print(f"   ⚠️  No improvement: {blip3o_r1:.1f}% vs {baseline_r1:.1f}%")
                print(f"   🔧 May need training adjustments or more data")
        
        # Expected vs actual
        print(f"\n🎯 Expected vs Actual:")
        if 'blip3o_global' in results:
            actual_r1 = results['blip3o_global'].get('recall@1', 0) * 100
            print(f"   Expected improvement: 0% → 60%+ recall")
            print(f"   Actual result: {baseline_r1:.1f}% → {actual_r1:.1f}% recall")
            
            if actual_r1 >= 60:
                print(f"   🎉 TARGET ACHIEVED! Recall ≥ 60%")
            elif actual_r1 >= 40:
                print(f"   📈 Good progress towards target")
            else:
                print(f"   🔧 More training or architecture tuning needed")
        
        print("="*80)


def load_coco_samples(coco_root: Path, num_samples: int = 1000) -> Tuple[List[Image.Image], List[List[str]], List[int]]:
    """Load COCO validation samples for evaluation."""
    logger.info(f"Loading {num_samples} COCO validation samples...")
    
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
        
        if not image_path.exists():
            continue
        
        try:
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
    """Main evaluation function for dual supervision recall testing."""
    parser = argparse.ArgumentParser(description="Dual Supervision BLIP3-o Recall Evaluation")
    parser.add_argument("--coco_root", type=str, required=True,
                       help="Path to MS-COCO dataset root directory")
    parser.add_argument("--blip3o_model_path", type=str, required=True,
                       help="Path to trained dual supervision BLIP3o model")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of COCO samples to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save results JSON file")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 5, 10],
                       help="K values for Recall@K computation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not BLIP3O_AVAILABLE:
        logger.error("BLIP3o modules not available")
        logger.error("Make sure src.modules are properly installed")
        sys.exit(1)
    
    # Convert paths
    coco_root = Path(args.coco_root)
    model_path = Path(args.blip3o_model_path)
    
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Initialize evaluator
    logger.info("Initializing Dual Supervision Recall Evaluator...")
    evaluator = DualSupervisionRecallEvaluator(
        device=args.device,
        blip3o_model_path=str(model_path)
    )
    
    # Load COCO samples
    logger.info(f"Loading {args.num_samples} COCO validation samples...")
    images, captions_per_image, image_ids = load_coco_samples(coco_root, args.num_samples)
    
    # Run comprehensive evaluation
    logger.info(f"Running dual supervision evaluation...")
    start_time = time.time()
    
    results = evaluator.run_comprehensive_evaluation(
        images=images,
        captions_per_image=captions_per_image,
        k_values=args.k_values
    )
    
    evaluation_time = time.time() - start_time
    
    # Print results
    evaluator.print_comparison_results(results)
    
    # Save results if requested
    if args.save_results:
        results_to_save = {
            'evaluation_info': {
                'model_path': str(model_path),
                'dataset': 'MS-COCO 2017 Validation',
                'num_images': len(images),
                'num_captions': sum(len(caps) for caps in captions_per_image),
                'k_values': args.k_values,
                'evaluation_time': evaluation_time,
                'architecture': 'dual_supervision',
            },
            'results': results,
            'timestamp': time.time(),
        }
        
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Results saved to: {save_path}")
    
    print(f"\n⏱️  Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Final summary
    if 'blip3o_global' in results:
        final_recall = results['blip3o_global'].get('recall@1', 0) * 100
        print(f"🎯 FINAL RESULT: {final_recall:.1f}% Recall@1 with dual supervision")
        
        if final_recall >= 60:
            print(f"🎉 SUCCESS! Achieved target recall performance!")
        elif final_recall >= 40:
            print(f"📈 Good progress! Continue training for better results.")
        else:
            print(f"🔧 Consider adjusting architecture or training parameters.")


if __name__ == "__main__":
    main()