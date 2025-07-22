#!/usr/bin/env python3
"""
BLIP3-o Patch-Level Evaluation Script
eval_blip3o_patch_recall.py

This script evaluates BLIP3-o patch-level models on image-to-text recall:
1. Loads trained BLIP3-o patch DiT model
2. Generates CLIP embeddings from EVA-CLIP conditioning
3. Measures Recall@1, Recall@5, Recall@10 for image-to-text retrieval
4. Compares with CLIP baseline
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoModel, CLIPImageProcessor
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import json
import argparse
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "src"))

def setup_paths():
    """Setup import paths"""
    try:
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.blip3o_patch_dit import BLIP3oPatchDiTModel, create_blip3o_patch_dit_model
        from src.modules.evaluation.blip3o_recall_evaluator import BLIP3oRecallEvaluator
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

class BLIP3oPatchEvaluator:
    """
    Evaluator for BLIP3-o patch-level models focused on image-to-text recall
    """
    
    def __init__(self, device="auto"):
        self.device = self._setup_device(device)
        self.clip_processor = None
        self.clip_model = None
        self.eva_processor = None
        self.eva_model = None
        self.blip3o_model = None
        self.recall_evaluator = None
        
        logger.info("BLIP3-o patch evaluator initialized")
    
    def _setup_device(self, device_arg):
        if device_arg == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                logger.info("Using CPU")
        else:
            device = torch.device(device_arg)
        return device
    
    def load_clip_models(self):
        """Load CLIP and EVA-CLIP models"""
        logger.info("Loading CLIP ViT-L/14...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_model.eval()
        
        logger.info("Loading EVA-CLIP-8B...")
        self.eva_model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B", 
            trust_remote_code=True
        ).to(self.device)
        self.eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.eva_model.eval()
        
        logger.info("✅ Models loaded successfully")
    
    def load_blip3o_model(self, model_path):
        """Load the BLIP3-o patch-level model"""
        logger.info(f"Loading BLIP3-o patch model from: {model_path}")
        
        model_path = Path(model_path)
        
        # Load config
        config_files = [
            model_path / "config.json",
            model_path / "blip3o_model_config.json"
        ]
        
        config_file = None
        for cf in config_files:
            if cf.exists():
                config_file = cf
                break
        
        if not config_file:
            raise FileNotFoundError(f"No config file found in {model_path}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        config = BLIP3oDiTConfig(**config_dict)
        
        # Create model
        from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model
        self.blip3o_model = create_blip3o_patch_dit_model(config)
        
        # Load weights
        weight_files = [
            model_path / "pytorch_model.bin",
            model_path / "model.safetensors",
            model_path / "pytorch_model.safetensors"
        ]
        
        weight_file = None
        for wf in weight_files:
            if wf.exists():
                weight_file = wf
                break
        
        if not weight_file:
            raise FileNotFoundError(f"No weight file found in {model_path}")
        
        logger.info(f"Loading weights from: {weight_file}")
        
        if weight_file.suffix == ".bin":
            state_dict = torch.load(weight_file, map_location=self.device)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(str(weight_file))
        
        # Load state dict
        missing_keys, unexpected_keys = self.blip3o_model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        
        self.blip3o_model = self.blip3o_model.to(self.device)
        self.blip3o_model.eval()
        
        # Initialize recall evaluator
        from src.modules.evaluation.blip3o_recall_evaluator import BLIP3oRecallEvaluator
        self.recall_evaluator = BLIP3oRecallEvaluator(
            model=self.blip3o_model,
            device=self.device
        )
        
        logger.info("✅ BLIP3-o patch model loaded successfully")
    
    def extract_clip_text_embeddings(self, captions):
        """Extract CLIP text embeddings"""
        with torch.no_grad():
            inputs = self.clip_processor(
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_features = self.clip_model.get_text_features(**inputs)
            # Apply visual projection to match image embedding space
            text_features = self.clip_model.visual_projection(text_features)
            text_features = F.normalize(text_features, p=2, dim=-1)
        
        return text_features.cpu()
    
    def extract_clip_baseline_embeddings(self, images):
        """Extract CLIP baseline embeddings for comparison"""
        global_embeddings = []
        
        with torch.no_grad():
            for img in images:
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                vision_features = self.clip_model.get_image_features(**inputs)
                # Apply visual projection
                vision_features = self.clip_model.visual_projection(vision_features)
                vision_features = F.normalize(vision_features, p=2, dim=-1)
                
                global_embeddings.append(vision_features.squeeze(0).cpu())
        
        return torch.stack(global_embeddings)
    
    def extract_eva_embeddings(self, images):
        """Extract EVA-CLIP embeddings for conditioning"""
        eva_embeddings = []
        
        with torch.no_grad():
            for img in images:
                inputs = self.eva_processor(images=img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                
                vision_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token)
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 4096]
                eva_embeddings.append(patch_embeddings.squeeze(0).cpu())
        
        return torch.stack(eva_embeddings)
    
    def generate_blip3o_embeddings(self, eva_embeddings, num_inference_steps=50):
        """Generate BLIP3-o embeddings using the patch model"""
        logger.info(f"Generating BLIP3-o embeddings for {eva_embeddings.shape[0]} images...")
        
        eva_embeddings = eva_embeddings.to(self.device)
        
        generated_embeddings = []
        batch_size = 4  # Process in small batches
        
        with torch.no_grad():
            for i in range(0, eva_embeddings.shape[0], batch_size):
                end_idx = min(i + batch_size, eva_embeddings.shape[0])
                batch_eva = eva_embeddings[i:end_idx]
                
                # Generate CLIP patch embeddings
                generated_patches = self.blip3o_model.generate(
                    eva_features=batch_eva,
                    num_inference_steps=num_inference_steps,
                )  # [B, 256, 1024]
                
                # Pool to global features
                global_features = generated_patches.mean(dim=1)  # [B, 1024]
                
                # Apply CLIP visual projection
                global_features = self.clip_model.visual_projection(global_features)  # [B, 768]
                global_features = F.normalize(global_features, p=2, dim=-1)
                
                generated_embeddings.append(global_features.cpu())
        
        result = torch.cat(generated_embeddings, dim=0)
        logger.info(f"Generated embeddings: {result.shape}")
        
        return result
    
    def compute_recall(self, image_embeddings, text_embeddings, image_to_text_mapping, k_values=[1, 5, 10]):
        """Compute Recall@K metrics"""
        # Ensure embeddings are normalized
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(image_embeddings, text_embeddings.t())
        
        logger.info(f"Similarity matrix: {similarity_matrix.shape}")
        logger.info(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
        
        # Compute recall for each K
        recall_results = {}
        
        for k in k_values:
            correct_retrievals = 0
            total_queries = len(image_to_text_mapping)
            
            _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
            
            for img_idx, correct_text_indices in enumerate(image_to_text_mapping):
                retrieved_indices = top_k_indices[img_idx].cpu().numpy()
                if any(ret_idx in correct_text_indices for ret_idx in retrieved_indices):
                    correct_retrievals += 1
            
            recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0
            recall_results[f'recall@{k}'] = recall_at_k
            
            logger.info(f"Recall@{k}: {correct_retrievals}/{total_queries} = {recall_at_k:.4f} ({recall_at_k*100:.2f}%)")
        
        return recall_results
    
    def evaluate_method(self, images, captions_per_image, method, k_values=[1, 5, 10], num_inference_steps=50):
        """Evaluate a specific method"""
        logger.info(f"Evaluating method: {method}")
        
        # Extract text embeddings
        all_captions = [caption for caption_list in captions_per_image for caption in caption_list]
        text_embeddings = self.extract_clip_text_embeddings(all_captions)
        
        # Create image-to-text mapping
        image_to_text_mapping = []
        text_idx = 0
        for caption_list in captions_per_image:
            current_indices = []
            for _ in caption_list:
                current_indices.append(text_idx)
                text_idx += 1
            image_to_text_mapping.append(current_indices)
        
        # Extract image embeddings
        if method == "clip_baseline":
            image_embeddings = self.extract_clip_baseline_embeddings(images)
            method_desc = "CLIP ViT-L/14 baseline"
            
        elif method == "blip3o_patch":
            if self.blip3o_model is None:
                raise ValueError("BLIP3-o model not loaded")
            
            # Extract EVA embeddings for conditioning
            eva_embeddings = self.extract_eva_embeddings(images)
            
            # Generate BLIP3-o embeddings
            image_embeddings = self.generate_blip3o_embeddings(eva_embeddings, num_inference_steps)
            
            method_desc = f"BLIP3-o Patch DiT (EVA → DiT → CLIP)"
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Image embeddings: {image_embeddings.shape}")
        logger.info(f"Text embeddings: {text_embeddings.shape}")
        
        # Compute recall
        recall_results = self.compute_recall(image_embeddings, text_embeddings, image_to_text_mapping, k_values)
        
        # Add method info
        recall_results.update({
            'method': method,
            'method_description': method_desc,
            'embedding_dim': image_embeddings.shape[-1],
        })
        
        return recall_results, image_embeddings


def load_coco_samples(coco_root, num_samples=1000):
    """Load COCO validation samples"""
    logger.info(f"Loading {num_samples} COCO samples...")
    
    coco_root = Path(coco_root)
    annotations_file = coco_root / "annotations" / "captions_val2017.json"
    images_dir = coco_root / "images" / "val2017"
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    images_info = {img['id']: img for img in coco_data['images']}
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
            captions_per_image.append(captions[:5])  # Max 5 captions
            image_ids.append(image_id)
            loaded_count += 1
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            continue
    
    logger.info(f"Loaded {len(images)} images with {sum(len(caps) for caps in captions_per_image)} captions")
    return images, captions_per_image, image_ids


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate BLIP3-o Patch Model")
    parser.add_argument("--coco_root", type=str, required=True, help="COCO dataset root")
    parser.add_argument("--model_path", type=str, required=True, help="BLIP3-o model path")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--save_results", type=str, default=None, help="Save results path")
    
    args = parser.parse_args()
    
    # Setup imports
    if not setup_paths():
        return 1
    
    # Initialize evaluator
    evaluator = BLIP3oPatchEvaluator(device=args.device)
    
    # Load models
    evaluator.load_clip_models()
    evaluator.load_blip3o_model(args.model_path)
    
    # Load COCO samples
    images, captions_per_image, image_ids = load_coco_samples(args.coco_root, args.num_samples)
    
    # Evaluate methods
    methods = ["clip_baseline", "blip3o_patch"]
    results = {}
    
    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {method.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            method_results, embeddings = evaluator.evaluate_method(
                images=images,
                captions_per_image=captions_per_image,
                method=method,
                num_inference_steps=args.num_inference_steps
            )
            
            results[method] = method_results
            
            # Print results
            print(f"\n📊 {method.upper()} Results:")
            for k in [1, 5, 10]:
                if f'recall@{k}' in method_results:
                    recall = method_results[f'recall@{k}']
                    print(f"  Recall@{k:2d}: {recall:.4f} ({recall*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {method}: {e}")
            results[method] = {'error': str(e)}
    
    # Print comparison
    print(f"\n{'='*80}")
    print("📊 BLIP3-o PATCH EVALUATION RESULTS")
    print(f"{'='*80}")
    
    if 'clip_baseline' in results and 'blip3o_patch' in results:
        clip_r1 = results['clip_baseline'].get('recall@1', 0) * 100
        blip3o_r1 = results['blip3o_patch'].get('recall@1', 0) * 100
        
        print(f"CLIP Baseline R@1:    {clip_r1:5.1f}%")
        print(f"BLIP3-o Patch R@1:    {blip3o_r1:5.1f}%")
        
        if blip3o_r1 > 0:
            improvement = blip3o_r1 - clip_r1
            print(f"Improvement:          {improvement:+5.1f}%")
            
            if blip3o_r1 >= clip_r1 * 0.8:  # Within 80% of CLIP
                print("🎉 SUCCESS: Model performs well!")
                print("   ✅ Patch-level training effective")
                print("   ✅ Image-to-text recall working")
            else:
                print("⚠️  Model needs improvement")
        
        # Performance analysis
        print(f"\n📈 Performance Analysis:")
        if blip3o_r1 > 25:
            print("   🚀 EXCELLENT: Strong recall performance")
        elif blip3o_r1 > 15:
            print("   ✅ GOOD: Solid recall performance")
        elif blip3o_r1 > 5:
            print("   🔄 FAIR: Model is learning")
        else:
            print("   ⚠️  NEEDS WORK: Low recall performance")
    
    # Save results
    if args.save_results:
        save_data = {
            'evaluation_info': {
                'model_path': args.model_path,
                'coco_root': args.coco_root,
                'num_samples': len(images),
                'num_inference_steps': args.num_inference_steps,
                'approach': 'patch_level_training',
                'evaluation_metric': 'image_to_text_recall',
            },
            'results': results,
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Results saved to: {args.save_results}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)