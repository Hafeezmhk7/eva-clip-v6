#!/usr/bin/env python3
"""
BLIP3-o Enhanced Patch-Level Evaluation Script
eval_blip3o_patch_recall.py

Enhanced features:
- Support for enhanced trained models
- Better error handling and compatibility
- Improved evaluation metrics
- Enhanced model loading
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
    """Setup import paths with enhanced error handling"""
    try:
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        from src.modules.models.blip3o_patch_dit import BLIP3oPatchDiTModel, create_blip3o_patch_dit_model
        logger.info("✅ BLIP3-o modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("💡 Make sure src/modules are properly set up")
        return False

class BLIP3oPatchEvaluator:
    """
    Enhanced evaluator for BLIP3-o patch-level models
    """
    
    def __init__(self, device="auto"):
        self.device = self._setup_device(device)
        self.clip_processor = None
        self.clip_model = None
        self.eva_processor = None
        self.eva_model = None
        self.blip3o_model = None
        self.model_info = {}
        
        logger.info("Enhanced BLIP3-o patch evaluator initialized")
    
    def _setup_device(self, device_arg):
        if device_arg == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                logger.info("Using CPU")
        else:
            device = torch.device(device_arg)
        return device
    
    def load_clip_models(self):
        """Load CLIP and EVA-CLIP models"""
        logger.info("Loading CLIP ViT-L/14...")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.clip_model.eval()
            logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise
        
        logger.info("Loading EVA-CLIP-8B...")
        try:
            self.eva_model = AutoModel.from_pretrained(
                "BAAI/EVA-CLIP-8B", 
                trust_remote_code=True
            ).to(self.device)
            self.eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.eva_model.eval()
            logger.info("✅ EVA-CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load EVA-CLIP model: {e}")
            raise
        
        logger.info("✅ All foundation models loaded successfully")
    
    def load_blip3o_model(self, model_path):
        """Load the BLIP3-o patch-level model with enhanced support"""
        logger.info(f"Loading enhanced BLIP3-o patch model from: {model_path}")
        
        model_path = Path(model_path)
        
        # Enhanced config loading
        config_files = [
            model_path / "enhanced_training_config.json",
            model_path / "config.json",
            model_path / "blip3o_model_config.json"
        ]
        
        config_file = None
        config_dict = None
        
        for cf in config_files:
            if cf.exists():
                config_file = cf
                with open(config_file, 'r') as f:
                    full_config = json.load(f)
                
                # Handle enhanced config format
                if cf.name == "enhanced_training_config.json":
                    config_dict = full_config.get('model_config', {})
                    self.model_info = {
                        'enhanced': True,
                        'training_mode': full_config.get('training_strategy', {}).get('mode', 'unknown'),
                        'enhanced_features': full_config.get('enhanced_hyperparameters', {}),
                        'architecture': full_config.get('architecture', 'unknown'),
                    }
                    logger.info("✅ Enhanced training config detected")
                else:
                    config_dict = full_config
                    self.model_info = {'enhanced': False}
                
                break
        
        if not config_file:
            raise FileNotFoundError(f"No config file found in {model_path}")
        
        logger.info(f"Using config: {config_file.name}")
        
        # Create model config
        from src.modules.config.blip3o_config import BLIP3oDiTConfig
        
        # Handle different config formats
        try:
            config = BLIP3oDiTConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Direct config creation failed: {e}")
            # Try with common defaults
            config_dict.update({
                'hidden_size': config_dict.get('hidden_size', 768),
                'num_hidden_layers': config_dict.get('num_hidden_layers', 12),
                'num_attention_heads': config_dict.get('num_attention_heads', 12),
                'intermediate_size': config_dict.get('intermediate_size', 3072),
            })
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
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(weight_file))
            except ImportError:
                logger.error("SafeTensors not installed, cannot load .safetensors file")
                raise
        
        # Clean up state dict if needed
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove DDP prefix
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            logger.info("Removed DDP prefix from state dict")
        
        # Load state dict
        missing_keys, unexpected_keys = self.blip3o_model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)}")
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    logger.warning(f"  Missing: {key}")
        
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    logger.warning(f"  Unexpected: {key}")
        
        self.blip3o_model = self.blip3o_model.to(self.device)
        self.blip3o_model.eval()
        
        # Log model info
        param_count = sum(p.numel() for p in self.blip3o_model.parameters())
        logger.info(f"✅ Enhanced BLIP3-o model loaded successfully")
        logger.info(f"   Parameters: {param_count:,}")
        logger.info(f"   Enhanced: {self.model_info.get('enhanced', False)}")
        
        if self.model_info.get('enhanced'):
            enhanced_features = self.model_info.get('enhanced_features', {})
            logger.info(f"   Training mode: {self.model_info.get('training_mode', 'unknown')}")
            logger.info(f"   Epochs: {enhanced_features.get('num_epochs', 'unknown')}")
            logger.info(f"   LR scheduler: {enhanced_features.get('lr_scheduler_type', 'unknown')}")
            logger.info(f"   Convergence optimized: {enhanced_features.get('optimized_for_convergence', False)}")
    
    def extract_clip_text_embeddings(self, captions):
        """Extract CLIP text embeddings"""
        logger.info(f"Extracting text embeddings for {len(captions)} captions...")
        
        with torch.no_grad():
            inputs = self.clip_processor(
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # get_text_features() returns features already in joint embedding space (768-dim)
            text_features = self.clip_model.get_text_features(**inputs)
            logger.info(f"CLIP text features shape: {text_features.shape}")
            
            # No projection needed - text features are already in joint space!
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            logger.info(f"Final text features shape: {text_features.shape}")
        
        return text_features.cpu()
    
    def extract_clip_baseline_embeddings(self, images):
        """Extract CLIP baseline embeddings for comparison"""
        logger.info(f"Extracting CLIP baseline embeddings for {len(images)} images...")
        
        global_embeddings = []
        batch_size = 8  # Process in batches
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                
                inputs = self.clip_processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # get_image_features() already returns projected features in joint space (768-dim)
                vision_features = self.clip_model.get_image_features(**inputs)
                logger.info(f"CLIP vision features shape: {vision_features.shape}")
                
                # No additional projection needed - already in joint space!
                vision_features = F.normalize(vision_features, p=2, dim=-1)
                
                global_embeddings.append(vision_features.cpu())
        
        result = torch.cat(global_embeddings, dim=0)
        logger.info(f"Final CLIP baseline embeddings shape: {result.shape}")
        return result
    
    def extract_eva_embeddings(self, images):
        """Extract EVA-CLIP embeddings for conditioning"""
        logger.info(f"Extracting EVA-CLIP embeddings for {len(images)} images...")
        
        eva_embeddings = []
        batch_size = 4  # Smaller batch for EVA-CLIP
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                
                inputs = self.eva_processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                
                vision_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token)
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [B, 256, 4096]
                eva_embeddings.append(patch_embeddings.cpu())
        
        return torch.cat(eva_embeddings, dim=0)
    
    def generate_blip3o_embeddings(self, eva_embeddings, num_inference_steps=50):
        """Generate BLIP3-o embeddings using the enhanced patch model"""
        logger.info(f"Generating enhanced BLIP3-o embeddings for {eva_embeddings.shape[0]} images...")
        logger.info(f"Inference steps: {num_inference_steps}")
        
        eva_embeddings = eva_embeddings.to(self.device)
        
        generated_embeddings = []
        batch_size = 2  # Conservative batch size
        
        with torch.no_grad():
            for i in range(0, eva_embeddings.shape[0], batch_size):
                end_idx = min(i + batch_size, eva_embeddings.shape[0])
                batch_eva = eva_embeddings[i:end_idx]
                
                try:
                    # Generate CLIP patch embeddings
                    generated_patches = self.blip3o_model.generate(
                        eva_features=batch_eva,
                        num_inference_steps=num_inference_steps,
                    )  # [B, 256, 1024]
                    
                    logger.info(f"Generated patches shape: {generated_patches.shape}")
                    
                    # Pool to global features (raw CLIP features, 1024-dim)
                    global_features = generated_patches.mean(dim=1)  # [B, 1024]
                    logger.info(f"Pooled global features shape: {global_features.shape}")
                    
                    # Project to joint embedding space (1024 → 768)
                    global_features = self.clip_model.visual_projection(global_features)  # [B, 768]
                    logger.info(f"After visual projection shape: {global_features.shape}")
                    
                    global_features = F.normalize(global_features, p=2, dim=-1)
                    
                    generated_embeddings.append(global_features.cpu())
                    
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {i//batch_size}: {e}")
                    # Add zero embeddings for failed batch
                    batch_size_actual = batch_eva.shape[0]
                    zero_embeddings = torch.zeros(batch_size_actual, 768)  # Match expected dimension
                    generated_embeddings.append(zero_embeddings)
        
        result = torch.cat(generated_embeddings, dim=0)
        logger.info(f"Final generated embeddings shape: {result.shape}")
        
        return result
    
    def compute_recall(self, image_embeddings, text_embeddings, image_to_text_mapping, k_values=[1, 5, 10]):
        """Compute Recall@K metrics with enhanced logging"""
        # Debug embedding shapes
        logger.info(f"Image embeddings shape: {image_embeddings.shape}")
        logger.info(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Ensure embeddings are normalized
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Verify dimensions match for matrix multiplication
        if image_embeddings.shape[1] != text_embeddings.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: images {image_embeddings.shape[1]} vs text {text_embeddings.shape[1]}")
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(image_embeddings, text_embeddings.t())
        
        logger.info(f"Similarity matrix: {similarity_matrix.shape}")
        logger.info(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
        logger.info(f"Mean similarity: {similarity_matrix.mean():.4f}")
        
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
        """Evaluate a specific method with enhanced error handling"""
        logger.info(f"Evaluating method: {method}")
        
        try:
            # Extract text embeddings
            all_captions = [caption for caption_list in captions_per_image for caption in caption_list]
            logger.info(f"Processing {len(all_captions)} captions...")
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
                
                enhanced_desc = " (Enhanced)" if self.model_info.get('enhanced') else ""
                method_desc = f"BLIP3-o Patch DiT{enhanced_desc} (EVA → DiT → CLIP)"
                
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
                'enhanced_model': self.model_info.get('enhanced', False),
            })
            
            return recall_results, image_embeddings
            
        except Exception as e:
            logger.error(f"Failed to evaluate method {method}: {e}")
            return {'error': str(e)}, None


def load_coco_samples(coco_root, num_samples=1000):
    """Load COCO validation samples with enhanced error handling"""
    logger.info(f"Loading {num_samples} COCO samples...")
    
    coco_root = Path(coco_root)
    annotations_file = coco_root / "annotations" / "captions_val2017.json"
    images_dir = coco_root / "images" / "val2017"
    
    # Verify files exist
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
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
    
    logger.info(f"Found {len(images_info)} images and {len(image_captions)} with captions")
    
    # Load samples
    images = []
    captions_per_image = []
    image_ids = []
    
    loaded_count = 0
    failed_count = 0
    
    for image_id, captions in image_captions.items():
        if loaded_count >= num_samples:
            break
        
        if image_id not in images_info:
            continue
        
        image_info = images_info[image_id]
        image_path = images_dir / image_info['file_name']
        
        if not image_path.exists():
            failed_count += 1
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            captions_per_image.append(captions[:5])  # Max 5 captions
            image_ids.append(image_id)
            loaded_count += 1
            
            if loaded_count % 100 == 0:
                logger.info(f"Loaded {loaded_count}/{num_samples} images...")
                
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            failed_count += 1
            continue
    
    logger.info(f"Successfully loaded {len(images)} images with {sum(len(caps) for caps in captions_per_image)} captions")
    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count} images")
    
    return images, captions_per_image, image_ids


def main():
    """Enhanced main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Enhanced BLIP3-o Patch Model")
    parser.add_argument("--coco_root", type=str, required=True, help="COCO dataset root")
    parser.add_argument("--model_path", type=str, required=True, help="Enhanced BLIP3-o model path")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save results JSON")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Enhanced BLIP3-o Patch Evaluation")
    logger.info("=" * 60)
    
    # Setup imports
    if not setup_paths():
        return 1
    
    # Initialize evaluator
    try:
        evaluator = BLIP3oPatchEvaluator(device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return 1
    
    # Load models
    try:
        logger.info("Loading foundation models...")
        evaluator.load_clip_models()
        
        logger.info("Loading enhanced BLIP3-o model...")
        evaluator.load_blip3o_model(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return 1
    
    # Load COCO samples
    try:
        logger.info("Loading COCO validation samples...")
        images, captions_per_image, image_ids = load_coco_samples(args.coco_root, args.num_samples)
    except Exception as e:
        logger.error(f"Failed to load COCO samples: {e}")
        return 1
    
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
            if 'error' not in method_results:
                print(f"\n📊 {method.upper()} Results:")
                for k in [1, 5, 10]:
                    if f'recall@{k}' in method_results:
                        recall = method_results[f'recall@{k}']
                        print(f"  Recall@{k:2d}: {recall:.4f} ({recall*100:.2f}%)")
                print(f"  Method: {method_results.get('method_description', method)}")
            else:
                print(f"\n❌ {method.upper()} Failed: {method_results['error']}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {method}: {e}")
            results[method] = {'error': str(e)}
    
    # Print enhanced comparison
    print(f"\n{'='*80}")
    print("📊 ENHANCED BLIP3-o PATCH EVALUATION RESULTS")
    print(f"{'='*80}")
    
    if 'clip_baseline' in results and 'blip3o_patch' in results:
        clip_results = results['clip_baseline']
        blip3o_results = results['blip3o_patch']
        
        if 'error' not in clip_results and 'error' not in blip3o_results:
            clip_r1 = clip_results.get('recall@1', 0) * 100
            blip3o_r1 = blip3o_results.get('recall@1', 0) * 100
            
            print(f"CLIP Baseline R@1:    {clip_r1:5.1f}%")
            enhanced_label = " (Enhanced)" if blip3o_results.get('enhanced_model') else ""
            print(f"BLIP3-o{enhanced_label} R@1:    {blip3o_r1:5.1f}%")
            
            if blip3o_r1 > 0:
                improvement = blip3o_r1 - clip_r1
                print(f"Improvement:          {improvement:+5.1f}%")
                
                if blip3o_r1 >= clip_r1:
                    print("🎉 SUCCESS: Enhanced model performs excellently!")
                    print("   ✅ Enhanced patch-level training effective")
                    print("   ✅ Image-to-text recall working well")
                elif blip3o_r1 >= clip_r1 * 0.9:
                    print("✅ GOOD: Enhanced model performs well!")
                    print("   ✅ Close to CLIP baseline performance")
                elif blip3o_r1 >= clip_r1 * 0.8:
                    print("⚠️  ACCEPTABLE: Enhanced model needs improvement")
                    print("   💡 Within 80% of CLIP performance")
                else:
                    print("❌ NEEDS WORK: Enhanced model underperforming")
                    print("   💡 Consider model architecture or training changes")
            
            # Enhanced performance analysis
            print(f"\n📈 Enhanced Performance Analysis:")
            if blip3o_r1 > 30:
                print("   🚀 OUTSTANDING: Exceptional recall performance")
            elif blip3o_r1 > 25:
                print("   🎉 EXCELLENT: Strong recall performance")
            elif blip3o_r1 > 20:
                print("   ✅ GOOD: Solid recall performance")
            elif blip3o_r1 > 15:
                print("   🔄 FAIR: Model is learning well")
            elif blip3o_r1 > 10:
                print("   📈 IMPROVING: Model shows promise")
            else:
                print("   ⚠️  NEEDS WORK: Low recall performance")
            
            # Enhanced training feedback
            if evaluator.model_info.get('enhanced'):
                print(f"\n🚀 Enhanced Training Analysis:")
                print(f"   • Training mode: {evaluator.model_info.get('training_mode', 'unknown')}")
                enhanced_features = evaluator.model_info.get('enhanced_features', {})
                if enhanced_features:
                    print(f"   • Convergence optimized: {enhanced_features.get('optimized_for_convergence', False)}")
                    print(f"   • LR scheduler: {enhanced_features.get('lr_scheduler_type', 'unknown')}")
                    print(f"   • Epochs: {enhanced_features.get('num_epochs', 'unknown')}")
        else:
            if 'error' in clip_results:
                print(f"❌ CLIP baseline error: {clip_results['error']}")
            if 'error' in blip3o_results:
                print(f"❌ Enhanced BLIP3-o error: {blip3o_results['error']}")
    
    # Save enhanced results
    if args.save_results:
        save_data = {
            'evaluation_info': {
                'model_path': args.model_path,
                'coco_root': args.coco_root,
                'num_samples': len(images),
                'num_inference_steps': args.num_inference_steps,
                'approach': 'enhanced_patch_level_training',
                'evaluation_metric': 'image_to_text_recall',
                'enhanced_model': evaluator.model_info.get('enhanced', False),
                'model_info': evaluator.model_info,
                'timestamp': time.time(),
            },
            'results': results,
        }
        
        # Ensure results directory exists
        results_path = Path(args.save_results)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Enhanced results saved to: {args.save_results}")
    
    # Final success check
    success = False
    if 'clip_baseline' in results and 'blip3o_patch' in results:
        clip_results = results['clip_baseline']
        blip3o_results = results['blip3o_patch']
        
        if 'error' not in clip_results and 'error' not in blip3o_results:
            success = True
    
    logger.info(f"\n🏁 Enhanced evaluation completed {'successfully' if success else 'with errors'}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)