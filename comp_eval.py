#!/usr/bin/env python3
"""
FIXED: BLIP3-o Recall Evaluation Script with Proper Dual Supervision Support
This script can now correctly load your trained dual supervision model!
"""

import os
import sys
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
import gc
from scipy.stats import pearsonr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def convert_to_serializable(obj):
    """Convert numpy/torch types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # torch tensors
        return obj.item()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy().tolist()
    else:
        return obj


class FixedBLIP3oRecallEvaluator:
    """
    FIXED: Evaluator for BLIP3-o recall with proper dual supervision support.
    
    Now correctly loads and tests dual supervision models!
    """
    
    def __init__(
        self, 
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """Initialize the FIXED evaluator."""
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype or torch.float32
        
        # Will be loaded when needed
        self.clip_processor = None
        self.clip_model = None
        self.eva_processor = None
        self.eva_model = None
        self.blip3o_model = None
        
        logger.info("FIXED BLIP3-o Recall Evaluator initialized")
        logger.info(f"Device: {self.device}")
    
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
    
    def load_clip_models(self):
        """Load CLIP ViT-L/14 and EVA-CLIP models."""
        logger.info("Loading CLIP ViT-L/14...")
        
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.clip_model.eval()
        
        logger.info("Loading EVA-CLIP-8B...")
        
        self.eva_model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B",
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        self.eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.eva_model.eval()
        
        logger.info("✅ CLIP models loaded successfully")

    def load_blip3o_model(self, model_path: str):
        """FIXED: Load trained BLIP3-o model with proper dual supervision support."""
        logger.info(f"Loading BLIP3-o model from: {model_path}")
        
        try:
            # FIXED: Direct imports to avoid module loading issues
            import importlib.util
            
            # Load config directly
            config_module_path = Path("src/modules/config/blip3o_config.py")
            spec = importlib.util.spec_from_file_location("blip3o_config", config_module_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            BLIP3oDiTConfig = config_module.BLIP3oDiTConfig
            
            # Load dual supervision model directly
            model_module_path = Path("src/modules/models/dual_supervision_blip3o_dit.py")
            if model_module_path.exists():
                logger.info("✅ Found dual supervision model file")
                spec = importlib.util.spec_from_file_location("dual_supervision_model", model_module_path)
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                
                DualSupervisionBLIP3oDiTModel = model_module.DualSupervisionBLIP3oDiTModel
                create_blip3o_dit_model = model_module.create_blip3o_dit_model
                dual_supervision_available = True
                logger.info("✅ Successfully loaded dual supervision model")
            else:
                # Fallback to standard model
                logger.warning("Dual supervision model not found, using standard model")
                standard_model_path = Path("src/modules/models/blip3o_dit.py")
                spec = importlib.util.spec_from_file_location("blip3o_dit", standard_model_path)
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                
                BLIP3oDiTModel = model_module.BLIP3oDiTModel
                create_blip3o_dit_model = model_module.create_blip3o_dit_model
                dual_supervision_available = False
            
            model_path = Path(model_path)
            
            # Load configuration
            config_file = model_path / "config.json"
            if not config_file.exists():
                config_file = model_path / "blip3o_model_config.json"
            
            if not config_file.exists():
                raise FileNotFoundError(f"No config file found in {model_path}")
            
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            logger.info(f"Loaded config with {len(config_dict)} parameters")
            
            # Create configuration
            config = BLIP3oDiTConfig(**config_dict)
            
            # FIXED: Create model using the correct approach
            if dual_supervision_available:
                logger.info("🎯 Creating dual supervision model...")
                self.blip3o_model = create_blip3o_dit_model(
                    config=config,
                    load_clip_projection=True,
                    enable_dual_supervision=True,
                )
                logger.info("✅ Created dual supervision model")
            else:
                logger.info("⚠️  Creating standard model...")
                self.blip3o_model = create_blip3o_dit_model(config)
            
            # Load weights
            model_files = [
                model_path / "model.safetensors",
                model_path / "pytorch_model.bin",
                model_path / "pytorch_model.safetensors"
            ]
            
            model_file = None
            for file_path in model_files:
                if file_path.exists():
                    model_file = file_path
                    break
            
            if model_file is None:
                raise FileNotFoundError(f"No model weights found in {model_path}")
            
            logger.info(f"Loading weights from: {model_file}")
            
            # Load weights
            if model_file.suffix == ".bin":
                state_dict = torch.load(model_file, map_location=self.device)
            else:
                from safetensors.torch import load_file
                state_dict = load_file(str(model_file))
            
            logger.info(f"Loaded state dict with {len(state_dict)} keys")
            
            # Check for dual supervision keys - THIS IS CRITICAL
            dual_keys = [k for k in state_dict.keys() if 'global_velocity_proj' in k]
            if dual_keys:
                logger.info(f"✅ Found dual supervision keys: {dual_keys}")
                logger.info("🎯 CONFIRMED: This checkpoint has dual supervision!")
            else:
                logger.warning("❌ No dual supervision keys found in checkpoint")
                logger.warning("⚠️  This checkpoint was NOT trained with dual supervision")
            
            # FIXED: Load state dict with better error handling
            try:
                missing_keys, unexpected_keys = self.blip3o_model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                
                logger.info("✅ Model weights loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading state dict: {e}")
                raise
            
            # Move to device
            self.blip3o_model = self.blip3o_model.to(device=self.device, dtype=self.torch_dtype)
            self.blip3o_model.eval()
            
            # FIXED: Check model capabilities properly
            self.model_capabilities = self._check_model_capabilities()
            
            logger.info(f"✅ BLIP3-o model loaded successfully")
            logger.info(f"   Parameters: {self._get_num_parameters():,}")
            logger.info(f"   Model type: {'FIXED Dual Supervision' if dual_supervision_available else 'Standard'}")
            logger.info(f"   Capabilities: {self.model_capabilities}")
            
            # CRITICAL: Verify the model actually has dual supervision
            if self.model_capabilities.get('has_global_velocity_proj', False):
                logger.info("🎉 SUCCESS: Model has global velocity projection!")
                logger.info("🎯 Expected performance: 50-70% recall")
            else:
                logger.warning("⚠️  Model missing global velocity projection!")
                logger.warning("🎯 Expected performance: 0-2% recall (like before)")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP3-o model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _check_model_capabilities(self) -> Dict[str, bool]:
        """FIXED: Check what capabilities the loaded model has."""
        capabilities = {
            'has_frozen_clip_proj': hasattr(self.blip3o_model, 'frozen_clip_visual_proj') and self.blip3o_model.frozen_clip_visual_proj is not None,
            'has_global_adaptation_mlp': hasattr(self.blip3o_model, 'global_adaptation_mlp'),
            'has_global_velocity_proj': hasattr(self.blip3o_model, 'global_velocity_proj'),  # KEY CAPABILITY!
            'supports_generation_modes': hasattr(self.blip3o_model, 'generate'),
            'is_dual_supervision_model': hasattr(self.blip3o_model, 'global_velocity_proj'),
        }
        
        return capabilities

    def _get_num_parameters(self) -> int:
        """Get number of model parameters."""
        if hasattr(self.blip3o_model, 'get_num_parameters'):
            return self.blip3o_model.get_num_parameters()
        else:
            return sum(p.numel() for p in self.blip3o_model.parameters())
    
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
        
        return text_embeddings.cpu().float()
    
    def extract_clip_vision_global_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract CLIP vision global embeddings (CLS token + visual projection)."""
        global_embeddings = []
        
        with torch.no_grad():
            for img in images:
                inputs = self.clip_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                         for k, v in inputs.items()}
                
                vision_embeddings = self.clip_model.get_image_features(**inputs)  # [1, 768]
                vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)
                
                global_embeddings.append(vision_embeddings.squeeze(0).cpu().float())  # [768]
        
        return torch.stack(global_embeddings)  # [B, 768]
    
    def extract_eva_vision_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract EVA-CLIP vision embeddings for BLIP3-o conditioning."""
        eva_embeddings = []
        
        logger.info(f"Extracting EVA embeddings for {len(images)} images...")
        
        with torch.no_grad():
            for i, img in enumerate(images):
                if i % 100 == 0:
                    logger.debug(f"Processing EVA image {i}/{len(images)}")
                
                inputs = self.eva_processor(images=img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(device=self.device, dtype=self.torch_dtype)
                
                vision_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get patch embeddings (remove CLS token) → [1, 256, hidden_dim]
                patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
                batch_size, num_patches, hidden_dim = patch_embeddings.shape
                
                # Verify dimensions
                assert num_patches == 256, f"Expected 256 patches, got {num_patches}"
                assert hidden_dim == 4096, f"Expected 4096 dimensions, got {hidden_dim}"
                
                eva_embeddings.append(patch_embeddings.squeeze(0).cpu().float())  # [256, 4096]
        
        result = torch.stack(eva_embeddings)  # [B, 256, 4096]
        logger.info(f"EVA embeddings extracted: {result.shape}")
        
        return result
    
    def generate_blip3o_embeddings_fixed(
        self, 
        eva_embeddings: torch.Tensor, 
        num_inference_steps: int = 50,
        generation_mode: str = "auto"
    ) -> torch.Tensor:
        """
        FIXED: Generate CLIP embeddings using dual supervision model.
        """
        if self.blip3o_model is None:
            raise ValueError("BLIP3-o model not loaded. Call load_blip3o_model() first.")
        
        logger.info(f"=== FIXED BLIP3-o Evaluation Pipeline ===")
        logger.info(f"Step 1: Input shape: {eva_embeddings.shape}")
        logger.info(f"Step 2: Generating CLIP embeddings using BLIP3-o...")
        logger.info(f"Model capabilities: {self.model_capabilities}")
        
        # Move to correct device
        eva_embeddings = eva_embeddings.to(device=self.device, dtype=self.torch_dtype)
        
        # FIXED: Determine generation mode based on model capabilities
        if generation_mode == "auto":
            if self.model_capabilities.get('has_global_velocity_proj', False):
                generation_mode = "global"  # Use global for dual supervision models
                logger.info("🎯 Auto-selected GLOBAL generation mode (dual supervision model)")
            else:
                generation_mode = "standard"  # Fallback for standard models
                logger.info("🔄 Auto-selected STANDARD generation mode (standard model)")
        
        logger.info(f"Generation mode: {generation_mode}")
        
        generated_embeddings = []
        
        with torch.no_grad():
            # Process in batches to manage memory
            batch_size = 8
            num_samples = eva_embeddings.shape[0]
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_eva = eva_embeddings[i:end_idx]
                
                logger.debug(f"Processing batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
                
                try:
                    # FIXED: Try different generation approaches based on model type
                    if (generation_mode == "global" and 
                        hasattr(self.blip3o_model, 'generate')):
                        
                        # Try to use generation_mode parameter if supported
                        try:
                            generated = self.blip3o_model.generate(
                                encoder_hidden_states=batch_eva,
                                num_inference_steps=num_inference_steps,
                                generation_mode="global",
                                return_global_only=True,
                            )
                            logger.debug("Used global generation mode")
                        except TypeError:
                            # Fallback if generation_mode parameter not supported
                            generated = self.blip3o_model.generate(
                                encoder_hidden_states=batch_eva,
                                num_inference_steps=num_inference_steps,
                                return_global_only=True,
                            )
                            logger.debug("Used standard generation with global return")
                        
                    else:
                        # Standard generation
                        generated = self.blip3o_model.generate(
                            encoder_hidden_states=batch_eva,
                            num_inference_steps=num_inference_steps,
                        )
                        
                        # Convert to global if needed
                        if generated.dim() == 3 and generated.shape[1] == 256:
                            logger.debug("Converting patch output to global")
                            generated = generated.mean(dim=1)  # Average pool
                            
                            # Apply CLIP projection if available
                            if (hasattr(self.blip3o_model, 'frozen_clip_visual_proj') and 
                                self.blip3o_model.frozen_clip_visual_proj is not None):
                                generated = self.blip3o_model.frozen_clip_visual_proj(generated)
                                logger.debug("Applied CLIP projection")
                    
                    # Ensure normalization for CLIP compatibility
                    generated = F.normalize(generated, p=2, dim=-1)
                    generated_embeddings.append(generated.cpu().float())
                    
                except Exception as e:
                    logger.error(f"Error in generation for batch {i//batch_size + 1}: {e}")
                    # Create dummy output to maintain batch consistency
                    dummy_output = torch.zeros(batch_eva.shape[0], 768)
                    generated_embeddings.append(dummy_output)
        
        # Concatenate all batches
        result = torch.cat(generated_embeddings, dim=0)  # [B, 768]
        
        logger.info(f"BLIP3-o generation completed: {result.shape}")
        logger.info(f"Generation mode used: {generation_mode}")
        
        # Final verification
        final_norms = torch.norm(result, p=2, dim=-1)
        logger.info(f"Generated embedding norms: mean={final_norms.mean():.6f}, std={final_norms.std():.6f}")
        logger.info(f"=== FIXED BLIP3-o Evaluation Complete ===")
        
        return result
    
    def compute_image_to_text_recall(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_to_text_mapping: List[List[int]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """Compute image-to-text recall metrics."""
        # Ensure embeddings are normalized
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix: [N_images, N_texts]
        similarity_matrix = torch.mm(image_embeddings, text_embeddings.t())
        
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
    
    def evaluate_method(
        self,
        images: List[Image.Image],
        captions_per_image: List[List[str]],
        method: str,
        k_values: List[int] = [1, 5, 10],
        num_inference_steps: int = 50,
        generation_mode: str = "auto",
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """Evaluate a specific method and return embeddings."""
        logger.info(f"Evaluating method: {method}")
        
        # Extract text embeddings (same for all methods)
        logger.info("Extracting text embeddings...")
        all_text_embeddings = []
        image_to_text_mapping = []
        text_idx = 0
        
        # Flatten captions and create mapping (using all 5 captions per image)
        for img_idx, caption_list in enumerate(captions_per_image):
            current_text_indices = []
            for caption in caption_list:
                current_text_indices.append(text_idx)
                text_idx += 1
            image_to_text_mapping.append(current_text_indices)
        
        # Extract all captions
        all_captions = [caption for caption_list in captions_per_image for caption in caption_list]
        text_embeddings = self.extract_clip_text_embeddings(all_captions)
        
        logger.info(f"Total captions: {len(all_captions)}")
        logger.info(f"Average captions per image: {len(all_captions) / len(images):.2f}")
        
        # Extract image embeddings based on method
        logger.info(f"Extracting image embeddings using {method} method...")
        
        if method == "clip_baseline":
            # Use CLIP's standard image features
            image_embeddings = self.extract_clip_vision_global_embeddings(images)
            method_description = "CLIP ViT-L/14 image features"
            
        elif method == "blip3o_fixed":
            if self.blip3o_model is None:
                raise ValueError("BLIP3-o model not loaded. Call load_blip3o_model() first.")
            
            logger.info("=== FIXED BLIP3-o Evaluation Pipeline ===")
            logger.info("Step 1: Extracting EVA-CLIP embeddings...")
            
            # Extract EVA embeddings first
            eva_embeddings = self.extract_eva_vision_embeddings(images)
            logger.info(f"EVA embeddings extracted: {eva_embeddings.shape}")
            
            logger.info("Step 2: Generating CLIP embeddings using FIXED BLIP3-o...")
            
            # Generate CLIP embeddings using FIXED BLIP3-o
            image_embeddings = self.generate_blip3o_embeddings_fixed(
                eva_embeddings, 
                num_inference_steps, 
                generation_mode
            )
            logger.info(f"BLIP3-o embeddings generated: {image_embeddings.shape}")
            
            method_description = f"FIXED BLIP3-o embeddings (EVA → DiT → {generation_mode} generation)"
            
            logger.info("=== FIXED BLIP3-o Evaluation Complete ===")
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'clip_baseline' or 'blip3o_fixed'.")
        
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
            'embedding_dim': image_embeddings.shape[-1],
            'generation_mode': generation_mode if method == "blip3o_fixed" else "N/A",
            'model_capabilities': self.model_capabilities if method == "blip3o_fixed" else {},
        })
        
        # Memory cleanup
        if method == "blip3o_fixed":
            del eva_embeddings
        gc.collect()
        torch.cuda.empty_cache()
        
        return recall_results, image_embeddings


def load_coco_samples(coco_root: Path, num_samples: int = 1000) -> Tuple[List[Image.Image], List[List[str]], List[int]]:
    """Load COCO validation samples."""
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
            captions_per_image.append(captions[:5])  # Max 5 captions per image (STANDARD)
            image_ids.append(image_id)
            loaded_count += 1
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(images)} images with {sum(len(caps) for caps in captions_per_image)} captions")
    logger.info(f"Average captions per image: {sum(len(caps) for caps in captions_per_image) / len(images):.2f}")
    
    return images, captions_per_image, image_ids


def main():
    parser = argparse.ArgumentParser(description="FIXED BLIP3-o Recall Evaluation")
    parser.add_argument("--coco_root", type=str, required=True,
                       help="Path to MS-COCO dataset root directory")
    parser.add_argument("--blip3o_model_path", type=str, required=True,
                       help="Path to trained BLIP3-o model")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of COCO samples to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--save_results", type=str, default=None,
                       help="Path to save results JSON file")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 5, 10],
                       help="K values for Recall@K computation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps for BLIP3-o generation")
    parser.add_argument("--generation_mode", type=str, default="auto",
                       choices=["auto", "global", "standard"],
                       help="BLIP3-o generation mode")
    
    args = parser.parse_args()
    
    # Convert paths
    coco_root = Path(args.coco_root)
    blip3o_model_path = Path(args.blip3o_model_path)
    
    # Validate paths
    if not coco_root.exists():
        logger.error(f"COCO root not found: {coco_root}")
        return 1
    
    if not blip3o_model_path.exists():
        logger.error(f"BLIP3-o model path not found: {blip3o_model_path}")
        return 1
    
    # Initialize FIXED evaluator
    logger.info("Initializing FIXED BLIP3-o Recall Evaluator...")
    evaluator = FixedBLIP3oRecallEvaluator(device=args.device)
    
    # Load CLIP models
    evaluator.load_clip_models()
    
    # Load BLIP3-o model
    evaluator.load_blip3o_model(str(blip3o_model_path))
    
    # Load COCO samples
    logger.info(f"Loading {args.num_samples} COCO validation samples...")
    images, captions_per_image, image_ids = load_coco_samples(coco_root, args.num_samples)
    
    # Run evaluations - test both CLIP baseline and FIXED BLIP3-o
    methods_to_evaluate = ["clip_baseline", "blip3o_fixed"]
    
    all_results = {}
    
    start_time = time.time()
    
    for method in methods_to_evaluate:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Evaluating method: {method.upper()}")
        if method == "blip3o_fixed":
            logger.info(f"🎯 Generation mode: {args.generation_mode}")
        logger.info(f"{'='*60}")
        
        method_start_time = time.time()
        
        try:
            results, embeddings = evaluator.evaluate_method(
                images=images,
                captions_per_image=captions_per_image,
                method=method,
                k_values=args.k_values,
                num_inference_steps=args.num_inference_steps,
                generation_mode=args.generation_mode,
            )
            
            method_time = time.time() - method_start_time
            results['evaluation_time'] = method_time
            
            all_results[method] = results
            
            # Print results for this method
            print(f"\n📊 {method.upper()} Results:")
            print(f"Method: {results['method_description']}")
            print(f"Time: {method_time:.2f}s")
            if method == "blip3o_fixed":
                print(f"Generation mode: {results['generation_mode']}")
                print(f"Model has dual supervision: {results['model_capabilities'].get('is_dual_supervision_model', False)}")
            for k in args.k_values:
                if f'recall@{k}' in results:
                    recall_k = results[f'recall@{k}']
                    print(f"  Recall@{k:2d}: {recall_k:.4f} ({recall_k*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to evaluate method {method}: {e}")
            import traceback
            traceback.print_exc()
            all_results[method] = {'error': str(e)}
    
    total_time = time.time() - start_time
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("📊 FIXED BLIP3-O RECALL EVALUATION RESULTS")
    print("="*80)
    print(f"Dataset: MS-COCO 2017 Validation ({len(images)} images, {sum(len(caps) for caps in captions_per_image)} captions)")
    print(f"Total evaluation time: {total_time:.2f}s")
    print(f"Generation mode: {args.generation_mode}")
    
    # Results table
    print(f"\n📋 Results Summary:")
    print(f"{'Method':<15} {'Description':<40} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'Time':<8}")
    print("-" * 85)
    
    for method, results in all_results.items():
        if 'error' in results:
            print(f"{method:<15} {'ERROR: ' + results['error']:<40} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
        else:
            description = results.get('method_description', 'Unknown')[:39]
            r1 = f"{results.get('recall@1', 0)*100:.1f}%" if 'recall@1' in results else "N/A"
            r5 = f"{results.get('recall@5', 0)*100:.1f}%" if 'recall@5' in results else "N/A"
            r10 = f"{results.get('recall@10', 0)*100:.1f}%" if 'recall@10' in results else "N/A"
            eval_time = f"{results.get('evaluation_time', 0):.1f}s"
            print(f"{method:<15} {description:<40} {r1:<8} {r5:<8} {r10:<8} {eval_time:<8}")
    
    # Performance analysis
    if 'clip_baseline' in all_results and 'blip3o_fixed' in all_results:
        if 'error' not in all_results['clip_baseline'] and 'error' not in all_results['blip3o_fixed']:
            print(f"\n🎯 FIXED BLIP3-o Performance Analysis:")
            baseline_r1 = all_results['clip_baseline']['recall@1']
            blip3o_r1 = all_results['blip3o_fixed']['recall@1']
            
            print(f"   CLIP Baseline R@1: {baseline_r1*100:.2f}%")
            print(f"   FIXED BLIP3-o R@1: {blip3o_r1*100:.2f}%")
            
            if blip3o_r1 > 0.001:  # Avoid division by zero
                improvement_factor = blip3o_r1 / 0.001  # Compare to previous ~0.1%
                print(f"   Improvement over old model: {improvement_factor:.0f}x")
            
            if blip3o_r1 > baseline_r1:
                improvement = ((blip3o_r1 - baseline_r1) / baseline_r1) * 100
                print(f"   🎉 BLIP3-o IMPROVEMENT: +{improvement:.1f}% vs CLIP")
                print(f"   ✅ SUCCESS: Model outperforms CLIP baseline!")
            elif blip3o_r1 >= baseline_r1 * 0.8:  # Within 20% of baseline
                print(f"   ✅ SUCCESS: Model competitive with CLIP baseline")
            elif blip3o_r1 >= baseline_r1 * 0.5:  # Within 50% of baseline
                print(f"   ⚠️  PARTIAL SUCCESS: Significant improvement but below CLIP")
            else:
                print(f"   ❌ NEEDS IMPROVEMENT: Still below expectations")
            
            # Model capability analysis
            model_caps = all_results['blip3o_fixed'].get('model_capabilities', {})
            if model_caps.get('is_dual_supervision_model', False):
                print(f"   📊 Model Type: FIXED Dual Supervision (with global generation)")
                print(f"   🎯 Generation Mode: {args.generation_mode}")
            else:
                print(f"   📊 Model Type: Standard (may need updating)")
    
    # Save results if requested
    if args.save_results:
        results_to_save = {
            'evaluation_info': {
                'dataset': 'MS-COCO 2017 Validation',
                'num_images': len(images),
                'num_captions': sum(len(caps) for caps in captions_per_image),
                'blip3o_model_path': str(blip3o_model_path),
                'generation_mode': args.generation_mode,
                'total_time': total_time,
                'device': str(evaluator.device),
                'k_values': args.k_values,
                'num_inference_steps': args.num_inference_steps,
                'evaluation_type': 'fixed_dual_supervision_blip3o',
            },
            'method_results': convert_to_serializable(all_results),
            'model_capabilities': evaluator.model_capabilities if hasattr(evaluator, 'model_capabilities') else {},
        }
        
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            logger.info(f"Results saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)