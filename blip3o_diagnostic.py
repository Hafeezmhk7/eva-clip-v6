#!/usr/bin/env python3
"""
BLIP3o Comprehensive Diagnostic Script
Identifies root causes of the evaluation failures.

Issues to investigate:
1. Why is patch averaging giving 9.8% instead of ~60%?
2. Why is BLIP3o giving 0% recall?
3. Are there bugs in embedding extraction or processing?
"""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, AutoModel
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import json
import time
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import BLIP3o modules
try:
    from src.modules.inference.blip3o_inference import BLIP3oInference
    BLIP3O_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BLIP3o not available: {e}")
    BLIP3O_AVAILABLE = False


class BLIP3oDiagnosticTool:
    """Comprehensive diagnostic tool for BLIP3o evaluation issues."""
    
    def __init__(self, device: str = "cuda", blip3o_model_path: str = None):
        self.device = torch.device(device)
        self.blip3o_model_path = blip3o_model_path
        
        # Load models
        self._load_clip_model()
        self._load_eva_model()
        
        if blip3o_model_path and BLIP3O_AVAILABLE:
            self._load_blip3o_model()
        else:
            self.blip3o_inference = None
    
    def _load_clip_model(self):
        """Load CLIP model."""
        logger.info("Loading CLIP ViT-L/14...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.clip_model.eval()
    
    def _load_eva_model(self):
        """Load EVA model."""
        logger.info("Loading EVA-CLIP-8B...")
        self.eva_model = AutoModel.from_pretrained("BAAI/EVA-CLIP-8B", trust_remote_code=True).to(self.device)
        self.eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.eva_model.eval()
    
    def _load_blip3o_model(self):
        """Load BLIP3o model."""
        try:
            logger.info(f"Loading BLIP3o from {self.blip3o_model_path}...")
            self.blip3o_inference = BLIP3oInference(
                model_path=self.blip3o_model_path,
                device=self.device
            )
            logger.info("BLIP3o loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP3o: {e}")
            self.blip3o_inference = None
    
    def test_patch_averaging_bug(self, test_image: Image.Image) -> Dict:
        """
        Test why patch averaging is giving 9.8% instead of ~60%.
        This should reveal the bug in patch averaging implementation.
        """
        logger.info("🔍 DIAGNOSING PATCH AVERAGING BUG")
        logger.info("=" * 50)
        
        results = {}
        
        with torch.no_grad():
            # Process image
            inputs = self.clip_processor(images=test_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get vision outputs
            vision_outputs = self.clip_model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract tokens
            all_tokens = vision_outputs.last_hidden_state  # [1, 257, 1024]
            cls_token = all_tokens[:, 0, :]  # [1, 1024] 
            patch_tokens = all_tokens[:, 1:, :]  # [1, 256, 1024]
            
            logger.info(f"All tokens shape: {all_tokens.shape}")
            logger.info(f"CLS token shape: {cls_token.shape}")
            logger.info(f"Patch tokens shape: {patch_tokens.shape}")
            
            # Method 1: Global token (working method)
            global_projected = self.clip_model.visual_projection(cls_token)  # [1, 768]
            global_normalized = F.normalize(global_projected, p=2, dim=-1)
            
            # Method 2: Patch averaging (broken method - let's debug)
            patch_averaged = patch_tokens.mean(dim=1)  # [1, 1024]
            patch_projected = self.clip_model.visual_projection(patch_averaged)  # [1, 768]
            patch_normalized = F.normalize(patch_projected, p=2, dim=-1)
            
            logger.info(f"Patch averaged shape: {patch_averaged.shape}")
            logger.info(f"Patch projected shape: {patch_projected.shape}")
            
            # Compare the embeddings
            cosine_sim = F.cosine_similarity(global_normalized, patch_normalized)
            l2_distance = torch.norm(global_normalized - patch_normalized, p=2)
            
            logger.info(f"Global vs Patch cosine similarity: {cosine_sim.item():.6f}")
            logger.info(f"Global vs Patch L2 distance: {l2_distance.item():.6f}")
            
            # Check norms
            global_norm = torch.norm(global_normalized, p=2)
            patch_norm = torch.norm(patch_normalized, p=2)
            logger.info(f"Global norm: {global_norm.item():.6f}")
            logger.info(f"Patch norm: {patch_norm.item():.6f}")
            
            # Statistical analysis of tokens
            logger.info(f"CLS token stats - mean: {cls_token.mean():.6f}, std: {cls_token.std():.6f}")
            logger.info(f"Patch tokens stats - mean: {patch_tokens.mean():.6f}, std: {patch_tokens.std():.6f}")
            logger.info(f"Patch averaged stats - mean: {patch_averaged.mean():.6f}, std: {patch_averaged.std():.6f}")
            
            # Check if averaging destroys important information
            patch_var = patch_tokens.var(dim=1).mean().item()  # Variance across patches
            logger.info(f"Variance across patches: {patch_var:.6f}")
            
            if cosine_sim.item() < 0.5:
                logger.warning("🚨 CRITICAL: Patch averaging produces very different embeddings!")
                logger.warning("This explains the 9.8% vs 61% performance gap!")
            
            results = {
                'global_vs_patch_similarity': cosine_sim.item(),
                'global_vs_patch_l2_distance': l2_distance.item(),
                'global_norm': global_norm.item(),
                'patch_norm': patch_norm.item(),
                'cls_token_mean': cls_token.mean().item(),
                'cls_token_std': cls_token.std().item(),
                'patch_tokens_mean': patch_tokens.mean().item(),
                'patch_tokens_std': patch_tokens.std().item(),
                'patch_variance': patch_var,
                'diagnosis': 'PATCH_AVERAGING_DESTROYS_INFORMATION' if cosine_sim.item() < 0.5 else 'PATCH_AVERAGING_OK'
            }
        
        return results
    
    def test_blip3o_pipeline(self, test_image: Image.Image) -> Dict:
        """
        Test why BLIP3o is giving 0% recall.
        This should reveal issues in the BLIP3o pipeline.
        """
        logger.info("🔍 DIAGNOSING BLIP3O PIPELINE")
        logger.info("=" * 50)
        
        if self.blip3o_inference is None:
            return {'error': 'BLIP3o model not loaded'}
        
        results = {}
        
        try:
            with torch.no_grad():
                # Step 1: Extract EVA embeddings
                eva_inputs = self.eva_processor(images=test_image, return_tensors="pt")
                pixel_values = eva_inputs['pixel_values'].to(self.device)
                
                eva_outputs = self.eva_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                eva_patches = eva_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 4096]
                logger.info(f"EVA patches shape: {eva_patches.shape}")
                logger.info(f"EVA patches range: [{eva_patches.min():.4f}, {eva_patches.max():.4f}]")
                logger.info(f"EVA patches mean: {eva_patches.mean():.6f}, std: {eva_patches.std():.6f}")
                
                # Step 2: Generate CLIP embeddings using BLIP3o
                generated_patches = self.blip3o_inference.generate(
                    eva_patches,  # [1, 256, 4096]
                    num_inference_steps=50,
                )  # Should be [1, 256, 1024]
                
                logger.info(f"Generated patches shape: {generated_patches.shape}")
                logger.info(f"Generated patches range: [{generated_patches.min():.4f}, {generated_patches.max():.4f}]")
                logger.info(f"Generated patches mean: {generated_patches.mean():.6f}, std: {generated_patches.std():.6f}")
                
                # Check for obvious issues
                has_nan = torch.isnan(generated_patches).any()
                has_inf = torch.isinf(generated_patches).any()
                is_all_zeros = torch.allclose(generated_patches, torch.zeros_like(generated_patches))
                is_all_same = generated_patches.std() < 1e-6
                
                logger.info(f"Generated patches - NaN: {has_nan}, Inf: {has_inf}, All zeros: {is_all_zeros}, All same: {is_all_same}")
                
                if has_nan or has_inf:
                    logger.error("🚨 CRITICAL: Generated patches contain NaN or Inf values!")
                
                if is_all_zeros:
                    logger.error("🚨 CRITICAL: Generated patches are all zeros!")
                
                if is_all_same:
                    logger.error("🚨 CRITICAL: Generated patches are all the same value!")
                
                # Step 3: Average and project
                generated_global = generated_patches.mean(dim=1)  # [1, 1024]
                generated_projected = self.clip_model.visual_projection(generated_global)  # [1, 768]
                generated_normalized = F.normalize(generated_projected, p=2, dim=-1)
                
                logger.info(f"Generated global shape: {generated_global.shape}")
                logger.info(f"Generated projected shape: {generated_projected.shape}")
                logger.info(f"Generated normalized norm: {torch.norm(generated_normalized):.6f}")
                
                # Compare with real CLIP embedding
                clip_inputs = self.clip_processor(images=test_image, return_tensors="pt")
                clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
                
                clip_outputs = self.clip_model.vision_model(
                    pixel_values=clip_inputs['pixel_values'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                clip_cls = clip_outputs.last_hidden_state[:, 0, :]  # [1, 1024]
                clip_projected = self.clip_model.visual_projection(clip_cls)  # [1, 768]
                clip_normalized = F.normalize(clip_projected, p=2, dim=-1)
                
                # Compare embeddings
                similarity = F.cosine_similarity(generated_normalized, clip_normalized)
                l2_dist = torch.norm(generated_normalized - clip_normalized, p=2)
                
                logger.info(f"BLIP3o vs CLIP similarity: {similarity.item():.6f}")
                logger.info(f"BLIP3o vs CLIP L2 distance: {l2_dist.item():.6f}")
                
                if similarity.item() < 0.1:
                    logger.error("🚨 CRITICAL: BLIP3o embeddings have no similarity to CLIP!")
                    logger.error("This explains the 0% recall performance!")
                
                results = {
                    'eva_shape': list(eva_patches.shape),
                    'eva_range': [eva_patches.min().item(), eva_patches.max().item()],
                    'eva_mean': eva_patches.mean().item(),
                    'eva_std': eva_patches.std().item(),
                    'generated_shape': list(generated_patches.shape),
                    'generated_range': [generated_patches.min().item(), generated_patches.max().item()],
                    'generated_mean': generated_patches.mean().item(),
                    'generated_std': generated_patches.std().item(),
                    'has_nan': has_nan.item(),
                    'has_inf': has_inf.item(),
                    'is_all_zeros': is_all_zeros.item(),
                    'is_all_same': is_all_same.item(),
                    'blip3o_vs_clip_similarity': similarity.item(),
                    'blip3o_vs_clip_l2_distance': l2_dist.item(),
                    'diagnosis': self._diagnose_blip3o_issues(similarity.item(), has_nan, has_inf, is_all_zeros, is_all_same)
                }
        
        except Exception as e:
            logger.error(f"Error in BLIP3o pipeline test: {e}")
            import traceback
            traceback.print_exc()
            results = {'error': str(e)}
        
        return results
    
    def _diagnose_blip3o_issues(self, similarity, has_nan, has_inf, is_all_zeros, is_all_same):
        """Diagnose specific BLIP3o issues."""
        if has_nan or has_inf:
            return "NUMERICAL_INSTABILITY"
        elif is_all_zeros:
            return "MODEL_NOT_GENERATING"
        elif is_all_same:
            return "MODEL_COLLAPSED"
        elif similarity < 0.05:
            return "COMPLETELY_WRONG_MAPPING"
        elif similarity < 0.3:
            return "POOR_MAPPING_QUALITY"
        else:
            return "MAPPING_REASONABLE"
    
    def test_text_embedding_consistency(self, test_captions: List[str]) -> Dict:
        """Test if text embeddings are working correctly."""
        logger.info("🔍 TESTING TEXT EMBEDDING CONSISTENCY")
        logger.info("=" * 50)
        
        with torch.no_grad():
            # Extract text embeddings
            inputs = self.clip_processor(text=test_captions, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_embeddings = self.clip_model.get_text_features(**inputs)
            text_normalized = F.normalize(text_embeddings, p=2, dim=-1)
            
            logger.info(f"Text embeddings shape: {text_embeddings.shape}")
            logger.info(f"Text embeddings range: [{text_embeddings.min():.4f}, {text_embeddings.max():.4f}]")
            logger.info(f"Text embeddings mean: {text_embeddings.mean():.6f}, std: {text_embeddings.std():.6f}")
            
            # Check normalization
            norms = torch.norm(text_normalized, p=2, dim=-1)
            logger.info(f"Text embedding norms: {norms}")
            
            # Test similarity between captions
            if len(test_captions) >= 2:
                sim_matrix = torch.mm(text_normalized, text_normalized.t())
                logger.info(f"Text similarity matrix:\n{sim_matrix}")
        
        return {
            'text_shape': list(text_embeddings.shape),
            'text_range': [text_embeddings.min().item(), text_embeddings.max().item()],
            'text_mean': text_embeddings.mean().item(),
            'text_std': text_embeddings.std().item(),
            'properly_normalized': torch.allclose(norms, torch.ones_like(norms), atol=1e-5).item()
        }
    
    def run_comprehensive_diagnosis(self, test_image_path: str, test_captions: List[str]) -> Dict:
        """Run all diagnostic tests."""
        logger.info("🚨 STARTING COMPREHENSIVE BLIP3O DIAGNOSIS")
        logger.info("=" * 60)
        
        # Load test image
        test_image = Image.open(test_image_path).convert('RGB')
        
        diagnosis = {
            'test_image_path': test_image_path,
            'test_captions': test_captions,
            'timestamp': time.time()
        }
        
        # Test 1: Patch averaging bug
        logger.info("\n1️⃣ TESTING PATCH AVERAGING BUG")
        patch_results = self.test_patch_averaging_bug(test_image)
        diagnosis['patch_averaging'] = patch_results
        
        # Test 2: BLIP3o pipeline
        logger.info("\n2️⃣ TESTING BLIP3O PIPELINE")
        blip3o_results = self.test_blip3o_pipeline(test_image)
        diagnosis['blip3o_pipeline'] = blip3o_results
        
        # Test 3: Text embeddings
        logger.info("\n3️⃣ TESTING TEXT EMBEDDINGS")
        text_results = self.test_text_embedding_consistency(test_captions)
        diagnosis['text_embeddings'] = text_results
        
        # Overall diagnosis
        logger.info("\n🏁 OVERALL DIAGNOSIS")
        logger.info("=" * 30)
        
        # Patch averaging issue
        if patch_results.get('global_vs_patch_similarity', 1.0) < 0.5:
            logger.error("❌ PATCH AVERAGING: Severe bug detected - this explains 9.8% vs 61% gap")
        else:
            logger.info("✅ PATCH AVERAGING: Working correctly")
        
        # BLIP3o issue
        if 'error' in blip3o_results:
            logger.error(f"❌ BLIP3o LOADING: {blip3o_results['error']}")
        else:
            blip3o_diag = blip3o_results.get('diagnosis', 'UNKNOWN')
            similarity = blip3o_results.get('blip3o_vs_clip_similarity', 0)
            
            if blip3o_diag in ['NUMERICAL_INSTABILITY', 'MODEL_NOT_GENERATING', 'MODEL_COLLAPSED']:
                logger.error(f"❌ BLIP3o GENERATION: Critical issue - {blip3o_diag}")
            elif similarity < 0.1:
                logger.error(f"❌ BLIP3o MAPPING: No similarity to CLIP ({similarity:.6f}) - explains 0% recall")
            else:
                logger.warning(f"⚠️  BLIP3o MAPPING: Poor quality ({similarity:.6f}) but not completely broken")
        
        return diagnosis


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BLIP3o Diagnostic Tool")
    parser.add_argument("--blip3o_model_path", type=str, required=True, help="Path to BLIP3o model")
    parser.add_argument("--test_image", type=str, required=True, help="Path to test image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--save_results", type=str, default="diagnosis_results.json", help="Save results to file")
    
    args = parser.parse_args()
    
    # Test captions
    test_captions = [
        "A person sitting on a bench",
        "A cat sleeping on a couch", 
        "A beautiful sunset over the ocean"
    ]
    
    # Initialize diagnostic tool
    diagnostic = BLIP3oDiagnosticTool(
        device=args.device,
        blip3o_model_path=args.blip3o_model_path
    )
    
    # Run diagnosis
    results = diagnostic.run_comprehensive_diagnosis(args.test_image, test_captions)
    
    # Save results
    with open(args.save_results, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Diagnosis completed. Results saved to {args.save_results}")
    
    # Print summary
    print("\n" + "="*60)
    print("🚨 DIAGNOSIS SUMMARY")
    print("="*60)
    
    patch_sim = results['patch_averaging'].get('global_vs_patch_similarity', 1.0)
    if patch_sim < 0.5:
        print("❌ PATCH AVERAGING BUG: Critical issue found")
        print(f"   Global vs Patch similarity: {patch_sim:.6f}")
        print("   This explains why patch method gives 9.8% instead of ~60%")
    
    if 'error' not in results['blip3o_pipeline']:
        blip3o_sim = results['blip3o_pipeline'].get('blip3o_vs_clip_similarity', 0)
        print(f"\n🤖 BLIP3o vs CLIP similarity: {blip3o_sim:.6f}")
        if blip3o_sim < 0.1:
            print("❌ BLIP3o CRITICAL FAILURE: No similarity to CLIP")
            print("   This explains the 0% recall performance")
            print("   Possible causes:")
            print("   • Model not trained properly")
            print("   • Architecture mismatch")
            print("   • Training data corruption")
            print("   • Loss function not working")
        
    print("="*60)


if __name__ == "__main__":
    main()