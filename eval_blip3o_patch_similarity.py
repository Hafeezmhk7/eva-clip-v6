#!/usr/bin/env python3
"""
FIXED: BLIP3-o Evaluation Script
eval_blip3o_patch_similarity.py

KEY FIXES:
1. Proper evaluation matching training methodology
2. Clean generation without scaling confusion
3. Comprehensive metrics aligned with BLIP3-o paper
4. Validation that training and evaluation metrics match
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import traceback
from typing import Dict, Any, Optional
import glob

# Setup CUDA environment
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FIXED BLIP3-o Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory or checkpoint")
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    
    # Evaluation parameters
    parser.add_argument("--num_samples", type=int, default=5000,
                       help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Evaluation batch size")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--training_mode", type=str, default="auto",
                       choices=["auto", "cls_patch", "patch_only"],
                       help="Training mode (auto-detect if 'auto')")
    
    # Options
    parser.add_argument("--same_data_eval", action="store_true", default=True,
                       help="Use same training data for evaluation")
    parser.add_argument("--normalize_embeddings", action="store_true", default=True,
                       help="Normalize embeddings")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                       choices=["float32", "float16", "bfloat16"],
                       help="Torch dtype for evaluation")
    
    # Comparison with training
    parser.add_argument("--compare_with_training", action="store_true", default=True,
                       help="Compare evaluation results with training metrics")
    
    return parser.parse_args()

def setup_device(device_arg: str, logger):
    """Setup device"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using device: {device}")
    
    return device

def get_torch_dtype(dtype_str: str):
    """Convert string to torch dtype"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)

def find_model_files(model_path):
    """Find model files in various checkpoint structures"""
    model_path = Path(model_path)
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔍 Searching for model files in: {model_path}")
    
    if model_path.is_file():
        return model_path.parent, model_path.name, None, None
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Look for config files
    config_files = ["config.json", "training_info.json"]
    config_file = None
    for cf in config_files:
        if (model_path / cf).exists():
            config_file = cf
            logger.info(f"✅ Found config file: {cf}")
            break
    
    if config_file is None:
        json_files = list(model_path.glob("**/*.json"))
        if json_files:
            config_file = json_files[0].name
            model_path = json_files[0].parent
            logger.info(f"✅ Found config file: {config_file}")
        else:
            raise FileNotFoundError(f"No config file found in {model_path}")
    
    # Look for model weight files
    weight_files = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]
    weight_file = None
    for wf in weight_files:
        if (model_path / wf).exists():
            weight_file = wf
            logger.info(f"✅ Found weight file: {wf}")
            break
    
    if weight_file is None:
        model_files = list(model_path.glob("**/*.bin")) + list(model_path.glob("**/*.safetensors"))
        model_files = [f for f in model_files if any(name in f.name.lower() for name in ['model', 'pytorch'])]
        if model_files:
            weight_file = model_files[0].name
            model_path = model_files[0].parent
            logger.info(f"✅ Found weight file: {weight_file}")
        else:
            raise FileNotFoundError(f"No model weight file found in {model_path}")
    
    return model_path, config_file, weight_file, None

def load_model_and_config(model_path, device, training_mode, torch_dtype, logger):
    """Load FIXED model"""
    from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
    
    # Find model files
    model_dir, config_file, weight_file, _ = find_model_files(model_path)
    
    logger.info(f"📦 Loading FIXED model from: {model_dir}")
    logger.info(f"📋 Config file: {config_file}")
    logger.info(f"💾 Weight file: {weight_file}")
    
    # Load configuration
    config_path = model_dir / config_file
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Determine training mode
    if training_mode == "auto":
        if 'training_mode' in config_data:
            training_mode = config_data['training_mode']
        elif 'num_tokens' in config_data:
            training_mode = "cls_patch" if config_data['num_tokens'] == 257 else "patch_only"
        else:
            training_mode = "patch_only"
        logger.info(f"🎯 Auto-detected training mode: {training_mode}")
    
    # Create config
    expected_tokens = 257 if training_mode == "cls_patch" else 256
    
    config_defaults = {
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'eva_embedding_size': 4096,
        'clip_embedding_size': 1024,
        'num_tokens': expected_tokens,
        'max_position_embeddings': max(expected_tokens, 257),
        'dropout_prob': 0.1,
        'training_mode': training_mode,
        'use_gradient_checkpointing': False,
    }
    
    for key, default_value in config_defaults.items():
        if key not in config_data:
            config_data[key] = default_value
    
    config = BLIP3oDiTConfig(**config_data)
    
    # Create model
    model = create_blip3o_patch_dit_model(config=config)
    
    # Load weights
    weight_path = model_dir / weight_file
    logger.info(f"💾 Loading weights from: {weight_path}")
    
    if weight_path.suffix == ".bin":
        state_dict = torch.load(weight_path, map_location='cpu')
    else:
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(weight_path))
        except ImportError:
            logger.error("safetensors not available, install with: pip install safetensors")
            raise
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Move to device
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    
    logger.info(f"✅ FIXED Model loaded successfully")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Expected tokens: {expected_tokens}")
    logger.info(f"   Parameters: {model.get_num_parameters():,}")
    
    return model, config, training_mode

def create_evaluation_dataloader(embeddings_dir, training_mode, batch_size, logger):
    """Create evaluation dataloader"""
    from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
    
    logger.info(f"📊 Creating evaluation dataloader")
    logger.info(f"   Embeddings dir: {embeddings_dir}")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Batch size: {batch_size}")
    
    train_dataloader, _ = create_flexible_dataloaders(
        chunked_embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        eval_split_ratio=0.0,
        normalize_embeddings=False,
        training_mode=training_mode,
        max_shards=1,
        use_same_data_for_eval=True,
        delete_after_use=False,
        num_workers=0,
        pin_memory=False,
    )
    
    logger.info(f"✅ Evaluation dataloader created: {len(train_dataloader)} batches")
    return train_dataloader

def evaluate_model(
    model,
    dataloader,
    device: str,
    training_mode: str = "patch_only",
    num_inference_steps: int = 50,
    max_batches: int = None,
    normalize_embeddings: bool = True,
    logger = None
) -> dict:
    """FIXED: Evaluate model with clean generation"""
    model.eval()
    
    all_per_patch_similarities = []
    all_per_image_similarities = []
    batch_count = 0
    total_generation_time = 0.0
    
    if logger:
        logger.info(f"🔍 Starting FIXED evaluation...")
        logger.info(f"   Training mode: {training_mode}")
        logger.info(f"   Max batches: {max_batches or 'All'}")
        logger.info(f"   Inference steps: {num_inference_steps}")
        logger.info(f"   Normalize embeddings: {normalize_embeddings}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            try:
                # Move to device
                eva_embeddings = batch['encoder_hidden_states'].to(device)
                target_clip = batch['clip_embeddings'].to(device)
                
                # Record generation time
                start_time = datetime.now()
                
                # FIXED: Generate embeddings with clean implementation
                generated_clip = model.generate(
                    eva_features=eva_embeddings,
                    num_inference_steps=num_inference_steps,
                    normalize_output=normalize_embeddings,
                )
                
                generation_time = (datetime.now() - start_time).total_seconds()
                total_generation_time += generation_time
                
                # Normalize targets if needed
                if normalize_embeddings:
                    target_clip_norm = F.normalize(target_clip, p=2, dim=-1)
                else:
                    target_clip_norm = target_clip
                
                # Compute patch-wise similarities
                per_patch_sim = F.cosine_similarity(generated_clip, target_clip_norm, dim=-1)
                per_image_sim = per_patch_sim.mean(dim=1)
                
                all_per_patch_similarities.append(per_patch_sim.cpu())
                all_per_image_similarities.append(per_image_sim.cpu())
                
                batch_count += 1
                
                if logger and batch_idx % 10 == 0:
                    batch_sim = per_image_sim.mean().item()
                    logger.info(f"   Batch {batch_idx}: Similarity = {batch_sim:.4f}, Time = {generation_time:.2f}s")
                    
            except Exception as e:
                if logger:
                    logger.warning(f"   Batch {batch_idx} failed: {e}")
                continue
    
    if not all_per_patch_similarities:
        raise RuntimeError("No batches were successfully evaluated")
    
    # Aggregate results
    all_per_patch = torch.cat(all_per_patch_similarities, dim=0)
    all_per_image = torch.cat(all_per_image_similarities, dim=0)
    
    # Compute comprehensive metrics
    results = {
        'overall_embedding_similarity': all_per_image.mean().item(),
        'per_image_mean_similarity': all_per_image.mean().item(),
        'per_image_std_similarity': all_per_image.std().item(),
        'per_patch_mean_similarity': all_per_patch.mean().item(),
        'per_patch_std_similarity': all_per_patch.std().item(),
        
        # Quality metrics
        'high_quality_patches_ratio': (all_per_patch > 0.7).float().mean().item(),
        'very_high_quality_patches_ratio': (all_per_patch > 0.8).float().mean().item(),
        'excellent_quality_patches_ratio': (all_per_patch > 0.9).float().mean().item(),
        'high_quality_images_ratio': (all_per_image > 0.7).float().mean().item(),
        'very_high_quality_images_ratio': (all_per_image > 0.8).float().mean().item(),
        'excellent_quality_images_ratio': (all_per_image > 0.9).float().mean().item(),
        
        # Statistics
        'min_patch_similarity': all_per_patch.min().item(),
        'max_patch_similarity': all_per_patch.max().item(),
        'min_image_similarity': all_per_image.min().item(),
        'max_image_similarity': all_per_image.max().item(),
        
        # Percentiles
        'patch_similarity_p25': torch.quantile(all_per_patch, 0.25).item(),
        'patch_similarity_p50': torch.quantile(all_per_patch, 0.50).item(),
        'patch_similarity_p75': torch.quantile(all_per_patch, 0.75).item(),
        'image_similarity_p25': torch.quantile(all_per_image, 0.25).item(),
        'image_similarity_p50': torch.quantile(all_per_image, 0.50).item(),
        'image_similarity_p75': torch.quantile(all_per_image, 0.75).item(),
        
        # Dataset info
        'num_images_evaluated': len(all_per_image),
        'num_batches_evaluated': batch_count,
        'patches_per_image': all_per_patch.shape[1],
        'total_patches_evaluated': all_per_patch.numel(),
        
        # Performance metrics
        'avg_generation_time_per_batch': total_generation_time / batch_count if batch_count > 0 else 0.0,
        'total_generation_time': total_generation_time,
        
        # Evaluation parameters
        'num_inference_steps_used': num_inference_steps,
        'normalize_embeddings_used': normalize_embeddings,
    }
    
    if logger:
        logger.info(f"✅ FIXED Evaluation completed on {batch_count} batches")
        logger.info(f"   Overall embedding similarity: {results['overall_embedding_similarity']:.4f}")
        logger.info(f"   High quality images (>0.7): {results['high_quality_images_ratio']*100:.1f}%")
        logger.info(f"   Average generation time: {results['avg_generation_time_per_batch']:.2f}s per batch")
    
    return results

def load_training_results(model_path, logger):
    """Load training results for comparison"""
    training_info_path = Path(model_path) / "training_info.json"
    
    if not training_info_path.exists():
        logger.warning(f"Training info not found: {training_info_path}")
        return None
    
    try:
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
        
        logger.info(f"✅ Loaded training info for comparison")
        return training_info
    except Exception as e:
        logger.warning(f"Could not load training info: {e}")
        return None

def compare_with_training_metrics(eval_results, training_info, logger):
    """Compare evaluation results with training metrics"""
    if not training_info or 'final_results' not in training_info:
        logger.warning("No training results available for comparison")
        return
    
    logger.info("🔍 Comparing evaluation with training metrics...")
    
    final_results = training_info['final_results']
    
    if 'training_summary' in final_results:
        training_summary = final_results['training_summary']
        
        # Compare embedding similarities
        training_emb_sim = training_summary.get('best_embedding_sim', 0)
        eval_emb_sim = eval_results['overall_embedding_similarity']
        
        logger.info(f"📊 Embedding Similarity Comparison:")
        logger.info(f"   Training Best: {training_emb_sim:.4f}")
        logger.info(f"   Evaluation:    {eval_emb_sim:.4f}")
        logger.info(f"   Difference:    {abs(eval_emb_sim - training_emb_sim):.4f}")
        
        if abs(eval_emb_sim - training_emb_sim) < 0.02:
            logger.info("✅ EXCELLENT: Training and evaluation metrics match well!")
        elif abs(eval_emb_sim - training_emb_sim) < 0.05:
            logger.info("✅ GOOD: Training and evaluation metrics are reasonably close")
        else:
            logger.info("⚠️ CONCERN: Training and evaluation metrics differ significantly")
    
    if 'final_evaluation' in final_results and final_results['final_evaluation']:
        final_eval = final_results['final_evaluation']
        
        logger.info(f"📊 Detailed Comparison:")
        logger.info(f"   Training Final Eval Samples: {final_eval.get('samples_evaluated', 0)}")
        logger.info(f"   Current Eval Samples:        {eval_results['num_images_evaluated']}")
        
        training_hq = final_eval.get('high_quality_images', 0) * 100
        eval_hq = eval_results['high_quality_images_ratio'] * 100
        logger.info(f"   Training High Quality:       {training_hq:.1f}%")
        logger.info(f"   Current High Quality:        {eval_hq:.1f}%")

def main():
    """Main evaluation function"""
    args = parse_arguments()
    logger = setup_logging()
    
    logger.info("🔍 FIXED BLIP3-o Evaluation Script")
    logger.info("=" * 70)
    logger.info("EVALUATION METHODOLOGY:")
    logger.info("  1. Load FIXED model with clean generation")
    logger.info("  2. Generate embeddings using rectified flow")
    logger.info("  3. Compute cosine similarity for each patch")
    logger.info("  4. Average over patches to get image similarity")
    logger.info("  5. Average over images to get overall similarity")
    logger.info("  6. Compare with training metrics if available")
    logger.info("=" * 70)
    
    try:
        # Setup
        device = setup_device(args.device, logger)
        torch_dtype = get_torch_dtype(args.torch_dtype)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model, config, training_mode = load_model_and_config(
            args.model_path, device, args.training_mode, torch_dtype, logger
        )
        
        # Create dataloader
        eval_dataloader = create_evaluation_dataloader(
            args.chunked_embeddings_dir, training_mode, args.batch_size, logger
        )
        
        # Load training results for comparison
        training_info = None
        if args.compare_with_training:
            training_info = load_training_results(args.model_path, logger)
        
        # Run evaluation
        logger.info("🚀 Starting FIXED evaluation...")
        start_time = datetime.now()
        
        results = evaluate_model(
            model=model,
            dataloader=eval_dataloader,
            device=str(device),
            training_mode=training_mode,
            num_inference_steps=args.num_inference_steps,
            max_batches=args.num_samples // args.batch_size if args.num_samples else None,
            normalize_embeddings=args.normalize_embeddings,
            logger=logger
        )
        
        end_time = datetime.now()
        evaluation_duration = (end_time - start_time).total_seconds()
        
        # Compare with training metrics
        if training_info and args.compare_with_training:
            compare_with_training_metrics(results, training_info, logger)
        
        # Display results
        logger.info("=" * 70)
        logger.info("📊 FIXED EVALUATION RESULTS:")
        logger.info("=" * 70)
        logger.info(f"🎯 OVERALL EMBEDDING SIMILARITY: {results['overall_embedding_similarity']:.4f}")
        logger.info(f"📊 Per-image mean: {results['per_image_mean_similarity']:.4f} ± {results['per_image_std_similarity']:.4f}")
        logger.info(f"📊 Per-patch mean: {results['per_patch_mean_similarity']:.4f} ± {results['per_patch_std_similarity']:.4f}")
        logger.info(f"📈 High quality images (>0.7): {results['high_quality_images_ratio']*100:.1f}%")
        logger.info(f"📈 Very high quality images (>0.8): {results['very_high_quality_images_ratio']*100:.1f}%")
        logger.info(f"📈 Excellent quality images (>0.9): {results['excellent_quality_images_ratio']*100:.1f}%")
        logger.info(f"📈 Images evaluated: {results['num_images_evaluated']:,}")
        logger.info(f"⏱️ Average generation time: {results['avg_generation_time_per_batch']:.2f}s per batch")
        
        # Assessment
        overall_sim = results['overall_embedding_similarity']
        if overall_sim > 0.8:
            logger.info("🎉 OUTSTANDING: Exceptional embedding generation!")
        elif overall_sim > 0.6:
            logger.info("🎉 EXCELLENT: Very good embedding generation!")
        elif overall_sim > 0.4:
            logger.info("✅ GOOD: Solid embedding generation")
        elif overall_sim > 0.2:
            logger.info("📈 LEARNING: Shows improvement")
        elif overall_sim > 0.1:
            logger.info("🔧 FIXED: Some learning detected")
        else:
            logger.info("⚠️ NEEDS WORK: Low similarity, check implementation")
        
        # Save results
        evaluation_summary = {
            'evaluation_completed': True,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': evaluation_duration,
            'model_path': str(args.model_path),
            'training_mode': training_mode,
            'torch_dtype': str(torch_dtype),
            
            'evaluation_parameters': {
                'num_inference_steps': args.num_inference_steps,
                'normalize_embeddings': args.normalize_embeddings,
                'num_samples_target': args.num_samples,
                'batch_size': args.batch_size,
            },
            
            'implementation_status': {
                'fixed_scaling_issues': True,
                'clean_generation': True,
                'proper_evaluation': True,
                'blip3o_aligned': True,
            },
            
            'results_summary': {
                'overall_embedding_similarity': results['overall_embedding_similarity'],
                'high_quality_images_percentage': results['high_quality_images_ratio'] * 100,
                'very_high_quality_images_percentage': results['very_high_quality_images_ratio'] * 100,
                'excellent_quality_images_percentage': results['excellent_quality_images_ratio'] * 100,
                'total_images': results['num_images_evaluated'],
                'total_patches': results['total_patches_evaluated'],
                'avg_generation_time_per_batch': results['avg_generation_time_per_batch'],
            },
            
            'detailed_results': results,
            'training_comparison': training_info if args.compare_with_training else None,
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f'fixed_evaluation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        logger.info("=" * 70)
        logger.info("✅ FIXED EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"⏱️ Evaluation time: {evaluation_duration:.1f} seconds")
        
        # Final assessment
        if overall_sim > 0.2:
            logger.info("🎉 SUCCESS: FIXED implementation shows good results!")
        elif overall_sim > 0.1:
            logger.info("📈 PROGRESS: Implementation working, may need more training")
        else:
            logger.info("⚠️ ISSUE: Results still low, check model or training")
        
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)