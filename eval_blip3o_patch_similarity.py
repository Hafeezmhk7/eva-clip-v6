#!/usr/bin/env python3
"""
BLIP3-o Patch-Level Cosine Similarity Evaluation Script
eval_blip3o_patch_similarity.py

Features:
1. Comprehensive patch-level cosine similarity evaluation
2. Support for both CLS+patch (257) and patch-only (256) modes
3. Per-patch, per-image, and global cosine similarity analysis
4. JSON reporting and visualization plots
5. Same-data evaluation for overfitting verification
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging for evaluation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="BLIP3-o Patch-Level Cosine Similarity Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained BLIP3-o model")
    parser.add_argument("--chunked_embeddings_dir", type=str, required=True,
                       help="Path to chunked embeddings directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")
    
    # Evaluation configuration
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Evaluation batch size")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps for generation")
    parser.add_argument("--training_mode", type=str, default="auto",
                       choices=["auto", "cls_patch", "patch_only"],
                       help="Training mode (auto-detect from model if 'auto')")
    
    # Evaluation options
    parser.add_argument("--same_data_eval", action="store_true", default=True,
                       help="Use same training data for evaluation (overfitting test)")
    parser.add_argument("--max_eval_shards", type=int, default=1,
                       help="Maximum number of shards to use for evaluation")
    parser.add_argument("--normalize_embeddings", action="store_true", default=True,
                       help="Normalize embeddings before computing similarity")
    
    # Output options
    parser.add_argument("--save_plots", action="store_true", default=True,
                       help="Save visualization plots")
    parser.add_argument("--save_detailed_results", action="store_true", default=True,
                       help="Save detailed per-image results")
    
    # Hardware configuration
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--torch_dtype", type=str, default="float32",
                       choices=["float32", "float16"],
                       help="Torch data type")
    
    return parser.parse_args()

def load_model_and_determine_mode(model_path, device, torch_dtype, training_mode, logger):
    """Load model and determine training mode"""
    from src.modules.models.blip3o_patch_dit import create_blip3o_patch_dit_model, BLIP3oDiTConfig
    from src.modules.losses.blip3o_flow_matching_loss import create_blip3o_flow_matching_loss
    
    model_path = Path(model_path)
    logger.info(f"📦 Loading model from: {model_path}")
    
    # Load model configuration
    config_files = [
        model_path / "config.json",
        model_path / "blip3o_model_config.json",
        model_path / "enhanced_training_config.json"
    ]
    
    config_data = None
    for config_file in config_files:
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            logger.info(f"✅ Loaded config from: {config_file}")
            break
    
    if config_data is None:
        raise FileNotFoundError(f"No config file found in {model_path}")
    
    # Create model config
    config = BLIP3oDiTConfig(**config_data)
    
    # Determine training mode
    if training_mode == "auto":
        if hasattr(config, 'training_mode'):
            training_mode = config.training_mode
        elif hasattr(config, 'num_tokens'):
            training_mode = "cls_patch" if config.num_tokens == 257 else "patch_only"
        else:
            training_mode = "cls_patch"  # Default
        logger.info(f"🎯 Auto-detected training mode: {training_mode}")
    
    # Create model
    model = create_blip3o_patch_dit_model(config=config)
    
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
    
    if weight_file is None:
        raise FileNotFoundError(f"No model weights found in {model_path}")
    
    logger.info(f"💾 Loading weights from: {weight_file}")
    
    # Load state dict
    if weight_file.suffix == ".bin":
        state_dict = torch.load(weight_file, map_location='cpu')
    else:
        from safetensors.torch import load_file
        state_dict = load_file(str(weight_file))
    
    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
    
    # Move to device and set dtype
    dtype = torch.float16 if torch_dtype == "float16" else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Expected tokens: {config.num_tokens if hasattr(config, 'num_tokens') else '257 or 256'}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Dtype: {dtype}")
    
    return model, config, training_mode

def create_evaluation_dataloader(embeddings_dir, training_mode, max_shards, batch_size, logger):
    """Create dataloader for evaluation"""
    from src.modules.datasets.blip3o_dataset import create_flexible_dataloaders
    
    logger.info(f"📊 Creating evaluation dataloader")
    logger.info(f"   Training mode: {training_mode}")
    logger.info(f"   Max shards: {max_shards}")
    logger.info(f"   Batch size: {batch_size}")
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_flexible_dataloaders(
        chunked_embeddings_dir=embeddings_dir,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        eval_split_ratio=0.0,  # Use all data for evaluation
        normalize_embeddings=False,  # We'll handle normalization in evaluation
        training_mode=training_mode,
        max_shards=max_shards,
        use_same_data_for_eval=True,  # Use same data
        delete_after_use=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=torch.cuda.is_available(),
    )
    
    # Use train_dataloader for same-data evaluation
    eval_dataloader = train_dataloader
    
    logger.info(f"✅ Evaluation dataloader created")
    
    return eval_dataloader

def run_comprehensive_evaluation(model, dataloader, training_mode, args, logger):
    """Run comprehensive patch-level cosine similarity evaluation"""
    from src.modules.evaluation.blip3o_detailed_evaluator import create_detailed_evaluator
    
    logger.info("🔍 Starting comprehensive evaluation...")
    
    # Create detailed evaluator
    evaluator = create_detailed_evaluator(
        model=model,
        training_mode=training_mode,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        normalize_embeddings=args.normalize_embeddings,
    )
    
    # Run detailed evaluation
    results = evaluator.evaluate_detailed_cosine_similarity(
        dataloader=dataloader,
        max_batches=args.num_samples // args.batch_size,
        save_dir=args.output_dir,
        save_plots=args.save_plots,
        plot_distribution=True,
        plot_per_image=True,
        plot_heatmap=True,
        same_data_eval=args.same_data_eval,
    )
    
    return results

def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("🔍 BLIP3-o Patch-Level Cosine Similarity Evaluation")
    logger.info("=" * 60)
    logger.info("🎯 FEATURES:")
    logger.info(f"  ✅ Model path: {args.model_path}")
    logger.info(f"  ✅ Training mode: {args.training_mode}")
    logger.info(f"  ✅ Same-data evaluation: {args.same_data_eval}")
    logger.info(f"  ✅ Max eval shards: {args.max_eval_shards}")
    logger.info(f"  ✅ Number of samples: {args.num_samples}")
    logger.info(f"  ✅ Patch-level cosine similarity analysis")
    logger.info(f"  ✅ Comprehensive visualization and JSON reporting")
    logger.info("=" * 60)
    
    try:
        # 1. Setup device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        
        logger.info(f"🎮 Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name()}")
        
        # 2. Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        # 3. Load model and determine mode
        model, config, training_mode = load_model_and_determine_mode(
            args.model_path, device, args.torch_dtype, args.training_mode, logger
        )
        
        # 4. Create evaluation dataloader
        eval_dataloader = create_evaluation_dataloader(
            args.chunked_embeddings_dir, training_mode, args.max_eval_shards, 
            args.batch_size, logger
        )
        
        # 5. Run comprehensive evaluation
        logger.info("🚀 Starting evaluation...")
        start_time = datetime.now()
        
        results = run_comprehensive_evaluation(
            model, eval_dataloader, training_mode, args, logger
        )
        
        end_time = datetime.now()
        evaluation_duration = (end_time - start_time).total_seconds()
        
        # 6. Process and display results
        logger.info("📊 EVALUATION RESULTS:")
        logger.info("=" * 40)
        
        if results:
            # Key metrics
            global_stats = results['global_patch_statistics']
            image_stats = results['per_image_statistics']
            quality = results['quality_assessment']
            
            logger.info(f"🎯 COSINE SIMILARITY ANALYSIS:")
            logger.info(f"   Global patch mean: {global_stats['mean_cosine_similarity']:.4f}")
            logger.info(f"   Per-image mean: {image_stats['mean_cosine_similarity']:.4f}")
            logger.info(f"   Quality level: {quality['quality_level']}")
            logger.info(f"   Quality message: {quality['quality_message']}")
            
            # Mode-specific results
            if 'mode_specific_analysis' in results and results['mode_specific_analysis']:
                mode_analysis = results['mode_specific_analysis']
                if 'cls_token_stats' in mode_analysis:
                    logger.info(f"   CLS token mean: {mode_analysis['cls_token_stats']['mean']:.4f}")
                if 'patch_only_stats' in mode_analysis:
                    logger.info(f"   Patch-only mean: {mode_analysis['patch_only_stats']['mean']:.4f}")
            
            # Quality thresholds
            quality_metrics = results['quality_metrics']
            logger.info(f"   High quality images (>0.7): {quality_metrics['high_quality_images_ratio']*100:.1f}%")
            logger.info(f"   Very high quality images (>0.8): {quality_metrics['very_high_quality_images_ratio']*100:.1f}%")
            logger.info(f"   Excellent quality images (>0.9): {quality_metrics['excellent_quality_images_ratio']*100:.1f}%")
            
            # Overfitting assessment
            if args.same_data_eval:
                overfitting_score = image_stats['mean_cosine_similarity']
                if overfitting_score > 0.8:
                    logger.info("🎉 EXCELLENT OVERFITTING: Model successfully learned the training data!")
                elif overfitting_score > 0.6:
                    logger.info("✅ GOOD OVERFITTING: Strong performance on training data")
                elif overfitting_score > 0.4:
                    logger.info("🔄 MODERATE OVERFITTING: Some learning detected")
                else:
                    logger.info("⚠️ LOW OVERFITTING: Model may need more training")
        
        # 7. Save evaluation summary
        evaluation_summary = {
            'evaluation_completed': True,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': evaluation_duration,
            'model_path': str(args.model_path),
            'embeddings_dir': args.chunked_embeddings_dir,
            'training_mode': training_mode,
            'evaluation_config': {
                'num_samples': args.num_samples,
                'batch_size': args.batch_size,
                'num_inference_steps': args.num_inference_steps,
                'same_data_eval': args.same_data_eval,
                'max_eval_shards': args.max_eval_shards,
                'normalize_embeddings': args.normalize_embeddings,
            },
            'results_summary': {
                'global_mean_similarity': results['global_patch_statistics']['mean_cosine_similarity'] if results else 0,
                'per_image_mean_similarity': results['per_image_statistics']['mean_cosine_similarity'] if results else 0,
                'quality_level': results['quality_assessment']['quality_level'] if results else 'unknown',
                'total_images': results['total_images'] if results else 0,
                'total_patches': results['total_patches'] if results else 0,
            },
            'files_generated': {
                'detailed_results': 'detailed_evaluation_*.json',
                'summary_results': 'evaluation_summary_*.json',
                'plots': 'various plot files if save_plots=True',
            }
        }
        
        with open(output_dir / 'patch_similarity_evaluation_summary.json', 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        logger.info("=" * 40)
        logger.info("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"⏱️ Evaluation time: {evaluation_duration:.1f} seconds")
        
        # 8. Print recommendations
        if results and 'quality_assessment' in results:
            recommendations = results['quality_assessment'].get('recommendations', [])
            if recommendations:
                logger.info("💡 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"   {i}. {rec}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        traceback.print_exc()
        
        # Save error info
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'evaluation_args': vars(args),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open('patch_similarity_evaluation_error.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        logger.error("💾 Error info saved to patch_similarity_evaluation_error.json")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)