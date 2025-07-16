#!/usr/bin/env python3
"""
COCO Embedding Extraction Script for Fast Recall Evaluation

This script extracts and saves embeddings for a subset of COCO images to enable
fast evaluation without recomputing embeddings every time.

Extracts:
- CLIP vision embeddings (with visual projection) 
- EVA-CLIP → BLIP3-o DiT generated embeddings (with visual projection)
- CLIP text embeddings for captions
- Metadata (image IDs, captions, etc.)

Usage:
    python extract_coco_embeddings.py --blip3o_model_path <path> --coco_root <path> [options]
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm
import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.modules.evaluation.evaluator import BLIP3oEvaluator
from src.modules.evaluation.coco_dataset import create_coco_dataloader


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract COCO Embeddings for Fast Recall Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--blip3o_model_path", type=str, required=True,
        help="Path to trained BLIP3-o DiT model directory"
    )
    parser.add_argument(
        "--coco_root", type=str, required=True,
        help="Path to MS-COCO dataset root directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of image-caption pairs to extract"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for processing (smaller for memory efficiency)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/coco_embeddings",
        help="Directory to save extracted embeddings"
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Custom name for output file (default: auto-generated)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--use_single_caption", action="store_true", default=True,
        help="Use only first caption per image (standard evaluation)"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--save_raw_embeddings", action="store_true",
        help="Also save raw embeddings before projection (for debugging)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_paths(args):
    """Validate input paths."""
    # Check BLIP3-o model path
    blip3o_path = Path(args.blip3o_model_path)
    if not blip3o_path.exists():
        raise FileNotFoundError(f"BLIP3-o model path not found: {blip3o_path}")
    
    # Check COCO path
    coco_path = Path(args.coco_root)
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO dataset path not found: {coco_path}")
    
    # Check for COCO structure
    images_dir = coco_path / "images" / "val2017"
    annotations_file = coco_path / "annotations" / "captions_val2017.json"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO images directory not found: {images_dir}")
    
    if not annotations_file.exists():
        raise FileNotFoundError(f"COCO annotations file not found: {annotations_file}")
    
    print(f"✅ Paths validated:")
    print(f"   BLIP3-o model: {blip3o_path}")
    print(f"   COCO dataset: {coco_path}")


def extract_embeddings(args, logger):
    """Main embedding extraction function."""
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Initialize evaluator
    logger.info("Initializing BLIP3-o evaluator...")
    evaluator = BLIP3oEvaluator(
        blip3o_model_path=args.blip3o_model_path,
        device=args.device,
    )
    
    # Create COCO dataloader
    logger.info(f"Creating COCO dataloader for {args.num_samples} samples...")
    dataloader = create_coco_dataloader(
        coco_root=args.coco_root,
        batch_size=args.batch_size,
        max_samples=args.num_samples,
        shuffle=True,  # Shuffle for diverse sample
        num_workers=4,
    )
    
    # Storage for embeddings
    embeddings_data = {
        'clip_vision_embeddings': [],      # CLIP vision + visual projection [N, 768]
        'generated_clip_embeddings': [],   # EVA->BLIP3o + visual projection [N, 768]
        'text_embeddings': [],             # CLIP text embeddings [N, 768]
        'image_ids': [],                   # Image IDs
        'captions': [],                    # Corresponding captions
        'image_paths': [],                 # Image file paths
        'metadata': {
            'extraction_date': str(datetime.datetime.now()),
            'num_samples': 0,
            'model_path': str(args.blip3o_model_path),
            'coco_root': str(args.coco_root),
            'embedding_dim': 768,
            'random_seed': args.random_seed,
            'use_single_caption': args.use_single_caption,
            'device': str(evaluator.device),
        }
    }
    
    # Optional raw embeddings (before projection)
    if args.save_raw_embeddings:
        embeddings_data.update({
            'clip_vision_raw': [],         # CLIP vision before projection [N, 1024]  
            'generated_clip_raw': [],      # Generated before projection [N, 1024]
        })
    
    logger.info("Starting embedding extraction...")
    
    total_processed = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
        images = batch['images']
        captions_batch = batch['captions']
        image_ids = batch['image_ids']
        image_paths = batch['image_paths']
        
        for i, (image, caption_list) in enumerate(zip(images, captions_batch)):
            if total_processed >= args.num_samples:
                break
                
            try:
                # Select caption
                if args.use_single_caption:
                    caption = caption_list[0]
                else:
                    # For now, stick with single caption for consistency
                    caption = caption_list[0]
                
                # Extract CLIP vision embeddings (with visual projection)
                clip_vision_emb = evaluator.extract_clip_vision_embeddings([image])
                clip_vision_final = clip_vision_emb.squeeze(0).cpu()  # [768]
                
                # Extract EVA-CLIP vision embeddings
                eva_vision_emb = evaluator.extract_eva_vision_embeddings([image])
                
                # Generate CLIP embeddings from EVA (with visual projection)
                generated_clip_emb = evaluator.generate_clip_from_eva(eva_vision_emb)
                generated_clip_final = generated_clip_emb.squeeze(0).cpu()  # [768]
                
                # Extract text embeddings
                text_emb = evaluator.extract_clip_text_embeddings([caption])
                text_final = text_emb.squeeze(0).cpu()  # [768]
                
                # Optional: Extract raw embeddings before projection
                if args.save_raw_embeddings:
                    clip_vision_raw = evaluator._extract_clip_vision_raw([image])
                    clip_raw_final = clip_vision_raw.squeeze(0).cpu()  # [1024]
                    
                    generated_clip_raw = evaluator._generate_clip_from_eva_raw(eva_vision_emb)
                    generated_raw_final = generated_clip_raw.squeeze(0).cpu()  # [1024]
                    
                    embeddings_data['clip_vision_raw'].append(clip_raw_final)
                    embeddings_data['generated_clip_raw'].append(generated_raw_final)
                
                # Store embeddings and metadata
                embeddings_data['clip_vision_embeddings'].append(clip_vision_final)
                embeddings_data['generated_clip_embeddings'].append(generated_clip_final)
                embeddings_data['text_embeddings'].append(text_final)
                embeddings_data['image_ids'].append(image_ids[i])
                embeddings_data['captions'].append(caption)
                embeddings_data['image_paths'].append(image_paths[i])
                
                total_processed += 1
                
                # Progress update
                if total_processed % 100 == 0:
                    logger.info(f"Processed {total_processed}/{args.num_samples} samples")
                
            except Exception as e:
                logger.error(f"Error processing image {image_ids[i]}: {e}")
                continue
        
        if total_processed >= args.num_samples:
            break
    
    # Convert lists to tensors
    logger.info("Converting embeddings to tensors...")
    
    embeddings_data['clip_vision_embeddings'] = torch.stack(embeddings_data['clip_vision_embeddings'])
    embeddings_data['generated_clip_embeddings'] = torch.stack(embeddings_data['generated_clip_embeddings'])
    embeddings_data['text_embeddings'] = torch.stack(embeddings_data['text_embeddings'])
    
    if args.save_raw_embeddings:
        embeddings_data['clip_vision_raw'] = torch.stack(embeddings_data['clip_vision_raw'])
        embeddings_data['generated_clip_raw'] = torch.stack(embeddings_data['generated_clip_raw'])
    
    # Update metadata
    embeddings_data['metadata']['num_samples'] = total_processed
    embeddings_data['metadata']['actual_shapes'] = {
        'clip_vision_embeddings': list(embeddings_data['clip_vision_embeddings'].shape),
        'generated_clip_embeddings': list(embeddings_data['generated_clip_embeddings'].shape),
        'text_embeddings': list(embeddings_data['text_embeddings'].shape),
    }
    
    # Validate embeddings
    logger.info("Validating extracted embeddings...")
    
    def validate_normalization(emb_tensor, name):
        norms = torch.norm(emb_tensor, p=2, dim=-1)
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
        logger.info(f"  {name}: mean norm = {mean_norm:.6f}, std = {std_norm:.6f}")
        return abs(mean_norm - 1.0) < 0.01
    
    clip_vision_normalized = validate_normalization(embeddings_data['clip_vision_embeddings'], "CLIP vision")
    generated_normalized = validate_normalization(embeddings_data['generated_clip_embeddings'], "Generated")
    text_normalized = validate_normalization(embeddings_data['text_embeddings'], "Text")
    
    embeddings_data['metadata']['normalization_check'] = {
        'clip_vision_normalized': clip_vision_normalized,
        'generated_normalized': generated_normalized,
        'text_normalized': text_normalized,
    }
    
    return embeddings_data


def save_embeddings(embeddings_data, args, logger):
    """Save extracted embeddings to disk."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    if args.output_name:
        output_name = args.output_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"coco_embeddings_{args.num_samples}samples_{timestamp}"
    
    # Save as pickle (most flexible)
    pickle_path = output_dir / f"{output_name}.pkl"
    logger.info(f"Saving embeddings to {pickle_path}...")
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    # Save metadata as JSON (human readable)
    json_path = output_dir / f"{output_name}_metadata.json"
    with open(json_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        metadata_json = embeddings_data['metadata'].copy()
        json.dump(metadata_json, f, indent=2)
    
    # Save embeddings as NPZ (for easy loading with numpy)
    npz_path = output_dir / f"{output_name}.npz"
    logger.info(f"Saving embeddings to {npz_path}...")
    
    npz_data = {
        'clip_vision_embeddings': embeddings_data['clip_vision_embeddings'].numpy(),
        'generated_clip_embeddings': embeddings_data['generated_clip_embeddings'].numpy(),
        'text_embeddings': embeddings_data['text_embeddings'].numpy(),
        'image_ids': np.array(embeddings_data['image_ids']),
        'captions': np.array(embeddings_data['captions']),
        'image_paths': np.array(embeddings_data['image_paths']),
    }
    
    if 'clip_vision_raw' in embeddings_data:
        npz_data['clip_vision_raw'] = embeddings_data['clip_vision_raw'].numpy()
        npz_data['generated_clip_raw'] = embeddings_data['generated_clip_raw'].numpy()
    
    np.savez_compressed(npz_path, **npz_data)
    
    logger.info("Embedding extraction completed!")
    logger.info(f"Files saved:")
    logger.info(f"  Primary data (pickle): {pickle_path}")
    logger.info(f"  Metadata (JSON): {json_path}")
    logger.info(f"  NumPy arrays (NPZ): {npz_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📦 EMBEDDING EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully extracted {embeddings_data['metadata']['num_samples']} samples")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Primary file: {output_name}.pkl")
    print("")
    print("📊 Embedding Shapes:")
    print(f"   CLIP vision (projected): {embeddings_data['clip_vision_embeddings'].shape}")
    print(f"   Generated (projected):   {embeddings_data['generated_clip_embeddings'].shape}")
    print(f"   Text embeddings:         {embeddings_data['text_embeddings'].shape}")
    
    if 'clip_vision_raw' in embeddings_data:
        print(f"   CLIP vision (raw):       {embeddings_data['clip_vision_raw'].shape}")
        print(f"   Generated (raw):         {embeddings_data['generated_clip_raw'].shape}")
    
    print("")
    print("🔍 Normalization Status:")
    norm_check = embeddings_data['metadata']['normalization_check']
    print(f"   CLIP vision: {'✅' if norm_check['clip_vision_normalized'] else '❌'}")
    print(f"   Generated:   {'✅' if norm_check['generated_normalized'] else '❌'}")
    print(f"   Text:        {'✅' if norm_check['text_normalized'] else '❌'}")
    
    print("")
    print("🚀 Next Steps:")
    print(f"   1. Use fast evaluation: python evaluate_from_embeddings.py --embeddings_file {pickle_path}")
    print("   2. Run multiple evaluation experiments quickly")
    print("   3. Compare different model configurations")
    
    return {
        'pickle_path': pickle_path,
        'npz_path': npz_path,
        'json_path': json_path,
        'num_samples': embeddings_data['metadata']['num_samples'],
    }


def main():
    """Main extraction function."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    print("📦 COCO Embedding Extraction for Fast Evaluation")
    print("=" * 60)
    print(f"🎯 Extracting {args.num_samples} image-caption pairs")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Device: {args.device}")
    print(f"🎲 Random seed: {args.random_seed}")
    print(f"📝 Single caption mode: {args.use_single_caption}")
    print("=" * 60)
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Extract embeddings
        embeddings_data = extract_embeddings(args, logger)
        
        # Save embeddings
        save_info = save_embeddings(embeddings_data, args, logger)
        
        print("✅ Embedding extraction completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Extraction interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)