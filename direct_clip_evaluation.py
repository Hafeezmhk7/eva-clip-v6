#!/usr/bin/env python3
"""
Simple CLIP Evaluation on COCO

Direct CLIP testing with clear output and recall@1,5,10 metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def load_coco_samples(coco_root, num_samples):
    """Load COCO samples with first caption per image."""
    print(f"📋 Loading COCO data from: {coco_root}")
    
    annotations_file = Path(coco_root) / "annotations" / "captions_val2017.json"
    images_dir = Path(coco_root) / "images" / "val2017"
    
    print(f"   Annotations: {annotations_file}")
    print(f"   Images: {images_dir}")
    
    # Check paths exist
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"   Total annotations: {len(coco_data['annotations'])}")
    
    # Group by image_id and take first caption
    image_to_captions = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_captions:
            image_to_captions[image_id] = ann['caption']  # First caption only
    
    print(f"   Unique images: {len(image_to_captions)}")
    
    # Take first N images
    image_ids = list(image_to_captions.keys())[:num_samples]
    
    # Create samples list
    samples = []
    for image_id in image_ids:
        image_path = images_dir / f"{image_id:012d}.jpg"
        if image_path.exists():
            samples.append({
                'image_id': image_id,
                'image_path': str(image_path),
                'caption': image_to_captions[image_id]
            })
    
    print(f"✅ Loaded {len(samples)} valid samples")
    return samples

def evaluate_clip(samples, device="cuda", batch_size=32):
    """Evaluate CLIP on COCO samples."""
    print(f"\n🔧 Setting up CLIP evaluation...")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Samples: {len(samples)}")
    
    # Load CLIP
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device)
    model.eval()
    
    print(f"✅ CLIP model loaded")
    
    # Extract embeddings
    all_image_embeddings = []
    all_text_embeddings = []
    
    print(f"\n⚡ Extracting embeddings...")
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Processing"):
        batch_samples = samples[i:i+batch_size]
        
        # Load images and captions
        images = []
        captions = []
        
        for sample in batch_samples:
            try:
                image = Image.open(sample['image_path']).convert('RGB')
                images.append(image)
                captions.append(sample['caption'])
            except Exception as e:
                print(f"   ⚠️  Failed to load image {sample['image_id']}: {e}")
                continue
        
        if not images:
            continue
        
        # Process with CLIP
        with torch.no_grad():
            inputs = processor(
                text=captions, 
                images=images, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            img_emb = outputs.image_embeds.cpu()  # Already normalized
            txt_emb = outputs.text_embeds.cpu()   # Already normalized
            
            all_image_embeddings.append(img_emb)
            all_text_embeddings.append(txt_emb)
    
    # Concatenate all embeddings
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    print(f"✅ Extracted embeddings: {image_embeddings.shape}")
    
    return image_embeddings, text_embeddings

def compute_recall(image_embeddings, text_embeddings):
    """Compute recall@1,5,10."""
    print(f"\n📊 Computing recall metrics...")
    
    # Compute similarity matrix
    sim_matrix = torch.mm(image_embeddings, text_embeddings.t())
    
    num_samples = sim_matrix.shape[0]
    print(f"   Similarity matrix: {sim_matrix.shape}")
    print(f"   Similarity range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
    
    # Diagonal (correct pairs) vs off-diagonal (incorrect pairs)
    diagonal = sim_matrix.diagonal()
    off_diagonal = sim_matrix.flatten()[~torch.eye(num_samples, dtype=bool).flatten()]
    
    similarity_gap = diagonal.mean() - off_diagonal.mean()
    
    print(f"   Correct pairs mean: {diagonal.mean():.4f}")
    print(f"   Incorrect pairs mean: {off_diagonal.mean():.4f}")
    print(f"   Similarity gap: {similarity_gap:.4f}")
    
    # Compute recall@k
    recalls = {}
    
    for k in [1, 5, 10]:
        # Get top-k indices for each image
        _, top_k = torch.topk(sim_matrix, k=k, dim=1)
        
        # Check if correct text is in top-k
        correct_in_topk = (top_k == torch.arange(num_samples).unsqueeze(1)).any(dim=1)
        recall_k = correct_in_topk.float().mean().item()
        
        recalls[f'R@{k}'] = recall_k
        print(f"   Recall@{k}: {recall_k:.4f} ({recall_k*100:.2f}%)")
    
    return recalls, sim_matrix, similarity_gap.item()

def main():
    args = parse_args()
    
    print("🚀 SIMPLE CLIP EVALUATION ON COCO")
    print("=" * 50)
    print(f"📁 COCO root: {args.coco_root}")
    print(f"📊 Samples: {args.num_samples}")
    print(f"🔧 Device: {args.device}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Load COCO data
        samples = load_coco_samples(args.coco_root, args.num_samples)
        
        # Evaluate CLIP
        image_embeddings, text_embeddings = evaluate_clip(
            samples, args.device, args.batch_size
        )
        
        # Compute recall
        recalls, sim_matrix, similarity_gap = compute_recall(image_embeddings, text_embeddings)
        
        # Final results - PROMINENT DISPLAY FOR JOB OUTPUT
        print(f"\n" + "=" * 70)
        print("🎯 FINAL RESULTS - CLIP EVALUATION ON COCO")
        print("=" * 70)
        print(f"📊 Samples evaluated: {len(samples)}")
        print(f"⏱️  Processing time: {time.time() - start_time:.1f} seconds")
        print(f"📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Print results clearly - LARGE AND VISIBLE
        print("🎯 RECALL METRICS (Image-to-Text Retrieval):")
        print("-" * 50)
        for metric, value in recalls.items():
            print(f"   {metric}: {value*100:.2f}%")
        print("-" * 50)
        
        # Validation check with PROMINENT STATUS
        r1 = recalls['R@1']
        r5 = recalls['R@5'] 
        r10 = recalls['R@10']
        
        print()
        if r1 >= 0.25:
            print("🎉 ✅ SUCCESS: CLIP BASELINE VALIDATED!")
            print(f"   R@1 = {r1*100:.1f}% (Expected: 25-35%)")
            print("   CLIP is working correctly on COCO!")
            status = "SUCCESS"
        elif r1 >= 0.15:
            print("⚠️  MODERATE PERFORMANCE:")
            print(f"   R@1 = {r1*100:.1f}% (Expected: 25-35%)")
            print("   Results are lower than expected but reasonable")
            status = "MODERATE"
        else:
            print("❌ LOW PERFORMANCE:")
            print(f"   R@1 = {r1*100:.1f}% (Expected: 25-35%)")
            print("   Results are significantly lower than expected")
            status = "LOW"
        
        # Additional metrics summary  
        print()
        print("📈 Performance Summary:")
        print(f"   • Image-Text Similarity Gap: {similarity_gap:.4f}")
        print(f"   • R@1 Performance: {status}")
        print(f"   • R@5 Performance: {r5*100:.1f}%")
        print(f"   • R@10 Performance: {r10*100:.1f}%")
        
        # Save results
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': vars(args),
            'recalls': recalls,
            'num_samples': len(samples),
            'processing_time': time.time() - start_time,
            'status': status,
            'similarity_gap': similarity_gap
        }
        
        results_file = f"clip_evaluation_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Final summary for job logs
        print("\n" + "=" * 70)
        print("🏁 EVALUATION COMPLETE - CHECK RESULTS ABOVE")
        print("=" * 70)
        print(f"STATUS: {status}")
        print(f"R@1: {r1*100:.2f}% | R@5: {r5*100:.2f}% | R@10: {r10*100:.2f}%")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())