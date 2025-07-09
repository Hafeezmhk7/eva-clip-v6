"""
Production Grid-Based Embedding Extraction for BLIP3-o
Place this file as: src/modules/extract_embeddings_production.py

Clean production version without debugging code.
Extracts grid features from EVA-CLIP-8B and CLIP ViT-L/14 for BLIP3-o training.
"""

import sys
import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoModel, CLIPImageProcessor
import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path

def setup_paths():
    """Setup paths for project structure"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Add import paths
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "src" / "data_hand"))
    
    return project_root

def load_models(device):
    """Load CLIP and EVA-CLIP models"""
    print("📦 Loading models...")
    
    # Load CLIP ViT-L/14
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16
    ).to(device)
    clip_model.eval()
    
    # Load EVA-CLIP-8B
    eva_model = AutoModel.from_pretrained(
        "BAAI/EVA-CLIP-8B", 
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    eva_model.eval()
    
    print("✅ Models loaded successfully")
    return clip_processor, clip_model, eva_processor, eva_model

def extract_clip_features(images, processor, model, device):
    """Extract CLIP ViT-L/14 patch grid features"""
    features = []
    
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device).half() if v.dtype == torch.float32 else v.to(device) 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get patch embeddings (remove CLS token)
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
            batch_size, num_patches, hidden_dim = patch_embeddings.shape
            
            # Reshape to spatial grid: [1, 16, 16, hidden_dim]
            grid_size = int(np.sqrt(num_patches))
            spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
            features.append(spatial_grid.squeeze().cpu().float())
    
    return torch.stack(features)

def extract_eva_features(images, processor, model, device):
    """Extract EVA-CLIP-8B patch grid features"""
    features = []
    
    for img in images:
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device).half()
        
        with torch.no_grad():
            vision_outputs = model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get patch embeddings (remove CLS token)
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
            batch_size, num_patches, hidden_dim = patch_embeddings.shape
            
            # Reshape to spatial grid: [1, 16, 16, hidden_dim]
            grid_size = int(np.sqrt(num_patches))
            spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
            features.append(spatial_grid.squeeze().cpu().float())
    
    return torch.stack(features)

def pool_to_blip3o_format(grid_features, target_tokens=64):
    """Pool 16x16 grids to 8x8 (64 tokens) using 2x2 average pooling"""
    batch_size, grid_h, grid_w, hidden_dim = grid_features.shape
    
    if grid_h * grid_w == target_tokens:
        return grid_features.reshape(batch_size, target_tokens, hidden_dim)
    
    # 2x2 average pooling: 16x16 -> 8x8
    grid_for_pooling = grid_features.permute(0, 3, 1, 2)  # [B, H, 16, 16]
    pooled = F.avg_pool2d(grid_for_pooling, kernel_size=2, stride=2)  # [B, H, 8, 8]
    result = pooled.permute(0, 2, 3, 1).reshape(batch_size, target_tokens, hidden_dim)
    
    return result

def main():
    """Main extraction function"""
    print("🚀 Production Grid Embedding Extraction for BLIP3-o")
    print("=" * 60)
    
    # Setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this script")
    
    device = torch.device('cuda')
    project_root = setup_paths()
    
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Import dataset
    from src.data_hand.dataset import BLIP3oWebDataset
    
    # Load models
    clip_processor, clip_model, eva_processor, eva_model = load_models(device)
    
    # Setup dataset
    data_file = project_root / "data" / "00000.tar"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print("📂 Creating dataset...")
    dataset = BLIP3oWebDataset(
        tar_paths=[str(data_file)],
        batch_size=8,  # Larger batch size for production
        shuffle=False,
        num_workers=0
    )
    
    dataloader = dataset.get_dataloader()
    print("✅ Dataset ready")
    
    # Extract embeddings
    print("🧠 Extracting embeddings...")
    
    all_clip_grids = []
    all_eva_grids = []
    all_clip_blip3o = []
    all_eva_blip3o = []
    all_captions = []
    all_keys = []
    
    total_samples = 0
    
    # Process all batches (no limit)
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        images = batch['image']
        captions = batch['caption']
        keys = batch['key']
        
        # Extract features
        clip_grids = extract_clip_features(images, clip_processor, clip_model, device)
        eva_grids = extract_eva_features(images, eva_processor, eva_model, device)
        
        # Pool to BLIP3-o format (64 tokens)
        clip_blip3o = pool_to_blip3o_format(clip_grids)
        eva_blip3o = pool_to_blip3o_format(eva_grids)
        
        # Store results
        all_clip_grids.append(clip_grids)
        all_eva_grids.append(eva_grids)
        all_clip_blip3o.append(clip_blip3o)
        all_eva_blip3o.append(eva_blip3o)
        all_captions.extend(captions)
        all_keys.extend(keys)
        
        total_samples += len(images)
        
        # Memory cleanup every 50 batches
        if batch_idx % 50 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()
            print(f"   Processed {total_samples} samples...")
    
    # Combine all results
    print(f"💾 Combining {total_samples} embeddings...")
    final_clip_grids = torch.cat(all_clip_grids, dim=0)
    final_eva_grids = torch.cat(all_eva_grids, dim=0)
    final_clip_blip3o = torch.cat(all_clip_blip3o, dim=0)
    final_eva_blip3o = torch.cat(all_eva_blip3o, dim=0)
    
    print(f"📊 Final embedding shapes:")
    print(f"   CLIP grids (16x16): {final_clip_grids.shape}")
    print(f"   EVA grids (16x16): {final_eva_grids.shape}")
    print(f"   CLIP BLIP3-o (64 tokens): {final_clip_blip3o.shape}")
    print(f"   EVA BLIP3-o (64 tokens): {final_eva_blip3o.shape}")
    
    # Save embeddings
    output_dir = project_root / "embeddings"
    output_dir.mkdir(exist_ok=True)
    
    embeddings_data = {
        # Full resolution grids (16x16)
        'clip_grid_embeddings': final_clip_grids,
        'eva_grid_embeddings': final_eva_grids,
        # BLIP3-o compatible format (64 tokens)
        'clip_blip3o_embeddings': final_clip_blip3o,
        'eva_blip3o_embeddings': final_eva_blip3o,
        # Metadata
        'captions': all_captions,
        'keys': all_keys,
        'total_samples': total_samples,
        'config': {
            'clip_model': 'openai/clip-vit-large-patch14',
            'eva_model': 'BAAI/EVA-CLIP-8B',
            'clip_dim': final_clip_grids.shape[-1],
            'eva_dim': final_eva_grids.shape[-1],
            'grid_size': 16,
            'blip3o_tokens': 64,
            'pooling_method': 'avg_pool2d_2x2'
        }
    }
    
    output_file = output_dir / "blip3o_grid_embeddings.pkl"
    
    print("💾 Saving embeddings...")
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print("✅ SUCCESS!")
    print(f"📁 Saved to: {output_file}")
    print(f"📊 File size: {file_size_mb:.2f} MB")
    print(f"🔢 Total samples: {total_samples}")
    print(f"📐 CLIP features: {final_clip_grids.shape[-1]}D")
    print(f"📐 EVA features: {final_eva_grids.shape[-1]}D")
    
    # Quick verification
    with open(output_file, 'rb') as f:
        loaded = pickle.load(f)
    
    print(f"✅ Verification: Successfully loaded {loaded['total_samples']} samples")
    print("🎉 Grid embedding extraction completed!")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)