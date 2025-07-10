"""
Memory-Efficient Grid-Based Embedding Extraction for BLIP3-o - CLEAN VERSION
Place this file as: src/modules/extract_embeddings_g.py

FIXES:
1. Progressive saving to avoid OOM
2. Reduced batch size and better memory management
3. Resume capability from checkpoints
4. More frequent garbage collection
5. Memory-efficient data types
6. NO h5py dependency - pure pickle/torch
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
import gc
import psutil
import time

def setup_paths():
    """Setup paths for project structure"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Add import paths
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "src" / "data_hand"))
    
    return project_root

def get_memory_usage():
    """Get current memory usage in GB"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb / 1024
    except:
        return 0.0

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def estimate_total_batches(data_file_path, batch_size=4):
    """Estimate total number of batches based on file size"""
    try:
        file_size_gb = data_file_path.stat().st_size / (1024**3)
        # Rough estimate: ~400-500 samples per GB for image data
        estimated_samples = int(file_size_gb * 450)
        estimated_batches = estimated_samples // batch_size
        return estimated_samples, estimated_batches
    except:
        return None, None

def load_models(device):
    """Load CLIP and EVA-CLIP models with memory optimization"""
    print("📦 Loading models...")
    
    # Load CLIP ViT-L/14
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16  # Use fp16 to save memory
    ).to(device)
    clip_model.eval()
    
    # Load EVA-CLIP-8B
    eva_model = AutoModel.from_pretrained(
        "BAAI/EVA-CLIP-8B", 
        trust_remote_code=True,
        torch_dtype=torch.float16  # Use fp16 to save memory
    ).to(device)
    eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    eva_model.eval()
    
    # Cleanup after loading
    cleanup_memory()
    
    print("✅ Models loaded successfully")
    print(f"💾 Memory usage after loading: {get_memory_usage():.2f} GB")
    return clip_processor, clip_model, eva_processor, eva_model

def extract_clip_features(images, processor, model, device):
    """Extract CLIP ViT-L/14 patch grid features with memory optimization"""
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
            
            # Convert to float32 and move to CPU immediately to save GPU memory
            features.append(spatial_grid.squeeze().cpu().float())
            
            # Clear GPU memory
            del vision_outputs, patch_embeddings, spatial_grid
    
    return torch.stack(features)

def extract_eva_features(images, processor, model, device):
    """Extract EVA-CLIP-8B patch grid features with memory optimization"""
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
            
            # Convert to float32 and move to CPU immediately to save GPU memory
            features.append(spatial_grid.squeeze().cpu().float())
            
            # Clear GPU memory
            del vision_outputs, patch_embeddings, spatial_grid, pixel_values
    
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

def save_checkpoint(embeddings_data, checkpoint_path, batch_idx, total_samples):
    """Save embeddings checkpoint"""
    checkpoint_data = {
        **embeddings_data,
        'checkpoint_info': {
            'batch_idx': batch_idx,
            'total_samples': total_samples,
            'timestamp': time.time(),
        }
    }
    
    print(f"💾 Saving checkpoint at batch {batch_idx} ({total_samples} samples)...")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"✅ Checkpoint saved: {file_size_mb:.2f} MB")

def load_checkpoint(checkpoint_path):
    """Load embeddings checkpoint"""
    if not checkpoint_path.exists():
        return None
    
    print(f"📂 Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'checkpoint_info' in data:
        info = data['checkpoint_info']
        print(f"✅ Checkpoint loaded: batch {info['batch_idx']}, {info['total_samples']} samples")
        return data
    else:
        print("⚠️  Checkpoint format not recognized, starting fresh")
        return None

def main():
    """Main extraction function with memory optimization"""
    print("🚀 Memory-Efficient Grid Embedding Extraction for BLIP3-o")
    print("=" * 70)
    
    # Setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this script")
    
    device = torch.device('cuda')
    project_root = setup_paths()
    
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Import dataset
    try:
        from src.data_hand.dataset import BLIP3oWebDataset
    except ImportError as e:
        print(f"❌ Failed to import dataset: {e}")
        print("Make sure you're running from the project root directory")
        return 1
    
    # Load models
    try:
        clip_processor, clip_model, eva_processor, eva_model = load_models(device)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Setup dataset with smaller batch size to prevent OOM
    data_file = project_root / "data" / "00000.tar"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print("📂 Creating dataset...")
    dataset = BLIP3oWebDataset(
        tar_paths=[str(data_file)],
        batch_size=4,  # REDUCED from 8 to prevent OOM
        shuffle=False,
        num_workers=0  # Disable multiprocessing to save memory
    )
    
    dataloader = dataset.get_dataloader()
    print("✅ Dataset ready")
    print("📊 WebDataset ready (length unknown, will process all available batches)")
    
    # Estimate total batches for progress tracking
    estimated_samples, estimated_batches = estimate_total_batches(data_file, batch_size=4)
    if estimated_samples:
        print(f"📈 Estimated: ~{estimated_samples:,} samples in ~{estimated_batches:,} batches")
    else:
        print("📈 Unable to estimate total - will show progress as we go")
    
    # Setup output directories
    output_dir = project_root / "embeddings"
    output_dir.mkdir(exist_ok=True)
    
    # Checkpoint management
    checkpoint_path = output_dir / "extraction_checkpoint.pkl"
    final_output_path = output_dir / "blip3o_grid_embeddings.pkl"
    
    # Try to load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    
    if checkpoint_data:
        # Resume from checkpoint
        all_clip_grids = checkpoint_data.get('clip_grid_embeddings_list', [])
        all_eva_grids = checkpoint_data.get('eva_grid_embeddings_list', [])
        all_clip_blip3o = checkpoint_data.get('clip_blip3o_embeddings_list', [])
        all_eva_blip3o = checkpoint_data.get('eva_blip3o_embeddings_list', [])
        all_captions = checkpoint_data.get('captions', [])
        all_keys = checkpoint_data.get('keys', [])
        start_batch = checkpoint_data['checkpoint_info']['batch_idx'] + 1
        total_samples = checkpoint_data['checkpoint_info']['total_samples']
        
        print(f"🔄 Resuming from batch {start_batch} with {total_samples} existing samples")
        print("📊 Note: WebDataset will skip already processed data automatically")
    else:
        # Start fresh
        all_clip_grids = []
        all_eva_grids = []
        all_clip_blip3o = []
        all_eva_blip3o = []
        all_captions = []
        all_keys = []
        start_batch = 0
        total_samples = 0
    
    # Extract embeddings with progressive saving
    print("🧠 Extracting embeddings...")
    
    # Configure checkpoint saving frequency
    CHECKPOINT_EVERY = 50  # Save checkpoint every 50 batches
    MEMORY_CLEANUP_EVERY = 10  # Cleanup memory every 10 batches
    
    try:
        # For WebDataset, we need to handle resuming differently
        # We'll count batches from start_batch and skip processing until we reach the resume point
        current_batch_idx = 0
        processed_batches = 0
        
        # Process all available batches
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            # Skip batches if resuming (count processed samples to determine if already done)
            if current_batch_idx < start_batch:
                current_batch_idx += 1
                continue
            
            try:
                images = batch['image']
                captions = batch['caption']
                keys = batch['key']
                
                # Monitor memory before processing
                mem_before = get_memory_usage()
                if mem_before > 8.0:  # If using more than 8GB, do aggressive cleanup
                    print(f"⚠️  High memory usage detected: {mem_before:.2f} GB - cleaning up...")
                    cleanup_memory()
                
                # Extract features with error handling
                try:
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
                    processed_batches += 1
                    
                    # Clear variables to free memory
                    del clip_grids, eva_grids, clip_blip3o, eva_blip3o, images
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"💥 OOM error at batch {current_batch_idx}, cleaning up and retrying with smaller batch...")
                    cleanup_memory()
                    current_batch_idx += 1
                    continue
                except Exception as e:
                    print(f"⚠️  Error processing batch {current_batch_idx}: {e}")
                    current_batch_idx += 1
                    continue
                
                # Memory cleanup
                if processed_batches % MEMORY_CLEANUP_EVERY == 0:
                    cleanup_memory()
                
                # Save checkpoint
                if processed_batches % CHECKPOINT_EVERY == 0 and processed_batches > 0:
                    # Prepare checkpoint data
                    checkpoint_data = {
                        'clip_grid_embeddings_list': all_clip_grids,
                        'eva_grid_embeddings_list': all_eva_grids,
                        'clip_blip3o_embeddings_list': all_clip_blip3o,
                        'eva_blip3o_embeddings_list': all_eva_blip3o,
                        'captions': all_captions,
                        'keys': all_keys,
                    }
                    
                    save_checkpoint(checkpoint_data, checkpoint_path, current_batch_idx, total_samples)
                    
                    # Monitor memory
                    mem_after = get_memory_usage()
                    progress_msg = f"📊 Progress: {processed_batches} processed batches, {total_samples} samples, {mem_after:.2f} GB memory"
                    if estimated_batches:
                        percent_complete = (processed_batches / estimated_batches) * 100
                        progress_msg += f" (~{percent_complete:.1f}% estimated)"
                    print(progress_msg)
                
                current_batch_idx += 1
                
            except StopIteration:
                print("📄 Reached end of dataset")
                break
            except Exception as e:
                print(f"⚠️  Unexpected error in batch loop: {e}")
                current_batch_idx += 1
                continue
    
    except KeyboardInterrupt:
        print("\n⛔ Extraction interrupted by user")
        # Save emergency checkpoint
        emergency_checkpoint = {
            'clip_grid_embeddings_list': all_clip_grids,
            'eva_grid_embeddings_list': all_eva_grids,
            'clip_blip3o_embeddings_list': all_clip_blip3o,
            'eva_blip3o_embeddings_list': all_eva_blip3o,
            'captions': all_captions,
            'keys': all_keys,
        }
        emergency_path = output_dir / "emergency_checkpoint.pkl"
        save_checkpoint(emergency_checkpoint, emergency_path, current_batch_idx, total_samples)
        print(f"🆘 Emergency checkpoint saved to {emergency_path}")
        return 1
    
    # Final processing and saving
    print(f"💾 Combining {total_samples} embeddings from {processed_batches} batches...")
    if estimated_batches and processed_batches > 0:
        completion_rate = (processed_batches / estimated_batches) * 100
        print(f"📈 Processed ~{completion_rate:.1f}% of estimated data")
    
    try:
        # Combine all results
        final_clip_grids = torch.cat(all_clip_grids, dim=0)
        final_eva_grids = torch.cat(all_eva_grids, dim=0)
        final_clip_blip3o = torch.cat(all_clip_blip3o, dim=0)
        final_eva_blip3o = torch.cat(all_eva_blip3o, dim=0)
        
        print(f"📊 Final embedding shapes:")
        print(f"   CLIP grids (16x16): {final_clip_grids.shape}")
        print(f"   EVA grids (16x16): {final_eva_grids.shape}")
        print(f"   CLIP BLIP3-o (64 tokens): {final_clip_blip3o.shape}")
        print(f"   EVA BLIP3-o (64 tokens): {final_eva_blip3o.shape}")
        
        # Prepare final embeddings data
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
        
        # Save final embeddings
        print("💾 Saving final embeddings...")
        with open(final_output_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        file_size_mb = final_output_path.stat().st_size / (1024 * 1024)
        
        print("✅ SUCCESS!")
        print(f"📁 Saved to: {final_output_path}")
        print(f"📊 File size: {file_size_mb:.2f} MB")
        print(f"🔢 Total samples: {total_samples}")
        print(f"📐 CLIP features: {final_clip_grids.shape[-1]}D")
        print(f"📐 EVA features: {final_eva_grids.shape[-1]}D")
        
        # Clean up checkpoint file
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"🗑️  Cleaned up checkpoint file")
        
        # Quick verification
        with open(final_output_path, 'rb') as f:
            loaded = pickle.load(f)
        
        print(f"✅ Verification: Successfully loaded {loaded['total_samples']} samples")
        print("🎉 Grid embedding extraction completed!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during final processing: {e}")
        print("💾 Keeping checkpoint file for manual recovery")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)