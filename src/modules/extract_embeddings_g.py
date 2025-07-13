"""
CHUNKED: Memory-Efficient Grid-Based Embedding Extraction for BLIP3-o
Place this file as: src/modules/extract_embeddings_chunked.py

NEW APPROACH:
1. Process each TAR file separately
2. Save individual pickle files per TAR (small chunks)
3. Avoid disk quota issues by processing one TAR at a time
4. Enable scaling to 30+ TAR files (~100k samples)

FIXED: Import issues resolved
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
import glob

def setup_paths():
    """Setup paths for project structure"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Add import paths
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "src" / "data_hand"))
    
    return project_root

def get_temp_directory():
    """Get the temp directory path for Snellius or other systems"""
    # Check for environment variable first
    if "BLIP3O_TEMP_DIR" in os.environ:
        temp_dir = Path(os.environ["BLIP3O_TEMP_DIR"])
    elif "TMPDIR" in os.environ:
        temp_dir = Path(os.environ["TMPDIR"])
    elif "SCRATCH_SHARED" in os.environ:
        temp_dir = Path(os.environ["SCRATCH_SHARED"])
    else:
        # Fallback to project embeddings directory
        temp_dir = setup_paths() / "embeddings"
    
    return temp_dir

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

def load_models(device):
    """Load CLIP and EVA-CLIP models with correct dimensions"""
    print("📦 Loading models...")
    
    # Load CLIP ViT-L/14 (should output 1024-dim features)
    print("   Loading CLIP ViT-L/14...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16
    ).to(device)
    clip_model.eval()
    
    # Verify CLIP dimensions
    clip_dim = clip_model.config.vision_config.hidden_size
    print(f"   CLIP ViT-L/14 dimension: {clip_dim}")
    assert clip_dim == 1024, f"Expected CLIP dimension 1024, got {clip_dim}"
    
    # Load EVA-CLIP-8B (should output 4096-dim features)
    print("   Loading EVA-CLIP-8B...")
    eva_model = AutoModel.from_pretrained(
        "BAAI/EVA-CLIP-8B", 
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    
    # Use CLIP processor for EVA-CLIP (common practice)
    print("   Using CLIP processor for EVA-CLIP preprocessing...")
    eva_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    eva_model.eval()
    
    # Verify EVA-CLIP dimensions  
    print("   Testing EVA-CLIP output dimensions...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    with torch.no_grad():
        try:
            dummy_output = eva_model.vision_model(dummy_input)
            eva_dim = dummy_output.last_hidden_state.shape[-1]
        except Exception as e:
            print(f"   Warning: Could not test with dummy input: {e}")
            # Assume 4096 for EVA-CLIP-8B based on architecture
            eva_dim = 4096
            print(f"   Assuming EVA dimension: {eva_dim} (EVA-CLIP-8B)")
    
    print(f"   EVA-CLIP-8B dimension: {eva_dim}")
    
    # Cleanup after loading
    cleanup_memory()
    
    print("✅ Models loaded successfully")
    print(f"   CLIP: {clip_dim}-dim, EVA-CLIP: {eva_dim}-dim")
    print(f"💾 Memory usage after loading: {get_memory_usage():.2f} GB")
    
    return clip_processor, clip_model, eva_processor, eva_model

def extract_clip_features(images, processor, model, device):
    """Extract CLIP ViT-L/14 patch grid features (1024-dim) - 256 tokens"""
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
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, 1024]
            batch_size, num_patches, hidden_dim = patch_embeddings.shape
            
            # Validate dimensions
            assert hidden_dim == 1024, f"Expected CLIP 1024-dim, got {hidden_dim}"
            assert num_patches == 256, f"Expected 256 patches (16x16), got {num_patches}"
            
            # Keep full 16x16 grid (256 tokens) - NO POOLING
            # Reshape to spatial grid: [1, 16, 16, 1024]
            grid_size = int(np.sqrt(num_patches))  # 16
            spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
            
            # Convert to float32 and move to CPU
            features.append(spatial_grid.squeeze().cpu().float())
            
            # Clear GPU memory
            del vision_outputs, patch_embeddings, spatial_grid
    
    return torch.stack(features)

def extract_eva_features(images, processor, model, device):
    """Extract EVA-CLIP-8B patch grid features - 256 tokens"""
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
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, hidden_dim]
            batch_size, num_patches, hidden_dim = patch_embeddings.shape
            
            # Validate dimensions
            assert num_patches == 256, f"Expected 256 patches (16x16), got {num_patches}"
            
            # Keep full 16x16 grid (256 tokens) - NO POOLING
            # Reshape to spatial grid: [1, 16, 16, hidden_dim]
            grid_size = int(np.sqrt(num_patches))  # 16
            spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
            
            # Convert to float32 and move to CPU
            features.append(spatial_grid.squeeze().cpu().float())
            
            # Clear GPU memory
            del vision_outputs, patch_embeddings, spatial_grid, pixel_values
    
    return torch.stack(features)

def format_to_blip3o_tokens(grid_features, target_tokens=256):
    """Format 16x16 grids to 256 tokens for BLIP3-o (NO POOLING)"""
    batch_size, grid_h, grid_w, hidden_dim = grid_features.shape
    
    # Keep full resolution - 16x16 = 256 tokens
    assert grid_h == 16 and grid_w == 16, f"Expected 16x16 grid, got {grid_h}x{grid_w}"
    assert grid_h * grid_w == target_tokens, f"Expected {target_tokens} tokens, got {grid_h * grid_w}"
    
    # Simply reshape to token format: [batch_size, 256, hidden_dim]
    result = grid_features.reshape(batch_size, target_tokens, hidden_dim)
    
    return result

def find_data_files():
    """Find downloaded tar files in temp directory or project directory"""
    temp_dir = get_temp_directory()
    project_root = setup_paths()
    
    # Check temp directory first
    tar_files = []
    
    print(f"🔍 Searching for dataset shards...")
    print(f"   Temp directory: {temp_dir}")
    
    # Look for downloaded shards list
    search_locations = []
    
    # 1. Check temp directory with blip3o_data subdirectory
    search_locations.append(temp_dir / "blip3o_data")
    search_locations.append(temp_dir / "data")
    search_locations.append(temp_dir)
    
    # 2. Check if TMPDIR has blip3o_data (common on Snellius)
    if "TMPDIR" in os.environ:
        tmpdir_path = Path(os.environ["TMPDIR"])
        search_locations.append(tmpdir_path / "blip3o_data")
        search_locations.append(tmpdir_path)
    
    # 3. Check SCRATCH_SHARED (Snellius)
    if "SCRATCH_SHARED" in os.environ:
        user = os.environ.get("USER", "user")
        scratch_path = Path(os.environ["SCRATCH_SHARED"]) / user
        search_locations.append(scratch_path / "blip3o_data")
        search_locations.append(scratch_path)
    
    # 4. Check project directory
    search_locations.append(project_root / "data")
    
    # Search each location
    for search_path in search_locations:
        if not search_path.exists():
            continue
            
        print(f"   Checking: {search_path}")
        
        # Look for tar files directly
        found_tars = list(search_path.glob("*.tar"))
        if found_tars and not tar_files:
            found_tars.sort()  # Sort numerically
            tar_files = [str(f) for f in found_tars]
            print(f"   ✅ Found {len(tar_files)} tar files directly")
            break
    
    if not tar_files:
        raise FileNotFoundError(
            "No tar files found!\n"
            "Please download dataset shards first:\n"
            "  python src/data_hand/download_data.py --shards 0 1 2 3 4 5 6 7 8 9\n"
            "\nOr check if files are in a different location."
        )
    
    # Validate files exist and show details
    valid_files = []
    total_size_gb = 0
    
    print(f"\n📊 Validating found files...")
    for tar_file in tar_files:
        tar_path = Path(tar_file)
        if tar_path.exists():
            size_gb = tar_path.stat().st_size / (1024**3)
            total_size_gb += size_gb
            valid_files.append(tar_file)
            print(f"   ✅ {tar_path.name}: {size_gb:.2f} GB")
        else:
            print(f"   ❌ Missing: {tar_file}")
    
    print(f"\n🎯 Using {len(valid_files)} tar files for CHUNKED extraction")
    print(f"📊 Total dataset size: {total_size_gb:.2f} GB")
    
    # Estimate samples
    estimated_samples = int(total_size_gb * 400000 / 1.0)  # Rough estimate
    print(f"📊 Estimated samples: ~{estimated_samples:,}")
    
    return valid_files

def process_single_tar(
    tar_file_path: str,
    shard_idx: int,
    clip_processor, clip_model, eva_processor, eva_model,
    device: torch.device,
    output_dir: Path,
    batch_size: int = 16
) -> dict:
    """
    Process a single TAR file and save embeddings.
    
    Args:
        tar_file_path: Path to the TAR file
        shard_idx: Index of this shard
        clip_processor, clip_model, eva_processor, eva_model: Loaded models
        device: Computing device
        output_dir: Output directory for embeddings
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with processing statistics
    """
    
    print(f"\n🔄 Processing shard {shard_idx}: {Path(tar_file_path).name}")
    
    # Import dataset - FIXED: Use the correct import path
    try:
        # Try to import the original WebDataset from the data_hand directory
        from dataset import BLIP3oWebDataset
        print("   ✅ Imported BLIP3oWebDataset from dataset.py")
    except ImportError:
        try:
            # Alternative import path
            from src.data_hand.dataset import BLIP3oWebDataset
            print("   ✅ Imported BLIP3oWebDataset from src.data_hand.dataset")
        except ImportError as e:
            print(f"   ❌ Failed to import BLIP3oWebDataset: {e}")
            print("   💡 Using simplified WebDataset approach...")
            
            # FALLBACK: Use webdataset directly without the custom wrapper
            import webdataset as wds
            from PIL import Image
            import io
            
            def decode_sample(sample):
                """Decode a sample from WebDataset"""
                try:
                    # Get image
                    for ext in ['jpg', 'jpeg', 'png', 'webp']:
                        if ext in sample:
                            image_data = sample[ext]
                            break
                    else:
                        return None
                    
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    
                    # Get caption
                    caption = ""
                    for caption_key in ['txt', 'caption', 'text']:
                        if caption_key in sample:
                            caption_data = sample[caption_key]
                            if isinstance(caption_data, bytes):
                                caption = caption_data.decode('utf-8').strip()
                            else:
                                caption = str(caption_data).strip()
                            break
                    
                    key = sample.get('__key__', 'unknown')
                    
                    return {
                        'image': image,
                        'caption': caption,
                        'key': key,
                    }
                except Exception as e:
                    print(f"     Warning: Failed to decode sample: {e}")
                    return None
            
            # Create WebDataset directly
            dataset = (wds.WebDataset([tar_file_path], empty_check=False)
                      .map(decode_sample)
                      .select(lambda x: x is not None))
            
            # Create simple dataloader
            def simple_collate(batch):
                images = [item['image'] for item in batch]
                captions = [item['caption'] for item in batch]
                keys = [item['key'] for item in batch]
                return {
                    'image': images,
                    'caption': captions,
                    'key': keys
                }
            
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=simple_collate)
            
            print(f"   ✅ Created simple WebDataset dataloader")
    
    # If we successfully imported BLIP3oWebDataset, use it
    if 'BLIP3oWebDataset' in locals():
        # Create dataset for this single TAR file
        dataset = BLIP3oWebDataset(
            tar_paths=[tar_file_path],  # Only this TAR file
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        dataloader = dataset.get_dataloader()
        print(f"   ✅ Created BLIP3oWebDataset dataloader")
    
    # Storage for this shard's embeddings
    shard_clip_embeddings = []
    shard_eva_embeddings = []
    shard_captions = []
    shard_keys = []
    
    total_samples = 0
    start_time = time.time()
    
    print(f"   📊 Processing batches...")
    
    # Process all batches in this TAR file
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Shard {shard_idx}", unit="batch")):
            try:
                images = batch['image']
                captions = batch['caption']
                keys = batch['key']
                
                # Extract features
                clip_grids = extract_clip_features(images, clip_processor, clip_model, device)
                cleanup_memory()
                
                eva_grids = extract_eva_features(images, eva_processor, eva_model, device)
                cleanup_memory()
                
                # Format to BLIP3-o token format (256 tokens)
                clip_blip3o = format_to_blip3o_tokens(clip_grids, target_tokens=256)
                eva_blip3o = format_to_blip3o_tokens(eva_grids, target_tokens=256)
                
                # Move to CPU and store
                shard_clip_embeddings.append(clip_blip3o.cpu())
                shard_eva_embeddings.append(eva_blip3o.cpu())
                shard_captions.extend(captions)
                shard_keys.extend(keys)
                
                total_samples += len(images)
                
                # Clear intermediate variables
                del clip_grids, eva_grids, clip_blip3o, eva_blip3o, images
                cleanup_memory()
                
                # Progress update
                if batch_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                    print(f"   Batch {batch_idx}: {total_samples} samples, {samples_per_sec:.1f} samples/sec, Mem: {get_memory_usage():.1f}GB")
            
            except Exception as e:
                print(f"   ⚠️  Error processing batch {batch_idx}: {e}")
                continue
    
    except Exception as e:
        print(f"   ❌ Error iterating through dataloader: {e}")
        return {
            'shard_idx': shard_idx,
            'total_samples': 0,
            'success': False,
            'error': str(e)
        }
    
    # Consolidate embeddings for this shard
    if shard_clip_embeddings:
        print(f"   🔄 Consolidating {total_samples} embeddings...")
        
        final_clip = torch.cat(shard_clip_embeddings, dim=0)
        final_eva = torch.cat(shard_eva_embeddings, dim=0)
        
        # Create shard data
        shard_data = {
            'clip_blip3o_embeddings': final_clip,
            'eva_blip3o_embeddings': final_eva,
            'captions': shard_captions,
            'keys': shard_keys,
            'total_samples': total_samples,
            'shard_idx': shard_idx,
            'source_tar': tar_file_path,
            'config': {
                'clip_model': 'openai/clip-vit-large-patch14',
                'eva_model': 'BAAI/EVA-CLIP-8B',
                'clip_dim': 1024,
                'eva_dim': final_eva.shape[2],
                'tokens': 256,
                'grid_size': '16x16',
                'pooling_method': 'none',
                'format_version': 'blip3o_256_tokens_chunked_v1',
                'extraction_time': time.time() - start_time,
            }
        }
        
        # Save this shard's embeddings
        shard_filename = f"embeddings_shard_{shard_idx:05d}.pkl"
        shard_path = output_dir / shard_filename
        
        print(f"   💾 Saving shard {shard_idx}...")
        with open(shard_path, 'wb') as f:
            pickle.dump(shard_data, f)
        
        file_size_mb = shard_path.stat().st_size / (1024 * 1024)
        
        print(f"   ✅ Shard {shard_idx} completed:")
        print(f"      File: {shard_filename}")
        print(f"      Size: {file_size_mb:.1f} MB")
        print(f"      Samples: {total_samples}")
        print(f"      Time: {time.time() - start_time:.1f}s")
        
        # Clear memory
        del shard_clip_embeddings, shard_eva_embeddings, final_clip, final_eva
        cleanup_memory()
        
        return {
            'shard_idx': shard_idx,
            'total_samples': total_samples,
            'file_size_mb': file_size_mb,
            'processing_time': time.time() - start_time,
            'output_path': str(shard_path),
            'success': True
        }
    
    else:
        print(f"   ❌ No embeddings extracted from shard {shard_idx}")
        return {
            'shard_idx': shard_idx,
            'total_samples': 0,
            'success': False
        }

def main():
    """Main extraction function with CHUNKED processing (one file per TAR)"""
    print("🚀 BLIP3-o CHUNKED Embedding Extraction (256 TOKENS)")
    print("=" * 80)
    print("NEW APPROACH:")
    print("  ✅ Process each TAR file separately")
    print("  ✅ Save individual pickle files (small chunks)")
    print("  ✅ Avoid disk quota issues")
    print("  ✅ Enable scaling to 30+ TAR files")
    print("  CLIP BLIP3-o: [N, 256, 1024] per shard")  
    print("  EVA BLIP3-o:  [N, 256, 4096] per shard")
    print("=" * 80)
    
    # Setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this script")
    
    device = torch.device('cuda')
    project_root = setup_paths()
    temp_dir = get_temp_directory()
    
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"📁 Temp directory: {temp_dir}")
    print(f"💾 Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Load models
    try:
        clip_processor, clip_model, eva_processor, eva_model = load_models(device)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Find TAR files
    try:
        tar_files = find_data_files()
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    # Setup output directory
    output_dir = temp_dir / "chunked_embeddings"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📤 Output directory: {output_dir}")
    
    # Process each TAR file separately
    print(f"\n🔄 Processing {len(tar_files)} TAR files...")
    
    processing_results = []
    total_samples_all = 0
    total_size_mb_all = 0
    
    for shard_idx, tar_file in enumerate(tar_files):
        print(f"\n" + "="*60)
        print(f"PROCESSING SHARD {shard_idx + 1}/{len(tar_files)}")
        print(f"="*60)
        
        result = process_single_tar(
            tar_file_path=tar_file,
            shard_idx=shard_idx,
            clip_processor=clip_processor,
            clip_model=clip_model,
            eva_processor=eva_processor,
            eva_model=eva_model,
            device=device,
            output_dir=output_dir,
            batch_size=16  # Smaller batch for memory efficiency
        )
        
        if result and result['success']:
            processing_results.append(result)
            total_samples_all += result['total_samples']
            total_size_mb_all += result['file_size_mb']
            
            print(f"✅ Shard {shard_idx} successful: {result['total_samples']} samples, {result['file_size_mb']:.1f} MB")
        else:
            print(f"❌ Shard {shard_idx} failed")
            if 'error' in result:
                print(f"   Error: {result['error']}")
    
    # Create manifest file
    manifest_data = {
        'total_shards': len(processing_results),
        'total_samples': total_samples_all,
        'total_size_mb': total_size_mb_all,
        'extraction_timestamp': time.time(),
        'shards': processing_results,
        'format_version': 'blip3o_256_tokens_chunked_v1',
        'usage': {
            'training_command': f'python train_blip3o_dit.py --chunked_embeddings_dir {output_dir}',
            'individual_files': [f"embeddings_shard_{i:05d}.pkl" for i in range(len(processing_results))]
        }
    }
    
    manifest_path = output_dir / "embeddings_manifest.json"
    with open(manifest_path, 'w') as f:
        import json
        json.dump(manifest_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ CHUNKED EXTRACTION COMPLETED!")
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   Successful shards: {len(processing_results)}/{len(tar_files)}")
    print(f"   Total samples: {total_samples_all:,}")
    print(f"   Total size: {total_size_mb_all:.1f} MB")
    print(f"   Average per shard: {total_samples_all//len(processing_results) if processing_results else 0:,} samples")
    print(f"   Output directory: {output_dir}")
    print(f"   Manifest file: {manifest_path}")
    print("\n🎉 Ready for chunked BLIP3-o training!")
    print("Use this command:")
    print(f"python train_blip3o_dit.py --chunked_embeddings_dir {output_dir}")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)