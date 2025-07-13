"""
UPDATED: Memory-Efficient Grid-Based Embedding Extraction for BLIP3-o with Temp Manager
Place this file as: src/modules/extract_embeddings_g.py

NEW FEATURES:
1. Uses SnelliusTempManager for structured temp directory management
2. Stores embeddings in persistent workspace (survives 14 days)
3. Uses job temp for processing and cache
4. Better disk space management and monitoring
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
    sys.path.insert(0, str(project_root / "src" / "utils"))
    
    return project_root

def setup_temp_manager():
    """Setup temp manager for structured directory management."""
    try:
        from temp_manager import setup_snellius_environment
        manager = setup_snellius_environment("blip3o_workspace")
        return manager
    except ImportError:
        print("⚠️  Temp manager not available, using fallback directories")
        return None

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

def find_data_files(temp_manager):
    """Find downloaded tar files using temp manager."""
    if temp_manager:
        datasets_dir = temp_manager.get_datasets_dir()
        print(f"🔍 Searching for dataset shards in: {datasets_dir}")
    else:
        # Fallback to old method
        print("🔍 Searching for dataset shards (fallback method)...")
        if "TMPDIR" in os.environ:
            datasets_dir = Path(os.environ["TMPDIR"]) / "blip3o_data"
        elif "SCRATCH_SHARED" in os.environ:
            user = os.environ.get("USER", "user")
            datasets_dir = Path(os.environ["SCRATCH_SHARED"]) / user / "blip3o_data"
        else:
            datasets_dir = Path(__file__).parent.parent.parent / "data"
    
    # Look for tar files
    tar_files = list(datasets_dir.glob("*.tar"))
    if tar_files:
        tar_files.sort()  # Sort numerically
        tar_files = [str(f) for f in tar_files]
        print(f"   ✅ Found {len(tar_files)} tar files")
        
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
    
    # Also check for downloaded_shards.txt file
    shard_list = datasets_dir / "downloaded_shards.txt"
    if shard_list.exists():
        print(f"   📋 Found shard list: {shard_list}")
        try:
            with open(shard_list, 'r') as f:
                listed_files = [line.strip() for line in f if line.strip()]
            
            # Validate files exist
            valid_files = [f for f in listed_files if Path(f).exists()]
            if valid_files:
                print(f"   ✅ Using {len(valid_files)} files from shard list")
                return valid_files
        except Exception as e:
            print(f"   ⚠️  Could not read shard list: {e}")
    
    raise FileNotFoundError(
        f"No TAR files found in {datasets_dir}!\n"
        "Please download dataset shards first:\n"
        "  python src/data_hand/download_data.py --shards 0 1 2 3 4 5 6 7 8 9\n"
        "\nOr check if files are in a different location."
    )

def process_single_tar(
    tar_file_path: str,
    shard_idx: int,
    clip_processor, clip_model, eva_processor, eva_model,
    device: torch.device,
    output_dir: Path,
    working_dir: Path,
    batch_size: int = 16
) -> dict:
    """
    Process a single TAR file and save embeddings using structured temp directories.
    
    Args:
        tar_file_path: Path to the TAR file
        shard_idx: Index of this shard
        clip_processor, clip_model, eva_processor, eva_model: Loaded models
        device: Computing device
        output_dir: Output directory for embeddings (persistent)
        working_dir: Working directory for processing (temp)
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with processing statistics
    """
    
    print(f"\n🔄 Processing shard {shard_idx}: {Path(tar_file_path).name}")
    
    # Import dataset
    try:
        from dataset import BLIP3oWebDataset
        print("   ✅ Imported BLIP3oWebDataset from dataset.py")
    except ImportError:
        try:
            from src.data_hand.dataset import BLIP3oWebDataset
            print("   ✅ Imported BLIP3oWebDataset from src.data_hand.dataset")
        except ImportError as e:
            print(f"   ❌ Failed to import BLIP3oWebDataset: {e}")
            print("   💡 Using simplified WebDataset approach...")
            
            # FALLBACK: Use webdataset directly
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
        dataset = BLIP3oWebDataset(
            tar_paths=[tar_file_path],
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
        
        # Save this shard's embeddings to PERSISTENT storage
        shard_filename = f"embeddings_shard_{shard_idx:05d}.pkl"
        shard_path = output_dir / shard_filename
        
        print(f"   💾 Saving shard {shard_idx} to persistent storage...")
        with open(shard_path, 'wb') as f:
            pickle.dump(shard_data, f)
        
        file_size_mb = shard_path.stat().st_size / (1024 * 1024)
        
        print(f"   ✅ Shard {shard_idx} completed:")
        print(f"      File: {shard_filename}")
        print(f"      Location: {shard_path}")
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
    """Main extraction function with structured temp directory management."""
    print("🚀 BLIP3-o CHUNKED Embedding Extraction (256 TOKENS) with Temp Manager")
    print("=" * 80)
    print("ENHANCED FEATURES:")
    print("  ✅ Structured temp directory management")
    print("  ✅ Persistent embeddings storage (14-day retention)")
    print("  ✅ Job-specific temp processing")
    print("  ✅ Automatic disk usage monitoring")
    print("  ✅ Smart cache management")
    print("=" * 80)
    
    # Setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this script")
    
    device = torch.device('cuda')
    project_root = setup_paths()
    
    # Setup temp manager
    temp_manager = setup_temp_manager()
    
    if temp_manager:
        # Use structured temp management
        embeddings_dir = temp_manager.create_embeddings_subdirectory("chunked_256_tokens")
        working_dir = temp_manager.get_working_dir()
        temp_manager.setup_model_cache()
        
        print(f"✅ Using structured temp management")
        print(f"📁 Embeddings dir (persistent): {embeddings_dir}")
        print(f"📁 Working dir (temp): {working_dir}")
        
        # Show disk usage
        temp_manager.print_status()
    else:
        # Fallback to old method
        if "TMPDIR" in os.environ:
            base_temp = Path(os.environ["TMPDIR"])
        elif "SCRATCH_SHARED" in os.environ:
            user = os.environ.get("USER", "user")
            base_temp = Path(os.environ["SCRATCH_SHARED"]) / user
        else:
            base_temp = Path("./temp")
        
        embeddings_dir = base_temp / "chunked_embeddings"
        working_dir = base_temp / "working"
        
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"⚠️  Using fallback temp management")
        print(f"📁 Embeddings dir: {embeddings_dir}")
        print(f"📁 Working dir: {working_dir}")
    
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Load models
    try:
        clip_processor, clip_model, eva_processor, eva_model = load_models(device)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Find TAR files
    try:
        tar_files = find_data_files(temp_manager)
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    print(f"📤 Output directory: {embeddings_dir}")
    
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
            output_dir=embeddings_dir,
            working_dir=working_dir,
            batch_size=16
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
        
        # Show disk usage after each shard
        if temp_manager and shard_idx % 5 == 0:  # Every 5 shards
            usage = temp_manager.get_disk_usage()
            persistent_usage = usage.get('embeddings', {}).get('total_size_gb', 0)
            print(f"   💾 Persistent storage usage: {persistent_usage:.2f} GB")
    
    # Create manifest file
    manifest_data = {
        'total_shards': len(processing_results),
        'total_samples': total_samples_all,
        'total_size_mb': total_size_mb_all,
        'extraction_timestamp': time.time(),
        'shards': processing_results,
        'format_version': 'blip3o_256_tokens_chunked_v1',
        'storage_info': {
            'embeddings_directory': str(embeddings_dir),
            'persistent_storage': True,
            'retention_policy': '14 days (scratch-shared)',
            'access_path': str(embeddings_dir),
        },
        'usage': {
            'training_command': f'python train_blip3o_dit.py --chunked_embeddings_dir {embeddings_dir}',
            'individual_files': [f"embeddings_shard_{i:05d}.pkl" for i in range(len(processing_results))]
        }
    }
    
    manifest_path = embeddings_dir / "embeddings_manifest.json"
    with open(manifest_path, 'w') as f:
        import json
        json.dump(manifest_data, f, indent=2)
    
    # Final status
    print("\n" + "=" * 80)
    print("✅ CHUNKED EXTRACTION COMPLETED!")
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   Successful shards: {len(processing_results)}/{len(tar_files)}")
    print(f"   Total samples: {total_samples_all:,}")
    print(f"   Total size: {total_size_mb_all:.1f} MB")
    print(f"   Average per shard: {total_samples_all//len(processing_results) if processing_results else 0:,} samples")
    print(f"   Embeddings location: {embeddings_dir}")
    print(f"   Manifest file: {manifest_path}")
    print(f"   Storage: Persistent (14-day retention on scratch-shared)")
    
    if temp_manager:
        print(f"\n📋 STORAGE DETAILS:")
        usage = temp_manager.get_disk_usage()
        for name, info in usage.items():
            if info.get('exists', False) and 'embeddings' in name:
                size_gb = info.get('total_size_gb', 0)
                print(f"   {name}: {size_gb:.2f} GB")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • Embeddings are in persistent storage (14-day retention)")
        print(f"   • Use the embeddings directory path for training")
        print(f"   • Copy final models to home directory for long-term storage")
        print(f"   • Monitor disk usage to avoid quotas")
    
    print("\n🎉 Ready for chunked BLIP3-o training!")
    print("Use this command:")
    print(f"python train_blip3o_dit.py --chunked_embeddings_dir {embeddings_dir}")
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