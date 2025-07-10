"""
UPDATED: Memory-Efficient Grid-Based Embedding Extraction for BLIP3-o
Place this file as: src/modules/extract_embeddings_g.py

UPDATES:
1. Uses temp directory for embeddings storage
2. 256 tokens (16x16 grid) instead of 64 tokens (no avg pooling)
3. Supports multiple tar files from temp directory
4. Proper validation and error handling
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
    # Check for Snellius temp directory
    if "TMPDIR" in os.environ:
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
    """Extract CLIP ViT-L/14 patch grid features (1024-dim) - UPDATED for 256 tokens"""
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
            
            # UPDATED: Keep full 16x16 grid (256 tokens) - NO POOLING
            # Reshape to spatial grid: [1, 16, 16, 1024]
            grid_size = int(np.sqrt(num_patches))  # 16
            spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
            
            # Convert to float32 and move to CPU
            features.append(spatial_grid.squeeze().cpu().float())
            
            # Clear GPU memory
            del vision_outputs, patch_embeddings, spatial_grid
    
    return torch.stack(features)

def extract_eva_features(images, processor, model, device):
    """Extract EVA-CLIP-8B patch grid features - UPDATED for 256 tokens"""
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
            print(f"   EVA patch embeddings shape: {patch_embeddings.shape}")
            assert num_patches == 256, f"Expected 256 patches (16x16), got {num_patches}"
            
            # Store the actual EVA dimension for later use
            if not hasattr(extract_eva_features, 'eva_dim'):
                extract_eva_features.eva_dim = hidden_dim
                print(f"   Detected EVA dimension: {hidden_dim}")
            
            # UPDATED: Keep full 16x16 grid (256 tokens) - NO POOLING
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
    
    # UPDATED: Keep full resolution - 16x16 = 256 tokens
    assert grid_h == 16 and grid_w == 16, f"Expected 16x16 grid, got {grid_h}x{grid_w}"
    assert grid_h * grid_w == target_tokens, f"Expected {target_tokens} tokens, got {grid_h * grid_w}"
    
    # Simply reshape to token format: [batch_size, 256, hidden_dim]
    result = grid_features.reshape(batch_size, target_tokens, hidden_dim)
    
    return result

def validate_extracted_embeddings(clip_blip3o, eva_blip3o, captions):
    """Validate extracted embeddings match UPDATED BLIP3-o requirements (256 tokens)"""
    print("🧪 Validating extracted embeddings...")
    
    # Check shapes
    assert clip_blip3o.dim() == 3, f"CLIP embeddings should be 3D, got {clip_blip3o.dim()}D"
    assert eva_blip3o.dim() == 3, f"EVA embeddings should be 3D, got {eva_blip3o.dim()}D"
    
    # UPDATED: Check for 256 tokens (16x16 grid)
    assert clip_blip3o.shape[1] == 256, f"Expected 256 CLIP tokens, got {clip_blip3o.shape[1]}"
    assert eva_blip3o.shape[1] == 256, f"Expected 256 EVA tokens, got {eva_blip3o.shape[1]}"
    
    # Check dimensions - be flexible about EVA dimension
    assert clip_blip3o.shape[2] == 1024, f"Expected CLIP 1024-dim, got {clip_blip3o.shape[2]}"
    
    eva_dim = eva_blip3o.shape[2]
    print(f"   Detected EVA dimension: {eva_dim}")
    
    # Check batch consistency
    assert clip_blip3o.shape[0] == eva_blip3o.shape[0], "Batch size mismatch between CLIP and EVA"
    assert len(captions) == clip_blip3o.shape[0], "Caption count mismatch"
    
    print(f"✅ Validation passed!")
    print(f"   CLIP BLIP3-o embeddings: {clip_blip3o.shape} (256 tokens, 1024-dim)")
    print(f"   EVA BLIP3-o embeddings: {eva_blip3o.shape} (256 tokens, {eva_dim}-dim)")
    print(f"   Captions: {len(captions)}")
    
    return eva_dim

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
        
        # Look for shard list file first
        shard_list_file = search_path / "downloaded_shards.txt"
        if shard_list_file.exists():
            print(f"   📋 Found shard list: {shard_list_file}")
            try:
                with open(shard_list_file, 'r') as f:
                    listed_files = [line.strip() for line in f if line.strip()]
                
                # Validate files exist
                valid_files = []
                for file_path in listed_files:
                    if Path(file_path).exists():
                        valid_files.append(file_path)
                    else:
                        print(f"   ⚠️  Listed file not found: {file_path}")
                
                if valid_files:
                    tar_files = valid_files
                    print(f"   ✅ Using {len(valid_files)} files from shard list")
                    break
            except Exception as e:
                print(f"   ⚠️  Could not read shard list: {e}")
        
        # Look for tar files directly
        found_tars = list(search_path.glob("*.tar"))
        if found_tars and not tar_files:
            found_tars.sort()  # Sort numerically
            tar_files = [str(f) for f in found_tars]
            print(f"   ✅ Found {len(tar_files)} tar files directly")
            break
    
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
    
    if not valid_files:
        raise FileNotFoundError(
            "No valid tar files found!\n"
            "Please download dataset shards first:\n"
            "  python src/data_hand/download_data.py --shards 0 1 2 3 4 5 6 7 8 9\n"
            "\nOr check if files are in a different location."
        )
    
    print(f"\n🎯 Using {len(valid_files)} tar files for extraction")
    print(f"📊 Total dataset size: {total_size_gb:.2f} GB")
    
    # Estimate samples
    estimated_samples = int(total_size_gb * 400000 / 1.0)  # Rough estimate
    print(f"📊 Estimated samples: ~{estimated_samples:,}")
    
    return valid_files

def save_checkpoint(embeddings_data, checkpoint_path, batch_idx, total_samples):
    """Save embeddings checkpoint to temp directory"""
    checkpoint_data = {
        **embeddings_data,
        'checkpoint_info': {
            'batch_idx': batch_idx,
            'total_samples': total_samples,
            'timestamp': time.time(),
            'format_version': 'blip3o_256_tokens_v1',
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
    """Main extraction function with UPDATED 256-token format and MEMORY OPTIMIZATION"""
    print("🚀 BLIP3-o Embedding Extraction - MEMORY OPTIMIZED (256 TOKENS)")
    print("=" * 80)
    print("UPDATED target format:")
    print("  CLIP BLIP3-o: [N, 256, 1024] - ViT-L/14 features (16x16 grid)")  
    print("  EVA BLIP3-o:  [N, 256, 4096] - EVA-CLIP-8B features (16x16 grid)")
    print("  NO POOLING - Full resolution maintained")
    print("  MEMORY OPTIMIZED - Small batches, frequent cleanup")
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
    
    # Check available space
    try:
        import shutil
        total, used, free = shutil.disk_usage(temp_dir)
        print(f"💽 Available space: {free / (1024**3):.1f} GB")
    except:
        pass
    
    # Import dataset
    try:
        from src.data_hand.dataset import BLIP3oWebDataset
    except ImportError as e:
        print(f"❌ Failed to import dataset: {e}")
        print("Make sure you're running from the project root directory")
        return 1
    
    # Load models with dimension validation
    try:
        clip_processor, clip_model, eva_processor, eva_model = load_models(device)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Find data files (multiple tar files)
    try:
        data_files = find_data_files()
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    print("📂 Creating dataset from multiple files...")
    dataset = BLIP3oWebDataset(
        tar_paths=data_files,
        batch_size=2,  # REDUCED: Smaller batch to prevent OOM (was 4)
        shuffle=False,
        num_workers=0
    )
    
    dataloader = dataset.get_dataloader()
    print("✅ Dataset ready")
    
    # Setup output directories in TEMP
    output_dir = temp_dir / "blip3o_embeddings"
    output_dir.mkdir(exist_ok=True)
    
    # Checkpoint management in temp directory
    checkpoint_path = output_dir / "extraction_checkpoint_256.pkl"
    final_output_path = output_dir / "blip3o_grid_embeddings_256.pkl"
    
    print(f"📤 Output directory: {output_dir}")
    print(f"📤 Final embeddings: {final_output_path}")
    
    # Try to load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    
    if checkpoint_data:
        # Resume from checkpoint
        all_clip_blip3o = checkpoint_data.get('clip_blip3o_embeddings_list', [])
        all_eva_blip3o = checkpoint_data.get('eva_blip3o_embeddings_list', [])
        all_captions = checkpoint_data.get('captions', [])
        all_keys = checkpoint_data.get('keys', [])
        start_batch = checkpoint_data['checkpoint_info']['batch_idx'] + 1
        total_samples = checkpoint_data['checkpoint_info']['total_samples']
        
        print(f"🔄 Resuming from batch {start_batch} with {total_samples} existing samples")
    else:
        # Start fresh
        all_clip_blip3o = []
        all_eva_blip3o = []
        all_captions = []
        all_keys = []
        start_batch = 0
        total_samples = 0
    
    # Extract embeddings with MEMORY OPTIMIZATION
    print("🧠 Extracting BLIP3-o compatible embeddings (256 tokens, MEMORY OPTIMIZED)...")
    
    # MEMORY OPTIMIZATION: More frequent checkpointing and cleanup
    CHECKPOINT_EVERY = 25  # REDUCED: Was 50, now 25
    MEMORY_CLEANUP_EVERY = 5  # REDUCED: Was 10, now 5
    MAX_MEMORY_GB = 50  # Stop and checkpoint if memory exceeds this
    
    # MEMORY OPTIMIZATION: Process in smaller chunks
    CHUNK_SIZE = 100  # Process 100 batches, then consolidate
    
    try:
        current_batch_idx = 0
        processed_batches = 0
        chunks_processed = 0
        
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            if current_batch_idx < start_batch:
                current_batch_idx += 1
                continue
            
            try:
                images = batch['image']
                captions = batch['caption']
                keys = batch['key']
                
                # MEMORY OPTIMIZATION: Monitor memory more frequently
                mem_before = get_memory_usage()
                if mem_before > MAX_MEMORY_GB:
                    print(f"⚠️  MEMORY LIMIT REACHED: {mem_before:.2f} GB > {MAX_MEMORY_GB} GB")
                    print("   Forcing checkpoint and cleanup...")
                    
                    # Force checkpoint
                    checkpoint_data = {
                        'clip_blip3o_embeddings_list': all_clip_blip3o,
                        'eva_blip3o_embeddings_list': all_eva_blip3o,
                        'captions': all_captions,
                        'keys': all_keys,
                    }
                    save_checkpoint(checkpoint_data, checkpoint_path, current_batch_idx, total_samples)
                    
                    # Aggressive cleanup
                    cleanup_memory()
                    
                    # Check if memory is reduced
                    mem_after = get_memory_usage()
                    print(f"   Memory after cleanup: {mem_after:.2f} GB")
                    
                    if mem_after > MAX_MEMORY_GB * 0.8:  # Still too high
                        print(f"   Memory still high, consolidating embeddings...")
                        
                        # Consolidate embeddings to free memory
                        if all_clip_blip3o:
                            consolidated_clip = torch.cat(all_clip_blip3o, dim=0)
                            consolidated_eva = torch.cat(all_eva_blip3o, dim=0)
                            
                            # Clear the lists
                            all_clip_blip3o.clear()
                            all_eva_blip3o.clear()
                            
                            # Add back as single tensor
                            all_clip_blip3o = [consolidated_clip]
                            all_eva_blip3o = [consolidated_eva]
                            
                            cleanup_memory()
                            print(f"   Memory after consolidation: {get_memory_usage():.2f} GB")
                
                # Extract features with UPDATED dimensions (256 tokens)
                clip_grids = extract_clip_features(images, clip_processor, clip_model, device)  # [B, 16, 16, 1024]
                
                # Clear GPU memory immediately after CLIP
                cleanup_memory()
                
                eva_grids = extract_eva_features(images, eva_processor, eva_model, device)      # [B, 16, 16, eva_dim]
                
                # Clear GPU memory immediately after EVA
                cleanup_memory()
                
                # Format to BLIP3-o token format (NO POOLING)
                clip_blip3o = format_to_blip3o_tokens(clip_grids, target_tokens=256)  # [B, 256, 1024]
                eva_blip3o = format_to_blip3o_tokens(eva_grids, target_tokens=256)    # [B, 256, eva_dim]
                
                # Move to CPU immediately to free GPU memory
                clip_blip3o = clip_blip3o.cpu()
                eva_blip3o = eva_blip3o.cpu()
                
                # Clear intermediate variables
                del clip_grids, eva_grids, images
                cleanup_memory()
                
                # Validate dimensions
                actual_eva_dim = eva_blip3o.shape[2]
                assert clip_blip3o.shape[2] == 1024, f"CLIP dimension error: {clip_blip3o.shape[2]}"
                assert clip_blip3o.shape[1] == 256, f"CLIP token count error: {clip_blip3o.shape[1]}"
                assert eva_blip3o.shape[1] == 256, f"EVA token count error: {eva_blip3o.shape[1]}"
                
                if processed_batches % 10 == 0:  # Log less frequently
                    print(f"   Batch {current_batch_idx}: CLIP {clip_blip3o.shape}, EVA {eva_blip3o.shape}, Mem: {get_memory_usage():.1f}GB")
                
                # Store the actual EVA dimension for later use
                if not hasattr(main, 'detected_eva_dim'):
                    main.detected_eva_dim = actual_eva_dim
                    print(f"   📏 Detected EVA dimension: {actual_eva_dim}")
                
                # Store results
                all_clip_blip3o.append(clip_blip3o)
                all_eva_blip3o.append(eva_blip3o)
                all_captions.extend(captions)
                all_keys.extend(keys)
                
                total_samples += len(batch['image'])
                processed_batches += 1
                
                # Clear variables
                del clip_blip3o, eva_blip3o
                
                # MEMORY OPTIMIZATION: More frequent cleanup
                if processed_batches % MEMORY_CLEANUP_EVERY == 0:
                    cleanup_memory()
                
                # MEMORY OPTIMIZATION: More frequent checkpointing
                if processed_batches % CHECKPOINT_EVERY == 0:
                    checkpoint_data = {
                        'clip_blip3o_embeddings_list': all_clip_blip3o,
                        'eva_blip3o_embeddings_list': all_eva_blip3o,
                        'captions': all_captions,
                        'keys': all_keys,
                    }
                    save_checkpoint(checkpoint_data, checkpoint_path, current_batch_idx, total_samples)
                    
                    mem_current = get_memory_usage()
                    print(f"📊 Progress: {processed_batches} batches, {total_samples} samples, Memory: {mem_current:.1f}GB")
                
                # MEMORY OPTIMIZATION: Chunk processing
                if processed_batches % CHUNK_SIZE == 0 and processed_batches > 0:
                    chunks_processed += 1
                    print(f"🔄 Completed chunk {chunks_processed} ({processed_batches} batches)")
                    
                    # Consolidate chunk to save memory
                    if len(all_clip_blip3o) > 1:
                        print("   Consolidating chunk...")
                        consolidated_clip = torch.cat(all_clip_blip3o, dim=0)
                        consolidated_eva = torch.cat(all_eva_blip3o, dim=0)
                        
                        # Replace lists with consolidated tensors
                        all_clip_blip3o = [consolidated_clip]
                        all_eva_blip3o = [consolidated_eva]
                        
                        cleanup_memory()
                        print(f"   Memory after consolidation: {get_memory_usage():.1f} GB")
                
                current_batch_idx += 1
                
            except Exception as e:
                print(f"⚠️  Error processing batch {current_batch_idx}: {e}")
                cleanup_memory()  # Clean up on error
                current_batch_idx += 1
                continue
    
    except KeyboardInterrupt:
        print("\n⛔ Extraction interrupted by user")
        emergency_checkpoint = {
            'clip_blip3o_embeddings_list': all_clip_blip3o,
            'eva_blip3o_embeddings_list': all_eva_blip3o,
            'captions': all_captions,
            'keys': all_keys,
        }
        emergency_path = output_dir / "emergency_checkpoint_256.pkl"
        save_checkpoint(emergency_checkpoint, emergency_path, current_batch_idx, total_samples)
        return 1
    
    # Final processing and saving
    print(f"💾 Combining {total_samples} embeddings...")
    
    try:
        # Final consolidation
        if len(all_clip_blip3o) > 1:
            print("🔄 Final consolidation...")
            final_clip_blip3o = torch.cat(all_clip_blip3o, dim=0)    # [N, 256, 1024]
            final_eva_blip3o = torch.cat(all_eva_blip3o, dim=0)      # [N, 256, 4096]
        else:
            final_clip_blip3o = all_clip_blip3o[0] if all_clip_blip3o else torch.empty(0, 256, 1024)
            final_eva_blip3o = all_eva_blip3o[0] if all_eva_blip3o else torch.empty(0, 256, 4096)
        
        # Clear lists to save memory
        all_clip_blip3o.clear()
        all_eva_blip3o.clear()
        cleanup_memory()
        
        # Validate final embeddings
        validate_extracted_embeddings(final_clip_blip3o, final_eva_blip3o, all_captions)
        
        # Get actual EVA dimension
        actual_eva_dim = final_eva_blip3o.shape[2] if final_eva_blip3o.numel() > 0 else 4096
        
        # Prepare final embeddings data in UPDATED BLIP3-o format (256 tokens)
        embeddings_data = {
            # BLIP3-o compatible format (UPDATED for 256 tokens)
            'clip_blip3o_embeddings': final_clip_blip3o,    # [N, 256, 1024]
            'eva_blip3o_embeddings': final_eva_blip3o,      # [N, 256, 4096]
            
            # Metadata
            'captions': all_captions,
            'keys': all_keys,
            'total_samples': total_samples,
            'data_files': data_files,  # Record source files
            
            # UPDATED Configuration for verification
            'config': {
                'clip_model': 'openai/clip-vit-large-patch14',
                'eva_model': 'BAAI/EVA-CLIP-8B',
                'clip_dim': 1024,  # ViT-L/14 dimension
                'eva_dim': actual_eva_dim,   # Actual EVA-CLIP dimension
                'tokens': 256,     # UPDATED: 16x16 grid tokens
                'grid_size': '16x16',  # Full resolution grid
                'pooling_method': 'none',  # NO POOLING
                'format_version': 'blip3o_256_tokens_v1_memory_optimized',
                'temp_directory': str(temp_dir),
                'num_shards': len(data_files),
                'memory_optimized': True,
            }
        }
        
        # Save final embeddings to TEMP directory
        print("💾 Saving final BLIP3-o compatible embeddings to TEMP...")
        with open(final_output_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        file_size_mb = final_output_path.stat().st_size / (1024 * 1024)
        
        print("=" * 80)
        print("✅ SUCCESS! BLIP3-o compatible embeddings created (256 TOKENS, MEMORY OPTIMIZED)!")
        print(f"📁 Saved to TEMP: {final_output_path}")
        print(f"📊 File size: {file_size_mb:.2f} MB")
        print(f"🔢 Total samples: {total_samples}")
        print(f"📐 CLIP BLIP3-o: {final_clip_blip3o.shape} (256 tokens, 1024-dim)")
        print(f"📐 EVA BLIP3-o:  {final_eva_blip3o.shape} (256 tokens, {actual_eva_dim}-dim)")
        print(f"📁 Source files: {len(data_files)} tar files")
        print(f"💾 Peak memory usage: {get_memory_usage():.1f} GB")
        print("=" * 80)
        print("🎉 Ready for BLIP3-o DiT training with 256 tokens!")
        print("Use this command:")
        print(f"python train_blip3o_dit.py --embeddings_path {final_output_path} --output_dir ./checkpoints/blip3o-dit")
        print("=" * 80)
        
        # Clean up checkpoint file
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"🗑️  Cleaned up checkpoint file")
        
        # Quick verification
        with open(final_output_path, 'rb') as f:
            loaded = pickle.load(f)
        
        print(f"✅ Verification: Successfully loaded {loaded['total_samples']} samples")
        print(f"   Keys: {list(loaded.keys())}")
        print(f"   CLIP shape: {loaded['clip_blip3o_embeddings'].shape}")
        print(f"   EVA shape: {loaded['eva_blip3o_embeddings'].shape}")
        print(f"   Format version: {loaded['config']['format_version']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during final processing: {e}")
        print("💾 Keeping checkpoint file for manual recovery")
        import traceback
        traceback.print_exc()
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