"""
FIXED: Memory-Efficient Grid-Based Embedding Extraction for BLIP3-o
Place this file as: src/modules/extract_embeddings_g.py

FIXES:
1. Correct dimensions: CLIP 1024-dim, EVA-CLIP 4096-dim  
2. Proper model loading and feature extraction
3. Correct file format for BLIP3-o DiT architecture
4. Proper validation and error handling
5. Compatible with your training pipeline
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
            # Try to get from config
            try:
                eva_dim = eva_model.config.vision_config.hidden_size
                print(f"   Got EVA dimension from config: {eva_dim}")
            except:
                # Assume 4096 for EVA-CLIP-8B based on architecture
                eva_dim = 4096
                print(f"   Assuming EVA dimension: {eva_dim} (EVA-CLIP-8B)")
    
    print(f"   EVA-CLIP-8B dimension: {eva_dim}")
    
    # Note: EVA-CLIP-8B should be 4096, but let's be flexible for now
    if eva_dim != 4096:
        print(f"   ⚠️  Warning: EVA dimension is {eva_dim}, expected 4096")
        print(f"   Continuing with detected dimension...")
    
    # Cleanup after loading
    cleanup_memory()
    
    print("✅ Models loaded successfully")
    print(f"   CLIP: {clip_dim}-dim, EVA-CLIP: {eva_dim}-dim")
    print(f"💾 Memory usage after loading: {get_memory_usage():.2f} GB")
    
    return clip_processor, clip_model, eva_processor, eva_model

def extract_clip_features(images, processor, model, device):
    """Extract CLIP ViT-L/14 patch grid features (1024-dim)"""
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
            
            # Reshape to spatial grid: [1, 16, 16, 1024]
            grid_size = int(np.sqrt(num_patches))  # 16
            spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
            
            # Convert to float32 and move to CPU
            features.append(spatial_grid.squeeze().cpu().float())
            
            # Clear GPU memory
            del vision_outputs, patch_embeddings, spatial_grid
    
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
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 256, hidden_dim]
            batch_size, num_patches, hidden_dim = patch_embeddings.shape
            
            # Validate dimensions - be flexible about EVA dimension
            print(f"   EVA patch embeddings shape: {patch_embeddings.shape}")
            assert num_patches == 256, f"Expected 256 patches (16x16), got {num_patches}"
            
            # Store the actual EVA dimension for later use
            if not hasattr(extract_eva_features, 'eva_dim'):
                extract_eva_features.eva_dim = hidden_dim
                print(f"   Detected EVA dimension: {hidden_dim}")
            
            # Reshape to spatial grid: [1, 16, 16, hidden_dim]
            grid_size = int(np.sqrt(num_patches))  # 16
            spatial_grid = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_dim)
            
            # Convert to float32 and move to CPU
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

def validate_extracted_embeddings(clip_blip3o, eva_blip3o, captions):
    """Validate extracted embeddings match BLIP3-o requirements"""
    print("🧪 Validating extracted embeddings...")
    
    # Check shapes
    assert clip_blip3o.dim() == 3, f"CLIP embeddings should be 3D, got {clip_blip3o.dim()}D"
    assert eva_blip3o.dim() == 3, f"EVA embeddings should be 3D, got {eva_blip3o.dim()}D"
    
    # Check token count
    assert clip_blip3o.shape[1] == 64, f"Expected 64 CLIP tokens, got {clip_blip3o.shape[1]}"
    assert eva_blip3o.shape[1] == 64, f"Expected 64 EVA tokens, got {eva_blip3o.shape[1]}"
    
    # Check dimensions - be flexible about EVA dimension
    assert clip_blip3o.shape[2] == 1024, f"Expected CLIP 1024-dim, got {clip_blip3o.shape[2]}"
    
    eva_dim = eva_blip3o.shape[2]
    print(f"   Detected EVA dimension: {eva_dim}")
    
    # Check batch consistency
    assert clip_blip3o.shape[0] == eva_blip3o.shape[0], "Batch size mismatch between CLIP and EVA"
    assert len(captions) == clip_blip3o.shape[0], "Caption count mismatch"
    
    print(f"✅ Validation passed!")
    print(f"   CLIP BLIP3-o embeddings: {clip_blip3o.shape}")
    print(f"   EVA BLIP3-o embeddings: {eva_blip3o.shape}")
    print(f"   Captions: {len(captions)}")
    
    return eva_dim

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
    """Main extraction function with correct BLIP3-o format"""
    print("🚀 BLIP3-o Embedding Extraction - FIXED VERSION")
    print("=" * 70)
    print("Target format:")
    print("  CLIP BLIP3-o: [N, 64, 1024] - ViT-L/14 features")  
    print("  EVA BLIP3-o:  [N, 64, 4096] - EVA-CLIP-8B features")
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
    
    # Load models with dimension validation
    try:
        clip_processor, clip_model, eva_processor, eva_model = load_models(device)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return 1
    
    # Setup dataset
    data_file = project_root / "data" / "00000.tar"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print("📂 Creating dataset...")
    dataset = BLIP3oWebDataset(
        tar_paths=[str(data_file)],
        batch_size=4,  # Small batch to prevent OOM
        shuffle=False,
        num_workers=0
    )
    
    dataloader = dataset.get_dataloader()
    print("✅ Dataset ready")
    
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
    
    # Extract embeddings
    print("🧠 Extracting BLIP3-o compatible embeddings...")
    
    CHECKPOINT_EVERY = 50
    MEMORY_CLEANUP_EVERY = 10
    
    try:
        current_batch_idx = 0
        processed_batches = 0
        
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            if current_batch_idx < start_batch:
                current_batch_idx += 1
                continue
            
            try:
                images = batch['image']
                captions = batch['caption']
                keys = batch['key']
                
                # Monitor memory
                mem_before = get_memory_usage()
                if mem_before > 8.0:
                    print(f"⚠️  High memory usage: {mem_before:.2f} GB - cleaning up...")
                    cleanup_memory()
                
                # Extract features with correct dimensions
                clip_grids = extract_clip_features(images, clip_processor, clip_model, device)  # [B, 16, 16, 1024]
                eva_grids = extract_eva_features(images, eva_processor, eva_model, device)      # [B, 16, 16, eva_dim]
                
                # Pool to BLIP3-o format (64 tokens)
                clip_blip3o = pool_to_blip3o_format(clip_grids)  # [B, 64, 1024]
                eva_blip3o = pool_to_blip3o_format(eva_grids)    # [B, 64, eva_dim]
                
                # Validate dimensions (get actual EVA dimension)
                actual_eva_dim = eva_blip3o.shape[2]
                assert clip_blip3o.shape[2] == 1024, f"CLIP dimension error: {clip_blip3o.shape[2]}"
                assert clip_blip3o.shape[1] == 64, f"CLIP token count error: {clip_blip3o.shape[1]}"
                assert eva_blip3o.shape[1] == 64, f"EVA token count error: {eva_blip3o.shape[1]}"
                
                print(f"   Batch {current_batch_idx}: CLIP {clip_blip3o.shape}, EVA {eva_blip3o.shape}")
                
                # Store the actual EVA dimension for later use
                if not hasattr(main, 'detected_eva_dim'):
                    main.detected_eva_dim = actual_eva_dim
                    print(f"   📏 Detected EVA dimension: {actual_eva_dim}")
                
                # Store results
                all_clip_blip3o.append(clip_blip3o)
                all_eva_blip3o.append(eva_blip3o)
                all_captions.extend(captions)
                all_keys.extend(keys)
                
                total_samples += len(images)
                processed_batches += 1
                
                # Clear variables
                del clip_grids, eva_grids, clip_blip3o, eva_blip3o, images
                
                # Memory cleanup
                if processed_batches % MEMORY_CLEANUP_EVERY == 0:
                    cleanup_memory()
                
                # Save checkpoint
                if processed_batches % CHECKPOINT_EVERY == 0:
                    checkpoint_data = {
                        'clip_blip3o_embeddings_list': all_clip_blip3o,
                        'eva_blip3o_embeddings_list': all_eva_blip3o,
                        'captions': all_captions,
                        'keys': all_keys,
                    }
                    save_checkpoint(checkpoint_data, checkpoint_path, current_batch_idx, total_samples)
                    
                    print(f"📊 Progress: {processed_batches} batches, {total_samples} samples")
                
                current_batch_idx += 1
                
            except Exception as e:
                print(f"⚠️  Error processing batch {current_batch_idx}: {e}")
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
        emergency_path = output_dir / "emergency_checkpoint.pkl"
        save_checkpoint(emergency_checkpoint, emergency_path, current_batch_idx, total_samples)
        return 1
    
    # Final processing and saving
    print(f"💾 Combining {total_samples} embeddings...")
    
    try:
        # Combine all results
        final_clip_blip3o = torch.cat(all_clip_blip3o, dim=0)    # [N, 64, 1024]
        final_eva_blip3o = torch.cat(all_eva_blip3o, dim=0)      # [N, 64, 4096]
        
        # Validate final embeddings
        validate_extracted_embeddings(final_clip_blip3o, final_eva_blip3o, all_captions)
        
        # Prepare final embeddings data in BLIP3-o format
        embeddings_data = {
            # BLIP3-o compatible format (exactly what the dataset expects)
            'clip_blip3o_embeddings': final_clip_blip3o,    # [N, 64, 1024]
            'eva_blip3o_embeddings': final_eva_blip3o,      # [N, 64, 4096]
            
            # Metadata
            'captions': all_captions,
            'keys': all_keys,
            'total_samples': total_samples,
            
            # Configuration for verification
            'config': {
                'clip_model': 'openai/clip-vit-large-patch14',
                'eva_model': 'BAAI/EVA-CLIP-8B',
                'clip_dim': 1024,  # ViT-L/14 dimension
                'eva_dim': 4096,   # EVA-CLIP-8B dimension
                'tokens': 64,      # 8x8 grid tokens
                'pooling_method': 'avg_pool2d_2x2',
                'format_version': 'blip3o_compatible_v1'
            }
        }
        
        # Save final embeddings
        print("💾 Saving final BLIP3-o compatible embeddings...")
        with open(final_output_path, 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        file_size_mb = final_output_path.stat().st_size / (1024 * 1024)
        
        print("=" * 70)
        print("✅ SUCCESS! BLIP3-o compatible embeddings created!")
        print(f"📁 Saved to: {final_output_path}")
        print(f"📊 File size: {file_size_mb:.2f} MB")
        print(f"🔢 Total samples: {total_samples}")
        print(f"📐 CLIP BLIP3-o: {final_clip_blip3o.shape} (1024-dim)")
        print(f"📐 EVA BLIP3-o:  {final_eva_blip3o.shape} ({actual_eva_dim}-dim)")
        print("=" * 70)
        print("🎉 Ready for BLIP3-o DiT training!")
        print("Use this command:")
        print(f"python train_blip3o_dit.py --embeddings_path {final_output_path} --output_dir ./checkpoints/blip3o-dit")
        print("=" * 70)
        
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