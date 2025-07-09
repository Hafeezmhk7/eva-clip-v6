#!/usr/bin/env python3
"""
Test script to check if your embeddings file is compatible with BLIP3-o training.
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F

def test_embeddings_file(embeddings_path):
    """Test if embeddings file has the right format."""
    print(f"🧪 Testing embeddings file: {embeddings_path}")
    print("=" * 60)
    
    try:
        # Load the file
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        print("✅ File loaded successfully!")
        print(f"📊 Keys in data: {list(data.keys())}")
        
        # Check for required keys
        required_keys = ['eva_blip3o_embeddings', 'clip_blip3o_embeddings']
        alternative_keys = {
            'eva_blip3o_embeddings': ['eva_grid_embeddings', 'eva_embeddings', 'eva_blip3o_embeddings'],
            'clip_blip3o_embeddings': ['clip_grid_embeddings', 'clip_embeddings', 'clip_blip3o_embeddings']
        }
        
        eva_embeddings = None
        clip_embeddings = None
        
        # PRIORITIZE the BLIP3-o compatible 64-token versions
        eva_priority_keys = ['eva_blip3o_embeddings', 'eva_embeddings', 'eva_grid_embeddings']
        clip_priority_keys = ['clip_blip3o_embeddings', 'clip_embeddings', 'clip_grid_embeddings']
        
        # Find EVA embeddings (prioritize 64-token version)
        for key in eva_priority_keys:
            if key in data:
                eva_embeddings = data[key]
                print(f"📍 Found EVA embeddings with key: '{key}'")
                if key == 'eva_blip3o_embeddings':
                    print("   ✅ Using BLIP3-o compatible 64-token version")
                elif key == 'eva_grid_embeddings':
                    print("   ⚠️  Using 16x16 grid version - will need conversion")
                break
        
        # Find CLIP embeddings (prioritize 64-token version)
        for key in clip_priority_keys:
            if key in data:
                clip_embeddings = data[key]
                print(f"📍 Found CLIP embeddings with key: '{key}'")
                if key == 'clip_blip3o_embeddings':
                    print("   ✅ Using BLIP3-o compatible 64-token version")
                elif key == 'clip_grid_embeddings':
                    print("   ⚠️  Using 16x16 grid version - will need conversion")
                break
        
        if eva_embeddings is None:
            print("❌ No EVA embeddings found!")
            print(f"   Looking for keys: {alternative_keys['eva_blip3o_embeddings']}")
            return False
            
        if clip_embeddings is None:
            print("❌ No CLIP embeddings found!")
            print(f"   Looking for keys: {alternative_keys['clip_blip3o_embeddings']}")
            return False
        
        # Convert to tensors if needed
        if isinstance(eva_embeddings, np.ndarray):
            eva_embeddings = torch.from_numpy(eva_embeddings)
        if isinstance(clip_embeddings, np.ndarray):
            clip_embeddings = torch.from_numpy(clip_embeddings)
        
        # Check shapes
        print(f"📐 EVA embeddings shape: {eva_embeddings.shape}")
        print(f"📐 CLIP embeddings shape: {clip_embeddings.shape}")
        
        # Validate shapes
        eva_shape = eva_embeddings.shape
        clip_shape = clip_embeddings.shape
        
        issues = []
        
        # Check if we have the same number of samples
        if eva_shape[0] != clip_shape[0]:
            issues.append(f"❌ Sample count mismatch: EVA {eva_shape[0]} vs CLIP {clip_shape[0]}")
        else:
            print(f"✅ Sample count matches: {eva_shape[0]} samples")
        
        # Check EVA dimensions
        if len(eva_shape) == 3:
            if eva_shape[1] == 64 and eva_shape[2] == 1280:
                print("✅ EVA embeddings have correct BLIP3-o format: [N, 64, 1280]")
            elif eva_shape[1] == 64:
                print(f"⚠️  EVA embeddings have 64 tokens but wrong dimension: [N, 64, {eva_shape[2]}] (expected 1280)")
                print("   This might be due to model version differences")
            else:
                issues.append(f"❌ EVA shape should be [N, 64, 1280], got [N, {eva_shape[1]}, {eva_shape[2]}]")
        elif len(eva_shape) == 4:
            print(f"⚠️  EVA embeddings are in 16x16 grid format: [N, 16, 16, {eva_shape[3]}]")
            print("   Can be converted to BLIP3-o format using 2x2 pooling")
            if eva_shape[3] != 1280:
                print(f"   ⚠️  Dimension is {eva_shape[3]} instead of expected 1280")
        else:
            issues.append(f"❌ EVA embeddings should be 3D [N, 64, 1280] or 4D [N, 16, 16, 1280], got {len(eva_shape)}D")
        
        # Check CLIP dimensions
        if len(clip_shape) == 3:
            if clip_shape[1] == 64 and clip_shape[2] == 768:
                print("✅ CLIP embeddings have correct BLIP3-o format: [N, 64, 768]")
            elif clip_shape[1] == 64:
                print(f"⚠️  CLIP embeddings have 64 tokens but wrong dimension: [N, 64, {clip_shape[2]}] (expected 768)")
                print("   This might be due to model version differences")
            else:
                issues.append(f"❌ CLIP shape should be [N, 64, 768], got [N, {clip_shape[1]}, {clip_shape[2]}]")
        elif len(clip_shape) == 4:
            print(f"⚠️  CLIP embeddings are in 16x16 grid format: [N, 16, 16, {clip_shape[3]}]")
            print("   Can be converted to BLIP3-o format using 2x2 pooling")
            if clip_shape[3] != 768:
                print(f"   ⚠️  Dimension is {clip_shape[3]} instead of expected 768")
        else:
            issues.append(f"❌ CLIP embeddings should be 3D [N, 64, 768] or 4D [N, 16, 16, 768], got {len(clip_shape)}D")
        
        # Check data statistics
        print(f"📈 EVA embeddings stats:")
        print(f"   Mean: {eva_embeddings.mean().item():.4f}")
        print(f"   Std: {eva_embeddings.std().item():.4f}")
        print(f"   Min: {eva_embeddings.min().item():.4f}")
        print(f"   Max: {eva_embeddings.max().item():.4f}")
        
        print(f"📈 CLIP embeddings stats:")
        print(f"   Mean: {clip_embeddings.mean().item():.4f}")
        print(f"   Std: {clip_embeddings.std().item():.4f}")
        print(f"   Min: {clip_embeddings.min().item():.4f}")
        print(f"   Max: {clip_embeddings.max().item():.4f}")
        
        # Check other metadata
        if 'captions' in data:
            print(f"📝 Found {len(data['captions'])} captions")
        if 'keys' in data:
            print(f"🔑 Found {len(data['keys'])} keys")
        
        print("=" * 60)
        
        # Check if we have BLIP3-o versions available
        has_blip3o_versions = 'eva_blip3o_embeddings' in data and 'clip_blip3o_embeddings' in data
        if has_blip3o_versions:
            print("📋 IMPORTANT: Your file contains BOTH 16x16 grids AND 64-token versions!")
            print("   The test picked up the 16x16 versions, but 64-token versions exist:")
            if 'eva_blip3o_embeddings' in data:
                blip3o_eva_shape = data['eva_blip3o_embeddings'].shape if hasattr(data['eva_blip3o_embeddings'], 'shape') else np.array(data['eva_blip3o_embeddings']).shape
                print(f"   EVA BLIP3-o shape: {blip3o_eva_shape}")
            if 'clip_blip3o_embeddings' in data:
                blip3o_clip_shape = data['clip_blip3o_embeddings'].shape if hasattr(data['clip_blip3o_embeddings'], 'shape') else np.array(data['clip_blip3o_embeddings']).shape
                print(f"   CLIP BLIP3-o shape: {blip3o_clip_shape}")
        
        if issues:
            print("⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
            
            if has_blip3o_versions:
                print("\n🎯 GOOD NEWS: You have BLIP3-o compatible versions!")
                print("   The compatible file creation will use those instead.")
            
            print("\n💡 SUGGESTED FIXES:")
            print("   1. Create a compatible file with proper keys and format")
            print("   2. Run with --debug mode first to test with current data")
            print("   3. Check your embedding extraction model versions")
            
            # Specific dimension issue warnings
            if eva_shape[-1] != 1280:
                print(f"   4. ⚠️  EVA dimension is {eva_shape[-1]} instead of 1280")
                print("      - Check if you're using EVA-CLIP-8B (should be 1280)")
                print("      - Current dimension suggests a different EVA model")
                
            if clip_shape[-1] != 768:
                print(f"   5. ⚠️  CLIP dimension is {clip_shape[-1]} instead of 768") 
                print("      - Check if you're using CLIP ViT-L/14 (should be 768)")
                print("      - Current dimension suggests a different CLIP model")
            
            return False
        else:
            print("🎉 SUCCESS! Your embeddings file is compatible with BLIP3-o training!")
            print(f"✅ Ready to train with {eva_shape[0]} samples")
            return True
            
    except Exception as e:
        print(f"❌ Error loading embeddings file: {e}")
        import traceback
        print("Full error:")
        traceback.print_exc()
        return False

def create_compatible_embeddings_file(input_path, output_path):
    """Create a compatible embeddings file if needed."""
    print(f"🔧 Creating compatible embeddings file...")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create new data structure
    new_data = {}
    
    # Function to pool 16x16 grids to 8x8 (64 tokens)
    def pool_to_64_tokens(grid_embeddings):
        """Pool 16x16 grids to 8x8 (64 tokens) using 2x2 average pooling"""
        if isinstance(grid_embeddings, np.ndarray):
            grid_embeddings = torch.from_numpy(grid_embeddings)
        
        if len(grid_embeddings.shape) == 4:  # [N, 16, 16, D]
            batch_size, grid_h, grid_w, hidden_dim = grid_embeddings.shape
            
            if grid_h == 16 and grid_w == 16:
                # 2x2 average pooling: 16x16 -> 8x8
                grid_for_pooling = grid_embeddings.permute(0, 3, 1, 2)  # [B, D, 16, 16]
                pooled = torch.nn.functional.avg_pool2d(grid_for_pooling, kernel_size=2, stride=2)  # [B, D, 8, 8]
                result = pooled.permute(0, 2, 3, 1).reshape(batch_size, 64, hidden_dim)  # [B, 64, D]
                return result
            elif grid_h * grid_w == 64:
                return grid_embeddings.reshape(batch_size, 64, hidden_dim)
        elif len(grid_embeddings.shape) == 3:  # Already [N, 64, D]
            return grid_embeddings
        
        return grid_embeddings
    
    # Find and convert embeddings
    eva_embeddings = None
    clip_embeddings = None
    
    # Check if we already have the BLIP3-o compatible versions
    if 'eva_blip3o_embeddings' in data and 'clip_blip3o_embeddings' in data:
        print("✅ BLIP3-o compatible embeddings already exist")
        eva_embeddings = data['eva_blip3o_embeddings']
        clip_embeddings = data['clip_blip3o_embeddings']
    else:
        print("🔄 Converting grid embeddings to BLIP3-o format...")
        
        # Find EVA embeddings and convert
        eva_keys = ['eva_grid_embeddings', 'eva_embeddings', 'eva_features']
        for key in eva_keys:
            if key in data:
                print(f"   Converting EVA from key: '{key}'")
                eva_embeddings = pool_to_64_tokens(data[key])
                break
        
        # Find CLIP embeddings and convert
        clip_keys = ['clip_grid_embeddings', 'clip_embeddings', 'clip_features']
        for key in clip_keys:
            if key in data:
                print(f"   Converting CLIP from key: '{key}'")
                clip_embeddings = pool_to_64_tokens(data[key])
                break
    
    if eva_embeddings is None or clip_embeddings is None:
        raise ValueError("Could not find EVA or CLIP embeddings in the data")
    
    # Store in new format with correct keys
    new_data['eva_blip3o_embeddings'] = eva_embeddings
    new_data['clip_blip3o_embeddings'] = clip_embeddings
    
    # Copy other metadata
    for key in ['captions', 'keys', 'config', 'total_samples']:
        if key in data:
            new_data[key] = data[key]
    
    # Add default metadata if missing
    num_samples = eva_embeddings.shape[0]
    if 'captions' not in new_data:
        new_data['captions'] = [f"sample_{i}" for i in range(num_samples)]
    
    if 'keys' not in new_data:
        new_data['keys'] = [f"key_{i}" for i in range(num_samples)]
    
    if 'total_samples' not in new_data:
        new_data['total_samples'] = num_samples
    
    # Update config
    if 'config' not in new_data:
        new_data['config'] = {}
    
    new_data['config'].update({
        'eva_dim': eva_embeddings.shape[-1],
        'clip_dim': clip_embeddings.shape[-1],
        'blip3o_tokens': 64,
        'converted_from_grids': True,
    })
    
    # Save compatible file
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)
    
    print(f"✅ Compatible embeddings saved to: {output_path}")
    print(f"   EVA shape: {eva_embeddings.shape}")
    print(f"   CLIP shape: {clip_embeddings.shape}")
    
    return output_path

if __name__ == "__main__":
    import sys
    import os
    
    # Test your embeddings file - try common locations
    possible_paths = [
        "embeddings/fixed_grid_embeddings.pkl",
        "data/embeddings/fixed_grid_embeddings.pkl", 
        "embeddings/blip3o_grid_embeddings.pkl",
        "data/embeddings/blip3o_grid_embeddings.pkl"
    ]
    
    embeddings_path = None
    
    if len(sys.argv) > 1:
        embeddings_path = sys.argv[1]
    else:
        # Auto-detect embeddings file
        for path in possible_paths:
            if os.path.exists(path):
                embeddings_path = path
                print(f"🔍 Found embeddings file: {path}")
                break
        
        if embeddings_path is None:
            print("❌ No embeddings file found in common locations:")
            for path in possible_paths:
                print(f"   - {path}")
            print("\nUsage: python test_embeddings.py <path_to_embeddings.pkl>")
            sys.exit(1)
    
    print(f"Testing embeddings file: {embeddings_path}")
    
    success = test_embeddings_file(embeddings_path)
    
    if not success:
        print("\n🔧 Would you like to try creating a compatible file? (y/n)")
        response = input().lower()
        if response == 'y':
            output_path = embeddings_path.replace('.pkl', '_compatible.pkl')
            create_compatible_embeddings_file(embeddings_path, output_path)
            print(f"\n🧪 Testing the new compatible file...")
            test_embeddings_file(output_path)