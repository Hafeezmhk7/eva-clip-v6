#!/usr/bin/env python3
"""
Verification and Recovery Script for BLIP3-o Embeddings
Usage: python verify_embeddings.py [embeddings_directory]

This script will:
1. Check what embedding files exist
2. Verify their integrity 
3. Show what's missing
4. Provide recovery suggestions
"""

import sys
import json
import pickle
from pathlib import Path
import torch

def check_embeddings_directory(embeddings_dir: Path):
    """Check what's in the embeddings directory"""
    print(f"🔍 Checking embeddings directory: {embeddings_dir}")
    
    if not embeddings_dir.exists():
        print(f"❌ Directory does not exist: {embeddings_dir}")
        return False
    
    # List all files
    all_files = list(embeddings_dir.iterdir())
    print(f"📁 Found {len(all_files)} files:")
    
    manifest_file = None
    shard_files = []
    other_files = []
    
    for file_path in all_files:
        if file_path.name == "embeddings_manifest.json":
            manifest_file = file_path
            print(f"   📋 {file_path.name} (manifest)")
        elif file_path.name.startswith("embeddings_shard_") and file_path.name.endswith(".pkl"):
            shard_files.append(file_path)
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   💾 {file_path.name} ({size_mb:.1f} MB)")
        else:
            other_files.append(file_path)
            print(f"   📄 {file_path.name}")
    
    return manifest_file, shard_files, other_files

def check_manifest(manifest_file: Path):
    """Check the manifest file"""
    print(f"\n📋 Checking manifest file: {manifest_file}")
    
    try:
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print(f"✅ Manifest loaded successfully")
        print(f"   Format version: {manifest.get('format_version', 'unknown')}")
        print(f"   Total shards (expected): {manifest.get('total_shards', 'unknown')}")
        print(f"   Total samples: {manifest.get('total_samples', 'unknown')}")
        print(f"   Total size: {manifest.get('total_size_mb', 'unknown')} MB")
        
        # Check shard details
        shards_info = manifest.get('shards', [])
        print(f"   Shard details: {len(shards_info)} entries")
        
        if 'verification' in manifest:
            verification = manifest['verification']
            print(f"   Verification info:")
            print(f"     Expected files: {verification.get('total_expected', 'unknown')}")
            print(f"     Valid files: {verification.get('valid_files', 'unknown')}")
            print(f"     Missing files: {verification.get('missing_files', 'unknown')}")
        
        return manifest
        
    except Exception as e:
        print(f"❌ Error reading manifest: {e}")
        return None

def verify_shard_file(shard_path: Path, expected_shard_idx: int = None):
    """Verify a single shard file"""
    try:
        # Check file size
        size_mb = shard_path.stat().st_size / (1024 * 1024)
        
        # Try to load the file
        with open(shard_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check required keys
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print(f"   ❌ {shard_path.name}: Missing keys {missing_keys}")
            return False
        
        # Check data shapes
        clip_emb = data['clip_blip3o_embeddings']
        eva_emb = data['eva_blip3o_embeddings']
        captions = data['captions']
        
        # Verify types
        if not isinstance(clip_emb, torch.Tensor):
            print(f"   ❌ {shard_path.name}: CLIP embeddings not a tensor")
            return False
            
        if not isinstance(eva_emb, torch.Tensor):
            print(f"   ❌ {shard_path.name}: EVA embeddings not a tensor")
            return False
        
        # Check shapes
        if clip_emb.shape[1] != 256:
            print(f"   ❌ {shard_path.name}: CLIP tokens {clip_emb.shape[1]} != 256")
            return False
            
        if eva_emb.shape[1] != 256:
            print(f"   ❌ {shard_path.name}: EVA tokens {eva_emb.shape[1]} != 256")
            return False
        
        # Check sample count consistency
        if clip_emb.shape[0] != eva_emb.shape[0] or clip_emb.shape[0] != len(captions):
            print(f"   ❌ {shard_path.name}: Sample count mismatch")
            print(f"      CLIP: {clip_emb.shape[0]}, EVA: {eva_emb.shape[0]}, Captions: {len(captions)}")
            return False
        
        sample_count = clip_emb.shape[0]
        
        # Check shard index if provided
        actual_shard_idx = data.get('shard_idx', 'unknown')
        if expected_shard_idx is not None and actual_shard_idx != expected_shard_idx:
            print(f"   ⚠️  {shard_path.name}: Shard index mismatch (expected {expected_shard_idx}, got {actual_shard_idx})")
        
        print(f"   ✅ {shard_path.name}: Valid ({sample_count} samples, {size_mb:.1f} MB)")
        print(f"      CLIP: {clip_emb.shape}, EVA: {eva_emb.shape}")
        print(f"      Shard index: {actual_shard_idx}")
        
        return {
            'valid': True,
            'samples': sample_count,
            'size_mb': size_mb,
            'shard_idx': actual_shard_idx,
            'clip_shape': clip_emb.shape,
            'eva_shape': eva_emb.shape
        }
        
    except Exception as e:
        print(f"   ❌ {shard_path.name}: Error - {e}")
        return False

def get_expected_shard_files(embeddings_dir: Path, manifest: dict = None):
    """Get list of expected shard files"""
    expected_files = []
    
    if manifest and 'total_shards' in manifest:
        total_shards = manifest['total_shards']
        print(f"📋 Expected {total_shards} shard files from manifest")
        
        for i in range(total_shards):
            filename = f"embeddings_shard_{i:05d}.pkl"
            expected_files.append(embeddings_dir / filename)
    
    else:
        # Try to guess from existing files
        existing_shards = list(embeddings_dir.glob("embeddings_shard_*.pkl"))
        if existing_shards:
            # Get the highest shard number
            max_shard = -1
            for shard_path in existing_shards:
                try:
                    # Extract shard number from filename
                    shard_num_str = shard_path.stem.split('_')[-1]
                    shard_num = int(shard_num_str)
                    max_shard = max(max_shard, shard_num)
                except:
                    continue
            
            if max_shard >= 0:
                print(f"📋 Guessing {max_shard + 1} shard files based on existing files")
                for i in range(max_shard + 1):
                    filename = f"embeddings_shard_{i:05d}.pkl"
                    expected_files.append(embeddings_dir / filename)
    
    return expected_files

def suggest_recovery(manifest: dict, existing_shards: list, expected_shards: list):
    """Suggest recovery actions"""
    print(f"\n💡 RECOVERY SUGGESTIONS:")
    
    existing_indices = set()
    for shard_path in existing_shards:
        try:
            shard_num_str = shard_path.stem.split('_')[-1]
            shard_num = int(shard_num_str)
            existing_indices.add(shard_num)
        except:
            continue
    
    expected_indices = set(range(len(expected_shards)))
    missing_indices = expected_indices - existing_indices
    
    if not missing_indices:
        print("✅ All expected shard files are present!")
        return
    
    print(f"❌ Missing {len(missing_indices)} shard files:")
    for idx in sorted(missing_indices):
        print(f"   • embeddings_shard_{idx:05d}.pkl")
    
    print(f"\n🔧 RECOVERY OPTIONS:")
    
    # Option 1: Re-run extraction for missing shards
    print("1. Re-run extraction for missing shards only:")
    print("   - The improved script can skip existing valid shards")
    print("   - Only missing shards will be processed")
    print("   - Run: python src/modules/extract_embeddings_g.py")
    
    # Option 2: Check if TAR files still exist
    if manifest and 'shards' in manifest:
        print("\n2. Check original TAR files:")
        for shard_info in manifest['shards']:
            shard_idx = shard_info.get('shard_idx', -1)
            if shard_idx in missing_indices:
                source_tar = shard_info.get('source_tar', 'unknown')
                print(f"   • Shard {shard_idx}: {source_tar}")
        print("   - Verify these TAR files still exist")
        print("   - Re-download if necessary")
    
    # Option 3: Check for disk space issues
    print("\n3. Check for disk space issues:")
    print("   - Verify sufficient space in embeddings directory")
    print("   - Check disk quotas (especially in home directory)")
    print("   - Consider using scratch-shared directories")
    
    # Option 4: Manual verification
    print("\n4. Manual verification:")
    print("   - Check job logs for error messages")
    print("   - Look for any partially created files")
    print("   - Verify permissions on embeddings directory")

def main():
    """Main verification function"""
    print("🔍 BLIP3-o Embeddings Verification and Recovery Tool")
    print("=" * 60)
    
    # Get embeddings directory
    if len(sys.argv) > 1:
        embeddings_dir = Path(sys.argv[1])
    else:
        # Try to auto-detect
        possible_dirs = [
            Path("/scratch-shared") / Path.home().name / "blip3o_workspace" / "embeddings" / "chunked_256_tokens",
            Path.home() / ".cache" / "blip3o_workspace" / "embeddings" / "chunked_256_tokens",
            Path("./embeddings/chunked_256_tokens"),
        ]
        
        embeddings_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                embeddings_dir = dir_path
                print(f"🎯 Auto-detected embeddings directory: {embeddings_dir}")
                break
        
        if embeddings_dir is None:
            print("❌ Could not find embeddings directory!")
            print("Usage: python verify_embeddings.py <embeddings_directory>")
            print("\nOr create embeddings first with:")
            print("  python src/modules/extract_embeddings_g.py")
            return 1
    
    # Check the directory
    result = check_embeddings_directory(embeddings_dir)
    if not result:
        return 1
    
    manifest_file, shard_files, other_files = result
    
    # Check manifest
    manifest = None
    if manifest_file:
        manifest = check_manifest(manifest_file)
    else:
        print("\n❌ No manifest file found!")
    
    # Get expected shard files
    expected_shards = get_expected_shard_files(embeddings_dir, manifest)
    
    print(f"\n💾 SHARD FILE VERIFICATION:")
    print(f"   Expected shards: {len(expected_shards)}")
    print(f"   Found shards: {len(shard_files)}")
    
    # Verify each existing shard
    valid_shards = []
    invalid_shards = []
    total_samples = 0
    total_size_mb = 0
    
    for shard_path in sorted(shard_files):
        # Extract expected shard index
        try:
            shard_num_str = shard_path.stem.split('_')[-1]
            expected_idx = int(shard_num_str)
        except:
            expected_idx = None
        
        result = verify_shard_file(shard_path, expected_idx)
        if result and result['valid']:
            valid_shards.append(shard_path)
            total_samples += result['samples']
            total_size_mb += result['size_mb']
        else:
            invalid_shards.append(shard_path)
    
    # Summary
    print(f"\n📊 VERIFICATION SUMMARY:")
    print(f"   Total expected: {len(expected_shards)}")
    print(f"   Valid shards: {len(valid_shards)}")
    print(f"   Invalid shards: {len(invalid_shards)}")
    print(f"   Missing shards: {len(expected_shards) - len(shard_files)}")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Total size: {total_size_mb:.1f} MB")
    
    # Recovery suggestions
    if len(valid_shards) != len(expected_shards):
        suggest_recovery(manifest, shard_files, expected_shards)
    else:
        print(f"\n🎉 SUCCESS! All embedding shards are present and valid!")
        print(f"   Ready for training with: {total_samples:,} samples")
        print(f"   Use: python train_blip3o_dit.py --chunked_embeddings_dir {embeddings_dir}")
    
    return 0 if len(valid_shards) == len(expected_shards) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)