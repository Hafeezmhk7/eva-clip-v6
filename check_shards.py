#!/usr/bin/env python3
"""
Quick script to check what dataset shards are available
Place this file as: check_shards.py

This script finds and validates dataset shards for BLIP3-o embedding extraction.
"""

import os
import sys
from pathlib import Path
import glob

def get_temp_directory():
    """Get the temp directory path for Snellius or other systems"""
    # Check for Snellius temp directory
    if "TMPDIR" in os.environ:
        temp_dir = Path(os.environ["TMPDIR"])
    elif "SCRATCH_SHARED" in os.environ:
        temp_dir = Path(os.environ["SCRATCH_SHARED"])
    else:
        # Fallback to project embeddings directory
        temp_dir = Path(__file__).parent / "embeddings"
    
    return temp_dir

def find_dataset_shards():
    """Find all available dataset shards"""
    print("🔍 Searching for BLIP3-o dataset shards...")
    print("=" * 50)
    
    # Search locations in order of preference
    search_locations = []
    
    # 1. Environment-based locations
    if "TMPDIR" in os.environ:
        tmpdir = Path(os.environ["TMPDIR"])
        search_locations.append(tmpdir / "blip3o_data")
        search_locations.append(tmpdir)
        print(f"📁 TMPDIR: {tmpdir}")
    
    if "SCRATCH_SHARED" in os.environ:
        user = os.environ.get("USER", "user")
        scratch = Path(os.environ["SCRATCH_SHARED"]) / user
        search_locations.append(scratch / "blip3o_data")
        search_locations.append(scratch)
        print(f"📁 SCRATCH_SHARED: {scratch}")
    
    # 2. Temp directory
    temp_dir = get_temp_directory()
    search_locations.append(temp_dir / "blip3o_data")
    search_locations.append(temp_dir / "data")
    search_locations.append(temp_dir)
    print(f"📁 Temp directory: {temp_dir}")
    
    # 3. Project directory
    project_root = Path(__file__).parent
    search_locations.append(project_root / "data")
    search_locations.append(project_root)
    print(f"📁 Project root: {project_root}")
    
    print(f"\n🔍 Searching {len(search_locations)} locations...")
    
    all_found_shards = {}  # location -> list of shards
    
    for location in search_locations:
        if not location.exists():
            print(f"   ⏭️  Skipping (not found): {location}")
            continue
        
        print(f"   📂 Checking: {location}")
        
        # Look for downloaded_shards.txt first
        shard_list_file = location / "downloaded_shards.txt"
        if shard_list_file.exists():
            print(f"      📋 Found shard list: {shard_list_file}")
            try:
                with open(shard_list_file, 'r') as f:
                    listed_files = [line.strip() for line in f if line.strip()]
                
                valid_files = []
                for file_path in listed_files:
                    if Path(file_path).exists():
                        valid_files.append(file_path)
                
                if valid_files:
                    all_found_shards[str(location)] = valid_files
                    print(f"      ✅ {len(valid_files)} shards from list")
                    continue
            except Exception as e:
                print(f"      ⚠️  Could not read shard list: {e}")
        
        # Look for .tar files directly
        tar_files = list(location.glob("*.tar"))
        if tar_files:
            tar_files.sort()
            all_found_shards[str(location)] = [str(f) for f in tar_files]
            print(f"      ✅ {len(tar_files)} .tar files found")
        else:
            print(f"      ❌ No .tar files found")
    
    return all_found_shards

def validate_shards(shard_files):
    """Validate and show details of shard files"""
    print(f"\n📊 Validating {len(shard_files)} shard files...")
    print("-" * 60)
    
    valid_shards = []
    total_size_gb = 0
    
    for i, shard_file in enumerate(shard_files):
        shard_path = Path(shard_file)
        
        if not shard_path.exists():
            print(f"   {i:2d}. ❌ MISSING: {shard_path.name}")
            continue
        
        try:
            size_bytes = shard_path.stat().st_size
            size_gb = size_bytes / (1024**3)
            total_size_gb += size_gb
            
            # Extract shard number from filename (e.g., "00001.tar" -> 1)
            shard_num = "?"
            try:
                name_parts = shard_path.stem.split('.')
                if name_parts[0].isdigit():
                    shard_num = int(name_parts[0])
            except:
                pass
            
            print(f"   {i:2d}. ✅ Shard {shard_num:>3}: {shard_path.name:<15} ({size_gb:6.2f} GB)")
            valid_shards.append(shard_file)
            
        except Exception as e:
            print(f"   {i:2d}. ❌ ERROR: {shard_path.name} - {e}")
    
    print("-" * 60)
    print(f"📊 SUMMARY:")
    print(f"   Valid shards: {len(valid_shards)}")
    print(f"   Total size: {total_size_gb:.2f} GB")
    
    if total_size_gb > 0:
        # Estimate samples (rough: 400 samples per MB)
        estimated_samples = int(total_size_gb * 400000)
        print(f"   Estimated samples: ~{estimated_samples:,}")
        
        # Estimate embedding size (5MB per sample for 256 tokens)
        estimated_output_gb = (estimated_samples * 5) / 1024
        print(f"   Estimated output: ~{estimated_output_gb:.1f} GB")
    
    return valid_shards, total_size_gb

def show_extraction_readiness(num_shards, total_size_gb):
    """Show whether ready for extraction"""
    print(f"\n🎯 EXTRACTION READINESS:")
    print("=" * 50)
    
    if num_shards == 0:
        print("❌ NOT READY - No dataset shards found")
        print("\n📋 To download shards:")
        print("   python src/data_hand/download_data.py --shards 0 1 2 3 4 5 6 7 8 9")
        return False
    
    elif num_shards < 3:
        print(f"⚠️  PARTIALLY READY - Only {num_shards} shards found")
        print("   You can proceed, but consider downloading more shards for better training data")
    
    elif num_shards >= 10:
        print(f"✅ EXCELLENT - {num_shards} shards found (full dataset)")
    
    else:
        print(f"✅ READY - {num_shards} shards found")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Check requirements: python check_requirements.py")
    print(f"   2. Extract embeddings: sbatch job_scripts/extract_emb.job")
    print(f"   3. Or test locally: python src/modules/extract_embeddings_g.py")
    
    print(f"\n⏱️  Estimated processing time: ~{total_size_gb / 10:.1f} hours")
    print(f"💾 Expected output size: ~{(total_size_gb * 400000 * 5) / (1024 * 1024):.1f} GB")
    
    return True

def main():
    """Main function to check dataset shards"""
    print("🔍 BLIP3-o Dataset Shard Checker")
    print("=" * 50)
    
    # Find all shards
    all_shards = find_dataset_shards()
    
    if not all_shards:
        print("\n❌ No dataset shards found in any location!")
        print("\n📋 To download shards:")
        print("   python src/data_hand/download_data.py --shards 0 1 2 3 4 5 6 7 8 9")
        return 1
    
    # Show findings
    print(f"\n📊 FOUND SHARDS IN {len(all_shards)} LOCATIONS:")
    print("=" * 60)
    
    best_location = None
    best_shards = []
    most_shards = 0
    
    for location, shards in all_shards.items():
        print(f"\n📁 {location}:")
        valid_shards, size_gb = validate_shards(shards)
        
        if len(valid_shards) > most_shards:
            most_shards = len(valid_shards)
            best_location = location
            best_shards = valid_shards
    
    print(f"\n🎯 BEST LOCATION: {best_location}")
    print(f"   Using {len(best_shards)} shards")
    
    # Show readiness
    show_extraction_readiness(len(best_shards), 
                            sum(Path(f).stat().st_size for f in best_shards) / (1024**3))
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)