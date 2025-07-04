#!/usr/bin/env python3
"""
Explore the downloaded BLIP3o dataset
Place this as: explore_data.py (or run directly)
"""

import tarfile
import io
from PIL import Image
import json
import os

def explore_blip3o_shard(tar_path="./data/00000.tar", num_samples=5):
    """
    Explore the contents of a BLIP3o shard
    
    Args:
        tar_path: Path to the downloaded .tar file
        num_samples: Number of samples to show
    """
    
    print("🔍 Exploring BLIP3o Dataset Shard")
    print("=" * 50)
    print(f"📁 File: {tar_path}")
    
    if not os.path.exists(tar_path):
        print(f"❌ File not found: {tar_path}")
        return
    
    # Get file size
    file_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"📊 Size: {file_size_mb:.1f} MB")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # Get all members (files inside the tar)
            members = tar.getmembers()
            print(f"📦 Total files in shard: {len(members)}")
            
            # Analyze file types
            file_types = {}
            sample_keys = set()
            
            for member in members:
                if member.isfile():
                    # Extract file extension
                    ext = member.name.split('.')[-1]
                    file_types[ext] = file_types.get(ext, 0) + 1
                    
                    # Extract sample key (filename without extension)
                    key = '.'.join(member.name.split('.')[:-1])
                    sample_keys.add(key)
            
            print(f"\n📋 File types found:")
            for ext, count in sorted(file_types.items()):
                print(f"  .{ext}: {count:,} files")
            
            print(f"\n🔢 Unique samples: {len(sample_keys):,}")
            
            # Show some sample data
            print(f"\n🔬 Examining first {num_samples} samples:")
            print("-" * 50)
            
            samples_shown = 0
            sample_keys_list = sorted(list(sample_keys))[:num_samples]
            
            for sample_key in sample_keys_list:
                if samples_shown >= num_samples:
                    break
                    
                print(f"\n📝 Sample {samples_shown + 1}: {sample_key}")
                
                # Try to find image and text for this sample
                image_exts = ['jpg', 'png', 'jpeg']
                text_exts = ['txt', 'caption']
                
                # Look for image
                image_data = None
                for ext in image_exts:
                    try:
                        image_member = tar.getmember(f"{sample_key}.{ext}")
                        image_data = tar.extractfile(image_member)
                        if image_data:
                            image = Image.open(image_data)
                            print(f"  🖼️ Image: {image.size[0]}x{image.size[1]} pixels ({ext.upper()})")
                            break
                    except KeyError:
                        continue
                
                # Look for caption
                caption_text = None
                for ext in text_exts:
                    try:
                        text_member = tar.getmember(f"{sample_key}.{ext}")
                        text_data = tar.extractfile(text_member)
                        if text_data:
                            caption_text = text_data.read().decode('utf-8').strip()
                            break
                    except KeyError:
                        continue
                
                if caption_text:
                    # Truncate long captions for display
                    display_caption = caption_text[:100] + "..." if len(caption_text) > 100 else caption_text
                    print(f"  💬 Caption: \"{display_caption}\"")
                    print(f"  📏 Caption length: {len(caption_text)} characters")
                
                samples_shown += 1
            
            # Summary statistics
            print(f"\n📈 Dataset Summary:")
            print(f"  • Total samples: ~{len(sample_keys):,}")
            print(f"  • File size: {file_size_mb:.1f} MB")
            print(f"  • Avg size per sample: {file_size_mb/len(sample_keys):.3f} MB")
            
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")
        return
    
    print(f"\n✅ Exploration complete!")
    
    # Next steps
    print(f"\n🚀 Next Steps:")
    print(f"  1. 🔄 Cache EVA-CLIP and CLIP features:")
    print(f"     python scripts/cache_features.py --shards 0")
    print(f"  2. 🧪 Test the dataset loading:")
    print(f"     python src/data/dataset.py")
    print(f"  3. 📖 Read Lumina-Next paper")
    print(f"  4. 🏗️ Implement flow matching pipeline")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore BLIP3o dataset")
    parser.add_argument("--tar_path", default="./data/00000.tar", help="Path to tar file")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to show")
    
    args = parser.parse_args()
    
    explore_blip3o_shard(args.tar_path, args.samples)