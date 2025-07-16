#!/usr/bin/env python3
"""
Download MS-COCO 2017 Validation Dataset for BLIP3-o Evaluation
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import zipfile
import json
from tqdm import tqdm

def download_with_progress(url: str, filename: str):
    """Download file with progress bar."""
    
    class ProgressBar:
        def __init__(self):
            self.pbar = None

        def __call__(self, block_num, block_size, total_size):
            if not self.pbar:
                self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename)
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.close()

    urllib.request.urlretrieve(url, filename, ProgressBar())

def download_coco_val2017(data_dir: str = "./data/coco"):
    """Download MS-COCO 2017 validation dataset."""
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading MS-COCO 2017 Validation Dataset")
    print("=" * 50)
    print(f"Download directory: {data_dir.absolute()}")
    
    # URLs for COCO 2017 validation
    files_to_download = [
        {
            "url": "http://images.cocodataset.org/zips/val2017.zip",
            "filename": "val2017.zip",
            "extract_to": "images",
            "description": "Validation Images (1GB)",
            "size_gb": 1.0
        },
        {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", 
            "filename": "annotations_trainval2017.zip",
            "extract_to": ".",
            "description": "Annotations (1MB)",
            "size_gb": 0.001
        }
    ]
    
    total_size_gb = sum(f["size_gb"] for f in files_to_download)
    print(f"Total download size: ~{total_size_gb:.1f} GB")
    print()
    
    # Check available disk space
    try:
        import shutil
        free_space_gb = shutil.disk_usage(data_dir.parent).free / (1024**3)
        print(f"Available disk space: {free_space_gb:.1f} GB")
        
        if free_space_gb < total_size_gb * 2:  # Need 2x space for extraction
            print("⚠️  Warning: Low disk space. Ensure you have enough space for download + extraction.")
        print()
    except:
        pass
    
    for file_info in files_to_download:
        url = file_info["url"]
        filename = file_info["filename"] 
        extract_to = file_info["extract_to"]
        description = file_info["description"]
        
        file_path = data_dir / filename
        
        print(f"📦 {description}")
        print(f"   URL: {url}")
        print(f"   File: {filename}")
        
        # Check if file already exists
        if file_path.exists():
            print(f"   ✅ File already exists: {file_path}")
        else:
            print(f"   ⬇️  Downloading...")
            try:
                download_with_progress(url, str(file_path))
                print(f"   ✅ Downloaded: {file_path}")
            except Exception as e:
                print(f"   ❌ Download failed: {e}")
                continue
        
        # Extract file
        extract_dir = data_dir / extract_to
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        if filename.endswith('.zip'):
            print(f"   📂 Extracting to {extract_dir}...")
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Get list of files to extract
                    file_list = zip_ref.namelist()
                    
                    # Extract with progress
                    for file in tqdm(file_list, desc="Extracting"):
                        zip_ref.extract(file, extract_dir)
                
                print(f"   ✅ Extracted to: {extract_dir}")
                
                # Optionally remove zip file to save space
                response = input(f"   🗑️  Remove {filename} to save space? (y/N): ")
                if response.lower() == 'y':
                    file_path.unlink()
                    print(f"   🗑️  Removed: {filename}")
                    
            except Exception as e:
                print(f"   ❌ Extraction failed: {e}")
                continue
        
        print()
    
    # Verify structure
    print("🔍 Verifying dataset structure...")
    
    expected_structure = {
        "images/val2017": "directory",
        "annotations/captions_val2017.json": "file"
    }
    
    all_good = True
    for path, expected_type in expected_structure.items():
        full_path = data_dir / path
        
        if expected_type == "directory":
            if full_path.is_dir():
                file_count = len(list(full_path.glob("*.jpg")))
                print(f"   ✅ {path}: {file_count} images")
            else:
                print(f"   ❌ {path}: Directory not found")
                all_good = False
                
        elif expected_type == "file":
            if full_path.is_file():
                file_size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {path}: {file_size_mb:.1f} MB")
                
                # Validate annotations file
                if "captions" in path:
                    try:
                        with open(full_path, 'r') as f:
                            annotations = json.load(f)
                        print(f"      📊 {len(annotations['images'])} images, {len(annotations['annotations'])} captions")
                    except Exception as e:
                        print(f"      ⚠️  Could not validate annotations: {e}")
            else:
                print(f"   ❌ {path}: File not found")
                all_good = False
    
    print()
    if all_good:
        print("✅ COCO dataset downloaded and verified successfully!")
        print(f"📁 Dataset location: {data_dir.absolute()}")
        print()
        print("🚀 Next steps:")
        print("1. Verify setup: python setup_evaluation.py")
        print("2. Run alignment eval: python evaluate_alignment.py --blip3o_model_path <model> --coco_root ./data/coco")
        print("3. Run recall eval: python evaluate_recall.py --blip3o_model_path <model> --coco_root ./data/coco")
    else:
        print("❌ Some files are missing. Please check the download and extraction.")
        return False
    
    return True

def main():
    """Main download function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MS-COCO 2017 validation dataset")
    parser.add_argument("--data_dir", type=str, default="./data/coco", 
                       help="Directory to download COCO data (default: ./data/coco)")
    parser.add_argument("--check_only", action="store_true",
                       help="Only check if dataset exists, don't download")
    
    args = parser.parse_args()
    
    if args.check_only:
        data_dir = Path(args.data_dir)
        images_dir = data_dir / "images" / "val2017"
        annotations_file = data_dir / "annotations" / "captions_val2017.json"
        
        if images_dir.exists() and annotations_file.exists():
            image_count = len(list(images_dir.glob("*.jpg")))
            print(f"✅ COCO dataset found: {image_count} images")
            return True
        else:
            print(f"❌ COCO dataset not found in {data_dir}")
            return False
    else:
        return download_coco_val2017(args.data_dir)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)