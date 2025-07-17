#!/usr/bin/env python3
"""
Download MS-COCO 2017 Validation Dataset

This script automatically downloads and extracts the COCO val2017 dataset
including images and captions for evaluation.
"""

import os
import urllib.request
import zipfile
import json
from pathlib import Path
from tqdm import tqdm
import argparse

def download_with_progress(url: str, filepath: Path):
    """Download file with progress bar."""
    class ProgressBar:
        def __init__(self):
            self.pbar = None

        def __call__(self, block_num, block_size, total_size):
            if not self.pbar:
                self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name)
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.close()

    print(f"📥 Downloading {filepath.name}...")
    urllib.request.urlretrieve(url, filepath, ProgressBar())

def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file with progress bar."""
    print(f"📦 Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.infolist()
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_to)

def download_coco_val2017(coco_root: str = "./data/coco"):
    """Download COCO val2017 dataset."""
    coco_path = Path(coco_root)
    coco_path.mkdir(parents=True, exist_ok=True)
    
    print("🎯 MS-COCO 2017 Validation Dataset Downloader")
    print("=" * 50)
    print(f"📁 Download directory: {coco_path.absolute()}")
    
    # URLs for COCO val2017
    urls = {
        "images": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    # Download images
    images_zip = coco_path / "val2017.zip"
    if not images_zip.exists():
        download_with_progress(urls["images"], images_zip)
    else:
        print(f"✅ Images zip already exists: {images_zip}")
    
    # Download annotations
    annotations_zip = coco_path / "annotations_trainval2017.zip"
    if not annotations_zip.exists():
        download_with_progress(urls["annotations"], annotations_zip)
    else:
        print(f"✅ Annotations zip already exists: {annotations_zip}")
    
    # Extract images
    images_dir = coco_path / "images"
    if not (images_dir / "val2017").exists():
        extract_zip(images_zip, coco_path)
        print(f"✅ Images extracted to: {images_dir / 'val2017'}")
    else:
        print(f"✅ Images already extracted: {images_dir / 'val2017'}")
    
    # Extract annotations
    annotations_dir = coco_path / "annotations"
    if not (annotations_dir / "captions_val2017.json").exists():
        extract_zip(annotations_zip, coco_path)
        print(f"✅ Annotations extracted to: {annotations_dir}")
    else:
        print(f"✅ Annotations already extracted: {annotations_dir}")
    
    # Verify download
    val_images = images_dir / "val2017"
    captions_file = annotations_dir / "captions_val2017.json"
    
    if val_images.exists() and captions_file.exists():
        # Count images
        num_images = len(list(val_images.glob("*.jpg")))
        
        # Count captions
        with open(captions_file, 'r') as f:
            coco_data = json.load(f)
        num_annotations = len(coco_data['annotations'])
        num_image_entries = len(coco_data['images'])
        
        print("\n✅ Download Complete!")
        print("=" * 30)
        print(f"📊 Validation images: {num_images:,}")
        print(f"📊 Image entries: {num_image_entries:,}")
        print(f"📊 Captions: {num_annotations:,}")
        print(f"📁 Dataset ready at: {coco_path.absolute()}")
        
        # Show directory structure
        print("\n📁 Directory Structure:")
        print(f"{coco_path}/")
        print("├── images/")
        print("│   └── val2017/")
        print(f"│       ├── 000000000139.jpg")
        print(f"│       └── ... ({num_images:,} images)")
        print("└── annotations/")
        print("    └── captions_val2017.json")
        
        # Clean up zip files (optional)
        cleanup = input("\n🗑️  Delete zip files to save space? (y/N): ").lower().strip()
        if cleanup == 'y':
            images_zip.unlink(missing_ok=True)
            annotations_zip.unlink(missing_ok=True)
            print("✅ Zip files deleted")
        
        return True
    else:
        print("❌ Download verification failed")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO val2017 dataset")
    parser.add_argument("--coco_root", type=str, default="./data/coco", 
                       help="Directory to download COCO data")
    args = parser.parse_args()
    
    success = download_coco_val2017(args.coco_root)
    if success:
        print("\n🎉 Ready for embedding extraction!")
    else:
        print("\n❌ Download failed")
        exit(1)