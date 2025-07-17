#!/usr/bin/env python3
"""
Quick COCO Data Verification
Check if downloaded COCO data is properly structured and ready for extraction.
"""

import json
from pathlib import Path
import sys

def check_coco_data(coco_root="./data/coco"):
    """Check if COCO data is properly downloaded and structured."""
    
    print("🔍 COCO Data Verification")
    print("=" * 30)
    
    coco_path = Path(coco_root)
    
    # Check directories
    images_dir = coco_path / "images" / "val2017"
    annotations_dir = coco_path / "annotations"
    
    print(f"📁 COCO root: {coco_path.absolute()}")
    print(f"📁 Images dir: {images_dir}")
    print(f"📁 Annotations dir: {annotations_dir}")
    
    # Check images
    if images_dir.exists():
        jpg_files = list(images_dir.glob("*.jpg"))
        print(f"✅ Images directory exists")
        print(f"📊 Number of images: {len(jpg_files):,}")
        
        # Show some example files
        if jpg_files:
            print(f"📝 Example images:")
            for i, img_file in enumerate(sorted(jpg_files)[:3]):
                print(f"   {i+1}. {img_file.name}")
    else:
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    # Check annotations
    annotations_file = annotations_dir / "captions_val2017.json"
    if annotations_file.exists():
        print(f"✅ Annotations file exists")
        
        # Load and check annotations
        try:
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            
            num_images = len(coco_data['images'])
            num_captions = len(coco_data['annotations'])
            
            print(f"📊 Image entries: {num_images:,}")
            print(f"📊 Caption entries: {num_captions:,}")
            print(f"📊 Avg captions per image: {num_captions/num_images:.1f}")
            
            # Show example annotation
            if coco_data['annotations']:
                example = coco_data['annotations'][0]
                print(f"📝 Example caption: '{example['caption'][:60]}...'")
                
        except Exception as e:
            print(f"❌ Error reading annotations: {e}")
            return False
    else:
        print(f"❌ Annotations file not found: {annotations_file}")
        return False
    
    # Final verification
    if len(jpg_files) > 4000 and num_captions > 20000:
        print(f"\n✅ COCO DATA VERIFICATION: PASSED")
        print(f"   Ready for embedding extraction!")
        return True
    else:
        print(f"\n❌ COCO DATA VERIFICATION: FAILED")
        print(f"   Insufficient data found")
        return False

if __name__ == "__main__":
    coco_root = sys.argv[1] if len(sys.argv) > 1 else "./data/coco"
    success = check_coco_data(coco_root)
    sys.exit(0 if success else 1)