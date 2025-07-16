"""
MS-COCO 2017 Validation Dataset Loader for BLIP3-o Evaluation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class COCOEvaluationDataset(Dataset):
    """
    MS-COCO 2017 Validation Dataset for BLIP3-o evaluation.
    
    This dataset loads images and their corresponding captions for evaluation tasks:
    - Alignment evaluation (cosine similarity)
    - Recall evaluation (retrieval tasks)
    """
    
    def __init__(
        self,
        coco_root: Union[str, Path],
        split: str = "val2017",
        max_samples: Optional[int] = None,
        max_captions_per_image: int = 5,
        image_transform=None,
    ):
        """
        Initialize COCO evaluation dataset.
        
        Args:
            coco_root: Root directory containing COCO data
            split: Dataset split (val2017 for validation)
            max_samples: Maximum number of samples to load (None for all)
            max_captions_per_image: Maximum captions per image
            image_transform: Image preprocessing transform
        """
        self.coco_root = Path(coco_root)
        self.split = split
        self.max_samples = max_samples
        self.max_captions_per_image = max_captions_per_image
        self.image_transform = image_transform
        
        # Paths
        self.images_dir = self.coco_root / "images" / split
        self.annotations_file = self.coco_root / "annotations" / f"captions_{split}.json"
        
        # Validate paths
        self._validate_paths()
        
        # Load annotations
        self.data = self._load_annotations()
        
        logger.info(f"COCO {split} dataset loaded: {len(self.data)} samples")
    
    def _validate_paths(self):
        """Validate that required COCO files exist."""
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        logger.info(f"COCO paths validated:")
        logger.info(f"  Images: {self.images_dir}")
        logger.info(f"  Annotations: {self.annotations_file}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load and process COCO annotations."""
        logger.info("Loading COCO annotations...")
        
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image info mapping
        images_info = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        image_captions = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_captions:
                image_captions[image_id] = []
            image_captions[image_id].append(ann['caption'])
        
        # Create dataset samples
        samples = []
        for image_id, captions in image_captions.items():
            if image_id not in images_info:
                continue
            
            image_info = images_info[image_id]
            image_path = self.images_dir / image_info['file_name']
            
            # Check if image file exists
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                continue
            
            # Limit captions per image
            captions = captions[:self.max_captions_per_image]
            
            sample = {
                'image_id': image_id,
                'image_path': str(image_path),
                'captions': captions,
                'width': image_info['width'],
                'height': image_info['height'],
            }
            samples.append(sample)
            
            # Apply max_samples limit
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        logger.info(f"Loaded {len(samples)} image-caption pairs")
        return samples
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample from the dataset."""
        sample = self.data[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Apply image transform if provided
            if self.image_transform is not None:
                image = self.image_transform(image)
        
        except Exception as e:
            logger.error(f"Error loading image {sample['image_path']}: {e}")
            # Return a dummy black image
            image = Image.new('RGB', (224, 224), color='black')
            if self.image_transform is not None:
                image = self.image_transform(image)
        
        return {
            'image_id': sample['image_id'],
            'image': image,
            'captions': sample['captions'],
            'image_path': sample['image_path'],
            'width': sample['width'],
            'height': sample['height'],
        }
    
    def get_all_captions(self) -> List[str]:
        """Get all unique captions in the dataset."""
        all_captions = []
        for sample in self.data:
            all_captions.extend(sample['captions'])
        return list(set(all_captions))
    
    def get_image_caption_pairs(self) -> List[Tuple[int, str]]:
        """Get all image-caption pairs as (image_idx, caption) tuples."""
        pairs = []
        for img_idx, sample in enumerate(self.data):
            for caption in sample['captions']:
                pairs.append((img_idx, caption))
        return pairs


def create_coco_dataloader(
    coco_root: Union[str, Path],
    split: str = "val2017",
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    max_captions_per_image: int = 5,
    num_workers: int = 4,
    shuffle: bool = False,
    image_transform=None,
) -> DataLoader:
    """
    Create a DataLoader for COCO evaluation dataset.
    
    Args:
        coco_root: Root directory containing COCO data
        split: Dataset split
        batch_size: Batch size
        max_samples: Maximum samples to load
        max_captions_per_image: Maximum captions per image
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle data
        image_transform: Image preprocessing transform
        
    Returns:
        DataLoader for COCO dataset
    """
    dataset = COCOEvaluationDataset(
        coco_root=coco_root,
        split=split,
        max_samples=max_samples,
        max_captions_per_image=max_captions_per_image,
        image_transform=image_transform,
    )
    
    def collate_fn(batch):
        """Custom collate function for COCO batch."""
        return {
            'image_ids': [item['image_id'] for item in batch],
            'images': [item['image'] for item in batch],  # List of PIL Images or tensors
            'captions': [item['captions'] for item in batch],  # List of caption lists
            'image_paths': [item['image_path'] for item in batch],
            'widths': [item['width'] for item in batch],
            'heights': [item['height'] for item in batch],
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def download_coco_val2017(download_dir: Union[str, Path]) -> Path:
    """
    Helper function to download MS-COCO 2017 validation data.
    
    Args:
        download_dir: Directory to download COCO data
        
    Returns:
        Path to downloaded COCO root directory
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 MS-COCO 2017 Validation Download Instructions:")
    print("=" * 50)
    print("Please download the following files manually:")
    print()
    print("1. Validation Images (1GB):")
    print("   http://images.cocodataset.org/zips/val2017.zip")
    print("   Extract to: {}/images/val2017/".format(download_dir))
    print()
    print("2. Validation Annotations (1MB):")
    print("   http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    print("   Extract captions_val2017.json to: {}/annotations/".format(download_dir))
    print()
    print("Expected directory structure:")
    print(f"{download_dir}/")
    print("├── images/")
    print("│   └── val2017/")
    print("│       ├── 000000000139.jpg")
    print("│       └── ...")
    print("└── annotations/")
    print("    └── captions_val2017.json")
    print()
    
    return download_dir


if __name__ == "__main__":
    # Test the dataset loader
    import sys
    
    if len(sys.argv) > 1:
        coco_root = sys.argv[1]
        
        print(f"🧪 Testing COCO dataset loader...")
        print(f"COCO root: {coco_root}")
        
        try:
            # Test dataset creation
            dataset = COCOEvaluationDataset(
                coco_root=coco_root,
                max_samples=10,  # Small test
            )
            
            print(f"✅ Dataset created: {len(dataset)} samples")
            
            # Test dataloader
            dataloader = create_coco_dataloader(
                coco_root=coco_root,
                batch_size=2,
                max_samples=10,
                num_workers=0,  # Avoid multiprocessing issues
            )
            
            print(f"✅ DataLoader created")
            
            # Test loading a batch
            batch = next(iter(dataloader))
            print(f"✅ Batch loaded:")
            print(f"   Images: {len(batch['images'])}")
            print(f"   Captions: {len(batch['captions'])}")
            print(f"   Sample caption: {batch['captions'][0][0]}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Show download instructions
        download_coco_val2017("./data/coco")