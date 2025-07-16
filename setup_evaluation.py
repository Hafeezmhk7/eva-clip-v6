#!/usr/bin/env python3
"""
Setup script for BLIP3-o DiT evaluation.
Verifies environment, dependencies, and data availability.
"""

import sys
import os
import subprocess
from pathlib import Path
import json
import torch

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_pytorch():
    """Check PyTorch installation."""
    try:
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🎮 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"🎮 CUDA version: {torch.version.cuda}")
            print(f"🎮 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test basic operations
        x = torch.randn(2, 3)
        if torch.cuda.is_available():
            x = x.cuda()
            print("✅ PyTorch GPU operations working")
        else:
            print("✅ PyTorch CPU operations working")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
        return False

def check_dependencies():
    """Check required dependencies."""
    required_packages = [
        'transformers',
        'PIL', 
        'numpy',
        'tqdm',
        'requests',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✅ {package} (Pillow): {PIL.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        if 'PIL' in missing_packages:
            missing_packages.remove('PIL')
            missing_packages.append('Pillow')
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_project_structure():
    """Check project structure."""
    project_root = Path(__file__).parent
    
    required_paths = [
        "src/modules/evaluation/__init__.py",
        "src/modules/evaluation/coco_dataset.py", 
        "src/modules/evaluation/metrics.py",
        "src/modules/evaluation/evaluator.py",
        "src/modules/models/blip3o_dit.py",
        "src/modules/inference/blip3o_inference.py",
        "evaluate_alignment.py",
        "evaluate_recall.py",
    ]
    
    missing_files = []
    
    for path in required_paths:
        file_path = project_root / path
        if file_path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files")
        return False
    else:
        print("✅ Project structure OK")
        return True

def test_model_loading():
    """Test loading models used in evaluation."""
    print("\n🧪 Testing model loading...")
    
    try:
        # Test CLIP loading
        from transformers import CLIPProcessor, CLIPModel
        print("✅ CLIP imports OK")
        
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("✅ CLIP processor loaded")
        
        # Test EVA-CLIP loading  
        from transformers import AutoModel
        print("✅ EVA-CLIP imports OK")
        
        # Don't actually load EVA-CLIP in setup (too large)
        print("⚠️  EVA-CLIP loading will be tested during evaluation")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def check_coco_dataset(coco_root=None):
    """Check COCO dataset availability."""
    if coco_root is None:
        # Check common locations
        possible_paths = [
            "./data/coco",
            "./coco",
            "~/data/coco",
            "/data/coco",
        ]
        
        for path in possible_paths:
            path = Path(path).expanduser()
            if path.exists():
                coco_root = path
                break
    
    if coco_root is None:
        print("⚠️  COCO dataset not found in common locations")
        print("📥 Download instructions:")
        print("   1. mkdir -p ./data/coco")
        print("   2. wget http://images.cocodataset.org/zips/val2017.zip")
        print("   3. unzip val2017.zip -d ./data/coco/images/")
        print("   4. wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print("   5. unzip annotations_trainval2017.zip -d ./data/coco/")
        return False
    
    coco_root = Path(coco_root)
    print(f"🔍 Checking COCO at: {coco_root}")
    
    # Check structure
    images_dir = coco_root / "images" / "val2017"
    annotations_file = coco_root / "annotations" / "captions_val2017.json"
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not annotations_file.exists():
        print(f"❌ Annotations file not found: {annotations_file}")
        return False
    
    # Count images
    image_files = list(images_dir.glob("*.jpg"))
    print(f"✅ Found {len(image_files)} images")
    
    # Check annotations
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        print(f"✅ Annotations loaded: {len(annotations['images'])} images, {len(annotations['annotations'])} captions")
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return False
    
    return True

def test_evaluation_pipeline(coco_root=None):
    """Test the evaluation pipeline with a small sample."""
    if coco_root is None:
        print("⚠️  Skipping pipeline test (no COCO dataset)")
        return True
    
    print("\n🧪 Testing evaluation pipeline...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test dataset loading
        from src.modules.evaluation.coco_dataset import COCOEvaluationDataset
        
        dataset = COCOEvaluationDataset(
            coco_root=coco_root,
            max_samples=5  # Very small test
        )
        print(f"✅ COCO dataset loaded: {len(dataset)} samples")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"✅ Sample loaded: {sample['image'].size}, {len(sample['captions'])} captions")
        
        # Test metrics
        from src.modules.evaluation.metrics import compute_cosine_similarity
        import torch
        
        emb_a = torch.randn(10, 512)
        emb_b = torch.randn(10, 512)
        sim = compute_cosine_similarity(emb_a, emb_b)
        print(f"✅ Metrics working: mean similarity = {sim.mean():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup verification."""
    print("🚀 BLIP3-o DiT Evaluation Setup")
    print("=" * 40)
    
    all_checks_passed = True
    
    # Basic checks
    print("\n📋 Basic Environment Checks")
    print("-" * 30)
    all_checks_passed &= check_python_version()
    all_checks_passed &= check_pytorch()
    all_checks_passed &= check_dependencies()
    
    # Project structure
    print("\n📁 Project Structure Check")
    print("-" * 30)
    all_checks_passed &= check_project_structure()
    
    # Model loading
    print("\n🤖 Model Loading Test")
    print("-" * 30)
    all_checks_passed &= test_model_loading()
    
    # COCO dataset
    print("\n🖼️  COCO Dataset Check")
    print("-" * 30)
    coco_available = check_coco_dataset()
    
    # Pipeline test
    if coco_available:
        all_checks_passed &= test_evaluation_pipeline("./data/coco")
    
    # Summary
    print("\n" + "=" * 40)
    if all_checks_passed and coco_available:
        print("✅ ALL CHECKS PASSED!")
        print("🚀 Ready to run evaluation!")
        print("\nNext steps:")
        print("1. Train your BLIP3-o DiT model")
        print("2. Run: python evaluate_alignment.py --blip3o_model_path <model> --coco_root ./data/coco")
        print("3. Run: python evaluate_recall.py --blip3o_model_path <model> --coco_root ./data/coco")
    elif all_checks_passed:
        print("⚠️  Environment OK, but COCO dataset missing")
        print("📥 Please download COCO val2017 dataset first")
    else:
        print("❌ Some checks failed")
        print("🔧 Please fix the issues above before running evaluation")
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)