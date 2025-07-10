"""
Debug script to test embedding extraction setup and identify issues
Run this before the full extraction to catch problems early
"""

import sys
import os
import torch
import psutil
from pathlib import Path
import argparse

def setup_paths():
    """Setup paths for project structure"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Add import paths
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    sys.path.insert(0, str(project_root / "src" / "data_hand"))
    
    return project_root

def check_system_resources():
    """Check available system resources"""
    print("🔍 System Resource Check")
    print("=" * 40)
    
    # CPU
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    print(f"💻 CPU cores: {cpu_count}")
    print(f"💾 RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    # GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🖥️ GPUs: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   GPU {i}: {props.name} - {memory_gb:.1f} GB")
            
        # Current GPU memory
        current_gpu = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_gpu) / (1024**3)
        cached = torch.cuda.memory_reserved(current_gpu) / (1024**3)
        print(f"   Current usage: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
    else:
        print("❌ No CUDA available")
        return False
    
    print("")
    return True

def test_dataset_loading(data_file, batch_size=2):
    """Test if dataset can be loaded"""
    print("📂 Dataset Loading Test")
    print("=" * 40)
    
    try:
        from src.data_hand.dataset import BLIP3oWebDataset
        
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return False
        
        print(f"📁 Testing data file: {data_file}")
        print(f"📊 File size: {data_file.stat().st_size / (1024**2):.1f} MB")
        
        # Create dataset
        dataset = BLIP3oWebDataset(
            tar_paths=[str(data_file)],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        dataloader = dataset.get_dataloader()
        print(f"✅ Dataset created successfully")
        
        # Test loading one batch
        print("🧪 Testing batch loading...")
        batch = next(iter(dataloader))
        
        print(f"✅ Batch loaded successfully")
        print(f"   Batch size: {len(batch['image'])}")
        print(f"   Sample keys: {batch['key'][:2]}")
        print(f"   Image sizes: {[img.size for img in batch['image'][:2]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading(device, test_eva=True):
    """Test if models can be loaded without OOM"""
    print("🤖 Model Loading Test")
    print("=" * 40)
    
    def print_memory():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            print(f"   GPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
    
    try:
        # Test CLIP loading
        print("📦 Testing CLIP ViT-L/14...")
        print_memory()
        
        from transformers import CLIPProcessor, CLIPModel
        
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16
        ).to(device)
        
        print("✅ CLIP loaded successfully")
        print_memory()
        
        # Clear CLIP
        del clip_model, clip_processor
        torch.cuda.empty_cache()
        
        if test_eva:
            # Test EVA loading
            print("📦 Testing EVA-CLIP-8B...")
            print_memory()
            
            from transformers import AutoModel
            
            eva_model = AutoModel.from_pretrained(
                "BAAI/EVA-CLIP-8B", 
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(device)
            
            print("✅ EVA-CLIP loaded successfully")
            print_memory()
            
            # Clear EVA
            del eva_model
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        if "out of memory" in str(e).lower():
            print("💡 Suggestions:")
            print("   - Increase GPU memory allocation")
            print("   - Use CPU for EVA model (slower but uses less GPU memory)")
            print("   - Process models one at a time")
        return False

def test_feature_extraction(data_file, device, num_samples=2):
    """Test feature extraction on a small batch"""
    print("🧠 Feature Extraction Test")
    print("=" * 40)
    
    try:
        # Load dataset
        from src.data_hand.dataset import BLIP3oWebDataset
        
        dataset = BLIP3oWebDataset(
            tar_paths=[str(data_file)],
            batch_size=num_samples,
            shuffle=False,
            num_workers=0
        )
        
        dataloader = dataset.get_dataloader()
        batch = next(iter(dataloader))
        images = batch['image'][:num_samples]  # Take only 2 samples
        
        print(f"🖼️ Testing with {len(images)} images")
        
        # Test CLIP extraction
        print("📊 Testing CLIP feature extraction...")
        from transformers import CLIPProcessor, CLIPModel
        
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16
        ).to(device)
        
        # Extract features for one image
        img = images[0]
        inputs = clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device).half() if v.dtype == torch.float32 else v.to(device) 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            vision_outputs = clip_model.vision_model(
                pixel_values=inputs['pixel_values'],
                output_hidden_states=True,
                return_dict=True
            )
            
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]
            print(f"✅ CLIP features extracted: {patch_embeddings.shape}")
        
        # Clean up
        del clip_model, clip_processor, inputs, vision_outputs, patch_embeddings
        torch.cuda.empty_cache()
        
        print("✅ Feature extraction test passed")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        if "out of memory" in str(e).lower():
            print("💡 GPU memory insufficient for feature extraction")
            print("   Try reducing batch size or using smaller models")
        return False

def run_comprehensive_test(data_file, device="cuda", test_extraction=True):
    """Run all tests"""
    print("🧪 Comprehensive Debug Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: System resources
    if check_system_resources():
        tests_passed += 1
        print("✅ System resources OK\n")
    else:
        print("❌ System resources insufficient\n")
    
    # Test 2: Dataset loading
    if test_dataset_loading(data_file):
        tests_passed += 1
        print("✅ Dataset loading OK\n")
    else:
        print("❌ Dataset loading failed\n")
        return False
    
    # Test 3: Model loading
    device_obj = torch.device(device)
    if test_model_loading(device_obj, test_eva=True):
        tests_passed += 1
        print("✅ Model loading OK\n")
    else:
        print("❌ Model loading failed\n")
        print("🔄 Trying without EVA-CLIP...")
        if test_model_loading(device_obj, test_eva=False):
            print("⚠️ CLIP works but EVA-CLIP fails\n")
        else:
            print("❌ Even CLIP loading fails\n")
            return False
    
    # Test 4: Feature extraction
    if test_extraction and test_feature_extraction(data_file, device_obj):
        tests_passed += 1
        print("✅ Feature extraction OK\n")
    else:
        print("❌ Feature extraction failed\n")
    
    print("📊 Test Results")
    print("=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready for full extraction")
        print("\n💡 Recommended settings:")
        print("   --batch_size 4")
        print("   --save_every 25")
        return True
    elif tests_passed >= 2:
        print("⚠️ Some tests failed but basic functionality works")
        print("\n💡 Recommended settings:")
        print("   --batch_size 2")
        print("   --save_every 10")
        print("   Consider using more memory or smaller models")
        return True
    else:
        print("❌ Critical issues found. Fix before proceeding.")
        return False

def main():
    """Main debug function"""
    parser = argparse.ArgumentParser(description="Debug embedding extraction setup")
    parser.add_argument("--data_file", type=str, default="data/00000.tar", help="Path to data file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip feature extraction test")
    
    args = parser.parse_args()
    
    project_root = setup_paths()
    data_file = project_root / args.data_file
    
    success = run_comprehensive_test(
        data_file=data_file,
        device=args.device,
        test_extraction=not args.skip_extraction
    )
    
    if success:
        print("\n🚀 Ready to run optimized extraction:")
        print("python src/modules/optimized_extract_embeddings.py")
    else:
        print("\n🛠️ Fix the issues above before running extraction")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())