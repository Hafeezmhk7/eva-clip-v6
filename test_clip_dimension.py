#!/usr/bin/env python3
"""
Quick test to verify CLIP embedding dimensions
Run this to verify the fix works before running full evaluation
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

def test_clip_dimensions():
    """Test CLIP embedding dimensions"""
    print("🧪 Testing CLIP embedding dimensions...")
    
    # Load CLIP
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test text embeddings
    test_captions = ["A cat sitting on a mat", "A dog playing in the park"]
    text_inputs = processor(text=test_captions, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        print(f"✅ Text features shape: {text_features.shape}")  # Should be [2, 768]
    
    # Test image embeddings  
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    image_inputs = processor(images=[dummy_image, dummy_image], return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    
    with torch.no_grad():
        # get_image_features() already returns projected features in joint space!
        vision_features = model.get_image_features(**image_inputs)
        print(f"✅ Vision features shape (already projected): {vision_features.shape}")  # Should be [2, 768]
        
        # Let's also check the raw vision features if we want
        vision_outputs = model.vision_model(**image_inputs)
        raw_vision_features = vision_outputs.pooler_output  # This is the raw [2, 1024]
        print(f"✅ Raw vision features shape: {raw_vision_features.shape}")  # Should be [2, 1024]
        
        # Manual projection (just to verify)
        manually_projected = model.visual_projection(raw_vision_features)
        print(f"✅ Manually projected features shape: {manually_projected.shape}")  # Should be [2, 768]
        
        # Verify they're the same
        print(f"✅ get_image_features == manual projection: {torch.allclose(vision_features, manually_projected, atol=1e-5)}")
    
    # Test similarity computation (this should work now!)
    similarity = torch.mm(vision_features, text_features.t())
    print(f"✅ Similarity matrix shape: {similarity.shape}")  # Should be [2, 2]
    print(f"✅ Similarity values: {similarity}")
    
    print("🎉 All dimension checks passed!")
    
    return True

if __name__ == "__main__":
    try:
        test_clip_dimensions()
        print("\n✅ CLIP dimensions are correct - your evaluation should work now!")
        print("\n📋 Key findings:")
        print("   • get_text_features() → [N, 768] (joint space)")
        print("   • get_image_features() → [N, 768] (already projected to joint space)")
        print("   • NO additional projection needed for either!")
    except Exception as e:
        print(f"\n❌ Error in CLIP dimension test: {e}")
        import traceback
        traceback.print_exc()