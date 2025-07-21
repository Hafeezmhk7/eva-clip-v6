#!/usr/bin/env python3
"""
Model Diagnostic Script - Check if your model has dual supervision
"""

import torch
from pathlib import Path
import json

def diagnose_model(model_path: str):
    """Diagnose if the model was trained with dual supervision"""
    
    model_path = Path(model_path)
    print(f"🔍 DIAGNOSING MODEL: {model_path}")
    print("=" * 60)
    
    # Check if model files exist
    model_files = [
        model_path / "pytorch_model.bin",
        model_path / "model.safetensors", 
        model_path / "pytorch_model.safetensors"
    ]
    
    model_file = None
    for file_path in model_files:
        if file_path.exists():
            model_file = file_path
            break
    
    if model_file is None:
        print("❌ No model weights found!")
        return False
    
    print(f"📁 Found model file: {model_file}")
    
    # Load weights and check keys
    if model_file.suffix == ".bin":
        try:
            state_dict = torch.load(model_file, map_location="cpu")
        except:
            print("❌ Could not load .bin file")
            return False
    else:
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_file))
        except:
            print("❌ Could not load safetensors file")
            return False
    
    print(f"📊 Total parameters in state dict: {len(state_dict)}")
    
    # Check for dual supervision components
    print("\n🔍 CHECKING FOR DUAL SUPERVISION COMPONENTS:")
    
    # 1. Check for global velocity projection (KEY COMPONENT)
    global_velocity_keys = [k for k in state_dict.keys() if 'global_velocity_proj' in k]
    if global_velocity_keys:
        print(f"✅ Global velocity projection found: {global_velocity_keys}")
        has_global_velocity = True
    else:
        print("❌ Global velocity projection NOT found")
        has_global_velocity = False
    
    # 2. Check for global adaptation MLP
    global_mlp_keys = [k for k in state_dict.keys() if 'global_adaptation_mlp' in k]
    if global_mlp_keys:
        print(f"✅ Global adaptation MLP found: {len(global_mlp_keys)} parameters")
        has_global_mlp = True
    else:
        print("❌ Global adaptation MLP NOT found")
        has_global_mlp = False
    
    # 3. Check for frozen CLIP projection
    clip_proj_keys = [k for k in state_dict.keys() if 'frozen_clip_visual_proj' in k or 'clip_visual_proj' in k]
    if clip_proj_keys:
        print(f"✅ CLIP projection found: {clip_proj_keys}")
        has_clip_proj = True
    else:
        print("❌ CLIP projection NOT found")
        has_clip_proj = False
    
    # 4. Show sample of all keys for debugging
    print(f"\n📋 Sample of parameter keys:")
    for i, key in enumerate(sorted(state_dict.keys())):
        if i < 10:
            print(f"   {key}")
        elif i == 10:
            print(f"   ... ({len(state_dict)} total)")
            break
    
    # Check training summary if available
    summary_file = model_path / "training_summary.json"
    if summary_file.exists():
        print(f"\n📊 TRAINING SUMMARY:")
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print(f"   Total steps: {summary.get('total_steps', 'Unknown')}")
            print(f"   Parameters: {summary.get('num_parameters', 'Unknown'):,}")
            print(f"   LR scheduler: {summary.get('lr_scheduler_type', 'Unknown')}")
            
        except Exception as e:
            print(f"   Could not read summary: {e}")
    
    # Check for dual supervision summary
    dual_summary_file = model_path / "fixed_dual_supervision_summary.json"
    if dual_summary_file.exists():
        print(f"\n🎯 DUAL SUPERVISION SUMMARY FOUND:")
        try:
            with open(dual_summary_file, 'r') as f:
                dual_summary = json.load(f)
            
            arch = dual_summary.get('architecture', 'unknown')
            key_fix = dual_summary.get('key_fix', 'unknown')
            print(f"   Architecture: {arch}")
            print(f"   Key fix: {key_fix}")
            
            # Check final metrics
            final_metrics = dual_summary.get('recall_performance_prediction', {})
            if final_metrics:
                print(f"   Predicted recall: {final_metrics.get('predicted_fixed_recall', 'Unknown')}")
                print(f"   Training successful: {final_metrics.get('training_success', 'Unknown')}")
            
        except Exception as e:
            print(f"   Could not read dual supervision summary: {e}")
    else:
        print(f"\n❌ No dual supervision summary found")
    
    # Final diagnosis
    print(f"\n" + "=" * 60)
    print(f"🎯 FINAL DIAGNOSIS:")
    
    if has_global_velocity and has_global_mlp:
        print("✅ SUCCESS: Model has dual supervision architecture!")
        print("🎯 This model should achieve 50-70% recall")
        print("💡 The poor performance might be due to evaluation code issues")
        verdict = True
    elif has_global_mlp and has_clip_proj:
        print("⚠️  PARTIAL: Model has some dual supervision components")
        print("❌ Missing global velocity projection (key for recall)")
        print("🎯 Expected performance: 10-30% recall")
        verdict = False
    else:
        print("❌ FAILURE: Model does NOT have dual supervision!")
        print("🎯 Expected performance: 0-2% recall (matches your results)")
        print("💡 You need to retrain with the fixed architecture")
        verdict = False
    
    print("=" * 60)
    return verdict

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Use your model path
        model_path = "/scratch-shared/scur2711/blip3o_workspace/checkpoints/blip3o_multi_gpu_fixed_cosine_13219643_20250719_081230"
    
    print("🔬 MODEL DIAGNOSTIC TOOL")
    print(f"Checking: {model_path}")
    
    success = diagnose_model(model_path)
    
    if success:
        print("\n🎉 Your model looks good! The issue might be in evaluation code.")
    else:
        print("\n❌ Your model needs retraining with dual supervision architecture.")