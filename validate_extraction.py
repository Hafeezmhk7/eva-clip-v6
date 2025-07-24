#!/usr/bin/env python3
"""
Quick Validation Script for Existing BLIP3-o Embeddings
validate_extraction.py

Run this to check if your current extraction is suitable for BLIP3-o training
"""

import os
import sys
import pickle
import torch
import json
from pathlib import Path
import argparse

def validate_extraction(embeddings_dir):
    """
    Validate existing BLIP3-o embeddings extraction
    
    Args:
        embeddings_dir: Path to embeddings directory
        
    Returns:
        Dictionary with validation results
    """
    embeddings_path = Path(embeddings_dir)
    
    print(f"🧪 VALIDATING EXTRACTION: {embeddings_path}")
    print("=" * 60)
    
    if not embeddings_path.exists():
        print(f"❌ Directory does not exist: {embeddings_path}")
        return {'valid': False, 'reason': 'directory_not_found'}
    
    # Look for shard files
    shard_files = list(embeddings_path.glob("embeddings_shard_*.pkl"))
    if not shard_files:
        shard_files = list(embeddings_path.glob("*.pkl"))
    
    if not shard_files:
        print("❌ No .pkl files found")
        return {'valid': False, 'reason': 'no_shards_found'}
    
    print(f"✅ Found {len(shard_files)} shard files")
    
    # Check first shard to validate format
    try:
        print(f"🔍 Checking first shard: {shard_files[0].name}")
        with open(shard_files[0], 'rb') as f:
            shard_data = pickle.load(f)
        
        # Check required keys
        required_keys = ['clip_blip3o_embeddings', 'eva_blip3o_embeddings', 'captions']
        missing_keys = [key for key in required_keys if key not in shard_data]
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
            return {'valid': False, 'reason': 'missing_keys', 'missing': missing_keys}
        
        clip_emb = shard_data['clip_blip3o_embeddings']
        eva_emb = shard_data['eva_blip3o_embeddings']
        captions = shard_data['captions']
        
        print(f"✅ CLIP embeddings shape: {clip_emb.shape}")
        print(f"✅ EVA embeddings shape: {eva_emb.shape}")
        print(f"✅ Number of captions: {len(captions)}")
        
        # Analyze token format
        num_tokens = clip_emb.shape[1]
        clip_dim = clip_emb.shape[2]
        eva_dim = eva_emb.shape[2]
        
        # Validate dimensions
        validation_results = {
            'valid': True,
            'num_tokens': num_tokens,
            'clip_dim': clip_dim,
            'eva_dim': eva_dim,
            'num_samples': clip_emb.shape[0],
            'num_shards': len(shard_files),
        }
        
        if clip_dim != 1024:
            print(f"❌ CLIP dimension should be 1024, got {clip_dim}")
            validation_results['valid'] = False
            validation_results['reason'] = 'wrong_clip_dim'
        
        if eva_dim != 4096:
            print(f"❌ EVA dimension should be 4096, got {eva_dim}")
            validation_results['valid'] = False
            validation_results['reason'] = 'wrong_eva_dim'
        
        # Determine format
        if num_tokens == 257:
            print("🎉 EXCELLENT! CLS+Patch format (257 tokens)")
            print("   ✅ CLS token at position [0]")
            print("   ✅ Patches at positions [1:257]")
            print("   ✅ Compatible with both cls_patch AND patch_only training")
            validation_results['format'] = 'cls_patch'
            validation_results['training_modes'] = ['cls_patch', 'patch_only']
            validation_results['recommendation'] = 'perfect_for_training'
            
        elif num_tokens == 256:
            print("⚠️  Patch-only format (256 tokens)")
            print("   ✅ Compatible with patch_only training")
            print("   ❌ NOT compatible with cls_patch training")
            print("   💡 Consider re-extracting with CLS token for flexibility")
            validation_results['format'] = 'patch_only'
            validation_results['training_modes'] = ['patch_only']
            validation_results['recommendation'] = 'reextract_for_cls_patch'
            
        else:
            print(f"❌ Unexpected token count: {num_tokens} (should be 256 or 257)")
            validation_results['valid'] = False
            validation_results['reason'] = 'wrong_token_count'
        
        # Check manifest
        manifest_file = embeddings_path / "embeddings_manifest.json"
        if manifest_file.exists():
            print(f"✅ Found manifest: {manifest_file}")
            try:
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                
                total_samples = manifest.get('total_samples', 0)
                tokens_per_sample = manifest.get('tokens_per_sample', num_tokens)
                include_cls = manifest.get('include_cls', num_tokens == 257)
                
                print(f"✅ Manifest info:")
                print(f"   Total samples: {total_samples:,}")
                print(f"   Tokens per sample: {tokens_per_sample}")
                print(f"   CLS included: {include_cls}")
                
                validation_results['total_samples'] = total_samples
                validation_results['manifest_valid'] = True
                
                if total_samples >= 50000:
                    print(f"🎉 EXCELLENT! {total_samples:,} samples - great for training!")
                elif total_samples >= 20000:
                    print(f"✅ GOOD! {total_samples:,} samples - should work well")
                elif total_samples >= 5000:
                    print(f"⚠️  {total_samples:,} samples - limited but OK for testing")
                else:
                    print(f"❌ Only {total_samples:,} samples - too few for robust training")
                    validation_results['recommendation'] = 'need_more_data'
                
            except Exception as e:
                print(f"⚠️  Manifest exists but couldn't parse: {e}")
                validation_results['manifest_valid'] = False
        else:
            print("⚠️  No manifest found (not critical, but recommended)")
            validation_results['manifest_valid'] = False
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Error validating shard: {e}")
        return {'valid': False, 'reason': 'shard_read_error', 'error': str(e)}

def print_recommendations(validation_results):
    """Print recommendations based on validation results"""
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    if not validation_results['valid']:
        print("❌ EXTRACTION IS NOT VALID")
        print("   👉 You MUST re-extract with the new script")
        print(f"   Reason: {validation_results.get('reason', 'unknown')}")
        return
    
    format_type = validation_results.get('format', 'unknown')
    
    if format_type == 'cls_patch':
        print("🎉 YOUR EXTRACTION IS PERFECT!")
        print("   ✅ You have CLS+patch format (257 tokens)")
        print("   ✅ Ready for BOTH training modes:")
        print("      • cls_patch mode (uses all 257 tokens)")
        print("      • patch_only mode (uses tokens [1:257])")
        print("\n🚀 NEXT STEPS:")
        print("   1. Test the dataset loading:")
        print(f"      python test_dataset_clean.py {validation_results.get('embeddings_dir', 'YOUR_PATH')} cls_patch")
        print("   2. If test passes, start training!")
        print("   3. No need to re-extract!")
        
    elif format_type == 'patch_only':
        print("⚠️  YOUR EXTRACTION IS PATCH-ONLY")
        print("   ✅ Good for patch_only training")
        print("   ❌ Cannot do cls_patch training")
        print("\n🤔 DECISION TIME:")
        print("   Option A: Keep current extraction")
        print("     • Only use --training_mode patch_only")
        print("     • Start training immediately")
        print("   Option B: Re-extract with new script")
        print("     • Get both 257 and 256 token versions")
        print("     • More flexible for experiments")
        print("\n💡 RECOMMENDATION: If you want to compare cls_patch vs patch_only,")
        print("    re-extract with the new script. Otherwise, your current extraction is fine!")
    
    # Sample size recommendations
    total_samples = validation_results.get('total_samples', 0)
    if total_samples < 20000:
        print(f"\n⚠️  SAMPLE COUNT WARNING:")
        print(f"   You have {total_samples:,} samples")
        print(f"   For robust BLIP3-o training, consider:")
        print(f"   • Extracting more shards")
        print(f"   • Using --max_shards=None (extract all available)")

def main():
    parser = argparse.ArgumentParser(description="Validate existing BLIP3-o embeddings")
    parser.add_argument("embeddings_dir", type=str, help="Path to embeddings directory")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    # Validate the extraction
    results = validate_extraction(args.embeddings_dir)
    results['embeddings_dir'] = args.embeddings_dir
    
    # Print recommendations
    print_recommendations(results)
    
    # Additional detailed analysis
    if args.detailed and results.get('valid', False):
        print("\n" + "=" * 60)
        print("📊 DETAILED ANALYSIS")
        print("=" * 60)
        print(f"Format: {results.get('format', 'unknown')}")
        print(f"Tokens per sample: {results.get('num_tokens', 0)}")
        print(f"CLIP dimension: {results.get('clip_dim', 0)}")
        print(f"EVA dimension: {results.get('eva_dim', 0)}")
        print(f"Samples per shard: ~{results.get('num_samples', 0)}")
        print(f"Total shards: {results.get('num_shards', 0)}")
        if results.get('total_samples'):
            print(f"Total samples: {results['total_samples']:,}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    return 0 if results.get('valid', False) else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)