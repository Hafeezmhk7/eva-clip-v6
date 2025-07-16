#!/usr/bin/env python3
"""
Test Script for Distance Evaluation (Task 3)
This script tests the distance evaluation functionality before running on real data.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.modules.evaluation.distance_metrics import (
        compute_comprehensive_distance_metrics,
        compute_per_sample_distances,
        analyze_distance_distribution,
        print_distance_metrics,
        compute_l2_distance,
        compute_l1_distance,
        compute_cosine_distance,
        compute_mse_distance,
        compute_mae_distance
    )
    print("✅ Successfully imported distance metrics functions")
except ImportError as e:
    print(f"❌ Failed to import distance metrics: {e}")
    print("Make sure you've added distance_metrics.py to src/modules/evaluation/")
    sys.exit(1)


def test_basic_distance_functions():
    """Test basic distance computation functions."""
    print("\n🧪 Testing basic distance functions...")
    
    # Create test data
    target = torch.randn(50, 768)  # Target embeddings
    predicted = target + torch.randn(50, 768) * 0.1  # Predicted with small noise
    
    # Test each distance function
    try:
        l2_dist = compute_l2_distance(target, predicted)
        print(f"   L2 distance: {l2_dist:.6f}")
        
        l1_dist = compute_l1_distance(target, predicted)
        print(f"   L1 distance: {l1_dist:.6f}")
        
        cosine_dist = compute_cosine_distance(target, predicted)
        print(f"   Cosine distance: {cosine_dist:.6f}")
        
        mse_dist = compute_mse_distance(target, predicted)
        print(f"   MSE distance: {mse_dist:.6f}")
        
        mae_dist = compute_mae_distance(target, predicted)
        print(f"   MAE distance: {mae_dist:.6f}")
        
        print("✅ All basic distance functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Basic distance functions failed: {e}")
        return False


def test_comprehensive_metrics():
    """Test comprehensive distance metrics computation."""
    print("\n🧪 Testing comprehensive metrics...")
    
    try:
        # Create test scenarios
        scenarios = [
            ("Perfect match", lambda x: x, "target == predicted"),
            ("Small noise", lambda x: x + torch.randn_like(x) * 0.1, "target + small noise"),
            ("Large noise", lambda x: x + torch.randn_like(x) * 0.5, "target + large noise"),
            ("Random", lambda x: torch.randn_like(x), "completely random"),
        ]
        
        for name, transform, description in scenarios:
            print(f"\n   Scenario: {name} ({description})")
            
            target = torch.randn(100, 768)
            predicted = transform(target)
            
            metrics = compute_comprehensive_distance_metrics(target, predicted)
            
            print(f"      L2 distance: {metrics['l2_distance_mean']:.6f}")
            print(f"      Cosine similarity: {metrics['cosine_similarity_mean']:.4f}")
            print(f"      MSE: {metrics['mse_distance']:.6f}")
            
            # Validate expected behavior
            if name == "Perfect match":
                assert metrics['l2_distance_mean'] < 1e-6, "Perfect match should have zero distance"
                assert metrics['cosine_similarity_mean'] > 0.999, "Perfect match should have cosine sim ~1"
            elif name == "Random":
                assert metrics['cosine_similarity_mean'] < 0.5, "Random should have low cosine similarity"
        
        print("✅ Comprehensive metrics work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive metrics failed: {e}")
        return False


def test_per_sample_analysis():
    """Test per-sample distance analysis."""
    print("\n🧪 Testing per-sample analysis...")
    
    try:
        target = torch.randn(50, 768)
        predicted = target + torch.randn(50, 768) * 0.2
        
        per_sample = compute_per_sample_distances(target, predicted)
        
        # Check that we get per-sample results
        assert 'l2_distances' in per_sample, "Missing L2 distances"
        assert 'cosine_similarities' in per_sample, "Missing cosine similarities"
        assert len(per_sample['l2_distances']) == 50, "Wrong number of samples"
        
        print(f"   Per-sample L2 distances: min={per_sample['l2_distances'].min():.4f}, "
              f"max={per_sample['l2_distances'].max():.4f}")
        print(f"   Per-sample cosine similarities: min={per_sample['cosine_similarities'].min():.4f}, "
              f"max={per_sample['cosine_similarities'].max():.4f}")
        
        print("✅ Per-sample analysis works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Per-sample analysis failed: {e}")
        return False


def test_distribution_analysis():
    """Test distance distribution analysis."""
    print("\n🧪 Testing distribution analysis...")
    
    try:
        target = torch.randn(200, 768)
        predicted = target + torch.randn(200, 768) * 0.3
        
        distribution = analyze_distance_distribution(target, predicted)
        
        # Check that we get distribution analysis
        assert 'l2_distances_percentiles' in distribution, "Missing L2 percentiles"
        assert 'l2_distances_histogram' in distribution, "Missing L2 histogram"
        
        percentiles = distribution['l2_distances_percentiles']
        print(f"   L2 distance percentiles: p25={percentiles['p25']:.4f}, "
              f"p50={percentiles['p50']:.4f}, p75={percentiles['p75']:.4f}")
        
        histogram = distribution['l2_distances_histogram']
        print(f"   Histogram bins: {len(histogram['bin_edges'])-1}")
        print(f"   Total count: {sum(histogram['counts'])}")
        
        print("✅ Distribution analysis works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Distribution analysis failed: {e}")
        return False


def test_different_embedding_sizes():
    """Test with different embedding dimensions."""
    print("\n🧪 Testing different embedding dimensions...")
    
    dimensions = [512, 768, 1024, 2048]
    
    try:
        for dim in dimensions:
            print(f"   Testing {dim}-dimensional embeddings...")
            target = torch.randn(50, dim)
            predicted = target + torch.randn(50, dim) * 0.1
            
            metrics = compute_comprehensive_distance_metrics(target, predicted)
            
            assert metrics['embedding_dimension'] == dim, f"Wrong dimension reported"
            assert metrics['num_samples'] == 50, f"Wrong sample count"
            
            print(f"      {dim}D: L2={metrics['l2_distance_mean']:.4f}, "
                  f"cosine_sim={metrics['cosine_similarity_mean']:.4f}")
        
        print("✅ Different embedding dimensions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Different embedding dimensions failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing edge cases...")
    
    try:
        # Test identical embeddings (zero distance)
        target = torch.randn(10, 768)
        predicted = target.clone()
        
        metrics = compute_comprehensive_distance_metrics(target, predicted)
        assert metrics['l2_distance_mean'] < 1e-6, "Identical embeddings should have zero distance"
        assert abs(metrics['cosine_similarity_mean'] - 1.0) < 1e-6, "Identical embeddings should have cosine sim = 1"
        
        # Test orthogonal embeddings (maximum cosine distance)
        target = torch.zeros(2, 4)
        target[0, :2] = 1.0  # [1, 1, 0, 0]
        target[1, 2:] = 1.0  # [0, 0, 1, 1]
        
        predicted = torch.zeros(2, 4)
        predicted[0, 2:] = 1.0  # [0, 0, 1, 1]
        predicted[1, :2] = 1.0  # [1, 1, 0, 0]
        
        metrics = compute_comprehensive_distance_metrics(target, predicted)
        print(f"   Orthogonal embeddings cosine similarity: {metrics['cosine_similarity_mean']:.6f}")
        # Should be close to 0 for orthogonal vectors
        
        # Test with very small embeddings
        target = torch.randn(5, 768) * 1e-8
        predicted = torch.randn(5, 768) * 1e-8
        
        metrics = compute_comprehensive_distance_metrics(target, predicted)
        # Should not crash with numerical issues
        
        print("✅ Edge cases handled correctly")
        return True
        
    except Exception as e:
        print(f"❌ Edge cases failed: {e}")
        return False


def test_print_functionality():
    """Test the printing and formatting functions."""
    print("\n🧪 Testing print functionality...")
    
    try:
        target = torch.randn(50, 768)
        predicted = target + torch.randn(50, 768) * 0.2
        
        metrics = compute_comprehensive_distance_metrics(target, predicted)
        
        print("   Testing print_distance_metrics...")
        print_distance_metrics(metrics, "Test Distance Metrics")
        
        print("✅ Print functionality works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Print functionality failed: {e}")
        return False


def run_integration_test():
    """Run a full integration test simulating real usage."""
    print("\n🧪 Running integration test...")
    
    try:
        # Simulate real-world scenario
        print("   Simulating BLIP3-o evaluation scenario...")
        
        # Simulate COCO validation set
        num_images = 100
        
        # Target: CLIP ViT-L/14 embeddings (768-dim with visual projection)
        target_embeddings = torch.randn(num_images, 768)
        target_embeddings = torch.nn.functional.normalize(target_embeddings, p=2, dim=-1)
        
        # Predicted: BLIP3-o generated embeddings (with some systematic bias and noise)
        bias = torch.randn(768) * 0.1  # Small systematic bias
        noise_scale = 0.15
        predicted_embeddings = target_embeddings + bias.unsqueeze(0) + torch.randn(num_images, 768) * noise_scale
        predicted_embeddings = torch.nn.functional.normalize(predicted_embeddings, p=2, dim=-1)
        
        # Run full evaluation
        print(f"   Evaluating {num_images} image embeddings...")
        
        # Comprehensive metrics
        metrics = compute_comprehensive_distance_metrics(target_embeddings, predicted_embeddings)
        
        # Per-sample analysis
        per_sample = compute_per_sample_distances(target_embeddings, predicted_embeddings)
        
        # Distribution analysis
        distribution = analyze_distance_distribution(target_embeddings, predicted_embeddings)
        
        # Print summary
        print(f"\n   📊 Integration Test Results:")
        print(f"      L2 distance: {metrics['l2_distance_mean']:.6f} ± {metrics['l2_distance_std']:.6f}")
        print(f"      Cosine similarity: {metrics['cosine_similarity_mean']:.4f} ± {metrics['cosine_similarity_std']:.4f}")
        print(f"      MSE: {metrics['mse_distance']:.6f}")
        print(f"      Samples processed: {metrics['num_samples']}")
        print(f"      Embedding dimension: {metrics['embedding_dimension']}")
        
        # Validate reasonable results
        assert 0.0 <= metrics['cosine_similarity_mean'] <= 1.0, "Cosine similarity out of range"
        assert metrics['l2_distance_mean'] > 0, "L2 distance should be positive"
        assert metrics['num_samples'] == num_images, "Sample count mismatch"
        
        print("✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Distance Evaluation Implementation")
    print("=" * 60)
    
    tests = [
        ("Basic Distance Functions", test_basic_distance_functions),
        ("Comprehensive Metrics", test_comprehensive_metrics),
        ("Per-Sample Analysis", test_per_sample_analysis),
        ("Distribution Analysis", test_distribution_analysis),
        ("Different Embedding Sizes", test_different_embedding_sizes),
        ("Edge Cases", test_edge_cases),
        ("Print Functionality", test_print_functionality),
        ("Integration Test", run_integration_test),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Distance evaluation implementation is ready to use")
        print("\nNext steps:")
        print("1. Copy distance_metrics.py to src/modules/evaluation/")
        print("2. Add methods to evaluator.py (see integration_patch.py)")
        print("3. Copy evaluate_distance.py to project root")
        print("4. Run: python evaluate_distance.py --blip3o_model_path <path> --coco_root <path>")
        return 0
    else:
        print(f"\n❌ {failed} tests failed")
        print("Please fix the issues before using the distance evaluation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)