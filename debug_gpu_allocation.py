#!/usr/bin/env python3
"""
GPU Diagnostics Script for BLIP3-o Training
File: gpu_diagnostics.py

Run this script to diagnose GPU access issues and get recommendations.
"""

import os
import sys
import torch
import subprocess
import json
from pathlib import Path

def check_cuda_installation():
    """Check CUDA installation and compatibility"""
    print("🔍 CUDA Installation Check")
    print("=" * 50)
    
    try:
        # Check PyTorch CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"PyTorch version: {torch.__version__}")
            
            # Check GPU count and details
            gpu_count = torch.cuda.device_count()
            print(f"GPU count: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        else:
            print("❌ CUDA not available in PyTorch")
            
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
    
    print()

def check_nvidia_smi():
    """Check nvidia-smi output"""
    print("🎮 NVIDIA SMI Check")
    print("=" * 50)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ nvidia-smi working")
            print("GPU Status:")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'A100' in line or 'V100' in line or 'H100' in line:
                    print(f"  {line.strip()}")
        else:
            print(f"❌ nvidia-smi failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timed out")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
    
    print()

def check_environment_variables():
    """Check relevant environment variables"""
    print("🌍 Environment Variables")
    print("=" * 50)
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'SLURM_GPUS',
        'SLURM_GPUS_ON_NODE', 
        'SLURM_LOCALID',
        'SLURM_PROCID',
        'LOCAL_RANK',
        'RANK',
        'WORLD_SIZE',
        'MASTER_ADDR',
        'MASTER_PORT',
        'NCCL_DEBUG',
        'PYTORCH_CUDA_ALLOC_CONF'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    print()

def check_slurm_allocation():
    """Check SLURM GPU allocation"""
    print("⚡ SLURM Allocation Check")
    print("=" * 50)
    
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ['SLURM_JOB_ID']
        print(f"SLURM Job ID: {job_id}")
        
        # Try to get job info
        try:
            result = subprocess.run(['scontrol', 'show', 'job', job_id], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Gres=' in line or 'TresPerNode=' in line:
                        print(f"  {line.strip()}")
            else:
                print(f"Could not get job info: {result.stderr}")
        except Exception as e:
            print(f"Error getting SLURM job info: {e}")
    else:
        print("Not running under SLURM")
    
    print()

def test_gpu_operations():
    """Test basic GPU operations"""
    print("🧪 GPU Operations Test")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU tests")
        return
    
    try:
        # Test tensor creation and operations
        device = torch.device('cuda:0')
        print(f"Testing operations on {device}")
        
        # Create tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        z = torch.mm(x, y)
        print("✅ Matrix multiplication successful")
        
        # Memory check
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        memory_cached = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"Memory: {memory_allocated:.1f} MB allocated, {memory_cached:.1f} MB cached")
        
        # Test multiple GPUs if available
        if torch.cuda.device_count() > 1:
            print("Testing multi-GPU operations...")
            for i in range(torch.cuda.device_count()):
                device_i = torch.device(f'cuda:{i}')
                test_tensor = torch.randn(100, 100, device=device_i)
                print(f"✅ GPU {i} accessible")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def test_distributed_setup():
    """Test distributed training setup"""
    print("🔗 Distributed Training Test")
    print("=" * 50)
    
    # Check if we're in a distributed environment
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        print("Distributed environment detected")
        
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
        
        # Test DDP initialization
        try:
            import torch.distributed as dist
            
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                backend = 'nccl'
            else:
                backend = 'gloo'
            
            print(f"Attempting to initialize DDP with {backend} backend...")
            
            # This would normally be done in the training script
            print("⚠️  DDP initialization test skipped (would conflict with actual training)")
            print(f"Recommended backend: {backend}")
            
        except Exception as e:
            print(f"❌ DDP test failed: {e}")
    else:
        print("Single-node environment")
    
    print()

def generate_recommendations():
    """Generate recommendations based on diagnostics"""
    print("💡 Recommendations")
    print("=" * 50)
    
    recommendations = []
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        recommendations.append("❌ CUDA not available:")
        recommendations.append("  • Check that you're running on a GPU node")
        recommendations.append("  • Verify CUDA drivers are installed")
        recommendations.append("  • Check module loading in your job script")
        recommendations.append("  • Try: module load CUDA/12.6.0")
    
    # Check GPU allocation
    if 'SLURM_GPUS' not in os.environ and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        recommendations.append("⚠️  No GPU allocation detected:")
        recommendations.append("  • Add #SBATCH --gpus=N to your job script")
        recommendations.append("  • Or use #SBATCH --gres=gpu:N")
        recommendations.append("  • Check if you're on a GPU partition")
    
    # Check environment setup
    if torch.cuda.is_available() and torch.cuda.device_count() == 0:
        recommendations.append("❌ CUDA available but no devices:")
        recommendations.append("  • CUDA_VISIBLE_DEVICES might be empty")
        recommendations.append("  • Check SLURM GPU allocation")
        recommendations.append("  • Try unsetting CUDA_VISIBLE_DEVICES")
    
    # Multi-GPU specific
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        if 'MASTER_ADDR' not in os.environ:
            recommendations.append("❌ Distributed training setup incomplete:")
            recommendations.append("  • MASTER_ADDR not set")
            recommendations.append("  • Use torchrun for proper DDP setup")
        
        if torch.cuda.is_available() and torch.cuda.device_count() < world_size:
            recommendations.append(f"❌ GPU count mismatch:")
            recommendations.append(f"  • WORLD_SIZE={world_size} but only {torch.cuda.device_count()} GPUs visible")
            recommendations.append("  • Check SLURM allocation vs torchrun parameters")
    
    if not recommendations:
        recommendations.append("✅ No major issues detected!")
        recommendations.append("✅ GPU setup looks good")
    
    for rec in recommendations:
        print(rec)
    
    print()

def create_debug_job_script():
    """Create a debugging job script"""
    print("📝 Creating Debug Job Script")
    print("=" * 50)
    
    debug_script = """#!/bin/bash
#SBATCH --job-name=gpu_debug
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=18
#SBATCH --time=0:30:00
#SBATCH --output=./gpu_debug_%j.out
#SBATCH --error=./gpu_debug_%j.err

echo "🔍 GPU Debug Job Started"
echo "========================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

# Load modules
module purge
module load 2024
module load Miniconda3/24.7.1-0
module load CUDA/12.6.0

echo "✅ Modules loaded"

# Activate environment
source activate eva_clip_env
echo "✅ Environment activated"

# Show environment
echo ""
echo "📊 Environment Check:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# Run diagnostics
echo ""
echo "🧪 Running GPU diagnostics..."
python gpu_diagnostics.py

echo ""
echo "🔍 Testing simple GPU operations..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        x = torch.randn(100, 100, device=f'cuda:{i}')
        print(f'  ✅ GPU {i} working')
else:
    print('❌ No CUDA available')
"

echo ""
echo "🏁 Debug job completed"
"""
    
    with open('debug_gpu.job', 'w') as f:
        f.write(debug_script)
    
    print("Created debug_gpu.job")
    print("Run with: sbatch debug_gpu.job")
    print()

def main():
    """Main diagnostics function"""
    print("🚀 BLIP3-o GPU Diagnostics")
    print("=" * 50)
    print()
    
    check_cuda_installation()
    check_nvidia_smi()
    check_environment_variables()
    check_slurm_allocation()
    test_gpu_operations()
    test_distributed_setup()
    generate_recommendations()
    create_debug_job_script()
    
    # Save results to file
    print("📁 Saving diagnostics to gpu_diagnostics_results.json")
    results = {
        'timestamp': str(torch.utils.data.default_collate.__globals__.get('time', __import__('time')).time()),
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'environment_vars': {k: os.environ.get(k) for k in [
            'CUDA_VISIBLE_DEVICES', 'SLURM_GPUS', 'WORLD_SIZE', 'LOCAL_RANK'
        ]},
    }
    
    if torch.cuda.is_available():
        results['gpus'] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            results['gpus'].append({
                'id': i,
                'name': props.name,
                'memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}"
            })
    
    with open('gpu_diagnostics_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Diagnostics complete!")

if __name__ == "__main__":
    main()