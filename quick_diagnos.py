#!/usr/bin/env python3
"""
Quick Training Diagnosis Script
Run this to diagnose why training is stuck after model loading
"""

import os
import sys
import time
import psutil
import subprocess
from pathlib import Path

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "Command timed out"
    except Exception as e:
        return "", str(e)

def check_gpu_usage():
    """Check GPU usage"""
    print("🎮 GPU Status:")
    stdout, stderr = run_command("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
    if stdout:
        lines = stdout.strip().split('\n')
        for line in lines:
            parts = line.split(', ')
            if len(parts) >= 5:
                gpu_id, name, mem_used, mem_total, utilization = parts
                mem_percent = (int(mem_used) / int(mem_total)) * 100
                print(f"   GPU {gpu_id}: {utilization}% util, {mem_percent:.1f}% memory ({mem_used}/{mem_total} MB)")
    else:
        print("   ❌ Could not get GPU info")

def check_processes():
    """Check for training processes"""
    print("\n🔍 Training Processes:")
    
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if any(keyword in cmdline.lower() for keyword in ['train_blip3o', 'blip3o', 'train']):
                training_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if training_processes:
        for proc in training_processes:
            print(f"   PID {proc['pid']}: {proc['name']} (CPU: {proc['cpu_percent']:.1f}%, MEM: {proc['memory_percent']:.1f}%)")
            cmdline = ' '.join(proc['cmdline'][:3]) + "..." if len(proc['cmdline']) > 3 else ' '.join(proc['cmdline'])
            print(f"      Command: {cmdline}")
    else:
        print("   ❌ No training processes found")

def check_disk_space():
    """Check disk space"""
    print("\n💾 Disk Space:")
    try:
        total, used, free = psutil.disk_usage('.')
        percent_used = (used / total) * 100
        print(f"   Current dir: {percent_used:.1f}% used ({free // (1024**3)} GB free)")
        
        # Check embeddings directory
        embeddings_dir = Path("embeddings/chunked_256_tokens")
        if embeddings_dir.exists():
            total, used, free = psutil.disk_usage(embeddings_dir)
            percent_used = (used / total) * 100
            print(f"   Embeddings dir: {percent_used:.1f}% used ({free // (1024**3)} GB free)")
        else:
            print("   ⚠️  Embeddings directory not found")
    except Exception as e:
        print(f"   ❌ Error checking disk: {e}")

def check_log_files():
    """Check for recent log output"""
    print("\n📋 Recent Log Activity:")
    
    log_patterns = [
        "slurm-*.out",
        "slurm-*.err", 
        "*training*.log",
        "blip3o*.log"
    ]
    
    recent_logs = []
    for pattern in log_patterns:
        for log_file in Path(".").glob(pattern):
            try:
                stat = log_file.stat()
                age_minutes = (time.time() - stat.st_mtime) / 60
                if age_minutes < 30:  # Modified in last 30 minutes
                    recent_logs.append((log_file, age_minutes, stat.st_size))
            except:
                continue
    
    if recent_logs:
        recent_logs.sort(key=lambda x: x[1])  # Sort by age
        for log_file, age, size in recent_logs:
            print(f"   📄 {log_file.name}: {age:.1f}m old, {size} bytes")
            
            # Show last few lines
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"      Last line: {lines[-1].strip()}")
            except:
                pass
    else:
        print("   ⚠️  No recent log files found")

def check_dataset():
    """Check dataset status"""
    print("\n📊 Dataset Status:")
    
    embeddings_dir = Path("embeddings/chunked_256_tokens")
    if not embeddings_dir.exists():
        print("   ❌ Embeddings directory not found")
        return
    
    # Check manifest
    manifest_file = embeddings_dir / "embeddings_manifest.json"
    if manifest_file.exists():
        try:
            import json
            with open(manifest_file) as f:
                manifest = json.load(f)
            print(f"   ✅ Manifest: {manifest.get('total_shards', 0)} shards, {manifest.get('total_samples', 0):,} samples")
        except Exception as e:
            print(f"   ❌ Could not read manifest: {e}")
    else:
        print("   ❌ No manifest file found")
    
    # Check shard files
    shard_files = list(embeddings_dir.glob("embeddings_shard_*.pkl"))
    if shard_files:
        print(f"   📁 Found {len(shard_files)} shard files")
        
        # Check first shard
        first_shard = sorted(shard_files)[0]
        try:
            size_mb = first_shard.stat().st_size / (1024 * 1024)
            print(f"   📄 First shard: {first_shard.name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"   ⚠️  Could not check first shard: {e}")
    else:
        print("   ❌ No shard files found")

def diagnose_stuck_training():
    """Main diagnosis function"""
    print("🔍 BLIP3-o Training Diagnosis")
    print("=" * 50)
    
    check_processes()
    check_gpu_usage()
    check_disk_space()
    check_log_files()
    check_dataset()
    
    print("\n" + "=" * 50)
    print("💡 LIKELY CAUSES:")
    print("=" * 50)
    
    # Analyze findings
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if any(keyword in cmdline.lower() for keyword in ['train_blip3o', 'blip3o']):
                training_processes.append(proc.info)
        except:
            continue
    
    if not training_processes:
        print("❌ PROCESS DIED: Training process is not running")
        print("   → Check error logs with: tail -100 slurm-*.err")
        print("   → Look for Python errors or out-of-memory issues")
    
    else:
        print("✅ PROCESS RUNNING: Training process is active")
        
        # Check GPU usage
        stdout, _ = run_command("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
        if stdout:
            gpu_utils = [int(x.strip()) for x in stdout.strip().split('\n')]
            max_util = max(gpu_utils) if gpu_utils else 0
            
            if max_util < 5:
                print("❌ NO GPU ACTIVITY: GPUs not being used")
                print("   → Likely stuck in data loading or initialization")
                print("   → Check dataset access with: ls -la embeddings/chunked_256_tokens/")
            elif max_util < 50:
                print("⚠️  LOW GPU ACTIVITY: Some GPU usage but low")
                print("   → Possibly slow data loading or small batch size")
            else:
                print("✅ GOOD GPU ACTIVITY: Training likely running")
                print("   → Check if logging is disabled or interval too high")
        
        # Check recent log activity
        recent_logs = []
        for log_file in Path(".").glob("*.log"):
            try:
                age_minutes = (time.time() - log_file.stat().st_mtime) / 60
                if age_minutes < 10:
                    recent_logs.append(log_file)
            except:
                continue
        
        if not recent_logs:
            print("⚠️  NO RECENT LOGS: No log files updated recently")
            print("   → Logging might be disabled or redirected")
            print("   → Check SLURM output: tail -f slurm-*.out")
    
    print("\n💡 IMMEDIATE ACTIONS:")
    print("1. Check latest logs: tail -f slurm-*.out")
    print("2. Monitor GPU: watch nvidia-smi")
    print("3. Check if first batch is loading: ls -la /tmp/*")
    print("4. If stuck >15min, restart with smaller batch size")

if __name__ == "__main__":
    diagnose_stuck_training()