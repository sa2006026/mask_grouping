#!/usr/bin/env python3
"""Simple GPU usage checker for SAM processing."""

import subprocess
import time
import re

def get_gpu_usage_simple():
    """Get GPU usage using simple nvidia-smi parsing."""
    try:
        # Run nvidia-smi with specific format
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        'id': int(parts[0]),
                        'name': parts[1],
                        'utilization': int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                        'memory_used_mb': int(parts[3]),
                        'memory_total_mb': int(parts[4]),
                        'temperature': int(parts[5]) if parts[5] != '[Not Supported]' else 0,
                        'power': float(parts[6]) if len(parts) > 6 and parts[6] != '[Not Supported]' else 0
                    })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def check_gpu_threshold(threshold=30):
    """Check if GPUs are below usage threshold."""
    gpus = get_gpu_usage_simple()
    
    if not gpus:
        print("❌ No GPUs found or nvidia-smi failed")
        return False
    
    print(f"🔍 GPU Usage Check (Target: ≤{threshold}%)")
    print("=" * 60)
    
    all_below_threshold = True
    
    for gpu in gpus:
        util = gpu['utilization']
        memory_used_gb = gpu['memory_used_mb'] / 1024
        memory_total_gb = gpu['memory_total_mb'] / 1024
        memory_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
        
        status = "✅" if util <= threshold else "❌"
        print(f"{status} GPU {gpu['id']}: {util:3d}% utilization")
        print(f"    Name: {gpu['name']}")
        print(f"    Memory: {memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB ({memory_percent:.1f}%)")
        print(f"    Temperature: {gpu['temperature']}°C")
        if gpu['power'] > 0:
            print(f"    Power: {gpu['power']:.0f}W")
        print()
        
        if util > threshold:
            all_below_threshold = False
    
    if all_below_threshold:
        print(f"🎉 All GPUs are below {threshold}% utilization threshold!")
    else:
        print(f"⚠️  Some GPUs are above {threshold}% threshold")
        print("\n💡 Suggestions to reduce GPU usage:")
        print("  • Use smaller SAM model: SAM_MODEL_TYPE=vit_l or vit_b")
        print("  • Reduce processing parameters: points_per_side=16, crop_n_layers=1")
        print("  • Use only one GPU: CUDA_VISIBLE_DEVICES=0")
        print("  • Process smaller images or batches")
    
    return all_below_threshold

def monitor_gpu(duration=60, interval=5, threshold=30):
    """Monitor GPU usage over time."""
    print(f"🔍 Monitoring GPU usage for {duration} seconds")
    print(f"🎯 Target: Keep all GPUs ≤{threshold}%")
    print(f"⏱️  Checking every {interval} seconds")
    print("=" * 60)
    
    start_time = time.time()
    violations = 0
    checks = 0
    
    try:
        while time.time() - start_time < duration:
            gpus = get_gpu_usage_simple()
            timestamp = time.strftime("%H:%M:%S")
            
            if gpus:
                print(f"\n📊 {timestamp}")
                max_util = 0
                above_threshold = False
                
                for gpu in gpus:
                    util = gpu['utilization']
                    memory_gb = gpu['memory_used_mb'] / 1024
                    temp = gpu['temperature']
                    
                    max_util = max(max_util, util)
                    if util > threshold:
                        above_threshold = True
                    
                    status = "✅" if util <= threshold else "⚠️"
                    print(f"  {status} GPU {gpu['id']}: {util:3d}% | {memory_gb:5.1f}GB | {temp:2d}°C")
                
                if above_threshold:
                    violations += 1
                    print(f"  ❌ VIOLATION: GPU usage above {threshold}%")
                
                checks += 1
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
    
    violation_rate = (violations / checks * 100) if checks > 0 else 0
    print(f"\n📋 Summary:")
    print(f"  Total checks: {checks}")
    print(f"  Violations: {violations} ({violation_rate:.1f}%)")
    print(f"  Success rate: {100-violation_rate:.1f}%")

def watch_gpu_realtime():
    """Watch GPU usage in real-time (like top)."""
    print("🔍 Real-time GPU monitoring (Press Ctrl+C to stop)")
    print("=" * 60)
    
    try:
        while True:
            # Clear screen (ANSI escape code)
            print("\033[2J\033[H", end="")
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"GPU Monitor - {timestamp}")
            print("=" * 60)
            
            gpus = get_gpu_usage_simple()
            
            if gpus:
                for gpu in gpus:
                    util = gpu['utilization']
                    memory_used_gb = gpu['memory_used_mb'] / 1024
                    memory_total_gb = gpu['memory_total_mb'] / 1024
                    memory_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
                    
                    # Create usage bar
                    bar_length = 20
                    filled = int(util / 100 * bar_length)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    
                    status = "✅" if util <= 30 else "⚠️" if util <= 70 else "🔥"
                    
                    print(f"{status} GPU {gpu['id']}: {gpu['name']}")
                    print(f"    Utilization: [{bar}] {util:3d}%")
                    print(f"    Memory:      {memory_used_gb:5.1f}GB / {memory_total_gb:.1f}GB ({memory_percent:.1f}%)")
                    print(f"    Temperature: {gpu['temperature']}°C")
                    if gpu['power'] > 0:
                        print(f"    Power:       {gpu['power']:.0f}W")
                    print()
            else:
                print("❌ No GPU data available")
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            check_gpu_threshold(threshold)
        elif command == "monitor":
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            monitor_gpu(duration, 5, threshold)
        elif command == "watch":
            watch_gpu_realtime()
        else:
            print("Usage: python3 simple_gpu_check.py [check|monitor|watch] [args...]")
            print("  check [threshold]           - Single check (default threshold: 30%)")
            print("  monitor [duration] [threshold] - Monitor for N seconds (default: 60s, 30%)")
            print("  watch                       - Real-time monitoring")
    else:
        # Default: single check with 30% threshold
        check_gpu_threshold(30) 