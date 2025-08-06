#!/usr/bin/env python3
"""
Create performance visualization charts for SAM backend comparison.
"""

import matplotlib.pyplot as plt
import numpy as np

def create_performance_charts():
    """Create performance comparison charts."""
    
    # Data from our benchmark
    backends = ['PyTorch', 'ONNX', 'TensorRT\nFP32', 'TensorRT\nFP16', 'TensorRT\nINT8']
    inference_times = [131.89, 114.01, 65.00, 37.14, 26.00]
    speedups = [1.00, 1.16, 2.03, 3.55, 5.07]
    throughput = [0.008, 0.009, 0.015, 0.027, 0.038]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SAM Performance Comparison: PyTorch vs ONNX vs TensorRT', fontsize=16, fontweight='bold')
    
    # Colors for different backends
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    # Chart 1: Inference Time
    bars1 = ax1.bar(backends, inference_times, color=colors, alpha=0.8)
    ax1.set_ylabel('Inference Time (seconds)', fontweight='bold')
    ax1.set_title('Inference Time Comparison', fontweight='bold')
    ax1.set_ylim(0, max(inference_times) * 1.1)
    
    # Add value labels on bars
    for bar, time in zip(bars1, inference_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Speedup
    bars2 = ax2.bar(backends, speedups, color=colors, alpha=0.8)
    ax2.set_ylabel('Speedup vs PyTorch', fontweight='bold')
    ax2.set_title('Performance Speedup', fontweight='bold')
    ax2.set_ylim(0, max(speedups) * 1.1)
    
    # Add value labels on bars
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{speedup:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # Chart 3: Throughput
    bars3 = ax3.bar(backends, throughput, color=colors, alpha=0.8)
    ax3.set_ylabel('Throughput (images/second)', fontweight='bold')
    ax3.set_title('Processing Throughput', fontweight='bold')
    ax3.set_ylim(0, max(throughput) * 1.1)
    
    # Add value labels on bars
    for bar, tput in zip(bars3, throughput):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{tput:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Chart 4: Cost Analysis (relative)
    baseline_cost = 100  # Percentage
    costs = [baseline_cost / speedup for speedup in speedups]
    
    bars4 = ax4.bar(backends, costs, color=colors, alpha=0.8)
    ax4.set_ylabel('Relative Cost (%)', fontweight='bold')
    ax4.set_title('Compute Cost Comparison', fontweight='bold')
    ax4.set_ylim(0, 110)
    
    # Add value labels on bars
    for bar, cost in zip(bars4, costs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{cost:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save high-quality image
    plt.savefig('sam_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Performance charts saved to: sam_performance_comparison.png")
    
    # Show the plot
    plt.show()

def create_real_world_impact_chart():
    """Create a chart showing real-world processing time impacts."""
    
    # Real-world scenario: satellite droplet analysis (2,795 masks)
    backends = ['PyTorch', 'ONNX', 'TensorRT FP16', 'TensorRT INT8']
    processing_times = [207, 178, 59, 41]  # seconds
    processing_minutes = [t/60 for t in processing_times]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Real-World Impact: Satellite Droplet Analysis (2,795 masks)', fontsize=14, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#96CEB4', '#FECA57']
    
    # Processing time in seconds
    bars1 = ax1.bar(backends, processing_times, color=colors, alpha=0.8)
    ax1.set_ylabel('Processing Time (seconds)', fontweight='bold')
    ax1.set_title('Total Processing Time', fontweight='bold')
    
    for bar, time in zip(bars1, processing_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time}s', ha='center', va='bottom', fontweight='bold')
    
    # Processing time in minutes for easier understanding
    bars2 = ax2.bar(backends, processing_minutes, color=colors, alpha=0.8)
    ax2.set_ylabel('Processing Time (minutes)', fontweight='bold')
    ax2.set_title('User Experience Impact', fontweight='bold')
    
    for bar, time in zip(bars2, processing_minutes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('real_world_impact.png', dpi=300, bbox_inches='tight')
    print("📊 Real-world impact chart saved to: real_world_impact.png")
    plt.show()

if __name__ == "__main__":
    print("🎨 Creating SAM Performance Visualization Charts...")
    
    try:
        # Create comprehensive performance charts
        create_performance_charts()
        
        # Create real-world impact visualization
        create_real_world_impact_chart()
        
        print("\n✅ All charts created successfully!")
        print("📁 Check the generated PNG files for detailed visualizations")
        
    except Exception as e:
        print(f"❌ Error creating charts: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib") 