#!/usr/bin/env python3
"""
Demo script to test ONNX vs PyTorch SAM performance.
This demonstrates the speed improvements from ONNX optimization.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from sam_config import SAMConfig, SAMBackend

def create_test_image():
    """Create a simple test image for benchmarking."""
    # Create a 512x512 RGB test image with some patterns
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add some geometric patterns for SAM to detect
    # Circle
    center = (256, 256)
    radius = 100
    y, x = np.ogrid[:512, :512]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    image[mask] = [255, 100, 100]  # Red circle
    
    # Rectangle
    image[150:200, 350:450] = [100, 255, 100]  # Green rectangle
    
    # Triangle-like shape
    for i in range(50):
        for j in range(50):
            if i + j > 25:
                image[100+i, 100+j] = [100, 100, 255]  # Blue triangle
    
    # Ensure the image is properly formatted for both backends
    return image.astype(np.uint8)

def test_pytorch_backend():
    """Test PyTorch backend performance."""
    print("=" * 50)
    print("🧠 TESTING PYTORCH BACKEND")
    print("=" * 50)
    
    # Configure for PyTorch
    config = SAMConfig(backend=SAMBackend.PYTORCH, model_type="vit_h")
    
    try:
        # Import and initialize
        from app import _initialize_pytorch_backend
        
        start_time = time.time()
        _initialize_pytorch_backend(config, Path("."))
        init_time = time.time() - start_time
        
        print(f"✅ PyTorch initialization: {init_time:.2f}s")
        
        # Test with sample image
        test_image = create_test_image()
        
        # Import the global generator
        from app import mask_generator
        
        print("🔍 Running PyTorch inference...")
        start_time = time.time()
        masks = mask_generator.generate(test_image)
        inference_time = time.time() - start_time
        
        print(f"✅ PyTorch inference: {inference_time:.2f}s")
        print(f"📊 Generated {len(masks)} masks")
        
        return {
            'backend': 'pytorch',
            'init_time': init_time,
            'inference_time': inference_time,
            'mask_count': len(masks)
        }
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return None

def test_onnx_backend():
    """Test ONNX backend performance."""
    print("=" * 50)
    print("⚡ TESTING ONNX BACKEND")
    print("=" * 50)
    
    # Configure for ONNX
    config = SAMConfig(backend=SAMBackend.ONNX, model_type="vit_h")
    
    try:
        # Check if ONNX models exist
        if not config.is_onnx_available():
            print("❌ ONNX models not found. Please run: python3 convert_model.py")
            return None
        
        # Import and initialize
        from app import _initialize_onnx_backend
        
        start_time = time.time()
        _initialize_onnx_backend(config)
        init_time = time.time() - start_time
        
        print(f"✅ ONNX initialization: {init_time:.2f}s")
        
        # Test with sample image
        test_image = create_test_image()
        
        # Import the global generator
        from app import mask_generator
        
        print("🚀 Running ONNX inference...")
        start_time = time.time()
        masks = mask_generator.generate(test_image)
        inference_time = time.time() - start_time
        
        print(f"✅ ONNX inference: {inference_time:.2f}s")
        print(f"📊 Generated {len(masks)} masks")
        
        return {
            'backend': 'onnx',
            'init_time': init_time,
            'inference_time': inference_time,
            'mask_count': len(masks)
        }
        
    except Exception as e:
        print(f"❌ ONNX test failed: {e}")
        return None

def compare_results(pytorch_result, onnx_result):
    """Compare performance between backends."""
    print("=" * 50)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    if not pytorch_result or not onnx_result:
        print("❌ Cannot compare - one or both backends failed")
        return
    
    # Initialization comparison
    init_speedup = pytorch_result['init_time'] / onnx_result['init_time']
    print(f"🚀 Initialization speedup: {init_speedup:.2f}x")
    print(f"   PyTorch: {pytorch_result['init_time']:.2f}s")
    print(f"   ONNX:    {onnx_result['init_time']:.2f}s")
    
    # Inference comparison
    inference_speedup = pytorch_result['inference_time'] / onnx_result['inference_time']
    print(f"⚡ Inference speedup: {inference_speedup:.2f}x")
    print(f"   PyTorch: {pytorch_result['inference_time']:.2f}s")
    print(f"   ONNX:    {onnx_result['inference_time']:.2f}s")
    
    # Mask count comparison
    print(f"🎯 Mask generation consistency:")
    print(f"   PyTorch: {pytorch_result['mask_count']} masks")
    print(f"   ONNX:    {onnx_result['mask_count']} masks")
    
    # Overall assessment
    total_pytorch = pytorch_result['init_time'] + pytorch_result['inference_time']
    total_onnx = onnx_result['init_time'] + onnx_result['inference_time']
    total_speedup = total_pytorch / total_onnx
    
    print(f"🏆 Total speedup: {total_speedup:.2f}x")
    print(f"   PyTorch total: {total_pytorch:.2f}s")
    print(f"   ONNX total:    {total_onnx:.2f}s")

def main():
    """Run the complete performance comparison."""
    print("🚀 SAM ONNX vs PyTorch Performance Demo")
    print("Testing with 512x512 synthetic image")
    print()
    
    # Test both backends
    pytorch_result = test_pytorch_backend()
    print()
    onnx_result = test_onnx_backend()
    print()
    
    # Compare results
    compare_results(pytorch_result, onnx_result)
    
    print()
    print("=" * 50)
    print("✅ Demo completed!")
    print("To use ONNX in production: SAM_BACKEND=onnx python3 app.py")
    print("=" * 50)

if __name__ == "__main__":
    main() 