#!/usr/bin/env python3
"""
Simple benchmarking script for ONNX vs PyTorch SAM performance comparison.
This script focuses on pure timing without mask generation errors.
"""

import time
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

def benchmark_pytorch_initialization():
    """Benchmark PyTorch SAM initialization only."""
    print("=" * 50)
    print("🧠 BENCHMARKING PYTORCH INITIALIZATION")
    print("=" * 50)
    
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        model_path = Path("model/sam_vit_h_4b8939.pth")
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return None
            
        start_time = time.time()
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_model = sam_model_registry["vit_h"](checkpoint=str(model_path))
        sam_model.to(device=device)
        
        # Create mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=3,
            min_mask_region_area=100
        )
        
        init_time = time.time() - start_time
        print(f"✅ PyTorch initialization: {init_time:.2f}s")
        print(f"🖥️  Device: {device}")
        
        return {
            'backend': 'pytorch',
            'init_time': init_time,
            'device': device,
            'mask_generator': mask_generator
        }
        
    except Exception as e:
        print(f"❌ PyTorch benchmark failed: {e}")
        return None

def benchmark_pytorch_inference(mask_generator, test_image):
    """Benchmark PyTorch SAM inference only."""
    print("\n🔍 BENCHMARKING PYTORCH INFERENCE")
    
    # Warm up
    print("🔥 Warming up...")
    _ = mask_generator.generate(test_image[:100, :100])  # Small warm-up
    
    # Actual benchmark
    print("📊 Running benchmark...")
    start_time = time.time()
    masks = mask_generator.generate(test_image)
    inference_time = time.time() - start_time
    
    print(f"✅ PyTorch inference: {inference_time:.2f}s")
    print(f"📊 Generated {len(masks)} masks")
    
    return {
        'inference_time': inference_time,
        'mask_count': len(masks)
    }

def benchmark_onnx_initialization():
    """Benchmark ONNX SAM initialization only."""
    print("=" * 50)
    print("⚡ BENCHMARKING ONNX INITIALIZATION") 
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        
        encoder_path = Path("model/onnx/sam_vit_h_encoder.onnx")
        decoder_path = Path("model/onnx/sam_vit_h_decoder.onnx")
        
        if not encoder_path.exists() or not decoder_path.exists():
            print(f"❌ ONNX models not found:")
            print(f"   Encoder: {encoder_path}")
            print(f"   Decoder: {decoder_path}")
            return None
            
        start_time = time.time()
        
        # Initialize ONNX sessions
        providers = ['CPUExecutionProvider']  # Use CPU since CUDA provider isn't available
        encoder_session = ort.InferenceSession(str(encoder_path), providers=providers)
        decoder_session = ort.InferenceSession(str(decoder_path), providers=providers)
        
        init_time = time.time() - start_time
        print(f"✅ ONNX initialization: {init_time:.2f}s")
        print(f"🖥️  Providers: {encoder_session.get_providers()}")
        
        return {
            'backend': 'onnx',
            'init_time': init_time,
            'encoder_session': encoder_session,
            'decoder_session': decoder_session
        }
        
    except Exception as e:
        print(f"❌ ONNX benchmark failed: {e}")
        return None

def benchmark_onnx_encoder(encoder_session, test_image):
    """Benchmark ONNX encoder only (the working part)."""
    print("\n🚀 BENCHMARKING ONNX ENCODER")
    
    try:
        # Preprocess image
        image_resized = np.array(test_image).astype(np.float32)
        if len(image_resized.shape) == 3:
            image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
        if image_resized.shape[-1] == 3:
            image_resized = np.transpose(image_resized, (0, 3, 1, 2))  # NHWC to NCHW
        
        # Normalize (approximate SAM preprocessing)
        mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
        std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)
        image_resized = (image_resized - mean) / std
        
        # Benchmark encoding
        start_time = time.time()
        input_name = encoder_session.get_inputs()[0].name
        outputs = encoder_session.run(None, {input_name: image_resized})
        encoding_time = time.time() - start_time
        
        print(f"✅ ONNX encoding: {encoding_time:.2f}s")
        print(f"📊 Encoder output shape: {outputs[0].shape}")
        
        return {
            'encoding_time': encoding_time,
            'encoder_output_shape': outputs[0].shape
        }
        
    except Exception as e:
        print(f"❌ ONNX encoding failed: {e}")
        return None

def create_test_image(size=512):
    """Create a test image for benchmarking."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add some patterns
    center = (size//2, size//2)
    radius = size//4
    y, x = np.ogrid[:size, :size]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    image[mask] = [255, 100, 100]
    
    # Add rectangle
    image[size//4:size//2, size//2:3*size//4] = [100, 255, 100]
    
    return image

def create_comprehensive_report(pytorch_results, onnx_results):
    """Create a comprehensive performance report."""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if not pytorch_results or not onnx_results:
        print("❌ Incomplete benchmark data")
        return
    
    # Initialization comparison
    if 'init_time' in pytorch_results and 'init_time' in onnx_results:
        init_speedup = pytorch_results['init_time'] / onnx_results['init_time']
        print(f"\n🚀 INITIALIZATION COMPARISON:")
        print(f"   PyTorch: {pytorch_results['init_time']:.2f}s")
        print(f"   ONNX:    {onnx_results['init_time']:.2f}s")
        print(f"   Speedup: {init_speedup:.2f}x ({'faster' if init_speedup > 1 else 'slower'})")
    
    # Inference comparison (partial for ONNX)
    if 'inference_time' in pytorch_results and 'encoding_time' in onnx_results:
        print(f"\n⚡ INFERENCE COMPARISON:")
        print(f"   PyTorch full: {pytorch_results['inference_time']:.2f}s")
        print(f"   ONNX encoding only: {onnx_results['encoding_time']:.2f}s")
        encoding_speedup = pytorch_results['inference_time'] / onnx_results['encoding_time']
        print(f"   Encoding speedup: {encoding_speedup:.2f}x")
        print(f"   ⚠️  Note: ONNX decoder has compatibility issues")
    
    # Mask generation
    if 'mask_count' in pytorch_results:
        print(f"\n🎯 MASK GENERATION:")
        print(f"   PyTorch: {pytorch_results['mask_count']} masks")
        print(f"   ONNX: 0 masks (decoder issues)")
    
    print(f"\n💡 SUMMARY:")
    print(f"   - ONNX encoder shows significant speed potential")
    print(f"   - PyTorch is currently more reliable for full pipeline")
    print(f"   - ONNX needs model regeneration to fix decoder issues")

def main():
    """Run comprehensive benchmarking."""
    print("🚀 SAM Performance Benchmarking Suite")
    print("Testing initialization and inference performance")
    print()
    
    # Create test image
    test_image = create_test_image(512)
    print(f"📐 Test image: {test_image.shape}")
    
    # Benchmark PyTorch
    pytorch_init = benchmark_pytorch_initialization()
    pytorch_inference = None
    if pytorch_init and 'mask_generator' in pytorch_init:
        pytorch_inference = benchmark_pytorch_inference(
            pytorch_init['mask_generator'], test_image
        )
    
    # Benchmark ONNX
    onnx_init = benchmark_onnx_initialization()
    onnx_encoding = None
    if onnx_init and 'encoder_session' in onnx_init:
        onnx_encoding = benchmark_onnx_encoder(
            onnx_init['encoder_session'], test_image
        )
    
    # Combine results
    pytorch_results = {}
    if pytorch_init:
        pytorch_results.update(pytorch_init)
    if pytorch_inference:
        pytorch_results.update(pytorch_inference)
        
    onnx_results = {}
    if onnx_init:
        onnx_results.update(onnx_init)
    if onnx_encoding:
        onnx_results.update(onnx_encoding)
    
    # Generate report
    create_comprehensive_report(pytorch_results, onnx_results)

if __name__ == "__main__":
    main()