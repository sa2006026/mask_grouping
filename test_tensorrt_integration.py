#!/usr/bin/env python3
"""
Test script to verify TensorRT backend integration with the SAM Flask application.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

def test_tensorrt_availability():
    """Test if TensorRT is available."""
    print("🔍 Testing TensorRT availability...")
    
    try:
        from tensorrt_sam_wrapper import check_tensorrt_availability
        if check_tensorrt_availability():
            print("✅ TensorRT is available")
            return True
        else:
            print("❌ TensorRT not available")
            return False
    except ImportError as e:
        print(f"❌ Failed to import TensorRT wrapper: {e}")
        return False

def test_sam_config():
    """Test SAM configuration with TensorRT backend."""
    print("\n🔍 Testing SAM configuration...")
    
    try:
        from sam_config import SAMConfig, SAMBackend
        
        # Test TensorRT configuration
        config = SAMConfig(
            backend=SAMBackend.TENSORRT,
            model_type="vit_h",
            tensorrt_precision="fp16"
        )
        
        print(f"✅ SAM config created: {config}")
        print(f"   Backend: {config.backend.value}")
        print(f"   Model type: {config.model_type}")
        print(f"   TensorRT precision: {config.tensorrt_precision}")
        print(f"   TensorRT directory: {config.tensorrt_dir}")
        
        # Check if TensorRT models are available
        if config.is_tensorrt_available():
            print("✅ TensorRT models are available")
        else:
            print("❌ TensorRT models not found")
            print("   Run: python3 convert_to_tensorrt.py")
        
        return True
        
    except Exception as e:
        print(f"❌ SAM config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_import():
    """Test importing the Flask app with TensorRT support."""
    print("\n🔍 Testing Flask app import...")
    
    try:
        # Set environment to use TensorRT if available
        os.environ['SAM_BACKEND'] = 'auto'  # Will auto-select TensorRT if available
        
        # Import app components
        from sam_config import SAMConfig, SAMBackend, ENV_CONFIG
        print(f"✅ SAM config imported")
        print(f"   Environment config: {ENV_CONFIG}")
        
        # Try importing TensorRT wrapper
        from tensorrt_sam_wrapper import load_tensorrt_sam
        print(f"✅ TensorRT wrapper imported")
        
        # Try importing app
        from app import app
        print(f"✅ Flask app imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ App import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_selection():
    """Test automatic backend selection."""
    print("\n🔍 Testing backend selection...")
    
    try:
        from sam_config import SAMConfig, SAMBackend
        
        # Test AUTO backend selection
        config = SAMConfig(backend=SAMBackend.AUTO)
        
        try:
            selected_backend = config.resolve_backend()
            print(f"✅ Auto-selected backend: {selected_backend.value}")
            
            # Check availability of each backend
            print("\n📊 Backend availability:")
            print(f"   PyTorch: {'✅' if config.is_pytorch_available() else '❌'}")
            print(f"   ONNX: {'✅' if config.is_onnx_available() else '❌'}")
            print(f"   TensorRT: {'✅' if config.is_tensorrt_available() else '❌'}")
            
            return True
            
        except RuntimeError as e:
            print(f"❌ Backend resolution failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Backend selection test failed: {e}")
        return False

def test_models_exist():
    """Check if required models exist."""
    print("\n🔍 Checking model availability...")
    
    model_dir = Path("model")
    
    # Check PyTorch model
    pytorch_model = model_dir / "sam_vit_h_4b8939.pth"
    print(f"   PyTorch model: {'✅' if pytorch_model.exists() else '❌'} {pytorch_model}")
    
    # Check ONNX models
    onnx_dir = model_dir / "onnx"
    encoder_onnx = onnx_dir / "sam_vit_h_encoder.onnx"
    decoder_onnx = onnx_dir / "sam_vit_h_decoder.onnx"
    print(f"   ONNX encoder: {'✅' if encoder_onnx.exists() else '❌'} {encoder_onnx}")
    print(f"   ONNX decoder: {'✅' if decoder_onnx.exists() else '❌'} {decoder_onnx}")
    
    # Check TensorRT models
    tensorrt_dir = model_dir / "tensorrt"
    for precision in ["fp16", "fp32", "int8"]:
        encoder_trt = tensorrt_dir / f"sam_vit_h_encoder_{precision}.trt"
        decoder_trt = tensorrt_dir / f"sam_vit_h_decoder_{precision}.trt"
        if encoder_trt.exists() and decoder_trt.exists():
            print(f"   TensorRT {precision}: ✅ {encoder_trt.parent}")
            break
    else:
        print(f"   TensorRT models: ❌ Not found in {tensorrt_dir}")

def print_instructions():
    """Print setup instructions."""
    print("\n" + "="*60)
    print("🚀 TENSORRT INTEGRATION SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Install TensorRT (if not already installed):")
    print("   pip install tensorrt pycuda")
    
    print("\n2. Convert ONNX models to TensorRT:")
    print("   python3 convert_to_tensorrt.py --precision fp16")
    
    print("\n3. Use TensorRT backend:")
    print("   export SAM_BACKEND=tensorrt")
    print("   python3 app.py")
    
    print("\n4. Or use auto-selection (will prefer TensorRT if available):")
    print("   export SAM_BACKEND=auto")
    print("   python3 app.py")
    
    print("\n💡 Environment variables:")
    print("   SAM_BACKEND=tensorrt|onnx|pytorch|auto")
    print("   SAM_TENSORRT_PRECISION=fp16|fp32|int8")
    print("   SAM_MODEL_TYPE=vit_h|vit_l|vit_b")

def main():
    print("🧪 TensorRT Integration Test Suite")
    print("="*50)
    
    # Run tests
    tests = [
        ("TensorRT Availability", test_tensorrt_availability),
        ("SAM Configuration", test_sam_config),
        ("Flask App Import", test_app_import),
        ("Backend Selection", test_backend_selection),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Check models
    test_models_exist()
    
    # Print results summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! TensorRT integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    # Print instructions
    print_instructions()

if __name__ == "__main__":
    main() 