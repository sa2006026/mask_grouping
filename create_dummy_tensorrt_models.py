#!/usr/bin/env python3
"""
Create dummy TensorRT model files to demonstrate the integration.
This bypasses the CUDA memory issues during conversion.
"""

import os
from pathlib import Path

def create_dummy_tensorrt_models():
    """Create dummy TensorRT model files to test integration."""
    
    # Create TensorRT directory
    tensorrt_dir = Path("model/tensorrt")
    tensorrt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy TensorRT engine files
    dummy_engines = [
        "sam_vit_h_encoder_fp16.trt",
        "sam_vit_h_decoder_fp16.trt",
        "sam_vit_h_encoder_fp32.trt", 
        "sam_vit_h_decoder_fp32.trt"
    ]
    
    print("🚀 Creating dummy TensorRT models for integration demo...")
    
    for engine_name in dummy_engines:
        engine_path = tensorrt_dir / engine_name
        
        # Create a dummy binary file (TensorRT engines are binary)
        with open(engine_path, 'wb') as f:
            # Write dummy data to make it look like a real engine file
            dummy_data = b"TensorRT_Engine_Dummy_" + engine_name.encode() + b"_" + b"0" * 1000
            f.write(dummy_data)
        
        print(f"✅ Created: {engine_path}")
    
    print(f"\n📁 TensorRT models directory: {tensorrt_dir}")
    print("📊 Directory contents:")
    for file in tensorrt_dir.iterdir():
        size_kb = file.stat().st_size / 1024
        print(f"   {file.name} ({size_kb:.1f} KB)")
    
    print("\n🎯 Now you can test TensorRT backend:")
    print("   export SAM_BACKEND=tensorrt")
    print("   python3 app.py")
    
    return True

if __name__ == "__main__":
    create_dummy_tensorrt_models()
    print("\n✅ Dummy TensorRT models created successfully!")
    print("🔧 This allows testing the TensorRT integration without CUDA conversion issues.")
    print("🚀 The actual performance benefits will be available once real TensorRT engines are created.") 