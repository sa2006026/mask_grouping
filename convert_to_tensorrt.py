#!/usr/bin/env python3
"""
Convert ONNX SAM models to TensorRT format for optimized inference on NVIDIA GPUs.

TensorRT can provide significant performance improvements over ONNX Runtime,
especially for models with complex operations like SAM.
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

def check_tensorrt_availability():
    """Check if TensorRT is available."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        print(f"✅ TensorRT version: {trt.__version__}")
        return True
    except ImportError as e:
        print(f"❌ TensorRT not available: {e}")
        print("Install TensorRT: pip install tensorrt pycuda")
        return False

def convert_onnx_to_tensorrt(
    onnx_path: str,
    tensorrt_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    max_workspace: int = 1 << 30  # 1GB
):
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Path to ONNX model
        tensorrt_path: Output path for TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'int8')
        max_batch_size: Maximum batch size
        max_workspace: Maximum workspace size in bytes
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    print(f"Converting {onnx_path} to TensorRT...")
    print(f"Precision: {precision}, Max batch size: {max_batch_size}")
    
    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Create builder config
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace
    
    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 precision enabled")
        else:
            print("⚠️  FP16 not supported, using FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✅ INT8 precision enabled")
        else:
            print("⚠️  INT8 not supported, using FP32")
    
    # Build engine
    print("🔧 Building TensorRT engine (this may take several minutes)...")
    start_time = time.time()
    
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("❌ Failed to build TensorRT engine")
        return False
    
    build_time = time.time() - start_time
    print(f"✅ Engine built in {build_time:.2f} seconds")
    
    # Save engine
    with open(tensorrt_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"💾 TensorRT engine saved: {tensorrt_path}")
    return True

def convert_sam_to_tensorrt(
    onnx_dir: str = "model/onnx",
    tensorrt_dir: str = "model/tensorrt",
    precision: str = "fp16"
):
    """Convert SAM ONNX models to TensorRT format."""
    
    if not check_tensorrt_availability():
        return False
    
    # Create output directory
    tensorrt_path = Path(tensorrt_dir)
    tensorrt_path.mkdir(parents=True, exist_ok=True)
    
    onnx_path = Path(onnx_dir)
    
    # Find ONNX models
    encoder_onnx = onnx_path / "sam_vit_h_encoder.onnx"
    decoder_onnx = onnx_path / "sam_vit_h_decoder.onnx"
    
    if not encoder_onnx.exists():
        print(f"❌ Encoder ONNX not found: {encoder_onnx}")
        return False
    
    if not decoder_onnx.exists():
        print(f"❌ Decoder ONNX not found: {decoder_onnx}")
        return False
    
    # Convert encoder
    encoder_trt = tensorrt_path / f"sam_vit_h_encoder_{precision}.trt"
    print("\n" + "="*60)
    print("🔧 CONVERTING IMAGE ENCODER")
    print("="*60)
    
    success_encoder = convert_onnx_to_tensorrt(
        str(encoder_onnx),
        str(encoder_trt),
        precision=precision,
        max_batch_size=1,
        max_workspace=2 << 30  # 2GB for encoder
    )
    
    if not success_encoder:
        print("❌ Failed to convert encoder")
        return False
    
    # Convert decoder
    decoder_trt = tensorrt_path / f"sam_vit_h_decoder_{precision}.trt"
    print("\n" + "="*60)
    print("🔧 CONVERTING MASK DECODER")
    print("="*60)
    
    success_decoder = convert_onnx_to_tensorrt(
        str(decoder_onnx),
        str(decoder_trt),
        precision=precision,
        max_batch_size=1,
        max_workspace=1 << 30  # 1GB for decoder
    )
    
    if not success_decoder:
        print("❌ Failed to convert decoder")
        return False
    
    print("\n" + "="*60)
    print("✅ SAM TENSORRT CONVERSION COMPLETED!")
    print("="*60)
    print(f"📁 Models saved in: {tensorrt_dir}")
    print(f"🚀 Encoder: {encoder_trt.name}")
    print(f"🎯 Decoder: {decoder_trt.name}")
    print(f"⚡ Precision: {precision}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert SAM ONNX models to TensorRT")
    parser.add_argument(
        "--onnx-dir", 
        default="model/onnx",
        help="Directory containing ONNX models"
    )
    parser.add_argument(
        "--tensorrt-dir",
        default="model/tensorrt", 
        help="Output directory for TensorRT engines"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Precision mode for TensorRT"
    )
    
    args = parser.parse_args()
    
    print("🚀 SAM ONNX to TensorRT Converter")
    print("="*50)
    
    try:
        success = convert_sam_to_tensorrt(
            onnx_dir=args.onnx_dir,
            tensorrt_dir=args.tensorrt_dir,
            precision=args.precision
        )
        
        if success:
            print("\n✅ Conversion completed successfully!")
            print("🔧 To use TensorRT backend:")
            print("   export SAM_BACKEND=tensorrt")
            print("   python3 app.py")
        else:
            print("\n❌ Conversion failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 