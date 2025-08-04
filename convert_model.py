#!/usr/bin/env python3
"""
Convenience script to convert the existing SAM model to ONNX format.

This script will find the existing PyTorch SAM model and convert it to ONNX
for faster inference performance.
"""

import os
import sys
from pathlib import Path
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from convert_to_onnx import convert_sam_to_onnx
from sam_config import SAMConfig

def main():
    """Main function to convert existing SAM model to ONNX."""
    parser = argparse.ArgumentParser(description="Convert existing SAM model to ONNX")
    parser.add_argument("--model-type", "-t", type=str, default="vit_h",
                       choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model type to convert")
    parser.add_argument("--output-dir", "-o", type=str, default="model/onnx",
                       help="Output directory for ONNX models")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force conversion even if ONNX models already exist")
    
    args = parser.parse_args()
    
    # Create SAM config to find model
    config = SAMConfig(model_type=args.model_type)
    
    # Check if PyTorch model exists
    pytorch_model_path = config.get_pytorch_model_path()
    if not pytorch_model_path:
        print(f"Error: PyTorch model {args.model_type} not found in {config.model_dir}")
        print("Please ensure the PyTorch model is downloaded first.")
        sys.exit(1)
    
    # Check if ONNX models already exist
    encoder_path, decoder_path = config.get_onnx_model_paths()
    if encoder_path and decoder_path and not args.force:
        print(f"ONNX models for {args.model_type} already exist in {config.onnx_dir}")
        print("Use --force to overwrite existing models.")
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            sys.exit(0)
    
    print(f"Converting SAM {args.model_type} model to ONNX...")
    print(f"Source: {pytorch_model_path}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Convert the model
        encoder_onnx_path, decoder_onnx_path = convert_sam_to_onnx(
            checkpoint_path=str(pytorch_model_path),
            model_type=args.model_type,
            output_dir=args.output_dir
        )
        
        print("\n✓ Conversion completed successfully!")
        print(f"Encoder ONNX model: {encoder_onnx_path}")
        print(f"Decoder ONNX model: {decoder_onnx_path}")
        
        # Test the converted models
        print("\nTesting converted models...")
        from onnx_sam_wrapper import load_onnx_sam
        
        test_generator = load_onnx_sam(
            model_dir=args.output_dir,
            model_type=args.model_type
        )
        print("✓ ONNX models loaded successfully!")
        
        print(f"\nTo use the ONNX backend, set the environment variable:")
        print(f"export SAM_BACKEND=onnx")
        print(f"export SAM_MODEL_TYPE={args.model_type}")
        
        print(f"\nOr start the server with ONNX backend:")
        print(f"SAM_BACKEND=onnx SAM_MODEL_TYPE={args.model_type} python app.py")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 