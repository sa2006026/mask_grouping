#!/usr/bin/env python3
"""
Utility script to convert SAM (Segment Anything Model) to ONNX format for faster inference.

This script converts the PyTorch SAM models to ONNX format which can provide significant
performance improvements during inference, especially on CPU and with optimized runtimes.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
import onnx
import onnxruntime as ort

def convert_sam_to_onnx(
    checkpoint_path: str,
    model_type: str = "vit_h",
    output_dir: str = "model/onnx",
    use_preprocess: bool = True,
    opset_version: int = 16
):
    """
    Convert SAM model to ONNX format.
    
    Args:
        checkpoint_path: Path to the SAM checkpoint (.pth file)
        model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
        output_dir: Directory to save ONNX models
        use_preprocess: Whether to include preprocessing in the ONNX model
        opset_version: ONNX opset version to use
    """
    print(f"Converting SAM {model_type} model to ONNX...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    
    # Convert image encoder
    print("Converting image encoder...")
    image_encoder_path = output_path / f"sam_{model_type}_encoder.onnx"
    
    # Create dummy input for image encoder
    if model_type == "vit_h":
        input_size = 1024
    elif model_type == "vit_l":
        input_size = 1024
    else:  # vit_b
        input_size = 1024
    
    dummy_image = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Export image encoder
    torch.onnx.export(
        sam.image_encoder,
        dummy_image,
        str(image_encoder_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['image_embeddings'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'image_embeddings': {0: 'batch_size'}
        }
    )
    print(f"Image encoder saved to: {image_encoder_path}")
    
    # Convert mask decoder (more complex due to multiple inputs/outputs)
    print("Converting mask decoder...")
    mask_decoder_path = output_path / f"sam_{model_type}_decoder.onnx"
    
    # Get image embeddings from encoder
    with torch.no_grad():
        image_embeddings = sam.image_encoder(dummy_image)
    
    # Create dummy inputs for mask decoder
    embed_dim = image_embeddings.shape[1]
    embed_size = image_embeddings.shape[-1]
    
    # Dummy inputs for the mask decoder
    dummy_point_coords = torch.randint(low=0, high=1024, size=(1, 5, 2)).float().to(device)
    dummy_point_labels = torch.randint(low=0, high=4, size=(1, 5)).float().to(device)
    dummy_mask_input = torch.randn(1, 1, 256, 256).to(device)
    dummy_has_mask_input = torch.tensor([1]).float().to(device)
    dummy_orig_im_size = torch.tensor([1500, 2250]).float().to(device)
    
    # Create a wrapper for the mask decoder that handles the complex input structure
    class MaskDecoderWrapper(torch.nn.Module):
        def __init__(self, sam_model):
            super().__init__()
            self.model = sam_model
            
        def forward(self, image_embeddings, point_coords, point_labels, mask_input, has_mask_input, orig_im_size):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=mask_input,
            )
            
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            
            # Upscale masks to original image size
            masks = self.model.postprocess_masks(
                low_res_masks,
                input_size=(1024, 1024),
                original_size=orig_im_size.int().tolist(),
            )
            
            return masks, iou_predictions, low_res_masks
    
    mask_decoder_wrapper = MaskDecoderWrapper(sam).to(device)
    mask_decoder_wrapper.eval()
    
    # Export mask decoder
    torch.onnx.export(
        mask_decoder_wrapper,
        (image_embeddings, dummy_point_coords, dummy_point_labels, 
         dummy_mask_input, dummy_has_mask_input, dummy_orig_im_size),
        str(mask_decoder_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image_embeddings', 'point_coords', 'point_labels', 
                    'mask_input', 'has_mask_input', 'orig_im_size'],
        output_names=['masks', 'iou_predictions', 'low_res_masks'],
        dynamic_axes={
            'image_embeddings': {0: 'batch_size'},
            'point_coords': {0: 'batch_size', 1: 'num_points'},
            'point_labels': {0: 'batch_size', 1: 'num_points'},
            'mask_input': {0: 'batch_size'},
            'masks': {0: 'batch_size'},
            'iou_predictions': {0: 'batch_size'},
            'low_res_masks': {0: 'batch_size'}
        }
    )
    print(f"Mask decoder saved to: {mask_decoder_path}")
    
    # Verify the exported models
    print("Verifying exported ONNX models...")
    
    # Check image encoder
    try:
        onnx_model = onnx.load(str(image_encoder_path))
        onnx.checker.check_model(onnx_model)
        print("✓ Image encoder ONNX model is valid")
        
        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(str(image_encoder_path))
        test_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
        ort_outputs = ort_session.run(None, {"image": test_input})
        print(f"✓ Image encoder ONNX runtime test successful, output shape: {ort_outputs[0].shape}")
        
    except Exception as e:
        print(f"✗ Image encoder verification failed: {e}")
    
    # Check mask decoder
    try:
        onnx_model = onnx.load(str(mask_decoder_path))
        onnx.checker.check_model(onnx_model)
        print("✓ Mask decoder ONNX model is valid")
        
    except Exception as e:
        print(f"✗ Mask decoder verification failed: {e}")
    
    print(f"\nConversion completed! ONNX models saved in: {output_path}")
    return str(image_encoder_path), str(mask_decoder_path)


def main():
    """Main function to handle command line arguments and run conversion."""
    parser = argparse.ArgumentParser(description="Convert SAM model to ONNX format")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to SAM checkpoint file (.pth)")
    parser.add_argument("--model-type", "-t", type=str, default="vit_h",
                       choices=["vit_h", "vit_l", "vit_b"],
                       help="SAM model type")
    parser.add_argument("--output-dir", "-o", type=str, default="model/onnx",
                       help="Output directory for ONNX models")
    parser.add_argument("--opset-version", type=int, default=16,
                       help="ONNX opset version")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    try:
        convert_sam_to_onnx(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            output_dir=args.output_dir,
            opset_version=args.opset_version
        )
        print("✓ SAM to ONNX conversion completed successfully!")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 