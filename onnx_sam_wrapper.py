"""
ONNX wrapper for SAM (Segment Anything Model) to provide faster inference.

This module provides an ONNX-based implementation of SAM that can be used as a 
drop-in replacement for the PyTorch version with significant performance improvements.
"""

import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import onnxruntime as ort
from pathlib import Path
import torch


class ONNXSamPredictor:
    """
    ONNX-based SAM predictor that provides the same interface as the PyTorch version
    but with improved inference performance.
    """
    
    def __init__(self, encoder_path: str, decoder_path: str, providers: Optional[List[str]] = None):
        """
        Initialize the ONNX SAM predictor.
        
        Args:
            encoder_path: Path to the ONNX encoder model
            decoder_path: Path to the ONNX decoder model  
            providers: List of execution providers for ONNX Runtime
        """
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        
        # Set up execution providers
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Initialize ONNX Runtime sessions
        try:
            self.encoder_session = ort.InferenceSession(encoder_path, providers=providers)
            self.decoder_session = ort.InferenceSession(decoder_path, providers=providers)
            print(f"ONNX SAM loaded with providers: {self.encoder_session.get_providers()}")
        except Exception as e:
            print(f"Failed to load ONNX models: {e}")
            raise
        
        # Get input/output names and shapes
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name
        
        self.decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]
        self.decoder_output_names = [out.name for out in self.decoder_session.get_outputs()]
        
        # Store current image embeddings
        self.features = None
        self.original_size = None
        self.input_size = None
        self.is_image_set = False
        
    def set_image(self, image: np.ndarray) -> None:
        """
        Process an image and compute embeddings for future mask predictions.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
        """
        assert len(image.shape) == 3 and image.shape[2] == 3, "Image must be in RGB format"
        
        # Store original size
        self.original_size = image.shape[:2]
        
        # Preprocess image
        input_image = self.preprocess_image(image)
        self.input_size = input_image.shape[-2:]
        
        # Run encoder
        start_time = time.time()
        encoder_inputs = {self.encoder_input_name: input_image}
        encoder_outputs = self.encoder_session.run([self.encoder_output_name], encoder_inputs)
        self.features = encoder_outputs[0]
        
        encoding_time = time.time() - start_time
        print(f"ONNX image encoding took: {encoding_time:.3f}s")
        
        self.is_image_set = True
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for SAM encoder.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            
        Returns:
            Preprocessed image (1, 3, 1024, 1024)
        """
        # Resize to 1024x1024 maintaining aspect ratio
        h, w = image.shape[:2]
        target_size = 1024
        
        # Calculate new size maintaining aspect ratio
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to 1024x1024
        padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
        padded[:new_h, :new_w] = resized
        
        # Convert to CHW format and normalize
        padded = padded.astype(np.float32) / 255.0
        
        # SAM preprocessing: normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        padded = (padded - mean) / std
        
        # Convert to NCHW format
        padded = np.transpose(padded, (2, 0, 1))
        padded = np.expand_dims(padded, axis=0)
        
        # Ensure float32 type for ONNX
        return padded.astype(np.float32)
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given prompts.
        
        Args:
            point_coords: Point coordinates (N, 2) in image coordinates
            point_labels: Point labels (N,) - 1 for foreground, 0 for background
            box: Bounding box (4,) in format [x1, y1, x2, y2]
            mask_input: Previous mask prediction (1, H, W)
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return mask logits
            
        Returns:
            Tuple of (masks, scores, logits)
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        
        # Prepare inputs
        coords, labels = self._prepare_prompts(point_coords, point_labels, box)
        
        # Prepare mask input
        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0], dtype=np.float32)
        else:
            # Resize mask input to 256x256
            mask_input = cv2.resize(mask_input.astype(np.float32), (256, 256))
            mask_input = mask_input.reshape(1, 1, 256, 256)
            has_mask_input = np.array([1], dtype=np.float32)
        
        # Prepare original image size
        orig_im_size = np.array(self.original_size, dtype=np.float32)
        
        # Run decoder
        start_time = time.time()
        decoder_inputs = {
            'image_embeddings': self.features,
            'point_coords': coords,
            'point_labels': labels,
            'mask_input': mask_input
        }
        
        decoder_outputs = self.decoder_session.run(self.decoder_output_names, decoder_inputs)
        masks, iou_predictions, low_res_masks = decoder_outputs
        
        decoding_time = time.time() - start_time
        print(f"ONNX mask decoding took: {decoding_time:.3f}s")
        
        # Process outputs
        if not multimask_output:
            # Return the mask with highest IoU score
            best_idx = np.argmax(iou_predictions, axis=1)
            masks = masks[np.arange(masks.shape[0]), best_idx:best_idx+1]
            iou_predictions = iou_predictions[np.arange(iou_predictions.shape[0]), best_idx:best_idx+1]
            low_res_masks = low_res_masks[np.arange(low_res_masks.shape[0]), best_idx:best_idx+1]
        
        # Convert to binary masks
        masks = masks > 0.0
        
        if return_logits:
            return masks, iou_predictions, low_res_masks
        else:
            return masks, iou_predictions, None
    
    def _prepare_prompts(
        self, 
        point_coords: Optional[np.ndarray], 
        point_labels: Optional[np.ndarray], 
        box: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare point coordinates and labels for the decoder."""
        coords_list = []
        labels_list = []
        
        # Add points
        if point_coords is not None:
            coords_list.append(point_coords)
            labels_list.append(point_labels)
        
        # Add box corners as points
        if box is not None:
            box_coords = np.array([[box[0], box[1]], [box[2], box[3]]])
            box_labels = np.array([2, 3])  # 2 and 3 are box corner labels
            coords_list.append(box_coords)
            labels_list.append(box_labels)
        
        # Combine all points
        if coords_list:
            coords = np.concatenate(coords_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
        else:
            # No prompts - use center point as default
            h, w = self.original_size
            coords = np.array([[w // 2, h // 2]])
            labels = np.array([1])
        
        # Pad to fixed size (5 points max) and add batch dimension
        max_points = 5
        padded_coords = np.zeros((1, max_points, 2), dtype=np.float32)
        padded_labels = np.zeros((1, max_points), dtype=np.float32)
        
        n_points = min(len(coords), max_points)
        padded_coords[0, :n_points] = coords[:n_points]
        padded_labels[0, :n_points] = labels[:n_points]
        
        return padded_coords, padded_labels


class ONNXSamAutomaticMaskGenerator:
    """
    ONNX-based automatic mask generator that replicates the functionality of
    SamAutomaticMaskGenerator but with improved performance.
    """
    
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize the ONNX automatic mask generator.
        
        Args:
            encoder_path: Path to ONNX encoder model
            decoder_path: Path to ONNX decoder model
            points_per_side: Number of points per side for point grid
            points_per_batch: Number of points to process in each batch
            pred_iou_thresh: IoU threshold for mask filtering
            stability_score_thresh: Stability score threshold for mask filtering
            stability_score_offset: Offset for stability score calculation
            box_nms_thresh: NMS threshold for duplicate mask removal
            crop_n_layers: Number of crop layers for processing
            crop_nms_thresh: NMS threshold for crop mask removal
            crop_overlap_ratio: Overlap ratio for crop processing
            crop_n_points_downscale_factor: Downscale factor for points in crops
            point_grids: Predefined point grids (if None, will be generated)
            min_mask_region_area: Minimum area for mask regions
            output_mode: Output format ("binary_mask" or "uncompressed_rle")
            providers: ONNX Runtime execution providers
        """
        self.predictor = ONNXSamPredictor(encoder_path, decoder_path, providers)
        
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        
        if point_grids is None:
            self.point_grids = self._build_point_grid(points_per_side)
        else:
            self.point_grids = point_grids
    
    def _build_point_grid(self, n_per_side: int) -> List[np.ndarray]:
        """Build a grid of evenly spaced points."""
        offset = 1 / (2 * n_per_side)
        points_one_side = np.linspace(offset, 1 - offset, n_per_side)
        points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
        points_y = np.tile(points_one_side[:, None], (1, n_per_side))
        grid = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
        return [grid]
    
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate masks for the input image.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            
        Returns:
            List of mask dictionaries with keys: segmentation, area, bbox, 
            predicted_iou, point_coords, stability_score, crop_box
        """
        print(f"ONNX SAM generating masks for image: {image.shape}")
        start_time = time.time()
        
        # Set image in predictor
        self.predictor.set_image(image)
        
        # Generate masks from point grids
        masks = []
        
        for points in self.point_grids:
            # Scale points to image size
            h, w = image.shape[:2]
            scaled_points = points * np.array([w, h])
            
            # Process points in batches
            for i in range(0, len(scaled_points), self.points_per_batch):
                batch_points = scaled_points[i:i + self.points_per_batch]
                batch_labels = np.ones(len(batch_points))
                
                # Predict masks for this batch
                try:
                    batch_masks, batch_ious, _ = self.predictor.predict(
                        point_coords=batch_points,
                        point_labels=batch_labels,
                        multimask_output=True
                    )
                    
                    # Process each mask in the batch
                    for j in range(batch_masks.shape[1]):  # For each mask output
                        for k in range(len(batch_points)):  # For each point
                            mask = batch_masks[k, j]
                            iou = batch_ious[k, j]
                            
                            # Apply thresholds
                            if iou < self.pred_iou_thresh:
                                continue
                            
                            # Calculate mask properties
                            mask_dict = self._mask_to_dict(
                                mask, iou, batch_points[k], image.shape[:2]
                            )
                            
                            if mask_dict is not None:
                                masks.append(mask_dict)
                                
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
        
        # Apply NMS to remove duplicate masks
        masks = self._remove_duplicates(masks)
        
        # Filter by area
        if self.min_mask_region_area > 0:
            masks = [m for m in masks if m["area"] >= self.min_mask_region_area]
        
        total_time = time.time() - start_time
        print(f"ONNX SAM generated {len(masks)} masks in {total_time:.3f}s")
        
        return masks
    
    def _mask_to_dict(
        self, 
        mask: np.ndarray, 
        iou: float, 
        point: np.ndarray, 
        original_size: Tuple[int, int]
    ) -> Optional[Dict[str, Any]]:
        """Convert mask to dictionary format."""
        # Remove batch dimension
        if len(mask.shape) == 3:
            mask = mask[0]
        
        # Calculate area
        area = int(np.sum(mask))
        if area == 0:
            return None
        
        # Calculate bounding box
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return None
            
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
        
        # Calculate stability score (simplified)
        stability_score = float(iou)
        
        return {
            "segmentation": mask.astype(bool),
            "area": area,
            "bbox": bbox,
            "predicted_iou": float(iou),
            "point_coords": [point.tolist()],
            "stability_score": stability_score,
            "crop_box": [0, 0, original_size[1], original_size[0]]  # [x1, y1, x2, y2]
        }
    
    def _remove_duplicates(self, masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate masks using NMS."""
        if len(masks) <= 1:
            return masks
        
        # Sort by IoU score
        masks = sorted(masks, key=lambda x: x["predicted_iou"], reverse=True)
        
        # Apply NMS
        keep = []
        for i, mask in enumerate(masks):
            overlaps = False
            mask1 = mask["segmentation"]
            
            for kept_mask in keep:
                mask2 = kept_mask["segmentation"]
                
                # Calculate IoU
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > self.box_nms_thresh:
                        overlaps = True
                        break
            
            if not overlaps:
                keep.append(mask)
        
        return keep


def load_onnx_sam(
    model_dir: str,
    model_type: str = "vit_h",
    providers: Optional[List[str]] = None
) -> ONNXSamAutomaticMaskGenerator:
    """
    Load ONNX SAM model and return automatic mask generator.
    
    Args:
        model_dir: Directory containing ONNX models
        model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
        providers: ONNX Runtime execution providers
        
    Returns:
        ONNXSamAutomaticMaskGenerator instance
    """
    model_path = Path(model_dir)
    encoder_path = model_path / f"sam_{model_type}_encoder.onnx"
    decoder_path = model_path / f"sam_{model_type}_decoder.onnx"
    
    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            f"ONNX models not found in {model_dir}. "
            f"Please run convert_to_onnx.py first to convert the PyTorch models."
        )
    
    return ONNXSamAutomaticMaskGenerator(
        encoder_path=str(encoder_path),
        decoder_path=str(decoder_path),
        providers=providers
    ) 