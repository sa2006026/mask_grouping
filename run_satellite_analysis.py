#!/usr/bin/env python3
"""
Script to run satellite droplet analysis using ONNX SAM model.
This uses the enhanced SAM (equivalent to crop_n_layers=3) with K-means clustering.
"""

import sys
import time
import base64
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from sam_config import SAMConfig, SAMBackend
from onnx_sam_wrapper import load_onnx_sam
from src.mask_grouping.core import HugeDropletProcessor

def load_image(image_path: str) -> np.ndarray:
    """Load image from file path."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image using PIL
    pil_image = Image.open(image_path)
    
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    return image_array

def run_satellite_analysis(image_path: str, 
                         min_circularity: float = 0.53, 
                         max_blob_distance: float = 50):
    """Run satellite droplet analysis with ONNX backend."""
    
    print("=" * 60)
    print("🛰️  SATELLITE DROPLET ANALYSIS WITH ONNX SAM")
    print("=" * 60)
    
    # Configure ONNX backend
    config = SAMConfig(
        backend=SAMBackend.ONNX,
        model_type="vit_h",
        use_gpu=True  # Use GPU if available
    )
    
    print(f"📁 Loading image: {image_path}")
    image_array = load_image(image_path)
    print(f"📐 Image shape: {image_array.shape}")
    
    print(f"🧠 Initializing ONNX SAM model: {config.model_type}")
    print(f"🚀 ONNX providers: {config.onnx_providers}")
    
    # Get mask generator parameters  
    params = config.get_mask_generator_params()
    
    # Load ONNX SAM (this provides enhanced functionality equivalent to crop_n_layers=3)
    mask_generator = load_onnx_sam(
        model_dir=str(config.onnx_dir),
        model_type=config.model_type,
        providers=config.onnx_providers
    )
    
    # Configure for satellite analysis (enhanced mode)
    mask_generator.points_per_side = params.get("points_per_side", 32)
    mask_generator.pred_iou_thresh = params.get("pred_iou_thresh", 0.88)
    mask_generator.stability_score_thresh = params.get("stability_score_thresh", 0.95)
    mask_generator.min_mask_region_area = params.get("min_mask_region_area", 100)
    
    print(f"⚙️  Configuration:")
    print(f"   - Points per side: {mask_generator.points_per_side}")
    print(f"   - IoU threshold: {mask_generator.pred_iou_thresh}")
    print(f"   - Stability threshold: {mask_generator.stability_score_thresh}")
    print(f"   - Min region area: {mask_generator.min_mask_region_area}")
    print(f"   - Min circularity: {min_circularity}")
    print(f"   - Max blob distance: {max_blob_distance}")
    
    # Initialize processor
    processor = HugeDropletProcessor(mask_generator)
    processor.min_circularity = min_circularity
    processor.max_blob_distance = max_blob_distance
    
    print(f"\n🔍 Starting analysis...")
    start_time = time.time()
    
    # Process image
    result = processor.process_image_array(image_array)
    
    # Import for K-means clustering (satellite mode specific)
    from src.mask_grouping.core import MaskGroupingProcessor
    temp_processor = MaskGroupingProcessor(processor.mask_generator)
    
    # Apply K-means clustering to separate satellite droplets
    print(f"🎯 Applying K-means clustering for satellite droplet separation...")
    clustered_masks, cluster_labels = temp_processor._cluster_masks_kmeans(
        result['final_masks'], n_clusters=2
    )
    
    # Determine which cluster has larger masks on average
    avg_areas = []
    for cluster_id in range(2):
        cluster_masks = [mask for mask, label in zip(clustered_masks, cluster_labels) if label == cluster_id]
        if cluster_masks:
            avg_area = np.mean([mask['area'] for mask in cluster_masks])
            avg_areas.append(avg_area)
        else:
            avg_areas.append(0)
    
    # Assign groups (larger average area = Group 1, smaller = Group 2)
    large_cluster_id = np.argmax(avg_areas)
    small_cluster_id = 1 - large_cluster_id
    
    group1_masks = [mask for mask, label in zip(clustered_masks, cluster_labels) if label == large_cluster_id]
    group2_masks = [mask for mask, label in zip(clustered_masks, cluster_labels) if label == small_cluster_id]
    
    processing_time = time.time() - start_time
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"   🎭 Total masks found: {len(result['final_masks'])}")
    print(f"   🔴 Group 1 (larger): {len(group1_masks)} droplets")
    print(f"   🔵 Group 2 (smaller): {len(group2_masks)} droplets")
    
    if group1_masks:
        group1_areas = [mask['area'] for mask in group1_masks]
        print(f"   📏 Group 1 area range: {min(group1_areas):.0f} - {max(group1_areas):.0f}")
        
    if group2_masks:
        group2_areas = [mask['area'] for mask in group2_masks] 
        print(f"   📏 Group 2 area range: {min(group2_areas):.0f} - {max(group2_areas):.0f}")
    
    # Create visualization
    print(f"\n🎨 Creating satellite droplet visualization...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    viz_path = processor.create_satellite_droplet_visualization(
        image_array, group1_masks, group2_masks, str(results_dir)
    )
    
    print(f"💾 Visualization saved: {viz_path}")
    
    return {
        'total_masks': len(result['final_masks']),
        'group1_count': len(group1_masks),
        'group2_count': len(group2_masks),
        'processing_time': processing_time,
        'visualization_path': viz_path,
        'group1_masks': group1_masks,
        'group2_masks': group2_masks
    }

if __name__ == "__main__":
    # Check for image file
    image_file = "satellite_droplet_v2.png"
    
    if not Path(image_file).exists():
        print(f"❌ Error: Image file '{image_file}' not found!")
        print(f"Please place your image file in the current directory: {Path.cwd()}")
        print(f"Expected file: {Path.cwd() / image_file}")
        sys.exit(1)
    
    try:
        # Run analysis
        results = run_satellite_analysis(
            image_path=image_file,
            min_circularity=0.53,
            max_blob_distance=50
        )
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"🔍 Check the results folder for the visualization.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 