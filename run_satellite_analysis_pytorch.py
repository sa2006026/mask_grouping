#!/usr/bin/env python3
"""
Script to run satellite droplet analysis using PyTorch SAM model with crop_n_layers=3.
This provides the enhanced SAM functionality equivalent to the ONNX approach.
"""

import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

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

def run_satellite_analysis_pytorch(image_path: str, 
                                 min_circularity: float = 0.53, 
                                 max_blob_distance: float = 50):
    """Run satellite droplet analysis with PyTorch backend and crop_n_layers=3."""
    
    print("=" * 60)
    print("🛰️  SATELLITE DROPLET ANALYSIS WITH PYTORCH SAM")
    print("=" * 60)
    
    # Import PyTorch SAM components
    try:
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        from src.mask_grouping.core import HugeDropletProcessor, MaskGroupingProcessor
        from sam_config import SAMConfig, SAMBackend
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure PyTorch and segment_anything are installed")
        return None
    
    # Configure PyTorch backend
    config = SAMConfig(
        backend=SAMBackend.PYTORCH,
        model_type="vit_h",
        use_gpu=True  # Use GPU if available
    )
    
    print(f"📁 Loading image: {image_path}")
    image_array = load_image(image_path)
    print(f"📐 Image shape: {image_array.shape}")
    
    # Check for model file
    model_path = config.get_pytorch_model_path()
    if not model_path:
        print(f"❌ PyTorch model not found. Expected location: {config.pytorch_model_dir}")
        print("Please download the SAM model using the convert_model.py script")
        return None
    
    print(f"🧠 Loading PyTorch SAM model: {config.model_type}")
    print(f"📂 Model path: {model_path}")
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
    print(f"🚀 Using device: {device}")
    
    sam_model = sam_model_registry[config.model_type](checkpoint=str(model_path))
    sam_model.to(device=device)
    
    # Get mask generator parameters
    params = config.get_mask_generator_params()
    
    # Create mask generator with crop_n_layers=3 (enhanced SAM for satellite analysis)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=params.get("points_per_side", 32),
        pred_iou_thresh=params.get("pred_iou_thresh", 0.88),
        stability_score_thresh=params.get("stability_score_thresh", 0.95),
        crop_n_layers=3,  # Enhanced SAM for satellite droplets
        min_mask_region_area=params.get("min_mask_region_area", 100)
    )
    
    print(f"⚙️  Configuration:")
    print(f"   - Points per side: {params.get('points_per_side', 32)}")
    print(f"   - IoU threshold: {params.get('pred_iou_thresh', 0.88)}")
    print(f"   - Stability threshold: {params.get('stability_score_thresh', 0.95)}")
    print(f"   - Crop layers: 3 (Enhanced SAM)")
    print(f"   - Min region area: {params.get('min_mask_region_area', 100)}")
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
    
    # Apply K-means clustering for satellite droplet separation
    print(f"🎯 Applying K-means clustering for satellite droplet separation...")
    temp_processor = MaskGroupingProcessor(processor.mask_generator)
    
    if len(result['final_masks']) < 2:
        print(f"⚠️  Warning: Only {len(result['final_masks'])} masks found. Need at least 2 for clustering.")
        if len(result['final_masks']) == 0:
            print("❌ No masks detected. Try adjusting the parameters.")
            return None
        # If only 1 mask, treat it as group 1
        group1_masks = result['final_masks']
        group2_masks = []
    else:
        clustered_masks, cluster_labels = temp_processor._cluster_masks_kmeans(
            result['final_masks'], n_clusters=2
        )
        
        # clustered_masks is a list of lists, where each sublist contains masks for that cluster
        # Determine which cluster has larger masks on average
        avg_areas = []
        for cluster_id in range(2):
            if cluster_id < len(clustered_masks) and len(clustered_masks[cluster_id]) > 0:
                cluster_mask_list = clustered_masks[cluster_id]
                avg_area = np.mean([mask['area'] for mask in cluster_mask_list])
                avg_areas.append(avg_area)
            else:
                avg_areas.append(0)
        
        # Assign groups (larger average area = Group 1, smaller = Group 2)
        large_cluster_id = np.argmax(avg_areas)
        small_cluster_id = 1 - large_cluster_id
        
        group1_masks = clustered_masks[large_cluster_id] if large_cluster_id < len(clustered_masks) else []
        group2_masks = clustered_masks[small_cluster_id] if small_cluster_id < len(clustered_masks) else []
    
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
        results = run_satellite_analysis_pytorch(
            image_path=image_file,
            min_circularity=0.53,
            max_blob_distance=50
        )
        
        if results:
            print(f"\n✅ Analysis completed successfully!")
            print(f"🔍 Check the results folder for the visualization.")
        else:
            print(f"\n❌ Analysis failed.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 