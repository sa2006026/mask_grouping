"""Flask web application for SAM mask grouping with overlap analysis."""

import os
import io
import base64
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Import our mask grouping utilities
import sys
sys.path.append(str(Path(__file__).parent / "src"))
from mask_grouping.core import MaskGroupingProcessor, HugeDropletProcessor
from mask_grouping.droplet_processor import DropletProcessor
from mask_grouping.utils import (
    convert_numpy_types,
    validate_file_upload,
    process_image_from_base64,
    masks_to_base64_images
)

def process_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to numpy array.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        RGB image as numpy array
    """
    # Convert bytes to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
    return image_array

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

# Global variables for SAM model
sam_model = None
mask_generator = None
mask_processor = None
droplet_processor = None
huge_droplet_processor = None

# Configuration for PyTorch-only mode
MODEL_TYPE = os.getenv('SAM_MODEL_TYPE', 'vit_h')
USE_GPU = os.getenv('SAM_USE_GPU', 'true').lower() == 'true'
MODEL_DIR = Path(__file__).parent / "model"

def initialize_sam():
    """Initialize the SAM model using PyTorch backend only."""
    global sam_model, mask_generator, mask_processor, droplet_processor, huge_droplet_processor
    
    if sam_model is None:
        print("Initializing SAM model with PyTorch backend...")
        print(f"Model type: {MODEL_TYPE}")
        print(f"GPU enabled: {USE_GPU}")
        
        # Create model directory
        MODEL_DIR.mkdir(exist_ok=True)
        
        # Download model if not available
        model_path = _get_pytorch_model_path()
        if not model_path.exists():
            print(f"Model not found at {model_path}")
            print("Downloading PyTorch model...")
            _download_pytorch_model()
        
        # Initialize PyTorch backend
        _initialize_pytorch_backend()
        
        print("SAM model initialized successfully!")
        print(f"Backend: PyTorch")
        print(f"Model type: {MODEL_TYPE}")
        print("All processors ready for analysis")

def _get_pytorch_model_path():
    """Get the path to the PyTorch SAM model."""
    model_files = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth", 
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    return MODEL_DIR / model_files[MODEL_TYPE]

def _download_pytorch_model():
    """Download PyTorch SAM model if not available."""
    model_files = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth", 
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    
    model_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    model_filename = model_files[MODEL_TYPE]
    model_path = MODEL_DIR / model_filename
    
    if not model_path.exists():
        import urllib.request
        model_url = model_urls[MODEL_TYPE]
        print(f"Downloading {model_filename}...")
        urllib.request.urlretrieve(model_url, str(model_path))
        print("Download completed!")

def _initialize_pytorch_backend():
    """Initialize PyTorch SAM backend."""
    global sam_model, mask_generator, droplet_processor, huge_droplet_processor, mask_processor
    
    # Get model path
    model_path = _get_pytorch_model_path()
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
    print(f"Using device: {device}")
    print(f"Loading PyTorch SAM model: {MODEL_TYPE} from {model_path}")
    
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=str(model_path))
    sam_model.to(device=device)
    
    # High-quality default parameters (32 points per side, 3 crop layers)
    points_per_side = 32
    pred_iou_thresh = 0.88
    stability_score_thresh = 0.95
    min_mask_region_area = 100
    
    print(f"⚙️  Configuration:")
    print(f"   - Points per side: {points_per_side}")
    print(f"   - Crop layers: 3 (multiple/huge_droplet modes)")
    print(f"   - IoU threshold: {pred_iou_thresh}")
    print(f"   - Stability threshold: {stability_score_thresh}")
    
    # Create mask generator for multiple mode (crop_n_layers=3)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=3,  # High quality with 3 crop layers
        min_mask_region_area=min_mask_region_area
    )
    
    # Create mask generator for droplet mode (crop_n_layers=1)
    droplet_mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        crop_n_layers=1,  # Basic SAM for droplet mode
        min_mask_region_area=0  # More permissive for droplets
    )
    
    # Create mask generator for huge droplet mode (crop_n_layers=3)
    huge_droplet_mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=3,  # High quality with 3 crop layers
        min_mask_region_area=min_mask_region_area
    )
    
    # Initialize our processors
    mask_processor = MaskGroupingProcessor(mask_generator)
    droplet_processor = DropletProcessor(droplet_mask_generator)
    huge_droplet_processor = HugeDropletProcessor(huge_droplet_mask_generator)


def generate_preview_image_with_masks(image_array, masks, mode):
    """Generate a preview image with bounding boxes overlaid on masks."""
    import cv2
    from io import BytesIO
    import base64
    
    # Create a copy of the original image
    preview_image = image_array.copy()
    if len(preview_image.shape) == 3:
        preview_rgb = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
    else:
        preview_rgb = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2RGB)
    
    # Draw bounding boxes for each mask
    colors = {
        'droplet': (0, 255, 0),        # Green for droplet mode
        'huge_droplet': (0, 255, 255), # Cyan for huge droplet mode
        'satellite_droplet': [(0, 255, 0), (255, 0, 0)],  # Green and Red for satellite mode
        'multiple': (255, 255, 0)      # Yellow for multiple mode
    }
    
    if mode == 'satellite_droplet':
        # For satellite mode, use clustering to determine colors
        from mask_grouping.core import MaskGroupingProcessor
        temp_processor = MaskGroupingProcessor(mask_generator)
        clustered_masks, _ = temp_processor._cluster_masks_kmeans(masks, n_clusters=2)
        
        # Draw group 1 (smaller) in green, group 2 (larger) in red
        if clustered_masks[0] and clustered_masks[1]:
            avg_area_0 = np.mean([mask['area'] for mask in clustered_masks[0]])
            avg_area_1 = np.mean([mask['area'] for mask in clustered_masks[1]])
            if avg_area_0 > avg_area_1:
                larger_cluster, smaller_cluster = clustered_masks[0], clustered_masks[1]
            else:
                larger_cluster, smaller_cluster = clustered_masks[1], clustered_masks[0]
        else:
            larger_cluster = clustered_masks[0] if clustered_masks[0] else clustered_masks[1]
            smaller_cluster = clustered_masks[1] if clustered_masks[0] else clustered_masks[0]
        
        # Apply 20% diameter filter to larger cluster for consistent visualization
        larger_cluster_original = larger_cluster.copy()
        larger_cluster_diameters = [2 * np.sqrt(mask['area'] / np.pi) for mask in larger_cluster_original]
        
        if larger_cluster_diameters:
            avg_diameter_larger_group = np.mean(larger_cluster_diameters)
            diameter_threshold = avg_diameter_larger_group * 1.2  # 20% larger than average
            
            # Filter out oversized masks from larger cluster
            filtered_larger_cluster = []
            for mask in larger_cluster_original:
                diameter = 2 * np.sqrt(mask['area'] / np.pi)
                if diameter <= diameter_threshold:
                    filtered_larger_cluster.append(mask)
            
            larger_cluster = filtered_larger_cluster
        
        # Draw bounding boxes for each group
        for mask in smaller_cluster:
            if 'bbox' in mask:
                x, y, w, h = mask['bbox']
                cv2.rectangle(preview_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        
        for mask in larger_cluster:
            if 'bbox' in mask:
                x, y, w, h = mask['bbox']
                cv2.rectangle(preview_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
    else:
        # For other modes, use single color
        color = colors.get(mode, (255, 255, 0))
        for mask in masks:
            if 'bbox' in mask:
                x, y, w, h = mask['bbox']
                cv2.rectangle(preview_rgb, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
    
    # No resizing - preserve original image dimensions for coordinate consistency
    
    # Convert to base64 for transmission
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


def generate_custom_preview_image(image_array, masks, mode):
    """Generate a custom preview image with bounding boxes based on user selections."""
    import cv2
    
    # Create a copy of the original image
    preview_image = image_array.copy()
    if len(preview_image.shape) == 3:
        preview_rgb = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
    else:
        preview_rgb = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2RGB)
    
    # Draw bounding boxes based on mode and mask properties
    for mask in masks:
        if 'bbox' not in mask:
            continue
            
        x, y, w, h = mask['bbox']
        
        # Choose color based on mode and mask properties
        if mode == 'satellite_droplet' and 'group' in mask:
            # Use group information from frontend
            color = (0, 255, 0) if mask['group'] == 1 else (255, 0, 0)  # Green for group 1, Red for group 2
        elif mode == 'multiple' and 'is_aggregate' in mask:
            # Orange for aggregate masks, Green for simple masks
            color = (255, 165, 0) if mask['is_aggregate'] else (0, 255, 0)
        elif mode == 'huge_droplet':
            color = (0, 255, 255)  # Cyan for huge droplet mode
        else:
            color = (0, 255, 0)  # Default green for droplet mode
        
        cv2.rectangle(preview_rgb, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
    
    # No resizing - preserve original image dimensions for coordinate consistency
    
    # Convert to base64 for transmission
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': sam_model is not None,
        'model_type': MODEL_TYPE if sam_model is not None else None,
        'backend': 'pytorch',
        'device': 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'
    })

@app.route('/')
def index():
    """Serve the main HTML interface."""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.route('/segment', methods=['POST'])
def segment_image():
    """Segment an uploaded image and return analysis based on mode."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Get image data and mode
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get mode (default to 'multiple' for backward compatibility)
        mode = data.get('mode', 'multiple')
        image_data = data['image']
        
        # Process image from base64
        image_array = process_image_from_base64(image_data)
        print(f"Processing image in {mode} mode: {image_array.shape}")
        
        if mode == 'droplet':
            # Process with droplet processor
            start_time = time.time()
            result = droplet_processor.process_image_array(image_array)
            processing_time = time.time() - start_time
            
            # Create droplet visualization
            droplet_viz_path = droplet_processor.create_droplet_visualization(
                image_array, result['final_masks'], app.config['RESULTS_FOLDER']
            )
            
            # Convert visualization to base64
            with open(droplet_viz_path, 'rb') as f:
                viz_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            response_data = {
                'success': True,
                'mode': 'droplet',
                'processing_time': float(processing_time),
                'statistics': convert_numpy_types(result['statistics']),
                'total_masks': int(result['total_masks']),
                'filtered_masks': int(result['filtered_masks']),
                'droplet_analysis': {
                    'image': f"data:image/png;base64,{viz_base64}",
                    'mask_count': result['statistics']['mask_count'],
                    'average_diameter': result['statistics']['average_diameter'],
                    'summary': result['statistics']
                }
            }
        elif mode == 'huge_droplet':
            # Process with huge droplet processor
            start_time = time.time()
            result = huge_droplet_processor.process_image_array(image_array)
            processing_time = time.time() - start_time
            
            # Create huge droplet visualization
            huge_droplet_viz_path = huge_droplet_processor.create_huge_droplet_visualization(
                image_array, result['final_masks'], app.config['RESULTS_FOLDER']
            )
            
            # Convert visualization to base64
            with open(huge_droplet_viz_path, 'rb') as f:
                viz_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            response_data = {
                'success': True,
                'mode': 'huge_droplet',
                'processing_time': float(processing_time),
                'statistics': convert_numpy_types(result['statistics']),
                'total_masks': int(result['total_masks']),
                'filtered_masks': int(result['filtered_masks']),
                'huge_droplet_analysis': {
                    'image': f"data:image/png;base64,{viz_base64}",
                    'mask_count': result['statistics']['mask_count'],
                    'average_diameter': result['statistics']['average_diameter'],
                    'summary': result['statistics']
                }
            }
        else:
            # Process with multiple mode processor (original functionality)
            start_time = time.time()
            result = mask_processor.process_image_array(image_array)
            processing_time = time.time() - start_time
            
            # Convert masks to base64 for web display
            mask_images = masks_to_base64_images(result['filtered_masks'])
            
            # Create overlap visualization
            smaller_masks = result.get('smaller_cluster_masks', [])
            overlap_viz_path = mask_processor.create_overlap_visualization(
                image_array, result['final_masks'], app.config['RESULTS_FOLDER'], smaller_masks
            )
            
            # Convert overlap visualization to base64
            with open(overlap_viz_path, 'rb') as f:
                overlap_viz_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            response_data = {
                'success': True,
                'mode': 'multiple',
                'processing_time': float(processing_time),
                'statistics': convert_numpy_types(result['statistics']),
                'total_masks': int(result['total_masks']),
                'filtered_masks': int(result['filtered_masks']),
                'overlap_analysis': {
                    'image': f"data:image/png;base64,{overlap_viz_base64}",
                    'total_large_masks': len(result['final_masks']),
                    'overlap_threshold': 80.0,
                    'summary': result['statistics']
                },
                'masks': mask_images[:50] if len(mask_images) > 50 else mask_images  # Limit for web display
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in segment_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/segment_file', methods=['POST'])
def segment_uploaded_file():
    """Segment an uploaded file and return analysis based on mode."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get mode (default to 'multiple' for backward compatibility)
        mode = request.form.get('mode', 'multiple')
        
        # Read and process image
        image_data = file.read()
        image_array = process_image_from_bytes(image_data)
        print(f"Processing uploaded file in {mode} mode: {image_array.shape}")
        
        if mode == 'droplet':
            # Process with droplet processor
            start_time = time.time()
            result = droplet_processor.process_image_array(image_array)
            processing_time = time.time() - start_time
            
            # Create droplet visualization
            droplet_viz_path = droplet_processor.create_droplet_visualization(
                image_array, result['final_masks'], app.config['RESULTS_FOLDER']
            )
            
            # Convert visualization to base64
            with open(droplet_viz_path, 'rb') as f:
                viz_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            response_data = {
                'success': True,
                'mode': 'droplet',
                'processing_time': float(processing_time),
                'statistics': convert_numpy_types(result['statistics']),
                'total_masks': int(result['total_masks']),
                'filtered_masks': int(result['filtered_masks']),
                'droplet_analysis': {
                    'image': f"data:image/png;base64,{viz_base64}",
                    'mask_count': result['statistics']['mask_count'],
                    'average_diameter': result['statistics']['average_diameter'],
                    'summary': result['statistics']
                }
            }
        elif mode == 'huge_droplet':
            # Process with huge droplet processor
            start_time = time.time()
            result = huge_droplet_processor.process_image_array(image_array)
            processing_time = time.time() - start_time
            
            # Create huge droplet visualization
            huge_droplet_viz_path = huge_droplet_processor.create_huge_droplet_visualization(
                image_array, result['final_masks'], app.config['RESULTS_FOLDER']
            )
            
            # Convert visualization to base64
            with open(huge_droplet_viz_path, 'rb') as f:
                viz_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            response_data = {
                'success': True,
                'mode': 'huge_droplet',
                'processing_time': float(processing_time),
                'statistics': convert_numpy_types(result['statistics']),
                'total_masks': int(result['total_masks']),
                'filtered_masks': int(result['filtered_masks']),
                'huge_droplet_analysis': {
                    'image': f"data:image/png;base64,{viz_base64}",
                    'mask_count': result['statistics']['mask_count'],
                    'average_diameter': result['statistics']['average_diameter'],
                    'summary': result['statistics']
                }
            }
        else:
            # Process with multiple mode processor (original functionality)
            start_time = time.time()
            result = mask_processor.process_image_array(image_array)
            processing_time = time.time() - start_time
            
            # Convert masks to base64 for web display
            mask_images = masks_to_base64_images(result['filtered_masks'])
            
            # Create overlap visualization
            smaller_masks = result.get('smaller_cluster_masks', [])
            overlap_viz_path = mask_processor.create_overlap_visualization(
                image_array, result['final_masks'], app.config['RESULTS_FOLDER'], smaller_masks
            )
            
            # Convert overlap visualization to base64
            with open(overlap_viz_path, 'rb') as f:
                overlap_viz_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            response_data = {
                'success': True,
                'mode': 'multiple',
                'processing_time': float(processing_time),
                'statistics': convert_numpy_types(result['statistics']),
                'total_masks': int(result['total_masks']),
                'filtered_masks': int(result['filtered_masks']),
                'overlap_analysis': {
                    'image': f"data:image/png;base64,{overlap_viz_base64}",
                    'total_large_masks': len(result['final_masks']),
                    'overlap_threshold': 80.0,
                    'summary': result['statistics']
                },
                'masks': mask_images[:50] if len(mask_images) > 50 else mask_images  # Limit for web display
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in segment_uploaded_file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_overlap', methods=['POST'])
def analyze_overlap():
    """Analyze overlap between masks (main endpoint matching mask_size_grouping.py)."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Get configuration from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get optional parameters
        overlap_threshold = data.get('overlap_threshold', 80.0)  # Default 80% like original script
        min_circularity = data.get('min_circularity', 0.55)  # Default from original script
        max_blob_distance = data.get('max_blob_distance', 50)  # Default from original script
        fluorescent_mode = data.get('fluorescent_mode', False)  # Default to non-fluorescent
        min_brightness_threshold = data.get('min_brightness_threshold', 200)  # Default brightness threshold
        
        # Update processor configuration
        mask_processor.overlap_threshold = overlap_threshold / 100.0  # Convert to decimal
        mask_processor.min_circularity = min_circularity
        mask_processor.max_blob_distance = max_blob_distance
        mask_processor.fluorescent_mode = fluorescent_mode
        mask_processor.min_brightness_threshold = min_brightness_threshold
        
        image_data = data['image']
        
        # Process image from base64
        image_array = process_image_from_base64(image_data)
        print(f"Analyzing overlap for image: {image_array.shape}")
        print(f"Configuration: overlap_threshold={overlap_threshold}%, min_circularity={min_circularity}, max_blob_distance={max_blob_distance}")
        print(f"Fluorescent mode: {fluorescent_mode}, min_brightness_threshold={min_brightness_threshold}")
        
        # Process with mask grouping (same as mask_size_grouping.py)
        start_time = time.time()
        result = mask_processor.process_image_array(image_array)
        processing_time = time.time() - start_time
        
        # Create overlap visualization (main output like original script)
        smaller_masks = result.get('smaller_cluster_masks', [])
        overlap_viz_path = mask_processor.create_overlap_visualization(
            image_array, result['final_masks'], app.config['RESULTS_FOLDER'], smaller_masks
        )
        
        # Convert overlap visualization to base64
        with open(overlap_viz_path, 'rb') as f:
            overlap_viz_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Calculate detailed statistics
        overlap_stats = result['statistics']
        total_overlaps = sum([mask.get('overlap_count', 0) for mask in result['final_masks']])
        
        # Generate detailed overlap counting statistics
        final_masks = result['final_masks']
        aggregate_masks = result.get('aggregate_masks', [])
        simple_masks = result.get('simple_masks', [])
        
        # Count overlaps by number (empty, single, double, triple, etc.)
        overlap_counts = {}
        aggregate_counts = {}
        simple_counts = {}
        
        for mask in final_masks:
            overlap_count = mask['overlap_count']
            is_aggregate = mask.get('is_aggregate_overlap', False)
            
            # Overall counts
            if overlap_count not in overlap_counts:
                overlap_counts[overlap_count] = 0
            overlap_counts[overlap_count] += 1
            
            # Separate counts for aggregate and simple
            if is_aggregate:
                if overlap_count not in aggregate_counts:
                    aggregate_counts[overlap_count] = 0
                aggregate_counts[overlap_count] += 1
            else:
                if overlap_count not in simple_counts:
                    simple_counts[overlap_count] = 0
                simple_counts[overlap_count] += 1
        
        # Create readable labels for overlap counts
        def get_overlap_label(count):
            if count == 0:
                return "Empty"
            elif count == 1:
                return "Single"
            elif count == 2:
                return "Double"
            elif count == 3:
                return "Triple"
            elif count == 4:
                return "Quadruple"
            elif count == 5:
                return "Quintuple"
            else:
                return f"{count} Overlaps"
        
        # Build statistical breakdown
        overlap_statistics = {
            'total_breakdown': {
                get_overlap_label(count): {
                    'count': masks,
                    'overlap_number': count
                }
                for count, masks in sorted(overlap_counts.items())
            },
            'aggregate_breakdown': {
                get_overlap_label(count): {
                    'count': masks,
                    'overlap_number': count
                }
                for count, masks in sorted(aggregate_counts.items())
            },
            'simple_breakdown': {
                get_overlap_label(count): {
                    'count': masks,
                    'overlap_number': count
                }
                for count, masks in sorted(simple_counts.items())
            },
            'summary_counts': {
                'total_masks': len(final_masks),
                'total_aggregate_masks': len(aggregate_masks),
                'total_simple_masks': len(simple_masks),
                'aggregate_percentage': (len(aggregate_masks) / len(final_masks) * 100) if final_masks else 0,
                'simple_percentage': (len(simple_masks) / len(final_masks) * 100) if final_masks else 0
            }
        }

        response_data = {
            'success': True,
            'processing_time': float(processing_time),
            'configuration': {
                'overlap_threshold': overlap_threshold,
                'min_circularity': min_circularity,
                'max_blob_distance': max_blob_distance,
                'fluorescent_mode': fluorescent_mode,
                'min_brightness_threshold': min_brightness_threshold,
                'model_type': 'vit_h',
                'crop_n_layers': 3,
                'points_per_side': 32
            },
            'results': {
                'total_masks_generated': int(result['total_masks']),
                'masks_after_filtering': int(result['filtered_masks']),
                'large_masks_analyzed': len(result['final_masks']),
                'total_overlaps_found': int(total_overlaps),
                'overall_overlap_percentage': float(overlap_stats.get('overlap_percentage', 0.0)),
                'overlap_analysis_image': f"data:image/png;base64,{overlap_viz_base64}",
                'original_image_for_hover': f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', image_array)[1]).decode('utf-8')}"
            },
            'mask_data_for_frontend': {
                'masks': [convert_numpy_types({
                    'bbox': mask['bbox'],
                    'area': mask['area'],
                    'diameter': float(np.sqrt(mask['area'] * 4 / np.pi)),
                    'overlap_count': mask.get('overlap_count', 0),
                    'is_aggregate': mask.get('is_aggregate_overlap', False),
                    'circularity': mask.get('circularity', 0),
                    'stability_score': mask.get('stability_score', 0)
                }) for mask in result['final_masks']]
            },
            'overlap_statistics': overlap_statistics,
            'detailed_statistics': convert_numpy_types(overlap_stats),
            'cluster_breakdown': {
                'cluster_0': overlap_stats.get('cluster_stats', {}).get('0', {}),
                'cluster_1': overlap_stats.get('cluster_stats', {}).get('1', {})
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_overlap: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download_result/<filename>')
def download_result(filename):
    """Download result file."""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/defaults', methods=['GET'])
def get_defaults():
    """Get default configuration values."""
    return jsonify({
        'modes': {
            'droplet': {
                'description': 'Basic droplet analysis with SAM crop_n_layers=1',
                'crop_n_layers': 1,
                'output': ['mask_count', 'average_diameter', 'diameter_statistics']
            },
            'huge_droplet': {
                'description': 'Enhanced droplet analysis with SAM crop_n_layers=3',
                'crop_n_layers': 3,
                'output': ['mask_count', 'average_diameter', 'diameter_statistics']
            },
            'multiple': {
                'description': 'Advanced overlap analysis with SAM crop_n_layers=3',
                'crop_n_layers': 3,
                'output': ['empty', 'single', 'double', 'triple', 'quadruple', 'aggregate']
            }
        },
        'overlap_threshold': 80.0,
        'min_circularity': 0.53,
        'max_blob_distance': 50,
        'fluorescent_mode': False,
        'min_brightness_threshold': 200,
        'model_type': MODEL_TYPE,
        'points_per_side': 32,
        'supported_formats': ['PNG', 'JPG', 'JPEG', 'TIFF', 'BMP'],
        'max_file_size_mb': 50
    })

@app.route('/backend_info', methods=['GET'])
def backend_info():
    """Get information about the SAM backend and configuration."""
    if sam_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'backend': 'pytorch',
        'model_type': MODEL_TYPE,
        'use_gpu': USE_GPU,
        'device': 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu',
        'pytorch_available': True,
        'onnx_available': False,
        'tensorrt_available': False,
        'model_path': str(_get_pytorch_model_path()),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'configuration': {
            'points_per_side': 32,
            'crop_layers_multiple': 3,
            'crop_layers_droplet': 1,
            'crop_layers_huge_droplet': 3,
            'pred_iou_thresh': 0.88,
            'stability_score_thresh': 0.95,
            'min_mask_region_area': 100
        }
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded SAM model."""
    if sam_model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_loaded': True,
        'model_type': MODEL_TYPE,
        'backend': 'pytorch',
        'device': 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu',
        'modes': {
            'droplet': {
                'crop_n_layers': 1,
                'processing_pipeline': [
                    'SAM mask generation (crop_n_layers=1)',
                    'Edge proximity filtering',
                    'Circularity filtering (>=0.53)',
                    'Blob distance filtering (<=50px)',
                    'Duplicate removal (20% overlap)',
                    'Droplet statistics calculation',
                    'Visualization generation'
                ]
            },
            'huge_droplet': {
                'crop_n_layers': 3,
                'processing_pipeline': [
                    'Enhanced SAM mask generation (crop_n_layers=3)',
                    'Edge proximity filtering',
                    'Circularity filtering (>=0.53)',
                    'Blob distance filtering (<=50px)',
                    'Duplicate removal (20% overlap)',
                    'Droplet statistics calculation',
                    'Enhanced visualization generation'
                ]
            },
            'multiple': {
                'crop_n_layers': 3,
                'processing_pipeline': [
                    'Enhanced SAM mask generation (crop_n_layers=3)',
                    'Edge proximity filtering',
                    'Circularity filtering (>=0.53)',
                    'Blob distance filtering (<=50px)',
                    'K-means clustering (2 clusters)',
                    'Brightness filtering (fluorescent mode)',
                    'Duplicate removal within clusters',
                    'Overlap analysis with statistical aggregation',
                    'Advanced visualization with overlap detection'
                ]
            }
        },
        'parameters': {
            'points_per_side': 32,
            'pred_iou_thresh': 0.9,
            'stability_score_thresh': 0.95,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 100
        }
    })

@app.route('/analyze_droplets', methods=['POST'])
def analyze_droplets():
    """Analyze droplets using basic SAM masking and filtering."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Get image data and configuration
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get optional parameters
        min_circularity = data.get('min_circularity', 0.53)
        max_blob_distance = data.get('max_blob_distance', 50)
        
        # Update processor configuration
        droplet_processor.min_circularity = min_circularity
        droplet_processor.max_blob_distance = max_blob_distance
        
        image_data = data['image']
        
        # Process image from base64
        image_array = process_image_from_base64(image_data)
        print(f"Analyzing droplets for image: {image_array.shape}")
        print(f"Configuration: min_circularity={min_circularity}, max_blob_distance={max_blob_distance}")
        
        # Process with droplet processor
        start_time = time.time()
        result = droplet_processor.process_image_array(image_array)
        processing_time = time.time() - start_time
        
        # Create droplet visualization
        droplet_viz_path = droplet_processor.create_droplet_visualization(
            image_array, result['final_masks'], app.config['RESULTS_FOLDER']
        )
        
        # Convert visualization to base64
        with open(droplet_viz_path, 'rb') as f:
            viz_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Extract droplet statistics
        droplet_stats = result['statistics']
        
        response_data = {
            'success': True,
            'mode': 'droplet',
            'processing_time': float(processing_time),
            'configuration': {
                'min_circularity': min_circularity,
                'max_blob_distance': max_blob_distance,
                'model_type': 'vit_h',
                'crop_n_layers': 1,
                'points_per_side': 32
            },
            'results': {
                'total_masks_generated': int(result['total_masks']),
                'masks_after_filtering': int(result['filtered_masks']),
                'final_droplet_count': droplet_stats['mask_count'],
                'average_diameter_pixels': float(droplet_stats['average_diameter']),
                'droplet_analysis_image': f"data:image/png;base64,{viz_base64}",
                'original_image_for_hover': f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', image_array)[1]).decode('utf-8')}"
            },
            'droplet_statistics': {
                'mask_count': droplet_stats['mask_count'],
                'average_diameter': droplet_stats['average_diameter'],
                'diameter_std': droplet_stats['diameter_std'],
                'min_diameter': droplet_stats['min_diameter'],
                'max_diameter': droplet_stats['max_diameter'],
                'total_area': droplet_stats['total_area'],
                'average_area': droplet_stats['average_area'],
                'diameter_distribution': droplet_stats['diameter_distribution']
            },
            'mask_data_for_frontend': {
                'masks': [convert_numpy_types({
                    'bbox': mask['bbox'],
                    'area': mask['area'],
                    'diameter': float(np.sqrt(mask['area'] * 4 / np.pi)),
                    'circularity': mask.get('circularity', 0),
                    'stability_score': mask.get('stability_score', 0)
                }) for mask in result['final_masks']]
            },
            'processing_details': convert_numpy_types(result['processing_details'])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_droplets: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_huge_droplets', methods=['POST'])
def analyze_huge_droplets():
    """Analyze droplets using enhanced SAM masking (crop_n_layers=3) and filtering."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Get image data and configuration
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get optional parameters
        min_circularity = data.get('min_circularity', 0.53)
        max_blob_distance = data.get('max_blob_distance', 50)
        
        # Update processor configuration
        huge_droplet_processor.min_circularity = min_circularity
        huge_droplet_processor.max_blob_distance = max_blob_distance
        
        image_data = data['image']
        
        # Process image from base64
        image_array = process_image_from_base64(image_data)
        print(f"Analyzing huge droplets for image: {image_array.shape}")
        print(f"Configuration: min_circularity={min_circularity}, max_blob_distance={max_blob_distance}")
        
        # Process with huge droplet processor
        start_time = time.time()
        result = huge_droplet_processor.process_image_array(image_array)
        processing_time = time.time() - start_time
        
        # Create huge droplet visualization
        huge_droplet_viz_path = huge_droplet_processor.create_huge_droplet_visualization(
            image_array, result['final_masks'], app.config['RESULTS_FOLDER']
        )
        
        # Convert visualization to base64
        with open(huge_droplet_viz_path, 'rb') as f:
            viz_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Extract droplet statistics
        droplet_stats = result['statistics']
        
        response_data = {
            'success': True,
            'mode': 'huge_droplet',
            'processing_time': float(processing_time),
            'configuration': {
                'min_circularity': min_circularity,
                'max_blob_distance': max_blob_distance,
                'model_type': 'vit_h',
                'crop_n_layers': 3,
                'points_per_side': 32
            },
            'results': {
                'total_masks_generated': int(result['total_masks']),
                'masks_after_filtering': int(result['filtered_masks']),
                'final_droplet_count': droplet_stats['mask_count'],
                'average_diameter_pixels': float(droplet_stats['average_diameter']),
                'huge_droplet_analysis_image': f"data:image/png;base64,{viz_base64}",
                'original_image_for_hover': f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', image_array)[1]).decode('utf-8')}"
            },
            'droplet_statistics': {
                'mask_count': droplet_stats['mask_count'],
                'average_diameter': droplet_stats['average_diameter'],
                'diameter_std': droplet_stats['diameter_std'],
                'min_diameter': droplet_stats['min_diameter'],
                'max_diameter': droplet_stats['max_diameter'],
                'total_area': droplet_stats['total_area'],
                'average_area': droplet_stats['average_area'],
                'diameter_distribution': droplet_stats['diameter_distribution']
            },
            'mask_data_for_frontend': {
                'masks': [convert_numpy_types({
                    'bbox': mask['bbox'],
                    'area': mask['area'],
                    'diameter': float(np.sqrt(mask['area'] * 4 / np.pi)),
                    'circularity': mask.get('circularity', 0),
                    'stability_score': mask.get('stability_score', 0)
                }) for mask in result['final_masks']]
            },
            'processing_details': convert_numpy_types(result['processing_details'])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_huge_droplets: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/statistics_summary', methods=['POST'])
def get_statistics_summary():
    """Get a clean summary of overlap group quantities and aggregate group information."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Get image data and configuration
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get optional parameters
        overlap_threshold = data.get('overlap_threshold', 80.0)
        min_circularity = data.get('min_circularity', 0.53)
        max_blob_distance = data.get('max_blob_distance', 50)
        
        # Update processor configuration
        mask_processor.overlap_threshold = overlap_threshold / 100.0
        mask_processor.min_circularity = min_circularity
        mask_processor.max_blob_distance = max_blob_distance
        
        image_data = data['image']
        
        # Process image from base64
        image_array = process_image_from_base64(image_data)
        print(f"Generating statistics summary for image: {image_array.shape}")
        
        # Process with mask grouping
        start_time = time.time()
        result = mask_processor.process_image_array(image_array)
        processing_time = time.time() - start_time
        
        # Extract key statistics from the result
        final_masks = result['final_masks']
        aggregate_masks = result.get('aggregate_masks', [])
        simple_masks = result.get('simple_masks', [])
        overlap_stats = result['statistics']['overlap_analysis']['overlap_count_statistics']
        
        # Build clean summary structure
        overlap_groups = {}
        for mask in final_masks:
            overlap_count = mask['overlap_count']
            is_aggregate = mask.get('is_aggregate_overlap', False)
            
            if overlap_count not in overlap_groups:
                overlap_groups[overlap_count] = {
                    'overlap_count': overlap_count,
                    'total_masks': 0,
                    'simple_masks': 0,
                    'aggregate_masks': 0,
                    'mask_details': []
                }
            
            overlap_groups[overlap_count]['total_masks'] += 1
            
            if is_aggregate:
                overlap_groups[overlap_count]['aggregate_masks'] += 1
            else:
                overlap_groups[overlap_count]['simple_masks'] += 1
            
            # Add mask details
            overlap_groups[overlap_count]['mask_details'].append({
                'mask_id': mask['original_sam_id'],
                'area': mask['area'],
                'type': 'aggregate' if is_aggregate else 'simple',
                'smaller_group_overlaps': mask.get('num_smaller_overlaps', 0),
                'larger_group_overlaps': mask.get('num_larger_overlaps', 0)
            })
        
        # Sort overlap groups by overlap count
        sorted_overlap_groups = dict(sorted(overlap_groups.items()))
        
        # Calculate aggregate group summary
        aggregate_summary = {
            'total_aggregate_masks': len(aggregate_masks),
            'aggregate_percentage': (len(aggregate_masks) / len(final_masks) * 100) if final_masks else 0,
            'overlap_distribution': {},
            'total_overlaps': sum([mask['overlap_count'] for mask in aggregate_masks]),
            'average_overlaps_per_mask': float(np.mean([mask['overlap_count'] for mask in aggregate_masks])) if aggregate_masks else 0,
            'larger_group_overlaps': sum([mask.get('num_larger_overlaps', 0) for mask in aggregate_masks]),
            'smaller_group_overlaps': sum([mask.get('num_smaller_overlaps', 0) for mask in aggregate_masks])
        }
        
        # Aggregate overlap distribution
        for mask in aggregate_masks:
            overlap_count = mask['overlap_count']
            if overlap_count not in aggregate_summary['overlap_distribution']:
                aggregate_summary['overlap_distribution'][overlap_count] = 0
            aggregate_summary['overlap_distribution'][overlap_count] += 1
        
        # Calculate simple group summary
        simple_summary = {
            'total_simple_masks': len(simple_masks),
            'simple_percentage': (len(simple_masks) / len(final_masks) * 100) if final_masks else 0,
            'overlap_distribution': {},
            'total_overlaps': sum([mask['overlap_count'] for mask in simple_masks]),
            'average_overlaps_per_mask': float(np.mean([mask['overlap_count'] for mask in simple_masks])) if simple_masks else 0,
            'larger_group_overlaps': sum([mask.get('num_larger_overlaps', 0) for mask in simple_masks]),
            'smaller_group_overlaps': sum([mask.get('num_smaller_overlaps', 0) for mask in simple_masks])
        }
        
        # Simple overlap distribution
        for mask in simple_masks:
            overlap_count = mask['overlap_count']
            if overlap_count not in simple_summary['overlap_distribution']:
                simple_summary['overlap_distribution'][overlap_count] = 0
            simple_summary['overlap_distribution'][overlap_count] += 1
        
        # Overall summary statistics
        overall_summary = {
            'total_masks_analyzed': len(final_masks),
            'masks_with_no_overlaps': overlap_groups.get(0, {}).get('total_masks', 0),
            'masks_with_overlaps': len(final_masks) - overlap_groups.get(0, {}).get('total_masks', 0),
            'max_overlaps_in_single_mask': max([mask['overlap_count'] for mask in final_masks]) if final_masks else 0,
            'total_overlap_instances': sum([mask['overlap_count'] for mask in final_masks]),
            'overlap_threshold_used': overlap_threshold,
            'processing_time_seconds': round(processing_time, 2)
        }
        
        # Create comprehensive response
        response_data = {
            'success': True,
            'summary': {
                'overall': overall_summary,
                'aggregate_groups': aggregate_summary,
                'simple_groups': simple_summary
            },
            'overlap_groups_detail': sorted_overlap_groups,
            'configuration': {
                'overlap_threshold': overlap_threshold,
                'min_circularity': min_circularity,
                'max_blob_distance': max_blob_distance
            },
            'raw_statistics': {
                'total_masks_generated': result['total_masks'],
                'masks_after_filtering': result['filtered_masks'],
                'processing_pipeline_summary': result['statistics']['processing_summary']
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_statistics_summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_satellite_droplets', methods=['POST'])
def analyze_satellite_droplets():
    """Analyze droplets using enhanced SAM masking with two-group classification and Excel export."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Get image data and configuration
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get optional parameters
        min_circularity = data.get('min_circularity', 0.53)
        max_blob_distance = data.get('max_blob_distance', 50)
        
        # Update processor configuration
        huge_droplet_processor.min_circularity = min_circularity
        huge_droplet_processor.max_blob_distance = max_blob_distance
        
        image_data = data['image']
        
        # Process image from base64
        image_array = process_image_from_base64(image_data)
        print(f"Analyzing satellite droplets for image: {image_array.shape}")
        print(f"Configuration: min_circularity={min_circularity}, max_blob_distance={max_blob_distance}")
        
        # Process with huge droplet processor but add K-means clustering for satellite mode
        start_time = time.time()
        result = huge_droplet_processor.process_image_array(image_array)
        
        # Add K-means clustering specifically for satellite mode
        from mask_grouping.core import MaskGroupingProcessor
        temp_processor = MaskGroupingProcessor(huge_droplet_processor.mask_generator)
        
        # Apply K-means clustering to the filtered masks
        clustered_masks, cluster_labels = temp_processor._cluster_masks_kmeans(result['final_masks'], n_clusters=2)
        
        # Determine which cluster has larger masks on average for proper grouping
        if clustered_masks[0] and clustered_masks[1]:
            avg_area_0 = np.mean([mask['area'] for mask in clustered_masks[0]])
            avg_area_1 = np.mean([mask['area'] for mask in clustered_masks[1]])
            
            if avg_area_0 > avg_area_1:
                larger_cluster = clustered_masks[0]
                smaller_cluster = clustered_masks[1]
                print(f"Cluster 0 identified as LARGER masks (avg area: {avg_area_0:.0f})")
                print(f"Cluster 1 identified as SMALLER masks (avg area: {avg_area_1:.0f})")
            else:
                larger_cluster = clustered_masks[1]
                smaller_cluster = clustered_masks[0]
                print(f"Cluster 1 identified as LARGER masks (avg area: {avg_area_1:.0f})")
                print(f"Cluster 0 identified as SMALLER masks (avg area: {avg_area_0:.0f})")
        else:
            # Fallback if one cluster is empty
            larger_cluster = clustered_masks[0] if clustered_masks[0] else clustered_masks[1]
            smaller_cluster = clustered_masks[1] if clustered_masks[0] else clustered_masks[0]
        
        print(f"K-means clustering results: {len(larger_cluster)} larger masks, {len(smaller_cluster)} smaller masks")
        
        processing_time = time.time() - start_time
        
        # Create satellite droplet visualization with two-group colors
        satellite_viz_path = huge_droplet_processor.create_satellite_droplet_visualization(
            image_array, smaller_cluster, larger_cluster, app.config['RESULTS_FOLDER']
        )
        
        # Convert visualization to base64
        with open(satellite_viz_path, 'rb') as f:
            viz_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Also convert original image to base64 for interactive hover (coordinates match bboxes)
        from PIL import Image as PILImage
        import io
        
        # Convert numpy array to PIL Image and then to base64
        pil_image = PILImage.fromarray(image_array)
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        original_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        
        # Extract droplet statistics
        droplet_stats = result['statistics']
        
        # Calculate two-group statistics using K-means clusters
        # Calculate diameters for each cluster
        def calculate_diameters_from_masks(masks):
            diameters = []
            for mask in masks:
                area = mask['area']
                # Diameter = 2 * sqrt(area / π)
                diameter = 2 * np.sqrt(area / np.pi)
                mask['diameter'] = diameter  # Store diameter in mask for later use
                diameters.append(diameter)
            return diameters
        
        # Group 1: Smaller cluster (smaller average area)
        group1_diameters = calculate_diameters_from_masks(smaller_cluster)
        
        # Group 2: Larger cluster (larger average area) - Apply additional 20% filter
        larger_cluster_original = larger_cluster.copy()
        group2_diameters_original = calculate_diameters_from_masks(larger_cluster_original)
        
        # Calculate average diameter of the larger group before filtering
        if group2_diameters_original:
            avg_diameter_larger_group = np.mean(group2_diameters_original)
            diameter_threshold = avg_diameter_larger_group * 1.2  # 20% larger than average
            
            print(f"🔍 Applying 20% diameter filter to larger group:")
            print(f"   📊 Original larger group count: {len(larger_cluster_original)}")
            print(f"   📏 Average diameter: {avg_diameter_larger_group:.1f}px")
            print(f"   🚫 Diameter threshold (120%): {diameter_threshold:.1f}px")
            
            # Filter out masks that are 20% larger than average diameter
            filtered_larger_cluster = []
            removed_oversized = []
            
            for mask in larger_cluster_original:
                diameter = 2 * np.sqrt(mask['area'] / np.pi)
                if diameter <= diameter_threshold:
                    filtered_larger_cluster.append(mask)
                else:
                    removed_oversized.append(mask)
            
            larger_cluster = filtered_larger_cluster
            print(f"   ✅ Filtered larger group count: {len(larger_cluster)}")
            print(f"   🗑️  Removed oversized masks: {len(removed_oversized)}")
            
            # Recalculate diameters for the filtered larger cluster
            group2_diameters = calculate_diameters_from_masks(larger_cluster)
        else:
            group2_diameters = []
        
        total_count = len(group1_diameters) + len(group2_diameters)
        
        if group1_diameters:
            group1_stats = {
                'count': len(group1_diameters),
                'percentage': (len(group1_diameters) / total_count * 100) if total_count > 0 else 0,
                'avg_diameter': np.mean(group1_diameters),
                'min_diameter': np.min(group1_diameters),
                'max_diameter': np.max(group1_diameters)
            }
        else:
            group1_diameters = []
            group1_stats = {'count': 0, 'percentage': 0, 'avg_diameter': 0, 'min_diameter': 0, 'max_diameter': 0}
        
        if group2_diameters:
            group2_stats = {
                'count': len(group2_diameters),
                'percentage': (len(group2_diameters) / total_count * 100) if total_count > 0 else 0,
                'avg_diameter': np.mean(group2_diameters),
                'min_diameter': np.min(group2_diameters),
                'max_diameter': np.max(group2_diameters)
            }
        else:
            group2_diameters = []
            group2_stats = {'count': 0, 'percentage': 0, 'avg_diameter': 0, 'min_diameter': 0, 'max_diameter': 0}
        
        print(f"K-means based grouping:")
        print(f"  Group 1 (Smaller): {len(group1_diameters)} masks, avg diameter: {group1_stats['avg_diameter']:.1f}px")
        print(f"  Group 2 (Larger): {len(group2_diameters)} masks, avg diameter: {group2_stats['avg_diameter']:.1f}px")
        
        # Report filtering results if applied
        if 'group2_diameters_original' in locals() and group2_diameters_original:
            removed_count = len(group2_diameters_original) - len(group2_diameters)
            if removed_count > 0:
                print(f"  🔧 Filtering summary: Removed {removed_count} oversized masks from larger group")
        
        # Debug: Print sample bounding boxes being sent to frontend
        print(f"\nDEBUG: Sample bounding boxes for frontend:")
        if smaller_cluster:
            sample_small = smaller_cluster[0]
            print(f"  Group 1 sample bbox: {sample_small['bbox']} (area: {sample_small['area']})")
        if larger_cluster:
            sample_large = larger_cluster[0]
            print(f"  Group 2 sample bbox: {sample_large['bbox']} (area: {sample_large['area']})")
        print(f"  Image shape: {image_array.shape}")
        print(f"  Total masks being sent: Group1={len(smaller_cluster)}, Group2={len(larger_cluster)}")
        
        response_data = {
            'success': True,
            'mode': 'satellite_droplet',
            'processing_time': float(processing_time),
            'configuration': {
                'min_circularity': min_circularity,
                'max_blob_distance': max_blob_distance,
                'model_type': 'vit_h',
                'crop_n_layers': 3,
                'points_per_side': 32,
                'clustering_method': 'K-means',
                'n_clusters': 2,
                'clustering_features': ['area', 'width', 'height', 'aspect_ratio', 'stability_score', 'circularity']
            },
            'results': {
                'total_masks_generated': int(result['total_masks']),
                'masks_after_filtering': int(result['filtered_masks']),
                'final_droplet_count': int(droplet_stats['mask_count']),
                'average_diameter_pixels': float(droplet_stats['average_diameter']),
                'satellite_droplet_analysis_image': f"data:image/png;base64,{viz_base64}",
                'original_image_for_hover': f"data:image/png;base64,{original_base64}"
            },
            'droplet_statistics': convert_numpy_types({
                'mask_count': droplet_stats['mask_count'],
                'average_diameter': droplet_stats['average_diameter'],
                'diameter_std': droplet_stats['diameter_std'],
                'min_diameter': droplet_stats['min_diameter'],
                'max_diameter': droplet_stats['max_diameter'],
                'total_area': droplet_stats['total_area'],
                'average_area': droplet_stats['average_area'],
                'diameter_distribution': droplet_stats['diameter_distribution']
            }),
            'group_statistics': convert_numpy_types({
                'group1': group1_stats,
                'group2': group2_stats
            }),
            'group1_diameters': [float(d) for d in group1_diameters],
            'group2_diameters': [float(d) for d in group2_diameters],
            'mask_data_for_frontend': {
                'group1_masks': [convert_numpy_types({
                    'bbox': mask['bbox'],
                    'area': mask['area'],
                    'diameter': float(np.sqrt(mask['area'] * 4 / np.pi)),
                    'group': 1
                }) for mask in smaller_cluster],
                'group2_masks': [convert_numpy_types({
                    'bbox': mask['bbox'],
                    'area': mask['area'],
                    'diameter': float(np.sqrt(mask['area'] * 4 / np.pi)),
                    'group': 2
                }) for mask in larger_cluster]
            },
            'clustering_info': {
                'method': 'K-means',
                'n_clusters': 2,
                'cluster_sizes': [len(smaller_cluster), len(larger_cluster)],
                'features_used': ['area', 'width', 'height', 'aspect_ratio', 'stability_score', 'circularity'],
                'preprocessing': ['log_transform_area', 'standard_scaler']
            },
            'processing_details': convert_numpy_types(result['processing_details'])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze_satellite_droplets: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/batch_process_files', methods=['POST'])
def batch_process_files():
    """Process multiple uploaded image files in batch mode."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        # Get mode and configuration
        mode = request.form.get('mode', 'multiple')
        overlap_threshold = float(request.form.get('overlap_threshold', 80.0))
        min_circularity = float(request.form.get('min_circularity', 0.53))
        max_blob_distance = float(request.form.get('max_blob_distance', 50))
        fluorescent_mode = request.form.get('fluorescent_mode', 'false').lower() == 'true'
        min_brightness_threshold = float(request.form.get('min_brightness_threshold', 200))
        
        print(f"Batch processing {len(files)} files in {mode} mode")
        print(f"Configuration: overlap_threshold={overlap_threshold}%, min_circularity={min_circularity}")
        
        # Process each file
        batch_results = []
        total_start_time = time.time()
        
        for file_idx, file in enumerate(files):
            if file.filename == '':
                continue
                
            try:
                file_start_time = time.time()
                
                # Read and process image
                image_data = file.read()
                image_array = process_image_from_bytes(image_data)
                print(f"Processing file {file_idx + 1}/{len(files)}: {file.filename} ({image_array.shape})")
                
                # Configure processors based on mode
                if mode == 'droplet':
                    processor = droplet_processor
                    processor.min_circularity = min_circularity
                    processor.max_blob_distance = max_blob_distance
                elif mode == 'huge_droplet':
                    processor = huge_droplet_processor
                    processor.min_circularity = min_circularity
                    processor.max_blob_distance = max_blob_distance
                elif mode == 'satellite_droplet':
                    processor = huge_droplet_processor
                    processor.min_circularity = min_circularity
                    processor.max_blob_distance = max_blob_distance
                else:  # multiple mode
                    processor = mask_processor
                    processor.overlap_threshold = overlap_threshold / 100.0
                    processor.min_circularity = min_circularity
                    processor.max_blob_distance = max_blob_distance
                    processor.fluorescent_mode = fluorescent_mode
                    processor.min_brightness_threshold = min_brightness_threshold
                
                # Process the image
                result = processor.process_image_array(image_array)
                file_processing_time = time.time() - file_start_time
                
                # Generate preview image with bounding boxes
                preview_image_data = generate_preview_image_with_masks(
                    image_array, result['final_masks'], mode
                )
                
                # Create file-specific result
                file_result = {
                    'filename': file.filename,
                    'file_index': file_idx,
                    'processing_time': float(file_processing_time),
                    'success': True,
                    'mode': mode,
                    'statistics': convert_numpy_types(result['statistics']),
                    'total_masks': int(result['total_masks']),
                    'filtered_masks': int(result['filtered_masks']),
                    'image_shape': list(image_array.shape),
                    'preview_image': preview_image_data
                }
                
                # Add mode-specific data
                if mode in ['droplet', 'huge_droplet']:
                    file_result.update({
                        'mask_count': result['statistics']['mask_count'],
                        'average_diameter': float(result['statistics']['average_diameter']),
                        'diameter_std': float(result['statistics']['diameter_std']),
                        'min_diameter': float(result['statistics']['min_diameter']),
                        'max_diameter': float(result['statistics']['max_diameter'])
                    })
                elif mode == 'satellite_droplet':
                    # Add K-means clustering for satellite mode
                    from mask_grouping.core import MaskGroupingProcessor
                    temp_processor = MaskGroupingProcessor(processor.mask_generator)
                    clustered_masks, _ = temp_processor._cluster_masks_kmeans(result['final_masks'], n_clusters=2)
                    
                    # Determine larger and smaller clusters
                    if clustered_masks[0] and clustered_masks[1]:
                        avg_area_0 = np.mean([mask['area'] for mask in clustered_masks[0]])
                        avg_area_1 = np.mean([mask['area'] for mask in clustered_masks[1]])
                        if avg_area_0 > avg_area_1:
                            larger_cluster, smaller_cluster = clustered_masks[0], clustered_masks[1]
                        else:
                            larger_cluster, smaller_cluster = clustered_masks[1], clustered_masks[0]
                    else:
                        larger_cluster = clustered_masks[0] if clustered_masks[0] else clustered_masks[1]
                        smaller_cluster = clustered_masks[1] if clustered_masks[0] else clustered_masks[0]
                    
                    # Calculate group statistics with 20% filter for larger group
                    group1_diameters = [2 * np.sqrt(mask['area'] / np.pi) for mask in smaller_cluster]
                    
                    # Apply 20% diameter filter to larger cluster
                    larger_cluster_original = larger_cluster.copy()
                    group2_diameters_original = [2 * np.sqrt(mask['area'] / np.pi) for mask in larger_cluster_original]
                    
                    if group2_diameters_original:
                        avg_diameter_larger_group = np.mean(group2_diameters_original)
                        diameter_threshold = avg_diameter_larger_group * 1.2  # 20% larger than average
                        
                        # Filter out oversized masks from larger cluster
                        filtered_larger_cluster = []
                        for mask in larger_cluster_original:
                            diameter = 2 * np.sqrt(mask['area'] / np.pi)
                            if diameter <= diameter_threshold:
                                filtered_larger_cluster.append(mask)
                        
                        larger_cluster = filtered_larger_cluster
                        group2_diameters = [2 * np.sqrt(mask['area'] / np.pi) for mask in larger_cluster]
                    else:
                        group2_diameters = []
                    
                    total_count = len(group1_diameters) + len(group2_diameters)
                    group1_percentage = (len(group1_diameters) / total_count * 100) if total_count > 0 else 0
                    group2_percentage = (len(group2_diameters) / total_count * 100) if total_count > 0 else 0
                    
                    file_result.update({
                        'mask_count': result['statistics']['mask_count'],
                        'average_diameter': float(result['statistics']['average_diameter']),
                        'group1_count': len(group1_diameters),
                        'group2_count': len(group2_diameters),
                        'group1_percentage': float(group1_percentage),
                        'group2_percentage': float(group2_percentage),
                        'group1_avg_diameter': float(np.mean(group1_diameters)) if group1_diameters else 0,
                        'group2_avg_diameter': float(np.mean(group2_diameters)) if group2_diameters else 0,
                        'group1_diameters': [float(d) for d in group1_diameters],
                        'group2_diameters': [float(d) for d in group2_diameters]
                    })
                else:  # multiple mode
                    overlap_stats = result['statistics']
                    final_masks = result['final_masks']
                    total_overlaps = sum([mask.get('overlap_count', 0) for mask in final_masks])
                    
                    file_result.update({
                        'total_large_masks': len(final_masks),
                        'total_overlaps_found': int(total_overlaps),
                        'overall_overlap_percentage': float(overlap_stats.get('overlap_percentage', 0.0)),
                        'aggregate_masks_count': len(result.get('aggregate_masks', [])),
                        'simple_masks_count': len(result.get('simple_masks', []))
                    })
                
                batch_results.append(file_result)
                print(f"✅ Completed {file.filename} in {file_processing_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error processing {file.filename}: {e}")
                batch_results.append({
                    'filename': file.filename,
                    'file_index': file_idx,
                    'success': False,
                    'error': str(e),
                    'mode': mode
                })
        
        total_processing_time = time.time() - total_start_time
        
        # Calculate batch summary statistics
        successful_results = [r for r in batch_results if r.get('success', False)]
        failed_results = [r for r in batch_results if not r.get('success', True)]
        
        batch_summary = {
            'total_files': len(files),
            'successful_files': len(successful_results),
            'failed_files': len(failed_results),
            'total_processing_time': float(total_processing_time),
            'average_processing_time': float(np.mean([r['processing_time'] for r in successful_results])) if successful_results else 0,
            'mode': mode,
            'configuration': {
                'overlap_threshold': overlap_threshold,
                'min_circularity': min_circularity,
                'max_blob_distance': max_blob_distance,
                'fluorescent_mode': fluorescent_mode,
                'min_brightness_threshold': min_brightness_threshold
            }
        }
        
        # Add mode-specific summary statistics
        if mode in ['droplet', 'huge_droplet'] and successful_results:
            batch_summary.update({
                'total_masks_found': sum([r['mask_count'] for r in successful_results]),
                'average_masks_per_image': float(np.mean([r['mask_count'] for r in successful_results])),
                'average_diameter_across_batch': float(np.mean([r['average_diameter'] for r in successful_results]))
            })
        elif mode == 'satellite_droplet' and successful_results:
            all_group1_diameters = []
            all_group2_diameters = []
            for r in successful_results:
                all_group1_diameters.extend(r.get('group1_diameters', []))
                all_group2_diameters.extend(r.get('group2_diameters', []))
            
            batch_summary.update({
                'total_masks_found': sum([r['mask_count'] for r in successful_results]),
                'total_group1_masks': len(all_group1_diameters),
                'total_group2_masks': len(all_group2_diameters),
                'batch_group1_avg_diameter': float(np.mean(all_group1_diameters)) if all_group1_diameters else 0,
                'batch_group2_avg_diameter': float(np.mean(all_group2_diameters)) if all_group2_diameters else 0,
                'all_group1_diameters': all_group1_diameters,
                'all_group2_diameters': all_group2_diameters
            })
        elif mode == 'multiple' and successful_results:
            batch_summary.update({
                'total_large_masks': sum([r.get('total_large_masks', 0) for r in successful_results]),
                'total_overlaps': sum([r.get('total_overlaps_found', 0) for r in successful_results]),
                'average_overlap_percentage': float(np.mean([r.get('overall_overlap_percentage', 0) for r in successful_results]))
            })
        
        print(f"🎉 Batch processing completed: {len(successful_results)}/{len(files)} files successful")
        
        return jsonify({
            'success': True,
            'batch_summary': batch_summary,
            'individual_results': batch_results,
            'failed_files': [r['filename'] for r in failed_results] if failed_results else []
        })
        
    except Exception as e:
        print(f"Error in batch_process_files: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/batch_process_base64', methods=['POST'])
def batch_process_base64():
    """Process multiple base64 encoded images in batch mode."""
    try:
        # Initialize SAM if not already done
        if sam_model is None:
            initialize_sam()
        
        # Get image data and configuration
        data = request.get_json()
        if not data or 'images' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        images = data['images']  # Expected to be list of {name: str, image: base64_string}
        if not images or len(images) == 0:
            return jsonify({'error': 'No images provided'}), 400
        
        # Get mode and configuration
        mode = data.get('mode', 'multiple')
        overlap_threshold = data.get('overlap_threshold', 80.0)
        min_circularity = data.get('min_circularity', 0.53)
        max_blob_distance = data.get('max_blob_distance', 50)
        fluorescent_mode = data.get('fluorescent_mode', False)
        min_brightness_threshold = data.get('min_brightness_threshold', 200)
        
        print(f"Batch processing {len(images)} base64 images in {mode} mode")
        print(f"Configuration: overlap_threshold={overlap_threshold}%, min_circularity={min_circularity}")
        
        # Process each image
        batch_results = []
        total_start_time = time.time()
        
        for img_idx, img_data in enumerate(images):
            try:
                img_start_time = time.time()
                
                # Get image name and data
                img_name = img_data.get('name', f'image_{img_idx + 1}')
                img_base64 = img_data.get('image', '')
                
                if not img_base64:
                    batch_results.append({
                        'image_name': img_name,
                        'image_index': img_idx,
                        'success': False,
                        'error': 'No image data provided',
                        'mode': mode
                    })
                    continue
                
                # Process image from base64
                image_array = process_image_from_base64(img_base64)
                print(f"Processing image {img_idx + 1}/{len(images)}: {img_name} ({image_array.shape})")
                
                # Configure processors based on mode
                if mode == 'droplet':
                    processor = droplet_processor
                    processor.min_circularity = min_circularity
                    processor.max_blob_distance = max_blob_distance
                elif mode == 'huge_droplet':
                    processor = huge_droplet_processor
                    processor.min_circularity = min_circularity
                    processor.max_blob_distance = max_blob_distance
                elif mode == 'satellite_droplet':
                    processor = huge_droplet_processor
                    processor.min_circularity = min_circularity
                    processor.max_blob_distance = max_blob_distance
                else:  # multiple mode
                    processor = mask_processor
                    processor.overlap_threshold = overlap_threshold / 100.0
                    processor.min_circularity = min_circularity
                    processor.max_blob_distance = max_blob_distance
                    processor.fluorescent_mode = fluorescent_mode
                    processor.min_brightness_threshold = min_brightness_threshold
                
                # Process the image
                result = processor.process_image_array(image_array)
                img_processing_time = time.time() - img_start_time
                
                # Create image-specific result
                img_result = {
                    'image_name': img_name,
                    'image_index': img_idx,
                    'processing_time': float(img_processing_time),
                    'success': True,
                    'mode': mode,
                    'statistics': convert_numpy_types(result['statistics']),
                    'total_masks': int(result['total_masks']),
                    'filtered_masks': int(result['filtered_masks']),
                    'image_shape': list(image_array.shape)
                }
                
                # Add mode-specific data (same logic as file processing)
                if mode in ['droplet', 'huge_droplet']:
                    img_result.update({
                        'mask_count': result['statistics']['mask_count'],
                        'average_diameter': float(result['statistics']['average_diameter']),
                        'diameter_std': float(result['statistics']['diameter_std']),
                        'min_diameter': float(result['statistics']['min_diameter']),
                        'max_diameter': float(result['statistics']['max_diameter'])
                    })
                elif mode == 'satellite_droplet':
                    # Add K-means clustering for satellite mode
                    from mask_grouping.core import MaskGroupingProcessor
                    temp_processor = MaskGroupingProcessor(processor.mask_generator)
                    clustered_masks, _ = temp_processor._cluster_masks_kmeans(result['final_masks'], n_clusters=2)
                    
                    # Determine larger and smaller clusters
                    if clustered_masks[0] and clustered_masks[1]:
                        avg_area_0 = np.mean([mask['area'] for mask in clustered_masks[0]])
                        avg_area_1 = np.mean([mask['area'] for mask in clustered_masks[1]])
                        if avg_area_0 > avg_area_1:
                            larger_cluster, smaller_cluster = clustered_masks[0], clustered_masks[1]
                        else:
                            larger_cluster, smaller_cluster = clustered_masks[1], clustered_masks[0]
                    else:
                        larger_cluster = clustered_masks[0] if clustered_masks[0] else clustered_masks[1]
                        smaller_cluster = clustered_masks[1] if clustered_masks[0] else clustered_masks[0]
                    
                    # Calculate group statistics with 20% filter for larger group
                    group1_diameters = [2 * np.sqrt(mask['area'] / np.pi) for mask in smaller_cluster]
                    
                    # Apply 20% diameter filter to larger cluster
                    larger_cluster_original = larger_cluster.copy()
                    group2_diameters_original = [2 * np.sqrt(mask['area'] / np.pi) for mask in larger_cluster_original]
                    
                    if group2_diameters_original:
                        avg_diameter_larger_group = np.mean(group2_diameters_original)
                        diameter_threshold = avg_diameter_larger_group * 1.2  # 20% larger than average
                        
                        # Filter out oversized masks from larger cluster
                        filtered_larger_cluster = []
                        for mask in larger_cluster_original:
                            diameter = 2 * np.sqrt(mask['area'] / np.pi)
                            if diameter <= diameter_threshold:
                                filtered_larger_cluster.append(mask)
                        
                        larger_cluster = filtered_larger_cluster
                        group2_diameters = [2 * np.sqrt(mask['area'] / np.pi) for mask in larger_cluster]
                    else:
                        group2_diameters = []
                    
                    img_result.update({
                        'mask_count': result['statistics']['mask_count'],
                        'average_diameter': float(result['statistics']['average_diameter']),
                        'group1_count': len(group1_diameters),
                        'group2_count': len(group2_diameters),
                        'group1_avg_diameter': float(np.mean(group1_diameters)) if group1_diameters else 0,
                        'group2_avg_diameter': float(np.mean(group2_diameters)) if group2_diameters else 0,
                        'group1_diameters': [float(d) for d in group1_diameters],
                        'group2_diameters': [float(d) for d in group2_diameters]
                    })
                else:  # multiple mode
                    overlap_stats = result['statistics']
                    final_masks = result['final_masks']
                    total_overlaps = sum([mask.get('overlap_count', 0) for mask in final_masks])
                    
                    img_result.update({
                        'total_large_masks': len(final_masks),
                        'total_overlaps_found': int(total_overlaps),
                        'overall_overlap_percentage': float(overlap_stats.get('overlap_percentage', 0.0)),
                        'aggregate_masks_count': len(result.get('aggregate_masks', [])),
                        'simple_masks_count': len(result.get('simple_masks', []))
                    })
                
                batch_results.append(img_result)
                print(f"✅ Completed {img_name} in {img_processing_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")
                batch_results.append({
                    'image_name': img_name,
                    'image_index': img_idx,
                    'success': False,
                    'error': str(e),
                    'mode': mode
                })
        
        total_processing_time = time.time() - total_start_time
        
        # Calculate batch summary statistics (same logic as file processing)
        successful_results = [r for r in batch_results if r.get('success', False)]
        failed_results = [r for r in batch_results if not r.get('success', True)]
        
        batch_summary = {
            'total_images': len(images),
            'successful_images': len(successful_results),
            'failed_images': len(failed_results),
            'total_processing_time': float(total_processing_time),
            'average_processing_time': float(np.mean([r['processing_time'] for r in successful_results])) if successful_results else 0,
            'mode': mode,
            'configuration': {
                'overlap_threshold': overlap_threshold,
                'min_circularity': min_circularity,
                'max_blob_distance': max_blob_distance,
                'fluorescent_mode': fluorescent_mode,
                'min_brightness_threshold': min_brightness_threshold
            }
        }
        
        # Add mode-specific summary statistics
        if mode in ['droplet', 'huge_droplet'] and successful_results:
            batch_summary.update({
                'total_masks_found': sum([r['mask_count'] for r in successful_results]),
                'average_masks_per_image': float(np.mean([r['mask_count'] for r in successful_results])),
                'average_diameter_across_batch': float(np.mean([r['average_diameter'] for r in successful_results]))
            })
        elif mode == 'satellite_droplet' and successful_results:
            all_group1_diameters = []
            all_group2_diameters = []
            for r in successful_results:
                all_group1_diameters.extend(r.get('group1_diameters', []))
                all_group2_diameters.extend(r.get('group2_diameters', []))
            
            batch_summary.update({
                'total_masks_found': sum([r['mask_count'] for r in successful_results]),
                'total_group1_masks': len(all_group1_diameters),
                'total_group2_masks': len(all_group2_diameters),
                'batch_group1_avg_diameter': float(np.mean(all_group1_diameters)) if all_group1_diameters else 0,
                'batch_group2_avg_diameter': float(np.mean(all_group2_diameters)) if all_group2_diameters else 0,
                'all_group1_diameters': all_group1_diameters,
                'all_group2_diameters': all_group2_diameters
            })
        elif mode == 'multiple' and successful_results:
            batch_summary.update({
                'total_large_masks': sum([r.get('total_large_masks', 0) for r in successful_results]),
                'total_overlaps': sum([r.get('total_overlaps_found', 0) for r in successful_results]),
                'average_overlap_percentage': float(np.mean([r.get('overall_overlap_percentage', 0) for r in successful_results]))
            })
        
        print(f"🎉 Batch processing completed: {len(successful_results)}/{len(images)} images successful")
        
        return jsonify({
            'success': True,
            'batch_summary': batch_summary,
            'individual_results': batch_results,
            'failed_images': [r['image_name'] for r in failed_results] if failed_results else []
        })
        
    except Exception as e:
        print(f"Error in batch_process_base64: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download_batch_excel', methods=['POST'])
def download_batch_excel():
    """Generate and download Excel file with batch processing results."""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        batch_results = data.get('batch_results', [])
        mode = data.get('mode', 'multiple')
        filename = data.get('filename', f'batch_{mode}_results.xlsx')
        
        if not batch_results:
            return jsonify({'error': 'No batch results provided'}), 400
        
        # Create Excel file in memory
        try:
            import pandas as pd
            from io import BytesIO
            
            excel_buffer = BytesIO()
            
            # Create different sheets based on mode
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                
                if mode in ['droplet', 'huge_droplet']:
                    # Create summary sheet
                    summary_data = []
                    for result in batch_results:
                        if result.get('success', False):
                            summary_data.append({
                                'Filename': result.get('image_name', result.get('filename', 'Unknown')),
                                'Mask Count': result.get('mask_count', 0),
                                'Average Diameter (px)': result.get('average_diameter', 0),
                                'Min Diameter (px)': result.get('min_diameter', 0),
                                'Max Diameter (px)': result.get('max_diameter', 0),
                                'Diameter Std Dev': result.get('diameter_std', 0),
                                'Processing Time (s)': result.get('processing_time', 0)
                            })
                    
                    if summary_data:
                        df_summary = pd.DataFrame(summary_data)
                        df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                elif mode == 'satellite_droplet':
                    # Create summary sheet
                    summary_data = []
                    all_group1_data = []
                    all_group2_data = []
                    
                    for result in batch_results:
                        if result.get('success', False):
                            filename = result.get('image_name', result.get('filename', 'Unknown'))
                            summary_data.append({
                                'Filename': filename,
                                'Total Masks': result.get('mask_count', 0),
                                'Group 1 Count': result.get('group1_count', 0),
                                'Group 2 Count': result.get('group2_count', 0),
                                'Group 1 Avg Diameter (px)': result.get('group1_avg_diameter', 0),
                                'Group 2 Avg Diameter (px)': result.get('group2_avg_diameter', 0),
                                'Processing Time (s)': result.get('processing_time', 0)
                            })
                            
                            # Collect diameter data
                            for diameter in result.get('group1_diameters', []):
                                all_group1_data.append({'Filename': filename, 'Diameter (px)': diameter})
                            for diameter in result.get('group2_diameters', []):
                                all_group2_data.append({'Filename': filename, 'Diameter (px)': diameter})
                    
                    if summary_data:
                        df_summary = pd.DataFrame(summary_data)
                        df_summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    if all_group1_data:
                        df_group1 = pd.DataFrame(all_group1_data)
                        df_group1.to_excel(writer, sheet_name='Group 1 Diameters', index=False)
                    
                    if all_group2_data:
                        df_group2 = pd.DataFrame(all_group2_data)
                        df_group2.to_excel(writer, sheet_name='Group 2 Diameters', index=False)
                
                elif mode == 'multiple':
                    # Create summary sheet
                    summary_data = []
                    for result in batch_results:
                        if result.get('success', False):
                            summary_data.append({
                                'Filename': result.get('image_name', result.get('filename', 'Unknown')),
                                'Total Large Masks': result.get('total_large_masks', 0),
                                'Total Overlaps Found': result.get('total_overlaps_found', 0),
                                'Overlap Percentage': result.get('overall_overlap_percentage', 0),
                                'Aggregate Masks': result.get('aggregate_masks_count', 0),
                                'Simple Masks': result.get('simple_masks_count', 0),
                                'Processing Time (s)': result.get('processing_time', 0)
                            })
                    
                    if summary_data:
                        df_summary = pd.DataFrame(summary_data)
                        df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add configuration sheet
                config_data = [
                    ['Mode', mode],
                    ['Total Files Processed', len(batch_results)],
                    ['Successful Files', len([r for r in batch_results if r.get('success', False)])],
                    ['Failed Files', len([r for r in batch_results if not r.get('success', True)])],
                ]
                
                # Add configuration parameters from first successful result
                successful_result = next((r for r in batch_results if r.get('success', False)), None)
                if successful_result and 'configuration' in data:
                    config = data['configuration']
                    config_data.extend([
                        ['Overlap Threshold (%)', config.get('overlap_threshold', 'N/A')],
                        ['Min Circularity', config.get('min_circularity', 'N/A')],
                        ['Max Blob Distance', config.get('max_blob_distance', 'N/A')],
                        ['Fluorescent Mode', config.get('fluorescent_mode', 'N/A')],
                        ['Min Brightness Threshold', config.get('min_brightness_threshold', 'N/A')]
                    ])
                
                df_config = pd.DataFrame(config_data, columns=['Parameter', 'Value'])
                df_config.to_excel(writer, sheet_name='Configuration', index=False)
                
                # Format the Excel file
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            excel_buffer.seek(0)
            
            return send_file(
                excel_buffer,
                download_name=filename,
                as_attachment=True,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
        except ImportError:
            # Fallback to CSV if pandas/openpyxl not available
            import csv
            from io import StringIO
            
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            
            # Write summary based on mode
            if mode in ['droplet', 'huge_droplet']:
                writer.writerow(['Filename', 'Mask Count', 'Average Diameter (px)', 'Min Diameter (px)', 'Max Diameter (px)', 'Processing Time (s)'])
                for result in batch_results:
                    if result.get('success', False):
                        writer.writerow([
                            result.get('image_name', result.get('filename', 'Unknown')),
                            result.get('mask_count', 0),
                            result.get('average_diameter', 0),
                            result.get('min_diameter', 0),
                            result.get('max_diameter', 0),
                            result.get('processing_time', 0)
                        ])
            elif mode == 'satellite_droplet':
                writer.writerow(['Filename', 'Total Masks', 'Group 1 Count', 'Group 2 Count', 'Group 1 Avg Diameter', 'Group 2 Avg Diameter', 'Processing Time (s)'])
                for result in batch_results:
                    if result.get('success', False):
                        writer.writerow([
                            result.get('image_name', result.get('filename', 'Unknown')),
                            result.get('mask_count', 0),
                            result.get('group1_count', 0),
                            result.get('group2_count', 0),
                            result.get('group1_avg_diameter', 0),
                            result.get('group2_avg_diameter', 0),
                            result.get('processing_time', 0)
                        ])
            elif mode == 'multiple':
                writer.writerow(['Filename', 'Total Large Masks', 'Total Overlaps', 'Overlap Percentage', 'Aggregate Masks', 'Simple Masks', 'Processing Time (s)'])
                for result in batch_results:
                    if result.get('success', False):
                        writer.writerow([
                            result.get('image_name', result.get('filename', 'Unknown')),
                            result.get('total_large_masks', 0),
                            result.get('total_overlaps_found', 0),
                            result.get('overall_overlap_percentage', 0),
                            result.get('aggregate_masks_count', 0),
                            result.get('simple_masks_count', 0),
                            result.get('processing_time', 0)
                        ])
            
            csv_content = csv_buffer.getvalue().encode('utf-8')
            csv_bytes = BytesIO(csv_content)
            csv_filename = filename.replace('.xlsx', '.csv')
            
            return send_file(
                csv_bytes,
                download_name=csv_filename,
                as_attachment=True,
                mimetype='text/csv'
            )
            
    except Exception as e:
        print(f"Error in download_batch_excel: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download_excel', methods=['POST'])
def download_excel():
    """Generate and download Excel file with droplet diameter data for two groups."""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        group1_diameters = data.get('group1_diameters', [])
        group2_diameters = data.get('group2_diameters', [])
        filename = data.get('filename', 'satellite_droplet_data.xlsx')
        
        # Create Excel file in memory
        try:
            import pandas as pd
            from io import BytesIO
            
            # Create a BytesIO object to store the Excel file
            excel_buffer = BytesIO()
            
            # Determine the maximum length to pad shorter lists
            max_length = max(len(group1_diameters), len(group2_diameters))
            
            # Pad shorter lists with empty strings
            group1_padded = group1_diameters + [''] * (max_length - len(group1_diameters))
            group2_padded = group2_diameters + [''] * (max_length - len(group2_diameters))
            
            # Create DataFrame
            df = pd.DataFrame({
                'Group 1 Diameters (px)': group1_padded,
                'Group 2 Diameters (px)': group2_padded
            })
            
            # Add summary statistics at the top
            summary_df = pd.DataFrame({
                'Group 1 Diameters (px)': [
                    'Summary Statistics:',
                    f'Count: {len(group1_diameters)}',
                    f'Average: {np.mean(group1_diameters):.2f}' if group1_diameters else 'Average: 0',
                    f'Min: {np.min(group1_diameters):.2f}' if group1_diameters else 'Min: 0',
                    f'Max: {np.max(group1_diameters):.2f}' if group1_diameters else 'Max: 0',
                    '',
                    'Raw Data:'
                ],
                'Group 2 Diameters (px)': [
                    'Summary Statistics:',
                    f'Count: {len(group2_diameters)}',
                    f'Average: {np.mean(group2_diameters):.2f}' if group2_diameters else 'Average: 0',
                    f'Min: {np.min(group2_diameters):.2f}' if group2_diameters else 'Min: 0',
                    f'Max: {np.max(group2_diameters):.2f}' if group2_diameters else 'Max: 0',
                    '',
                    'Raw Data:'
                ]
            })
            
            # Combine summary and data
            final_df = pd.concat([summary_df, df], ignore_index=True)
            
            # Write to Excel buffer
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                final_df.to_excel(writer, sheet_name='Satellite Droplet Data', index=False)
                
                # Get the workbook and worksheet to format
                workbook = writer.book
                worksheet = writer.sheets['Satellite Droplet Data']
                
                # Format headers
                from openpyxl.styles import Font, PatternFill, Alignment
                header_font = Font(bold=True)
                header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
                
                # Apply header formatting
                for cell in worksheet[1]:
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = Alignment(horizontal='center')
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Reset buffer position
            excel_buffer.seek(0)
            
            return send_file(
                excel_buffer,
                download_name=filename,
                as_attachment=True,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
        except ImportError:
            # Fallback: pandas/openpyxl not available, create simple CSV
            import csv
            from io import StringIO
            
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            
            # Write headers
            writer.writerow(['Group 1 Diameters (px)', 'Group 2 Diameters (px)'])
            
            # Write summary
            writer.writerow(['Summary Statistics:', 'Summary Statistics:'])
            writer.writerow([f'Count: {len(group1_diameters)}', f'Count: {len(group2_diameters)}'])
            writer.writerow([
                f'Average: {np.mean(group1_diameters):.2f}' if group1_diameters else 'Average: 0',
                f'Average: {np.mean(group2_diameters):.2f}' if group2_diameters else 'Average: 0'
            ])
            writer.writerow(['', ''])
            writer.writerow(['Raw Data:', 'Raw Data:'])
            
            # Write data
            max_length = max(len(group1_diameters), len(group2_diameters))
            for i in range(max_length):
                group1_val = group1_diameters[i] if i < len(group1_diameters) else ''
                group2_val = group2_diameters[i] if i < len(group2_diameters) else ''
                writer.writerow([group1_val, group2_val])
            
            # Convert to bytes
            csv_content = csv_buffer.getvalue().encode('utf-8')
            csv_bytes = BytesIO(csv_content)
            
            # Change filename to CSV
            csv_filename = filename.replace('.xlsx', '.csv')
            
            return send_file(
                csv_bytes,
                download_name=csv_filename,
                as_attachment=True,
                mimetype='text/csv'
            )
            
    except Exception as e:
        print(f"Error in download_excel: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_custom_result', methods=['POST'])
def generate_custom_result():
    """Generate a custom result image based on user's interactive mask selections."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required parameters
        original_image_b64 = data.get('original_image')
        visible_masks = data.get('visible_masks', [])
        mode = data.get('mode', 'multiple')
        
        if not original_image_b64:
            return jsonify({'error': 'Original image data required'}), 400
        
        # Process the original image
        image_array = process_image_from_base64(original_image_b64)
        
        # Generate custom result image with only visible masks
        custom_result_image = generate_custom_preview_image(image_array, visible_masks, mode)
        
        # Save the custom result to a temporary file for download
        timestamp = int(time.time() * 1000)
        filename = f"custom_{mode}_result_{timestamp}.png"
        
        # Ensure results folder exists
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        print(f"Generating custom result for {mode} mode with {len(visible_masks)} visible masks")
        
        # Convert base64 back to image and save
        if custom_result_image.startswith('data:image/jpeg;base64,'):
            img_data = custom_result_image.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            
            # Convert JPEG to PNG for consistency
            from PIL import Image
            import io
            jpg_image = Image.open(io.BytesIO(img_bytes))
            jpg_image.save(filepath, 'PNG')
        else:
            return jsonify({'error': 'Invalid image format received'}), 500
        
        return jsonify({
            'success': True,
            'download_url': f'/download_result/{filename}',
            'filename': filename,
            'visible_mask_count': len(visible_masks)
        })
        
    except Exception as e:
        print(f"Error generating custom result: {str(e)}")
        return jsonify({'error': f'Failed to generate custom result: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Mask Grouping Server...")
    print("Supporting droplet, huge droplet, and multiple analysis modes")
    print("=" * 50)
    
    # Initialize SAM model
    initialize_sam()
    
    print("=" * 50)
    print("Server ready! Starting Flask application...")
    print("Access the application at: http://localhost:5003")
    print("API endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /backend_info - Get backend information")
    print("  POST /segment - Segment image from base64 (supports mode parameter)")
    print("  POST /segment_file - Segment uploaded file (supports mode parameter)")
    print("  POST /analyze_overlap - Multiple mode overlap analysis endpoint")
    print("  POST /analyze_droplets - Droplet mode analysis endpoint")
    print("  POST /analyze_huge_droplets - Huge droplet mode analysis endpoint")
    print("  POST /analyze_satellite_droplets - Satellite droplet mode analysis endpoint")
    print("  POST /batch_process_files - Process multiple uploaded files in batch")
    print("  POST /batch_process_base64 - Process multiple base64 images in batch") 
    print("  POST /download_excel - Download Excel file with diameter data")
    print("  POST /download_batch_excel - Download Excel file with batch results")
    print("  GET  /defaults - Get default configuration and mode information")
    print("  GET  /model_info - Get model information for all modes")
    print("=" * 50)
    print("🚀 PyTorch SAM Backend:")
    print(f"  - Model type: {MODEL_TYPE}")
    print(f"  - GPU enabled: {USE_GPU}")
    print(f"  - Device: {'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'}")
    print(f"  - Points per side: 32")
    print(f"  - Crop layers: 3 (high quality)")
    print("=" * 50)
    print("Modes:")
    print("  - droplet: Basic SAM (crop_n_layers=1) + 4 filters, shows mask count & diameter")
    print("  - huge_droplet: Enhanced SAM (crop_n_layers=3) + 4 filters, shows mask count & diameter")
    print("  - satellite_droplet: Enhanced SAM (crop_n_layers=3) + 4 filters + K-means clustering + two-group analysis & Excel export")
    print("  - multiple: Advanced SAM (crop_n_layers=3) + clustering, shows overlap analysis")
    print("=" * 50)
    print("🚀 Batch Processing Features:")
    print("  - Upload and process multiple images simultaneously")
    print("  - Supports all analysis modes (droplet, huge_droplet, satellite_droplet, multiple)")
    print("  - Automatic aggregation of results across all images")
    print("  - Excel export with comprehensive batch statistics")
    print("  - Error handling for individual files (continues processing others)")
    print("  - Processing time tracking per image and total batch")
    print("=" * 50)
    
    # Start Flask application
    app.run(
        host='0.0.0.0',
        port=5003,
        debug=True,
        threaded=True
    ) 