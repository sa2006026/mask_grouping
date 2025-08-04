"""Utility functions for the mask grouping server."""

import io
import base64
from typing import Any, List, Dict
import numpy as np
from PIL import Image
from flask import Request


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def validate_file_upload(file: Any) -> bool:
    """
    Validate uploaded file type.
    
    Args:
        file: Flask file upload object
        
    Returns:
        True if file type is supported, False otherwise
    """
    if not file or not file.filename:
        return False
    
    # Check file extension
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    filename = file.filename.lower()
    
    return any(filename.endswith(ext) for ext in allowed_extensions)


def process_image_from_base64(image_data: str) -> np.ndarray:
    """
    Convert base64 image to numpy array.
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        RGB image as numpy array
    """
    # Remove data URL prefix if present
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
    return image_array


def masks_to_base64_images(masks: List[Dict]) -> List[Dict]:
    """
    Convert masks to base64 encoded PNG images for web display.
    
    Args:
        masks: List of mask dictionaries from SAM
        
    Returns:
        List of mask dictionaries with base64 image data
    """
    mask_images = []
    
    for i, mask in enumerate(masks):
        # Get the segmentation mask
        segmentation = mask['segmentation']
        
        # Convert boolean mask to uint8 (0 or 255)
        mask_image = (segmentation * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_mask = Image.fromarray(mask_image, mode='L')
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_mask.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        mask_data = {
            'id': mask.get('original_sam_id', i),
            'image': f"data:image/png;base64,{mask_base64}",
            'area': int(mask['area']),
            'bbox': convert_numpy_types(mask['bbox']),
            'stability_score': float(mask['stability_score']),
            'circularity': float(mask.get('circularity', 0.0))
        }
        
        # Add blob distance if available
        if 'blob_distance' in mask:
            mask_data['blob_distance'] = float(mask['blob_distance'])
        
        # Add overlap information if available
        if 'overlap_count' in mask:
            mask_data['overlap_count'] = int(mask['overlap_count'])
        
        if 'overlapping_masks' in mask:
            mask_data['overlapping_masks'] = convert_numpy_types(mask['overlapping_masks'])
        
        mask_images.append(mask_data)
    
    return mask_images


def create_error_response(message: str, status_code: int = 500) -> tuple:
    """
    Create standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Tuple of (error_dict, status_code)
    """
    return {
        'success': False,
        'error': message,
        'status_code': status_code
    }, status_code


def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create standardized success response.
    
    Args:
        data: Response data
        
    Returns:
        Success response dictionary
    """
    response = {
        'success': True,
        **data
    }
    return convert_numpy_types(response)


def validate_overlap_threshold(threshold: float) -> bool:
    """
    Validate overlap threshold value.
    
    Args:
        threshold: Overlap threshold percentage (0-100)
        
    Returns:
        True if valid, False otherwise
    """
    return 0 <= threshold <= 100


def validate_circularity(circularity: float) -> bool:
    """
    Validate circularity value.
    
    Args:
        circularity: Circularity value (0-1)
        
    Returns:
        True if valid, False otherwise
    """
    return 0 <= circularity <= 1


def validate_blob_distance(distance: float) -> bool:
    """
    Validate blob distance value.
    
    Args:
        distance: Maximum blob distance in pixels
        
    Returns:
        True if valid, False otherwise
    """
    return distance >= 0


def format_processing_time(seconds: float) -> str:
    """
    Format processing time for display.
    
    Args:
        seconds: Processing time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def calculate_memory_usage() -> Dict[str, float]:
    """
    Calculate current memory usage.
    
    Returns:
        Dictionary with memory usage statistics
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {
            'rss_mb': 0,
            'vms_mb': 0,
            'percent': 0
        }


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    import torch
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3  # GB
        info['memory_cached'] = torch.cuda.memory_reserved(0) / 1024**3  # GB
    
    return info


def cleanup_temp_files(directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
        
    Returns:
        Number of files deleted
    """
    import os
    import time
    from pathlib import Path
    
    if not os.path.exists(directory):
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0
    
    try:
        for file_path in Path(directory).glob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
    except Exception as e:
        print(f"Error cleaning up files: {e}")
    
    return deleted_count


def log_request_info(request: Request, processing_time: float = None) -> None:
    """
    Log request information for monitoring.
    
    Args:
        request: Flask request object
        processing_time: Processing time in seconds (optional)
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    info = {
        'method': request.method,
        'path': request.path,
        'remote_addr': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', 'Unknown'),
        'content_length': request.content_length
    }
    
    if processing_time is not None:
        info['processing_time'] = f"{processing_time:.3f}s"
    
    logger.info(f"Request: {info}")


def validate_image_size(image_array: np.ndarray, max_dimension: int = 4096) -> bool:
    """
    Validate image size to prevent memory issues.
    
    Args:
        image_array: Image as numpy array
        max_dimension: Maximum allowed dimension
        
    Returns:
        True if image size is acceptable, False otherwise
    """
    height, width = image_array.shape[:2]
    return height <= max_dimension and width <= max_dimension


def estimate_processing_time(image_shape: tuple, num_masks_estimate: int = None) -> float:
    """
    Estimate processing time based on image size.
    
    Args:
        image_shape: Image shape (height, width, channels)
        num_masks_estimate: Estimated number of masks (optional)
        
    Returns:
        Estimated processing time in seconds
    """
    height, width = image_shape[:2]
    pixels = height * width
    
    # Base time for SAM processing (rough estimate)
    base_time = 10 + (pixels / 1000000) * 5  # 10s base + 5s per megapixel
    
    # Additional time for mask processing
    if num_masks_estimate:
        mask_processing_time = num_masks_estimate * 0.01  # 10ms per mask
        base_time += mask_processing_time
    
    return base_time 