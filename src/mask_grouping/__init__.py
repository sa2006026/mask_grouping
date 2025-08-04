"""Mask grouping package for SAM-based mask analysis and overlap detection."""

__version__ = "1.0.0"
__author__ = "jimmy"
__email__ = "jimmy@example.com"

from .core import MaskGroupingProcessor
from .utils import (
    convert_numpy_types,
    validate_file_upload,
    process_image_from_base64,
    masks_to_base64_images
)

__all__ = [
    "MaskGroupingProcessor",
    "convert_numpy_types",
    "validate_file_upload", 
    "process_image_from_base64",
    "masks_to_base64_images"
] 