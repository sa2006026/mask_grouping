"""
Configuration module for SAM (Segment Anything Model) backends.

This module provides configuration management for choosing between PyTorch and ONNX
backends, along with performance and quality settings.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

class SAMBackend(Enum):
    """Available SAM inference backends."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    AUTO = "auto"  # Automatically choose the best available backend

class SAMConfig:
    """Configuration class for SAM model settings."""
    
    def __init__(
        self,
        backend: SAMBackend = SAMBackend.AUTO,
        model_type: str = "vit_h",
        use_gpu: bool = True,
        onnx_providers: Optional[List[str]] = None,
        performance_mode: bool = False,
        model_dir: str = "model",
        onnx_dir: str = "model/onnx"
    ):
        """
        Initialize SAM configuration.
        
        Args:
            backend: Inference backend to use
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            use_gpu: Whether to use GPU acceleration
            onnx_providers: ONNX Runtime execution providers
            performance_mode: Whether to prioritize speed over quality
            model_dir: Directory containing PyTorch models
            onnx_dir: Directory containing ONNX models
        """
        self.backend = backend
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.performance_mode = performance_mode
        self.model_dir = Path(model_dir)
        self.onnx_dir = Path(onnx_dir)
        
        # Set ONNX providers
        if onnx_providers is None:
            if use_gpu:
                self.onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                self.onnx_providers = ['CPUExecutionProvider']
        else:
            self.onnx_providers = onnx_providers
    
    def get_pytorch_model_path(self) -> Optional[Path]:
        """Get path to PyTorch model file."""
        model_files = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth", 
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        model_path = self.model_dir / model_files[self.model_type]
        return model_path if model_path.exists() else None
    
    def get_onnx_model_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        """Get paths to ONNX encoder and decoder models."""
        encoder_path = self.onnx_dir / f"sam_{self.model_type}_encoder.onnx"
        decoder_path = self.onnx_dir / f"sam_{self.model_type}_decoder.onnx"
        
        encoder_exists = encoder_path.exists()
        decoder_exists = decoder_path.exists()
        
        return (
            encoder_path if encoder_exists else None,
            decoder_path if decoder_exists else None
        )
    
    def is_onnx_available(self) -> bool:
        """Check if ONNX models are available."""
        encoder_path, decoder_path = self.get_onnx_model_paths()
        return encoder_path is not None and decoder_path is not None
    
    def is_pytorch_available(self) -> bool:
        """Check if PyTorch model is available."""
        return self.get_pytorch_model_path() is not None
    
    def resolve_backend(self) -> SAMBackend:
        """
        Resolve the actual backend to use based on availability.
        
        Returns:
            The backend that will be used
        """
        if self.backend == SAMBackend.PYTORCH:
            if not self.is_pytorch_available():
                raise RuntimeError(f"PyTorch model {self.model_type} not found in {self.model_dir}")
            return SAMBackend.PYTORCH
        
        elif self.backend == SAMBackend.ONNX:
            if not self.is_onnx_available():
                raise RuntimeError(f"ONNX models {self.model_type} not found in {self.onnx_dir}")
            return SAMBackend.ONNX
        
        elif self.backend == SAMBackend.AUTO:
            # Prefer ONNX for better performance, fallback to PyTorch
            if self.is_onnx_available():
                print(f"Auto-selected ONNX backend for {self.model_type}")
                return SAMBackend.ONNX
            elif self.is_pytorch_available():
                print(f"Auto-selected PyTorch backend for {self.model_type} (ONNX not available)")
                return SAMBackend.PYTORCH
            else:
                raise RuntimeError(f"No SAM models found for {self.model_type}")
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def get_mask_generator_params(self) -> Dict[str, Any]:
        """Get parameters for mask generator based on performance mode."""
        if self.performance_mode:
            # Optimized for speed
            return {
                "points_per_side": 16,  # Reduced from 32
                "pred_iou_thresh": 0.8,  # Slightly lower threshold
                "stability_score_thresh": 0.9,  # Slightly lower threshold
                "crop_n_layers": 1,  # Reduced layers for speed
                "min_mask_region_area": 500  # Filter small masks
            }
        else:
            # Optimized for quality (default)
            return {
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
                "crop_n_layers": 3,
                "min_mask_region_area": 100
            }
    
    @classmethod
    def from_env(cls) -> 'SAMConfig':
        """Create configuration from environment variables."""
        backend_str = os.getenv('SAM_BACKEND', 'auto').lower()
        backend = SAMBackend(backend_str)
        
        model_type = os.getenv('SAM_MODEL_TYPE', 'vit_h')
        use_gpu = os.getenv('SAM_USE_GPU', 'true').lower() == 'true'
        performance_mode = os.getenv('SAM_PERFORMANCE_MODE', 'false').lower() == 'true'
        model_dir = os.getenv('SAM_MODEL_DIR', 'model')
        onnx_dir = os.getenv('SAM_ONNX_DIR', 'model/onnx')
        
        return cls(
            backend=backend,
            model_type=model_type,
            use_gpu=use_gpu,
            performance_mode=performance_mode,
            model_dir=model_dir,
            onnx_dir=onnx_dir
        )
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"SAMConfig(backend={self.backend.value}, model_type={self.model_type}, "
            f"use_gpu={self.use_gpu}, performance_mode={self.performance_mode})"
        )


# Default configuration
DEFAULT_CONFIG = SAMConfig()

# Environment-based configuration
ENV_CONFIG = SAMConfig.from_env() 