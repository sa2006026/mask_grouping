#!/bin/bash

echo "🎯 SAM Mask Grouping - Low GPU Usage Mode"
echo "========================================"
echo "Target: Keep GPU usage below 30%"
echo ""

# Activate virtual environment if it exists
if [ -d "sam_conversion_env" ]; then
    echo "📦 Activating virtual environment..."
    source sam_conversion_env/bin/activate
fi

echo "🔧 GPU Optimization Settings:"
echo "  • Using smaller model (vit_l instead of vit_h)"
echo "  • Reduced parameters for lower GPU load"
echo "  • Using only one GPU"
echo "  • Optimized batch processing"
echo ""

# Set environment variables for low GPU usage
export SAM_MODEL_TYPE=vit_l           # Smaller model (faster, less VRAM)
export SAM_USE_GPU=true               # Still use GPU but optimized
export CUDA_VISIBLE_DEVICES=0         # Use only GPU 0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024  # Limit memory fragmentation

echo "🚀 Configuration:"
echo "  - Model type: $SAM_MODEL_TYPE (smaller than vit_h)"
echo "  - GPU device: $CUDA_VISIBLE_DEVICES (single GPU only)"
echo "  - Memory optimization: enabled"
echo ""

# Check current GPU usage before starting
echo "📊 Current GPU usage:"
python3 simple_gpu_check.py
echo ""

echo "💡 Tips for keeping GPU usage low:"
echo "  • Use 'droplet' mode instead of 'multiple' or 'huge_droplet'"
echo "  • Process images one at a time instead of batch processing"
echo "  • Use smaller input images (resize before processing)"
echo "  • Monitor GPU usage: python3 simple_gpu_check.py watch"
echo ""

read -p "Continue with low GPU usage settings? (y/N): " confirm
if [[ $confirm != [yY] ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "🌐 Starting server with low GPU usage settings..."
echo "   Monitor GPU usage in another terminal: python3 simple_gpu_check.py watch"
echo ""

# Start the application
python3 app.py 