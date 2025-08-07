#!/bin/bash

echo "🚀 SAM Mask Grouping Server - Performance Optimized"
echo "=================================================="
echo ""
echo "This script demonstrates how to run the server with different performance settings."
echo ""

# Activate virtual environment if it exists
if [ -d "sam_conversion_env" ]; then
    echo "📦 Activating virtual environment..."
    source sam_conversion_env/bin/activate
fi

echo "🎯 Available Performance Options:"
echo ""
echo "1. HIGH PERFORMANCE MODE (2-4x faster, slightly lower quality)"
echo "   - Points per side: 16 (vs 32)"
echo "   - Crop layers: 1 (vs 3)" 
echo "   - Larger batches for better GPU utilization"
echo "   - Recommended for batch processing or when speed is critical"
echo ""
echo "2. HIGH QUALITY MODE (slower but highest accuracy)"
echo "   - Points per side: 32"
echo "   - Crop layers: 3"
echo "   - Best for detailed analysis when time is not critical"
echo ""

read -p "Choose mode (1 for PERFORMANCE, 2 for QUALITY, or Enter for PERFORMANCE): " choice

case $choice in
    2)
        echo "🎨 Starting in HIGH QUALITY mode..."
        echo "Expected processing time: 300-500s per image"
        export SAM_PERFORMANCE_MODE=false
        ;;
    *)
        echo "⚡ Starting in HIGH PERFORMANCE mode..."
        echo "Expected processing time: 75-150s per image (2-4x faster!)"
        export SAM_PERFORMANCE_MODE=true
        ;;
esac

# Set other optimizations
export SAM_MODEL_TYPE=vit_h  # Use vit_l or vit_b for even more speed
export SAM_USE_GPU=true

echo ""
echo "🔧 Configuration:"
echo "  - Model type: $SAM_MODEL_TYPE"
echo "  - Performance mode: $SAM_PERFORMANCE_MODE"
echo "  - GPU enabled: $SAM_USE_GPU"
echo ""
echo "💡 Tip: You can also set these environment variables manually:"
echo "  export SAM_PERFORMANCE_MODE=true"
echo "  export SAM_MODEL_TYPE=vit_h"
echo "  python3 app.py"
echo ""
echo "🌐 Server will be available at: http://localhost:5003"
echo ""
echo "Starting server..."
python3 app.py 