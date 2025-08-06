#!/bin/bash
set -e

echo "🚀 Mask Grouping Server Setup Script"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Python 3.8+ is installed
check_python() {
    print_header "📋 Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
        
        # Check if version is 3.8+
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible"
        else
            print_error "Python 3.8+ is required. Please upgrade Python."
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+."
        exit 1
    fi
}

# Check for NVIDIA GPU and CUDA
check_gpu() {
    print_header "🖥️ Checking GPU and CUDA..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            print_status "CUDA $CUDA_VERSION detected"
        else
            print_warning "CUDA not found. GPU acceleration may not work optimally."
        fi
    else
        print_warning "NVIDIA GPU not detected. Will use CPU-only mode."
    fi
}

# Create virtual environment
create_venv() {
    print_header "🐍 Setting up virtual environment..."
    
    if [ ! -d "mask_env" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv mask_env
    else
        print_status "Virtual environment already exists"
    fi
    
    print_status "Activating virtual environment..."
    source mask_env/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Install dependencies
install_dependencies() {
    print_header "📦 Installing dependencies..."
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        print_status "Installing core dependencies..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install segment-anything opencv-python flask flask-cors numpy pillow
        pip install scikit-learn matplotlib pandas openpyxl
        
        # Optional dependencies
        print_status "Installing optional dependencies..."
        pip install onnx onnxruntime-gpu || pip install onnxruntime
        
        # TensorRT (may fail if not available)
        print_status "Attempting to install TensorRT..."
        pip install tensorrt || print_warning "TensorRT installation failed (this is optional)"
    fi
}

# Create necessary directories
create_directories() {
    print_header "📁 Creating directories..."
    
    mkdir -p model/onnx
    mkdir -p model/tensorrt
    mkdir -p uploads
    mkdir -p results
    mkdir -p static
    
    print_status "Directories created successfully"
}

# Download SAM model
download_model() {
    print_header "⬇️ Checking SAM model..."
    
    MODEL_FILE="model/sam_vit_h_4b8939.pth"
    MODEL_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    if [ ! -f "$MODEL_FILE" ]; then
        print_status "Downloading SAM vit_h model (~2.4GB)..."
        print_warning "This may take a while depending on your internet connection..."
        
        if command -v wget &> /dev/null; then
            wget -O "$MODEL_FILE" "$MODEL_URL"
        elif command -v curl &> /dev/null; then
            curl -L -o "$MODEL_FILE" "$MODEL_URL"
        else
            print_warning "wget or curl not found. Model will be downloaded on first run."
            return
        fi
        
        print_status "Model downloaded successfully"
    else
        print_status "SAM model already exists"
    fi
}

# Test installation
test_installation() {
    print_header "🧪 Testing installation..."
    
    print_status "Testing Python imports..."
    python3 -c "
import torch
import cv2
import numpy as np
import flask
print('✅ Core dependencies OK')

try:
    import segment_anything
    print('✅ Segment Anything OK')
except ImportError:
    print('⚠️ Segment Anything not found (will be installed on first run)')

try:
    import onnx
    import onnxruntime
    print('✅ ONNX OK')
except ImportError:
    print('⚠️ ONNX not available')

try:
    import tensorrt
    print('✅ TensorRT OK')
except ImportError:
    print('⚠️ TensorRT not available')
    
print('🎉 Installation test completed!')
"
}

# Create run script
create_run_script() {
    print_header "📝 Creating run script..."
    
    cat > run_server.sh << 'EOF'
#!/bin/bash
# Mask Grouping Server Runner

# Activate virtual environment
source mask_env/bin/activate

# Set default backend if not specified
if [ -z "$SAM_BACKEND" ]; then
    export SAM_BACKEND=auto
fi

echo "🚀 Starting Mask Grouping Server..."
echo "Backend: $SAM_BACKEND"
echo "Access at: http://localhost:5003"
echo "Press Ctrl+C to stop"

# Start the server
python3 app.py
EOF

    chmod +x run_server.sh
    print_status "Created run_server.sh script"
}

# Main installation process
main() {
    print_header "🚀 Starting Mask Grouping Server Setup"
    echo ""
    
    check_python
    check_gpu
    create_venv
    install_dependencies
    create_directories
    download_model
    test_installation
    create_run_script
    
    echo ""
    print_header "✅ Setup Complete!"
    echo ""
    print_status "🎉 Mask Grouping Server is ready to use!"
    echo ""
    echo "To start the server:"
    echo "  ./run_server.sh"
    echo ""
    echo "Or manually:"
    echo "  source mask_env/bin/activate"
    echo "  python3 app.py"
    echo ""
    echo "Then open: http://localhost:5003"
    echo ""
    print_status "For backend selection:"
    echo "  export SAM_BACKEND=pytorch    # Most reliable"
    echo "  export SAM_BACKEND=onnx       # 2x faster"
    echo "  export SAM_BACKEND=tensorrt   # 3.5x faster (GPU only)"
    echo "  export SAM_BACKEND=auto       # Automatic selection"
    echo ""
    print_status "Check README.md for detailed usage instructions!"
}

# Run main function
main "$@" 