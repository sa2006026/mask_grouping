# Mask Grouping Server

A powerful Flask web application for image segmentation and mask analysis using Facebook's Segment Anything Model (SAM) with support for multiple backends including PyTorch, ONNX, and TensorRT.

## 🌟 Features

- **Multi-Backend Support**: PyTorch, ONNX, and TensorRT backends
- **Multiple Analysis Modes**: 
  - Droplet analysis (basic and enhanced)
  - Satellite droplet analysis with K-means clustering
  - Advanced overlap analysis for multiple objects
- **High Performance**: Up to 3.5x speedup with TensorRT optimization
- **Web Interface**: User-friendly Flask web application
- **Export Capabilities**: Excel export for statistical analysis
- **Comprehensive API**: RESTful endpoints for all analysis modes

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for TensorRT)
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sa2006026/mask_grouping.git
cd mask_grouping_server
```

2. **Create and activate virtual environment**
```bash
python3 -m venv mask_env
source mask_env/bin/activate  # On Windows: mask_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download SAM model** (automatic on first run)
```bash
# The app will automatically download sam_vit_h_4b8939.pth on first startup
# Or manually download to model/ directory
```

5. **Start the server**
```bash
python3 app.py
```

6. **Access the application**
```
Open http://localhost:5003 in your browser
```

## 🔧 Backend Configuration

The server supports three backends with automatic fallback:

### PyTorch Backend (Default)
```bash
export SAM_BACKEND=pytorch
python3 app.py
```
- ✅ Most reliable and accurate
- ✅ Generates 2500-3000 high-quality masks
- ✅ Full compatibility with all analysis modes
- ⚠️ Slower inference (~130s for large images)

### ONNX Backend
```bash
# Convert PyTorch model to ONNX
python3 convert_to_onnx.py

# Run with ONNX backend
export SAM_BACKEND=onnx
python3 app.py
```
- ✅ 2-3x faster than PyTorch
- ✅ Cross-platform compatibility
- ⚠️ Requires model conversion step

### TensorRT Backend (Fastest)
```bash
# Install TensorRT (if not installed)
pip install tensorrt

# Convert ONNX to TensorRT
python3 convert_to_tensorrt.py --precision fp16

# Run with TensorRT backend
export SAM_BACKEND=tensorrt
python3 app.py
```
- 🚀 **3-5x faster than PyTorch**
- 🚀 **Optimized for NVIDIA GPUs**
- ⚠️ Requires NVIDIA GPU and CUDA
- ⚠️ May need GPU memory management

## 📊 Analysis Modes

### 1. Droplet Analysis
Basic droplet counting and size analysis
```bash
curl -X POST http://localhost:5003/analyze_droplets \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_data"}'
```

### 2. Huge Droplet Analysis
Enhanced analysis with crop_n_layers=3 for complex scenarios
```bash
curl -X POST http://localhost:5003/analyze_huge_droplets \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_data"}'
```

### 3. Satellite Droplet Analysis
Advanced K-means clustering with two-group classification
```bash
curl -X POST http://localhost:5003/analyze_satellite_droplets \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_data"}'
```

### 4. Multiple Object Overlap Analysis
Comprehensive overlap detection and statistical analysis
```bash
curl -X POST http://localhost:5003/analyze_overlap \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_data", "overlap_threshold": 80.0}'
```

## 🛠️ Advanced Setup

### GPU Memory Management

If you encounter CUDA memory issues:

1. **Check GPU usage**
```bash
nvidia-smi
```

2. **Free GPU memory**
```bash
# Kill processes using GPU memory
sudo kill <PID>

# Or use CPU-only mode
CUDA_VISIBLE_DEVICES="" python3 app.py
```

3. **Optimize memory usage**
```bash
# Use mixed precision
export SAM_TENSORRT_PRECISION=fp16

# Reduce batch size
export SAM_PERFORMANCE_MODE=false
```

### Model Conversion

#### PyTorch to ONNX
```bash
python3 convert_to_onnx.py \
  --model-type vit_h \
  --checkpoint model/sam_vit_h_4b8939.pth \
  --output-dir model/onnx
```

#### ONNX to TensorRT
```bash
python3 convert_to_tensorrt.py \
  --precision fp16 \
  --onnx-dir model/onnx \
  --tensorrt-dir model/tensorrt
```

### Performance Benchmarking
```bash
# Run comprehensive benchmark
python3 performance_benchmark.py

# Quick performance test
python3 simple_performance_comparison.py

# TensorRT demo
python3 tensorrt_performance_demo.py
```

## 🐳 Docker Deployment

### Build and run with Docker
```bash
# Build image
docker build -t mask-grouping-server .

# Run container
docker run -p 5003:5003 --gpus all mask-grouping-server
```

### Docker Compose
```bash
docker-compose up
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/backend_info` | GET | Backend information |
| `/model_info` | GET | Model configuration |
| `/defaults` | GET | Default parameters |
| `/segment` | POST | Basic segmentation |
| `/analyze_droplets` | POST | Droplet analysis |
| `/analyze_huge_droplets` | POST | Enhanced droplet analysis |
| `/analyze_satellite_droplets` | POST | Satellite droplet analysis |
| `/analyze_overlap` | POST | Overlap analysis |
| `/download_excel` | POST | Export Excel data |

## 🔧 Configuration Options

### Environment Variables

```bash
# Backend selection
export SAM_BACKEND=pytorch|onnx|tensorrt|auto

# Model configuration
export SAM_MODEL_TYPE=vit_h|vit_l|vit_b
export SAM_USE_GPU=true|false
export SAM_PERFORMANCE_MODE=true|false

# ONNX configuration
export SAM_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider

# TensorRT configuration
export SAM_TENSORRT_PRECISION=fp16|fp32
export SAM_TENSORRT_DIR=model/tensorrt
```

### Analysis Parameters

```python
{
    "overlap_threshold": 80.0,        # Overlap detection threshold (%)
    "min_circularity": 0.53,          # Minimum circularity filter
    "max_blob_distance": 50,          # Maximum blob distance (pixels)
    "fluorescent_mode": false,        # Enable fluorescent analysis
    "min_brightness_threshold": 200   # Brightness threshold
}
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
```bash
# Check GPU usage
nvidia-smi

# Kill GPU processes
sudo kill <PID>

# Use CPU backend
export SAM_BACKEND=pytorch
CUDA_VISIBLE_DEVICES="" python3 app.py
```

2. **ONNX model loading errors**
```bash
# Re-export ONNX models
python3 convert_to_onnx.py

# Check model files
ls -la model/onnx/
```

3. **TensorRT conversion fails**
```bash
# Check TensorRT installation
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Free GPU memory
nvidia-smi
sudo kill <GPU_PROCESS_PID>

# Retry conversion
python3 convert_to_tensorrt.py --precision fp16
```

4. **Port already in use**
```bash
# Kill existing process
sudo lsof -i :5003
sudo kill <PID>

# Or use different port
export FLASK_PORT=5004
python3 app.py
```

### Performance Optimization

1. **Enable performance mode**
```bash
export SAM_PERFORMANCE_MODE=true
```

2. **Use mixed precision**
```bash
export SAM_TENSORRT_PRECISION=fp16
```

3. **Optimize for your GPU**
```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Use appropriate TensorRT optimization
python3 convert_to_tensorrt.py --precision fp16 --optimization-level 3
```

## 📈 Performance Comparison

| Backend | Inference Time | Speedup | Memory Usage | Accuracy |
|---------|---------------|---------|--------------|----------|
| PyTorch | ~130s | 1.0x | 8GB | 100% |
| ONNX | ~65s | 2.0x | 6GB | 99% |
| TensorRT | ~37s | 3.5x | 5GB | 99% |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference
- [TensorRT](https://developer.nvidia.com/tensorrt) for GPU optimization
- [Flask](https://flask.palletsprojects.com/) for web framework

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**Made with ❤️ for scientific image analysis** 