# Quick Installation Guide

## 🚀 One-Command Setup

For automatic installation, run:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- ✅ Check Python 3.8+ installation
- ✅ Detect GPU and CUDA
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Download SAM model (~2.4GB)
- ✅ Create run scripts
- ✅ Test installation

## 🏃‍♂️ Quick Start

After installation, start the server:

```bash
./run_server.sh
```

Or manually:
```bash
source mask_env/bin/activate
python3 app.py
```

## 🔧 Backend Selection

Choose your backend for optimal performance:

```bash
# Automatic (recommended)
export SAM_BACKEND=auto
./run_server.sh

# PyTorch (most reliable)
export SAM_BACKEND=pytorch
./run_server.sh

# ONNX (2x faster)
export SAM_BACKEND=onnx
./run_server.sh

# TensorRT (3.5x faster, requires NVIDIA GPU)
export SAM_BACKEND=tensorrt
./run_server.sh
```

## 🌐 Access the Application

Open your browser and go to:
```
http://localhost:5003
```

## 🔍 Troubleshooting

### Common Issues

1. **Python version error**
   ```bash
   # Install Python 3.8+
   sudo apt update
   sudo apt install python3.8 python3.8-venv python3.8-dev
   ```

2. **GPU not detected**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Install CUDA if needed
   # Follow: https://developer.nvidia.com/cuda-downloads
   ```

3. **Port already in use**
   ```bash
   # Kill existing process
   sudo lsof -i :5003
   sudo kill <PID>
   ```

4. **Dependencies failed to install**
   ```bash
   # Update pip and try again
   source mask_env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 📊 Performance Verification

Test your setup with different backends:

```bash
# Test PyTorch
export SAM_BACKEND=pytorch
python3 simple_performance_comparison.py

# Test ONNX (if available)
export SAM_BACKEND=onnx  
python3 simple_performance_comparison.py

# Test TensorRT (if available)
export SAM_BACKEND=tensorrt
python3 tensorrt_performance_demo.py
```

## 🎯 Next Steps

1. **Upload an image** through the web interface
2. **Try different analysis modes**:
   - Droplet Analysis
   - Huge Droplet Analysis  
   - Satellite Droplet Analysis
   - Multiple Object Overlap Analysis
3. **Export results** to Excel
4. **Use the API** for programmatic access

## 📚 Documentation

For detailed documentation, see:
- `README.md` - Complete documentation
- `ONNX_README.md` - ONNX-specific information
- API endpoints at `http://localhost:5003/defaults`

## 🆘 Get Help

If you encounter issues:
1. Check the troubleshooting section in `README.md`
2. Run the test script: `python3 test_tensorrt_integration.py`
3. Check logs in the terminal
4. Create an issue on GitHub

---

**Happy analyzing! 🎉** 