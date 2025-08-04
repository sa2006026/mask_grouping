# SAM ONNX Optimization

This server now supports ONNX runtime for faster SAM (Segment Anything Model) inference, which can provide significant performance improvements, especially on CPU and with optimized hardware.

## Quick Start

### 1. Install ONNX Dependencies

The required dependencies are already in `requirements.txt`:

```bash
pip install onnxruntime>=1.16.0 onnx>=1.15.0
# For GPU support:
pip install onnxruntime-gpu>=1.16.0
```

### 2. Convert Your SAM Model to ONNX

Use the convenience script to convert your existing PyTorch SAM model:

```bash
# Convert the existing vit_h model to ONNX
python convert_model.py

# Or specify a different model type
python convert_model.py --model-type vit_l
python convert_model.py --model-type vit_b
```

### 3. Use ONNX Backend

Set environment variables to use the ONNX backend:

```bash
export SAM_BACKEND=onnx
export SAM_MODEL_TYPE=vit_h
python app.py
```

Or start the server directly with ONNX:

```bash
SAM_BACKEND=onnx python app.py
```

## Configuration Options

You can configure the SAM backend using environment variables:

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `SAM_BACKEND` | `pytorch`, `onnx`, `auto` | `auto` | Which backend to use |
| `SAM_MODEL_TYPE` | `vit_h`, `vit_l`, `vit_b` | `vit_h` | SAM model variant |
| `SAM_USE_GPU` | `true`, `false` | `true` | Enable GPU acceleration |
| `SAM_PERFORMANCE_MODE` | `true`, `false` | `false` | Optimize for speed over quality |
| `SAM_MODEL_DIR` | directory path | `model` | PyTorch models directory |
| `SAM_ONNX_DIR` | directory path | `model/onnx` | ONNX models directory |

## Performance Benefits

ONNX typically provides:

- **2-5x faster inference** on CPU
- **1.5-3x faster inference** on GPU
- **Lower memory usage** during inference
- **Better optimization** for production deployment

Performance improvements vary based on:
- Hardware (CPU vs GPU)
- Model size (vit_h > vit_l > vit_b)
- Image size and complexity
- ONNX Runtime version and optimizations

## API Endpoints

### Check Backend Status

```bash
curl http://localhost:5000/backend_info
```

Response:
```json
{
  "backend": "onnx",
  "model_type": "vit_h",
  "use_gpu": true,
  "performance_mode": false,
  "device": "cuda",
  "onnx_available": true,
  "pytorch_available": true,
  "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}
```

## Automatic Backend Selection

With `SAM_BACKEND=auto` (default), the system will:

1. **Prefer ONNX** if models are available (better performance)
2. **Fallback to PyTorch** if ONNX models are not found
3. **Download PyTorch model** if neither is available

## Converting Models Manually

For advanced users, you can use the conversion script directly:

```bash
python convert_to_onnx.py --checkpoint model/sam_vit_h_4b8939.pth --model-type vit_h
```

## Performance Mode

Enable performance mode for faster inference with slightly lower quality:

```bash
SAM_PERFORMANCE_MODE=true python app.py
```

Performance mode adjustments:
- Reduces `points_per_side` from 32 to 16
- Lowers IoU threshold from 0.88 to 0.8
- Reduces `crop_n_layers` to 1 for faster processing
- Filters out smaller masks (min area 500 vs 100)

## Troubleshooting

### ONNX Models Not Found
```bash
# Convert your existing PyTorch model
python convert_model.py

# Or force re-conversion
python convert_model.py --force
```

### GPU Issues
```bash
# Use CPU-only ONNX Runtime
pip uninstall onnxruntime-gpu
pip install onnxruntime

# Or disable GPU in config
SAM_USE_GPU=false python app.py
```

### Memory Issues
```bash
# Enable performance mode
SAM_PERFORMANCE_MODE=true python app.py

# Or use smaller model
SAM_MODEL_TYPE=vit_b python app.py
```

## File Structure

After conversion, your directory structure will be:

```
mask_grouping_server/
├── model/
│   ├── sam_vit_h_4b8939.pth          # PyTorch model
│   └── onnx/
│       ├── sam_vit_h_encoder.onnx    # ONNX encoder
│       └── sam_vit_h_decoder.onnx    # ONNX decoder
├── convert_model.py                   # Convenience conversion script
├── convert_to_onnx.py                # Low-level conversion utility
├── onnx_sam_wrapper.py               # ONNX SAM implementation
├── sam_config.py                     # Configuration management
└── app.py                            # Main server (now with ONNX support)
``` 