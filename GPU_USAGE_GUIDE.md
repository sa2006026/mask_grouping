# GPU Usage Monitoring & Control Guide

This guide helps you monitor and keep GPU usage below 30% when running SAM mask grouping.

## 🔍 Current Status

Your GPUs are currently at **100% utilization**, which needs to be reduced to below 30%.

## 📊 Quick GPU Check Commands

### 1. Basic Check (30% threshold)
```bash
python3 simple_gpu_check.py
```

### 2. Custom Threshold Check
```bash
python3 simple_gpu_check.py check 25    # Check against 25% threshold
```

### 3. Continuous Monitoring
```bash
python3 simple_gpu_check.py monitor 300 30    # Monitor for 5 minutes, 30% threshold
```

### 4. Real-time Monitoring (like `top`)
```bash
python3 simple_gpu_check.py watch
```

### 5. Standard nvidia-smi
```bash
nvidia-smi                    # Single check
watch -n 2 nvidia-smi        # Update every 2 seconds
```

## 🎯 Methods to Reduce GPU Usage Below 30%

### Method 1: Use Smaller SAM Model (Recommended)
```bash
export SAM_MODEL_TYPE=vit_l    # Smaller model (~30-50% less GPU usage)
# or
export SAM_MODEL_TYPE=vit_b    # Even smaller (~60-70% less GPU usage)
python3 app.py
```

### Method 2: Use Single GPU Only
```bash
export CUDA_VISIBLE_DEVICES=0    # Use only GPU 0
python3 app.py
```

### Method 3: Use Low GPU Usage Script
```bash
./run_low_gpu_usage.sh
```

### Method 4: Optimize SAM Parameters (Modify app.py)
Edit the `_initialize_pytorch_backend()` function:
```python
# Reduce these values for lower GPU usage:
points_per_side = 16        # Default: 32
crop_n_layers = 1          # Default: 3 (for multiple/huge_droplet modes)
points_per_batch = 32      # Default: 64
```

### Method 5: Use CPU-Only Mode
```bash
export SAM_USE_GPU=false
python3 app.py
```

## 📈 GPU Usage Comparison by Model Type

| Model Type | Relative GPU Usage | Relative Speed | Accuracy |
|------------|-------------------|----------------|----------|
| `vit_h`    | 100% (baseline)   | Slowest        | Highest  |
| `vit_l`    | ~60-70%          | Faster         | High     |
| `vit_b`    | ~30-40%          | Fastest        | Good     |

## 🚀 Quick Start for Low GPU Usage

### Option A: Automated Script
```bash
./run_low_gpu_usage.sh
```

### Option B: Manual Setup
```bash
# Activate environment
source sam_conversion_env/bin/activate

# Set low GPU usage configuration
export SAM_MODEL_TYPE=vit_l
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Start server
python3 app.py

# Monitor in another terminal
python3 simple_gpu_check.py watch
```

## 📊 Monitoring During Processing

### 1. Before Processing
```bash
python3 simple_gpu_check.py
```

### 2. During Processing (Real-time)
Open a second terminal:
```bash
cd /data3/megan_data/Jimmy/AI_droplet/mask_grouping
python3 simple_gpu_check.py watch
```

### 3. After Processing
```bash
python3 simple_gpu_check.py
```

## ⚠️ Current Issues & Solutions

### Issue: GPUs at 100% utilization
**Current Status**: Both A100 GPUs at 100%
**Target**: Reduce to ≤30%

**Immediate Solutions**:
1. **Kill current GPU processes**:
   ```bash
   # Find GPU processes
   nvidia-smi
   
   # Kill specific processes (use PID from nvidia-smi)
   kill 2981911 2981912
   ```

2. **Use smaller model**:
   ```bash
   export SAM_MODEL_TYPE=vit_l
   python3 app.py
   ```

3. **Use single GPU**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   python3 app.py
   ```

## 📋 Monitoring Checklist

- [ ] Check GPU usage before starting: `python3 simple_gpu_check.py`
- [ ] Use appropriate model size for your needs
- [ ] Monitor during processing: `python3 simple_gpu_check.py watch`
- [ ] Verify usage stays below 30%
- [ ] Adjust parameters if usage exceeds threshold

## 🔧 Advanced Configuration

### Environment Variables for GPU Control
```bash
# Model selection
export SAM_MODEL_TYPE=vit_l           # vit_h, vit_l, vit_b

# GPU settings
export SAM_USE_GPU=true               # true/false
export CUDA_VISIBLE_DEVICES=0         # 0, 1, 0,1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
```

### Application-Level Optimizations
1. **Use appropriate processing mode**:
   - `droplet`: Lowest GPU usage
   - `huge_droplet`: Medium GPU usage  
   - `multiple`: Highest GPU usage

2. **Process images individually** instead of batch processing

3. **Resize large images** before processing

## 🎯 Success Criteria

✅ **Target Achieved When**:
- All GPUs show ≤30% utilization
- Memory usage is reasonable (<50GB per GPU)
- Temperature stays below 80°C
- Processing still completes successfully

## 📞 Quick Help Commands

```bash
# Check current status
python3 simple_gpu_check.py

# Start with low GPU usage
./run_low_gpu_usage.sh

# Monitor in real-time
python3 simple_gpu_check.py watch

# Get optimization suggestions
python3 simple_gpu_check.py check 30
```

## 🔄 Troubleshooting

### Problem: Still high GPU usage after optimization
**Solutions**:
1. Use `vit_b` instead of `vit_l`
2. Switch to CPU-only mode temporarily
3. Process smaller images
4. Reduce batch sizes in API calls

### Problem: Performance too slow with optimizations
**Solutions**:
1. Use `vit_l` as compromise between speed and GPU usage
2. Optimize image preprocessing
3. Use targeted processing modes (droplet vs multiple)

---

**Remember**: Monitor GPU usage continuously during processing to ensure it stays below 30%! 