# ONNX Inference Issues Analysis

## Root Cause: Model Export vs Runtime Mismatch

### 🔍 The Problem

The ONNX models were exported with **different input specifications** than what they actually accept:

#### Expected Inputs (from conversion script):
```python
input_names=['image_embeddings', 'point_coords', 'point_labels', 
            'mask_input', 'has_mask_input', 'orig_im_size']  # 6 inputs
```

#### Actual Model Inputs (from runtime analysis):
```
- image_embeddings: tensor(float) ['batch_size', 256, 64, 64]
- point_coords: tensor(float) ['batch_size', 'num_points', 2] 
- point_labels: tensor(float) ['batch_size', 'num_points']
- mask_input: tensor(float) ['batch_size', 1, 256, 256]
# Only 4 inputs - missing has_mask_input and orig_im_size!
```

## Specific Issues Identified

### 1. **Input Count Mismatch**
- Export script expects: 6 inputs
- Actual model accepts: 4 inputs
- **Missing inputs**: `has_mask_input`, `orig_im_size`

### 2. **Data Type Issues**
```
ONNX encoding failed: Unexpected input data type. 
Actual: (tensor(double)) , expected: (tensor(float))
```
- The encoder expects `float32` but receives `float64` (double)

### 3. **Broadcasting/Shape Issues**
```
BroadcastIterator::Init: axis == 1 || axis == largest was false. 
Attempting to broadcast an axis by a dimension other than 1. 6 by 256
```
- Shape mismatch in the prompt encoder's Where operation
- Likely related to point coordinate processing

## Technical Analysis

### Export Process Issues

The `convert_to_onnx.py` script defines a `MaskDecoderWrapper` that expects 6 inputs:

```python
def forward(self, image_embeddings, point_coords, point_labels, 
           mask_input, has_mask_input, orig_im_size):
```

But during ONNX export, some of these inputs are being optimized away or merged, resulting in a model that only accepts 4 inputs.

### Why This Happens

1. **ONNX Optimization**: During export, ONNX may optimize away unused inputs
2. **Constant Folding**: `orig_im_size` might be folded into the model as a constant
3. **Input Simplification**: `has_mask_input` may be simplified away if it's always the same value

## Solutions

### Immediate Fix Options

#### Option 1: Regenerate ONNX Models
```bash
# Delete existing models
rm -rf model/onnx/

# Regenerate with proper export settings
python3 convert_model.py --model-type vit_h
```

#### Option 2: Fix the Wrapper Code
Update `onnx_sam_wrapper.py` to match actual model inputs:

```python
# Remove the problematic inputs
decoder_inputs = {
    'image_embeddings': self.features,
    'point_coords': coords.astype(np.float32),
    'point_labels': labels.astype(np.float32), 
    'mask_input': mask_input.astype(np.float32)
}
# Don't include has_mask_input and orig_im_size
```

#### Option 3: Fix Export Script
Modify `convert_to_onnx.py` to properly handle the input simplification during export.

## Performance Impact

### Current Status
- **PyTorch**: ✅ Works perfectly (213s for satellite analysis)
- **ONNX**: ❌ Cannot complete inference due to input mismatches
- **ONNX Encoder Only**: ✅ Works and shows speed potential (2.87s vs 4.17s init)

### Expected Performance (when fixed)
Based on documentation and partial results:
- **2-5x faster inference** on CPU
- **1.5-3x faster inference** on GPU
- **Lower memory usage**
- **Better production deployment**

## Recommendations

1. **Short-term**: Use PyTorch backend (reliable, working)
2. **Medium-term**: Fix ONNX export/import compatibility  
3. **Long-term**: Consider official SAM ONNX models if available

The ONNX implementation shows clear speed potential but needs the export/import pipeline to be fixed first.