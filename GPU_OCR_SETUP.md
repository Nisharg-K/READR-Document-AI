# GPU-Accelerated OCR Setup Guide

## Overview

This guide helps you set up **GPU-accelerated OCR** using NVIDIA CUDA for fast PDF text extraction.

## System Requirements

- **NVIDIA GPU** with CUDA Compute Capability ≥ 3.5 (check: https://developer.nvidia.com/cuda-gpus)
- **NVIDIA CUDA Toolkit** (11.8 or 12.1 recommended)
- **cuDNN** (optional but recommended for better performance)

## Step 1: Check GPU Support

```powershell
# Check if NVIDIA GPU is installed
nvidia-smi

# Example output:
# | NVIDIA-SMI 555.99                 Driver Version: 555.99    |
# | GPU Name                            TCC/WDDM             |
# |=============================================================|
# | 0  NVIDIA GeForce RTX 4090           WDDM                 |
```

If `nvidia-smi` doesn't work:
1. Install NVIDIA drivers: https://www.nvidia.com/Download/driverDetails.aspx
2. Restart your computer
3. Try `nvidia-smi` again

## Step 2: Install PyTorch with CUDA

Choose your CUDA version and install:

### Option A: CUDA 12.1 (Recommended for newer GPUs)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option B: CUDA 11.8 (Compatible with more GPUs)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Option C: CPU-only (Slower, testing only)
```powershell
pip install torch torchvision torchaudio
```

## Step 3: Install Dependencies

```powershell
# Install all requirements (already includes GPU packages)
pip install -r requirements.txt

# Or install individually:
pip install easyocr opencv-python pillow numpy
```

## Step 4: Verify GPU Detection

```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## Usage Examples

### Example 1: Basic PDF OCR
```python
from test.ocrmode import GPUOCRProcessor

processor = GPUOCRProcessor(languages=['en'], use_gpu=True)
stats = processor.ocr_pdf("your_pdf.pdf", output_txt="output.txt", zoom=2.0)
processor.print_stats(stats)
```

### Example 2: Using FastAPI Endpoints

With the GPU OCR routes integrated, you can:

```bash
# Extract text from PDF
curl -X POST "http://localhost:8000/api/ocr/extract-from-pdf" \
  -F "file=@your_file.pdf" \
  -F "zoom=2.0"

# Extract from image
curl -X POST "http://localhost:8000/api/ocr/extract-from-image" \
  -F "file=@your_image.jpg"

# Check GPU status
curl "http://localhost:8000/api/ocr/health"
```

### Example 3: Advanced Configuration

```python
from backend.gpu_ocr import ocr_service

# Extract with multiple languages
reader = easyocr.Reader(['en', 'es', 'fr'], gpu=True, quantize=True)

# Process with high precision (slower)
results = reader.readtext(image, batch_size=1)

# Process with speed priority (lower accuracy)
results = reader.readtext(image, batch_size=4)
```

## Performance Tips

### 1. **Adjust Batch Size**
- Larger batch size = faster but uses more memory
- Start with `batch_size=1` for RTX 3090/4090, increase if memory allows

### 2. **Zoom Level**
- `zoom=1.5`: Fast, medium quality
- `zoom=2.0`: Balanced (recommended)
- `zoom=3.0`: High quality but slower

### 3. **Language Configuration**
- Fewer languages = faster processing
- `['en']` = fastest
- `['en', 'es', 'fr']` = moderate speed
- `['en', 'es', 'fr', 'zh', ...]` = slowest

### 4. **VRAM Management**
```python
# Monitor GPU memory during processing
import torch
print(f"Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

## Expected Performance

With NVIDIA RTX 4090:
- **PDF Pages**: ~5-10 seconds per page (zoom=2.0)
- **Simple Images**: 0.5-2 seconds per image
- **Speedup over CPU**: 10-50x faster

With NVIDIA RTX 3090:
- **PDF Pages**: ~8-15 seconds per page
- **Speedup over CPU**: 8-30x faster

## Troubleshooting

### `CUDA out of memory` Error
```python
# Solution 1: Reduce batch size
reader.readtext(image, batch_size=1)

# Solution 2: Lower zoom level
processor.ocr_pdf("file.pdf", zoom=1.5)

# Solution 3: Clear cache
torch.cuda.empty_cache()
```

### `torch.cuda.is_available() returns False`
1. Run `nvidia-smi` to verify driver installation
2. Reinstall PyTorch with correct CUDA version
3. Check GPU compute capability at: https://developer.nvidia.com/cuda-gpus

### Slow Performance (CPU instead of GPU)
```python
# Verify GPU is being used:
import easyocr
reader = easyocr.Reader(['en'], gpu=True, verbose=True)
# Should show: "CUDA is available"
```

### EasyOCR Model Download Issues
```python
# Models download to ~/.EasyOCR/ (Linux/Mac) or C:\Users\[username]\.EasyOCR (Windows)
# Ensure you have internet connection on first run
# Or manually download using:
import easyocr
reader = easyocr.Reader(['en'])  # Downloads models on first run
```

## Reference Links

- PyTorch Installation: https://pytorch.org/get-started/locally/
- EasyOCR Documentation: https://github.com/JaidedAI/EasyOCR
- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN: https://developer.nvidia.com/cudnn

## Integration with Your Server

To add GPU OCR to your FastAPI server:

```python
# In server.py, add these lines:

from gpu_ocr_routes import router

# Add the OCR routes
app.include_router(router)

# Now you have endpoints:
# POST /api/ocr/extract-from-pdf
# POST /api/ocr/extract-from-image
# GET /api/ocr/health
```

## Next Steps

1. ✅ Install NVIDIA drivers and CUDA Toolkit
2. ✅ Run `pip install -r requirements.txt` with GPU PyTorch
3. ✅ Test with: `python test/ocrmode.py`
4. ✅ Monitor performance with `nvidia-smi`
5. ✅ Integrate into FastAPI for production use

Happy GPU-accelerated OCR processing! 🚀
