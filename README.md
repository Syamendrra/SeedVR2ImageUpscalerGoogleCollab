# SeedVR2 Image Upscaling Pipeline on Google Colab

This repository details a high-performance image upscaling, denoising, and sharpening pipeline using the **SeedVR2 7B GGUF** model. Optimized for execution on Nvidia GPU environments like Google Colab, it handles batch scaling of images up to 4K resolution.

---

## 📋 Pipeline Architecture

1. **Environment Check:** Diagnostics on PyTorch, CUDA activation, and available VRAM metrics.
2. **Installation:** Clones the `ComfyUI-SeedVR2_VideoUpscaler` repository and installs custom dependencies.
3. **CLI Verification:** Validates the engine parameters against the configuration arguments.
4. **ZIP Archive Import:** Secure extraction and isolation of batch source images.
5. **Batch Processing:** Iterative upscaling of images using resource-optimized scaling parameters.
6. **Verification & Packaging:** Validates output image dimensions, packages results into a downloadable ZIP archive, and purges workspace files to prevent VRAM and storage overflow.

---

## ⚙️ Key Configuration Options

You can customize execution profiles inside the batch processing loop by altering these CLI arguments:

### Quality & Resolution
* `--resolution "2160"`: Specifies target dimension bounds (e.g., `2160` for 4K class, `1080` for Full HD).
* `--max_resolution "3840"`: Restricts maximum boundary limits to prevent Out-Of-Memory (OOM) failures.

### Memory Optimization (VRAM Tuning)
* `--dit_offload_device "cpu"`: Swaps heavy model parameters back and forth to system RAM.
* `--vae_offload_device "cpu"` / `--tensor_offload_device "cpu"`: Offloads operations to maintain structural stability under lower VRAM limitations.
* `--vae_encode_tiled` / `--vae_decode_tiled`: Splits high-resolution processes into small tile matrices to maintain safe resource overheads.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (Tesla T4 or better recommended)
- PyTorch with CUDA support

### Manual Setup
```bash
# Clone Repository
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
cd ComfyUI-SeedVR2_VideoUpscaler

# Install Requirements
pip install -r requirements.txt
Execution
python inference_cli.py \
  /path/to/input.jpg \
  --output /path/to/output.png \
  --dit_model seedvr2_ema_7b_sharp-Q4_K_M.gguf \
  --resolution 2160 \
  --max_resolution 3840 \
  --vae_encode_tiled \
  --vae_decode_tiled \
  --attention_mode sdpa
