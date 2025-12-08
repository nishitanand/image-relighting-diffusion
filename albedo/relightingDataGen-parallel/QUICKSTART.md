# Quick Start Guide

Get the relighting pipeline running in 5 minutes!

## Prerequisites

- NVIDIA GPU with 24GB VRAM (e.g., A5000, RTX 3090, RTX 4090)
- CUDA 11.8 or later
- Python 3.10+

## Installation

### 1. Create Environment

```bash
cd /scratch1/manans/projects/relighting
conda create -n relighting python=3.10 -y
conda activate relighting
```

### 2. Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install SAM3
pip install git+https://github.com/facebookresearch/sam3.git
```

### 3. Download Data (Small Subset for Testing)

```bash
# Download 10 FFHQ images for quick testing
python scripts/download_ffhq.py --num-samples 10
```

This will download 10 face images to `data/raw/`.

### 4. Download Models

```bash
# This will attempt to download all models
# Some models will be downloaded automatically on first use
python scripts/download_models.py
```

**Note**: Model downloads may take time depending on your connection. If downloads fail, they'll be attempted automatically when the pipeline runs.

## Run Pipeline

### Quick Test (1 image)

```bash
python scripts/run_pipeline.py --num-samples 1
```

This will process a single image through all 4 stages:
1. SAM3 segmentation (~5-10s)
2. IntrinsicAnything albedo (~15-20s)
3. IC-Light shadows (~20-30s)
4. Qwen2.5-VL caption (~10-15s)

**Total**: ~50-75 seconds per image

### Process All Downloaded Images

```bash
python scripts/run_pipeline.py
```

## Check Results

After processing, check:

```bash
# Intermediate outputs
ls data/stage_1/  # Foreground, background, masks
ls data/stage_2/  # Albedo maps
ls data/stage_3/  # Shadow images
ls data/stage_4/  # Captions

# Final outputs
ls data/outputs/  # Input-output pairs + captions + metadata.json
```

View the metadata:

```bash
cat data/outputs/metadata.json
```

View a sample caption:

```bash
cat data/outputs/00000_caption.txt
```

## Visualize Results

You can open the images in any image viewer:

```bash
# View progression for image 00000
eog data/raw/00000.png &                    # Original
eog data/stage_1/00000_foreground.png &     # Segmented foreground
eog data/stage_2/00000_albedo.png &         # Albedo
eog data/outputs/00000_output.png &         # Final shadow image
```

## Common Issues

### Out of Memory

If you get OOM errors:

1. Edit `config/pipeline_config.yaml`:
   ```yaml
   memory:
     max_gpu_memory_gb: 20  # Reduce from 22
   ```

2. Process fewer images at a time:
   ```bash
   python scripts/run_pipeline.py --num-samples 1
   ```

### Model Download Failures

If automatic downloads fail:

**SAM3**: The model might not be available on HuggingFace yet. Check:
- https://github.com/facebookresearch/sam3

**IntrinsicAnything**: Download manually:
```bash
git clone https://github.com/zju3dv/IntrinsicAnything models/intrinsic_anything
```

**IC-Light**: Download manually:
```bash
# Will be downloaded automatically by diffusers on first use
```

**Qwen2.5-VL**: Downloaded automatically on first use

### Import Errors

Make sure you're in the correct conda environment:

```bash
conda activate relighting
```

Check installed packages:

```bash
pip list | grep torch
pip list | grep transformers
```

## Next Steps

1. **Process more data**: Increase `--num-samples` or remove it to process all
2. **Customize prompts**: Edit `config/model_config.yaml` to change caption prompts
3. **Adjust shadow parameters**: Modify `shadow_generation` in `config/model_config.yaml`
4. **Monitor GPU**: Use `nvidia-smi -l 1` in another terminal to watch memory usage

## Getting Help

- Check `logs/` directory for detailed error messages
- See `README.md` for full documentation
- Open an issue if you encounter problems

## Estimated Time

For 100 FFHQ images:
- **Download data**: ~2-5 minutes
- **Download models**: ~30-60 minutes (one-time)
- **Processing**: ~90-120 minutes (~60s per image)

**Total for first run**: ~2-3 hours
**Subsequent runs**: ~90-120 minutes (models already downloaded)
