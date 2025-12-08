# CSV Input Usage Guide

## 🎯 Overview

The pipeline has been updated to read image paths from CSV files (train/val/test splits) instead of requiring images in a `raw/` folder.

---

## ✅ Changes Made

### 1. CSV Support Added
- Pipeline can now read image paths from CSV files
- CSV must have an `image_path` column with full paths to images
- Works with your filtered FFHQ CSV files directly

### 2. Updated Files
- `src/pipeline/pipeline_runner.py` - Added CSV reading logic
- `scripts/run_pipeline.py` - Added `--csv` argument
- `requirements.txt` - Added `pandas>=2.0.0`

---

## 📋 CSV Format

Your CSV must have at minimum an `image_path` column:

```csv
image_path,lighting_score
/mnt/localssd/diffusion/filter_images/ffhq_github/ffhq-dataset/images1024x1024/64000/64665.png,0.2628
/mnt/localssd/diffusion/filter_images/ffhq_github/ffhq-dataset/images1024x1024/46000/46876.png,0.2535
```

**Good news:** Your existing CSV files already have this format! ✅

---

## 🚀 Usage

### Basic Usage with CSV

```bash
cd /mnt/localssd/diffusion/albedo/relightingDataGen

# Process training set
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv

# Process validation set
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/val_images.csv

# Process test set
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/test_images.csv
```

### Test with Limited Samples

```bash
# Process only first 10 images from train set
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv \
    --num-samples 10
```

### Old Behavior (still works)

```bash
# If you don't provide --csv, it reads from data/raw/ (old behavior)
python scripts/run_pipeline.py --config config/mvp_config.yaml
```

---

## 📁 Output Structure

Outputs are saved based on image index (not original image ID):

```
data/
├── stage_1/                      # Segmentation outputs
│   ├── 00000_foreground.png      # First image from CSV
│   ├── 00000_background.png
│   ├── 00000_mask.png
│   ├── 00001_foreground.png      # Second image from CSV
│   └── ...
├── stage_2/                      # Albedo extraction
│   ├── 00000_albedo.png
│   └── ...
├── stage_3/                      # Degradation synthesis
│   ├── 00000_degraded.png
│   └── ...
├── stage_3_5/                    # Background recombination
│   ├── 00000_composite.png       # Final output
│   └── ...
└── outputs/                      # Final organized outputs
    ├── 00000_input.png           # Original from CSV
    ├── 00000_output.png          # Final composite
    ├── 00000_albedo.png
    ├── 00000_metadata.json
    └── ...
```

**Note:** The pipeline uses sequential indices (0, 1, 2...) for output naming, reading images in order from the CSV.

---

## 🔧 Your Filtered Dataset

You have 3 CSV files ready to use:

### 1. Training Set (10,000 images)
```bash
--csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv
```

### 2. Validation Set (1,000 images)
```bash
--csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/val_images.csv
```

### 3. Test Set (1,000 images)
```bash
--csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/test_images.csv
```

All images have lighting scores > 0.21 (97%+) from your CLIP filtering!

---

## 📝 Complete Workflow Example

### Step 1: Install Dependencies

```bash
cd /mnt/localssd/diffusion/albedo/relightingDataGen

# Create/activate conda environment
conda create -n relighting python=3.10 -y
conda activate relighting

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements (includes pandas now)
pip install -r requirements.txt

# Install SAM2 (see next section for details)
pip install git+https://github.com/facebookresearch/sam2.git
```

### Step 2: Test on 1 Image

```bash
# Test with 1 training image
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv \
    --num-samples 1
```

### Step 3: Process Full Training Set

```bash
# Process all 10,000 training images
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv
```

**Estimated time:** ~5-15 seconds per image = ~14-40 hours for 10k images

### Step 4: Process Validation Set

```bash
# Process 1,000 validation images
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/val_images.csv
```

### Step 5: Process Test Set

```bash
# Process 1,000 test images
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/test_images.csv
```

---

## 🤖 SAM3 vs SAM2: Do You Need to Download Manually?

### Short Answer: **NO - Automatic Download** ✅

Both SAM3 and SAM2 models are **automatically downloaded** from HuggingFace when you first run the pipeline.

### How It Works:

#### SAM3 (Primary, with Text Prompting):
```python
# Code attempts to load SAM3 automatically:
self.model = build_sam3_image_model(
    model_id="facebook/sam3",  # Downloads from HuggingFace
    device=self.device
)
```

**Download location:** `~/.cache/huggingface/hub/`

**First run:** Will download ~2-4GB of model weights

**Subsequent runs:** Uses cached weights (no re-download)

#### SAM2 (Fallback, with Point Prompting):
```python
# If SAM3 fails, automatically loads SAM2:
self.processor = SAM2ImagePredictor.from_pretrained(
    "facebook/sam2.1-hiera-large"  # Downloads from HuggingFace
)
```

**Download location:** `~/.cache/huggingface/hub/`

**First run:** Will download ~600MB of model weights

**Subsequent runs:** Uses cached weights

### What You Need to Install:

```bash
# Option 1: Install SAM2 (stable, recommended)
pip install git+https://github.com/facebookresearch/sam2.git

# Option 2: Install SAM3 (optional, for text prompting)
pip install git+https://github.com/facebookresearch/sam3.git

# If SAM3 package is not available, pipeline automatically falls back to SAM2
```

### Current Status (as of Dec 2024):

- **SAM2**: ✅ Stable, publicly available
- **SAM3**: ⚠️ May not be publicly released yet

**Recommendation:** Install SAM2 only. The code will:
1. Try SAM3 → fail to import
2. Auto-fallback to SAM2 ✅
3. Download SAM2 weights automatically on first run

### Manual Download (Optional, if automatic fails):

If automatic download fails due to network issues:

```bash
# For SAM2
cd /mnt/localssd/diffusion/albedo/relightingDataGen
python -c "
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-large')
print('SAM2 downloaded successfully!')
"
```

This pre-downloads weights to cache before running the full pipeline.

---

## ⚙️ Configuration Adjustments

You may want to adjust paths in `config/mvp_config.yaml`:

```yaml
paths:
  data_root: "data"                                    # Where stage outputs are saved
  output_root: "data/outputs"                          # Final outputs
  models_root: "models"                                # Not used (HF cache used instead)
  logs_root: "logs"                                    # Log files

memory:
  max_gpu_memory_gb: 22                                # Adjust for your GPU
  clear_cache_between_stages: true                     # Keep enabled

degradation:
  soft_shading:
    weight: 0.8                                        # 80% use soft shading
    ambient_range: [0.6, 0.85]                         # Subtle shadows
```

---

## 🐛 Troubleshooting

### Issue: "CSV file must contain 'image_path' column"

**Solution:** Your CSV is missing the required column. Check with:
```bash
head -1 /path/to/your.csv
```

Should show `image_path` as one of the columns.

### Issue: Image file not found

**Solution:** CSV contains relative paths but pipeline expects absolute paths.

**Fix:** Your CSV already has absolute paths like:
```
/mnt/localssd/diffusion/filter_images/ffhq_github/ffhq-dataset/images1024x1024/...
```
So this should not be an issue! ✅

### Issue: SAM3 not found, falling back to SAM2

**This is expected!** The code automatically handles this:
```
WARNING: SAM3 not installed, falling back to SAM2
SAM2 fallback loaded successfully
```

**No action needed** - pipeline continues with SAM2.

### Issue: Out of memory

**Solution:** Reduce batch size or use smaller MiDaS model:

Edit `config/mvp_config.yaml`:
```yaml
memory:
  max_gpu_memory_gb: 16  # Reduce from 22

degradation:
  soft_shading:
    normal_estimation: "MiDaS_small"  # Already set to smallest
```

---

## 📊 Expected Performance

### Memory Usage (24GB GPU):
- SAM2/SAM3: ~4GB
- Albedo extraction: ~0.5GB
- MiDaS (normal estimation): ~2-4GB
- **Peak:** ~6-8GB

### Processing Speed:
- Segmentation: 5-10s/image
- Albedo: 0.1-0.5s/image
- Degradation: 3-5s/image (soft shading)
- **Total:** ~8-15s/image

### For 10,000 training images:
- **Optimistic:** 8s × 10,000 = 22 hours
- **Realistic:** 12s × 10,000 = 33 hours
- **Conservative:** 15s × 10,000 = 42 hours

---

## 🎯 Quick Reference

### Process 1 test image:
```bash
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv \
    --num-samples 1
```

### Process first 100 training images:
```bash
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv \
    --num-samples 100
```

### Process full training set:
```bash
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv
```

### Check outputs:
```bash
ls -lh data/outputs/        # Final outputs
ls -lh data/stage_3_5/      # Composite images
```

---

## ✨ Summary

✅ **CSV support added** - no need to copy images to `raw/` folder  
✅ **Pandas dependency added** to requirements.txt  
✅ **SAM2/SAM3 auto-download** from HuggingFace (no manual download needed)  
✅ **Works with your filtered dataset** - all 3 CSVs compatible  
✅ **Flexible usage** - process train/val/test separately with `--csv` argument  

You're ready to run the pipeline on your 12k filtered high-quality lighting images! 🚀

