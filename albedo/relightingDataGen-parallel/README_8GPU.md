# 8-GPU Parallel Relighting Pipeline

## 🚀 Quick Start

Process 12,000 images in **~2.5 hours** using 8 GPUs!

```bash
cd /mnt/localssd/diffusion/albedo/relightingDataGen-parallel

# Test with 16 images first (takes ~1 minute)
python scripts/run_multi_gpu.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv \
    --num-samples 16 \
    --num-gpus 8

# Then process everything (~2.5 hours)
./scripts/run_all_splits_8gpu.sh
```

## 📊 Performance

| Split      | Images | Single GPU | 8 GPUs     | Speedup |
|------------|--------|------------|------------|---------|
| Training   | 10,000 | 58 hours   | **2 hours**    | 29x     |
| Validation | 1,000  | 5.8 hours  | **12 min**     | 29x     |
| Test       | 1,000  | 5.8 hours  | **12 min**     | 29x     |
| **Total**  | **12,000** | **70 hours** | **2.5 hours** | **28x** |

## 📁 Outputs

### Image Outputs:
- `data-train/` - 10,000 relighted training images
- `data-val/` - 1,000 relighted validation images
- `data-test/` - 1,000 relighted test images

Each directory contains 7 files per image:
```
00000_output.png       ← Final composite (person + gray background) ⭐
00000_input.png        ← Original image
00000_albedo.png       ← Extracted albedo
00000_degraded_fg.png  ← Degraded foreground
00000_foreground.png   ← Segmented person
00000_background.png   ← Original background (for reference)
00000_metadata.json    ← Processing metadata
```

### CSV Outputs with Mappings:

**Original CSVs (unchanged):**
- `train_images.csv`
- `val_images.csv`
- `test_images.csv`

**New CSVs with output paths:**
- `train_images_with_relighting_outputs.csv`
- `val_images_with_relighting_outputs.csv`
- `test_images_with_relighting_outputs.csv`

Each new CSV has an additional `output_image_path` column mapping input → output.

## 🎯 What It Does

For each input image, the pipeline:

1. **Segments** the person using SAM3 (text prompt: "person")
2. **Extracts albedo** from the foreground using Multi-Scale Retinex
3. **Applies degradations** (soft shading, hard shadows, or specular reflection)
4. **Composites** the result onto a **gray background** (#808080)

**Result:** Realistic relighting with consistent gray background for training.

## 🔧 How It Works

### Multi-GPU Distribution:

For 10,000 training images across 8 GPUs:
- GPU 0: Images 0-1249 (1,250 images)
- GPU 1: Images 1250-2499 (1,250 images)
- GPU 2: Images 2500-3749 (1,250 images)
- ...
- GPU 7: Images 8750-9999 (1,250 images)

All GPUs process **simultaneously** with no contention!

### Memory Usage:
- Per GPU: ~4GB (only 5% of 80GB H100)
- All 8 GPUs: ~32GB total
- Plenty of headroom ✅

## 📝 Commands

### Process All Splits:
```bash
./scripts/run_all_splits_8gpu.sh
```

### Process Individual Splits:
```bash
# Training
python scripts/run_multi_gpu.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv \
    --num-gpus 8

# Validation
python scripts/run_multi_gpu.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/val_images.csv \
    --num-gpus 8

# Test
python scripts/run_multi_gpu.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/test_images.csv \
    --num-gpus 8
```

### Use Fewer GPUs:
```bash
# Use 4 GPUs instead of 8
python scripts/run_multi_gpu.py \
    --config config/mvp_config.yaml \
    --csv train_images.csv \
    --num-gpus 4
```

### Test with Sample:
```bash
# Test with 16 images (2 per GPU)
python scripts/run_multi_gpu.py \
    --config config/mvp_config.yaml \
    --csv train_images.csv \
    --num-samples 16 \
    --num-gpus 8
```

## 📊 Monitoring

### Watch GPU Usage:
```bash
watch -n 1 nvidia-smi

# You should see all 8 GPUs at ~4-5GB usage
```

### Count Completed Images:
```bash
# Training progress
watch -n 10 'echo "Train: $(ls data-train/*_output.png 2>/dev/null | wc -l) / 10000"'

# Validation progress
watch -n 10 'echo "Val: $(ls data-val/*_output.png 2>/dev/null | wc -l) / 1000"'

# Test progress
watch -n 10 'echo "Test: $(ls data-test/*_output.png 2>/dev/null | wc -l) / 1000"'
```

### Check CSV Mapping:
```bash
# Verify output paths were added
tail -20 /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images_with_relighting_outputs.csv
```

## ✅ Verification

After processing, verify the outputs:

```bash
# Check image counts
echo "Train: $(ls data-train/*_output.png 2>/dev/null | wc -l) / 10000"
echo "Val: $(ls data-val/*_output.png 2>/dev/null | wc -l) / 1000"
echo "Test: $(ls data-test/*_output.png 2>/dev/null | wc -l) / 1000"

# Check CSV row counts (should match)
echo "Train CSV: $(tail -n +2 /path/to/train_images_with_relighting_outputs.csv | wc -l)"
echo "Val CSV: $(tail -n +2 /path/to/val_images_with_relighting_outputs.csv | wc -l)"
echo "Test CSV: $(tail -n +2 /path/to/test_images_with_relighting_outputs.csv | wc -l)"

# View a sample output
display data-train/00000_output.png  # Should show person on gray background
```

## 🎨 Configuration

The gray background is configured in `config/mvp_config.yaml`:

```yaml
background:
  use_gray: true                  # Use gray background (not original)
  gray_color: [128, 128, 128]     # RGB values for #808080
```

To use original backgrounds instead:
```yaml
background:
  use_gray: false
```

## 📚 Documentation

- **PARALLEL_GUIDE.md** - Complete documentation
- **CSV_USAGE_GUIDE.md** - CSV input format details
- **SAM3_FIX.md** - SAM3 integration details

## 🛠️ Requirements

- Python 3.10+
- 8 GPUs (H100 80GB recommended)
- SAM3 model weights
- Conda environment with dependencies

## 🎯 Next Steps

After processing, you'll have:
- 12,000 relighted images ready for diffusion model training
- CSVs mapping original images to relighted outputs
- Organized train/val/test splits

Use these for training your lighting transformation diffusion model! 🚀

