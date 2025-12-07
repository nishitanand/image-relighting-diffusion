# FFHQ Lighting Filter - Quick Start Guide

## Your Dataset
- **Location**: `/mnt/localssd/diffusion/filter_images/ffhq_github/ffhq-dataset/images1024x1024`
- **Total Images**: 70,000 PNG images (1024x1024)
- **Structure**: 70 subdirectories (00000/ to 69000/), each containing 1000 images
- **Goal**: Select 50,000 images with best lighting quality

## What the Code Does

### Step-by-Step Process:

1. **Image Discovery** (`find_images()`)
   - Recursively scans the dataset directory
   - Finds all `.png` files across all subdirectories
   - Returns list of 70,000 image paths

2. **CLIP Model Loading** (`LightingImageFilter.__init__()`)
   - Loads OpenAI's CLIP model (ViT-B/32 by default)
   - Prepares text encodings for lighting-related prompts:
     * "beautiful lighting"
     * "good lighting"
     * "well lit face"
     * "professional lighting"
     * "natural light"
     * "illumination"
     * "bright and clear lighting"

3. **Batch Processing** (`compute_batch_scores()`)
   - Processes images in batches of 32-64 (configurable)
   - For each image:
     * Load and preprocess image
     * Encode with CLIP vision encoder
     * Compute cosine similarity with text embeddings
     * Average similarity across all 7 prompts
   - Returns lighting score (0-1, higher = better lighting)

4. **Ranking and Selection**
   - Sorts all 70,000 images by lighting score (descending)
   - Selects top 50,000 images
   - Rejects bottom 20,000 images

5. **Output Generation**
   - `filtered_images.txt`: Plain text list of 50k selected paths
   - `filtered_images.json`: JSON with paths and scores
   - `all_scores.csv`: Complete dataset with all scores

## How to Run

### Option 1: Simple Command

```bash
cd /mnt/localssd/diffusion/filter_images

python filter_lighting_images.py \
    --dataset_path /mnt/localssd/diffusion/filter_images/ffhq_github/ffhq-dataset/images1024x1024 \
    --output_dir ./ffhq_output \
    --num_images 50000 \
    --batch_size 64
```

### Option 2: Complete Pipeline (Recommended)

```bash
cd /mnt/localssd/diffusion/filter_images
./run_pipeline.sh
```

This runs:
1. Filtering (gets 50k images)
2. Verification (shows top/bottom/cutoff images)
3. Analysis (creates splits and visualizations)

## Expected Runtime

- **With GPU (CUDA)**:
  - ViT-B/32: ~1-2 hours for 70k images
  - ViT-B/16: ~2-3 hours
  - ViT-L/14: ~4-6 hours

- **Without GPU (CPU only)**:
  - Much slower (~10-20x), not recommended

## Verification

After filtering, verify the quality:

```bash
python verify_filtering.py \
    --results ./ffhq_output/all_scores.csv \
    --output_dir ./ffhq_verification
```

This shows you:
- **Top 10 images** - Should have beautiful/professional lighting
- **20 random samples** - Shows typical quality of filtered images
- **Bottom 20 from filtered set** - Images #49,980 to #50,000 (worst of selected, but should still have acceptable lighting)
- **Image #49,999** - The last selected image (cutoff point)
- **Bottom 10 overall** - Should be dark/poorly lit (these are rejected)

## Output Files Explained

### filtered_images.txt (50,000 lines)
```
/path/to/images1024x1024/03000/03456.png
/path/to/images1024x1024/12000/12789.png
...
```
Use this file to load your filtered dataset.

### filtered_images.json (50,000 entries)
```json
[
  {
    "image_path": "/path/to/03456.png",
    "lighting_score": 0.3456
  },
  ...
]
```
Contains both paths and scores for analysis.

### all_scores.csv (70,000 rows)
```csv
image_path,lighting_score
/path/to/00000.png,0.3456
/path/to/00001.png,0.2123
...
```
Complete dataset with all scores - useful for threshold adjustment.

## Understanding Scores

- **Score Range**: 0.0 to 1.0 (typically 0.15 to 0.40 for face images)
- **High scores (0.35+)**: Well-lit, clear, professional lighting
- **Medium scores (0.25-0.35)**: Decent lighting, moderate quality
- **Low scores (<0.25)**: Dark, underlit, poor lighting quality

The cutoff at 50k typically falls around 0.28-0.30 (varies by dataset).

## Customization

### Change Number of Images
```bash
--num_images 40000  # Select only 40k instead of 50k
```

### Adjust Batch Size (GPU Memory)
```bash
--batch_size 32   # Use if you have 8-12 GB GPU
--batch_size 64   # Use if you have 16-24 GB GPU
--batch_size 128  # Use if you have 40+ GB GPU
```

### Use Better CLIP Model (slower but more accurate)
```bash
--model_name ViT-L/14  # Larger, more accurate model
```

### Custom Text Prompts
Edit `filter_lighting_images.py` line 52:
```python
self.lighting_prompts = [
    "beautiful lighting",
    "your custom prompt here",
]
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `--batch_size 16`
- Use smaller model: `--model_name ViT-B/32`

### Too Slow
- Increase batch size (if you have GPU memory)
- Use smaller model
- Check GPU is being used: `nvidia-smi`

### No CUDA Available
Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

After filtering, you'll have 50k well-lit images ready for:
1. **Lighting transformation** - Apply your lighting change model
2. **Triplet creation** - (input, instruction, output) for diffusion training
3. **Model training** - Train your instruction-following diffusion model

## Questions?

Check the main README.md for full documentation.

