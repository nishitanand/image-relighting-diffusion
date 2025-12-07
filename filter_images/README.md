# CLIP-Based Image Filtering for Lighting Quality

Filter images from large datasets based on their lighting quality using CLIP (Contrastive Language-Image Pre-training). This tool is designed to help you select high-quality images with good lighting for training diffusion models or other computer vision tasks.

## Features

- 🔍 **CLIP-based filtering**: Uses semantic understanding to identify images with good lighting
- 📊 **Batch processing**: Efficiently processes large datasets with GPU acceleration
- 📈 **Score tracking**: Saves detailed scores for all processed images
- 🎨 **Visualization**: Analyze and visualize filtering results
- 📦 **Multiple output formats**: JSON, CSV, and text file outputs

## Installation

```bash
cd /mnt/localssd/diffusion/filter_images
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Filter 50k images from the FFHQ dataset:

```bash
python filter_lighting_images.py \
    --dataset_path /path/to/ffhq-dataset \
    --output_dir ./output \
    --num_images 50000 \
    --batch_size 32
```

### For FFHQ Dataset

If you have the FFHQ dataset downloaded from https://github.com/NVlabs/ffhq-dataset:

```bash
python filter_lighting_images.py \
    --dataset_path /mnt/localssd/diffusion/filter_images/ffhq_github/ffhq-dataset/images1024x1024 \
    --output_dir ./ffhq_filtered \
    --num_images 50000 \
    --batch_size 64 \
    --model_name ViT-B/32
```

Or use the provided pipeline script:

```bash
./run_pipeline.sh
```

## Command-Line Arguments

### `filter_lighting_images.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_path` | str | *required* | Path to the image dataset directory |
| `--output_dir` | str | `./filtered_output` | Directory to save filtered results |
| `--num_images` | int | `50000` | Number of top images to select |
| `--batch_size` | int | `32` | Batch size for processing (adjust based on GPU memory) |
| `--model_name` | str | `ViT-B/32` | CLIP model variant (`ViT-B/32`, `ViT-B/16`, `ViT-L/14`) |
| `--no_save_scores` | flag | - | Don't save all scores to CSV |
| `--copy_images` | flag | - | Copy filtered images to output directory |

## Output Files

After running the filtering script, you'll get:

1. **`filtered_images.txt`**: List of filtered image paths (one per line)
2. **`filtered_images.json`**: Detailed results with paths and scores
3. **`all_scores.csv`**: Complete scores for all processed images (optional)
4. **`filtered_images/`**: Copied images (if `--copy_images` is used)

### Example JSON Output

```json
[
  {
    "image_path": "/path/to/image001.jpg",
    "lighting_score": 0.3456
  },
  {
    "image_path": "/path/to/image002.jpg",
    "lighting_score": 0.3421
  }
]
```

## Analyzing Results

Use the `analyze_results.py` script to visualize and analyze your filtering results:

```bash
python analyze_results.py \
    --results_json ./output/filtered_images.json \
    --output_dir ./analysis \
    --create_grid \
    --grid_size 5 5 \
    --create_splits
```

This will generate:
- **Score distribution histogram**
- **Grid visualization of top images**
- **Train/val/test splits** (80/10/10 by default)
- **Statistics summary**

## Verifying Filter Quality

Use the `verify_filtering.py` script to verify that the filtering is working correctly:

```bash
python verify_filtering.py \
    --results ./output/all_scores.csv \
    --output_dir ./verification \
    --cutoff 50000
```

This will show you:
- **Top 10 highest scoring images** (best lighting)
- **20 random samples from filtered set** (typical quality)
- **Bottom 20 from filtered set** (worst of the 50k selected - images #49,980 to #50,000)
- **Images around the 50k cutoff** (49999th image and neighbors)
- **Bottom 10 lowest scoring images** (worst lighting/dark images - rejected)
- **Score statistics and gaps**

This helps verify that:
- High-scoring images actually have good lighting
- Random samples show typical filtered image quality
- Even the worst selected images (bottom 20) still have acceptable lighting
- Low-scoring rejected images are dark/poorly lit
- The 50k cutoff is reasonable

## Advanced Usage

### Using a Larger CLIP Model

For better accuracy, use a larger CLIP model (requires more GPU memory):

```bash
python filter_lighting_images.py \
    --dataset_path /path/to/dataset \
    --model_name ViT-L/14 \
    --batch_size 16
```

### Processing in Smaller Batches

If you encounter GPU memory issues:

```bash
python filter_lighting_images.py \
    --dataset_path /path/to/dataset \
    --batch_size 8
```

### Copy Filtered Images

To create a separate directory with only the filtered images:

```bash
python filter_lighting_images.py \
    --dataset_path /path/to/dataset \
    --copy_images
```

## How It Works

The filtering process uses CLIP to compute similarity scores between images and lighting-related text prompts:

1. **Text Prompts** (customizable in the code):
   - "beautiful lighting"
   - "good lighting"
   - "well lit face"
   - "professional lighting"
   - "natural light"
   - "illumination"
   - "bright and clear lighting"

2. **Scoring**: Each image receives a score based on its average similarity to all text prompts

3. **Selection**: Top N images with highest scores are selected

## Customization

### Adding Custom Text Prompts

Edit the `lighting_prompts` list in `filter_lighting_images.py`:

```python
self.lighting_prompts = [
    "beautiful lighting",
    "good lighting",
    # Add your custom prompts here
    "dramatic lighting",
    "soft ambient light",
]
```

### Using Different Image Extensions

Modify the `extensions` parameter in the `find_images()` function:

```python
image_paths = find_images(dataset_path, extensions=['.jpg', '.png', '.webp'])
```

## Performance Tips

1. **GPU Memory**: Adjust batch size based on your GPU:
   - 24GB VRAM: batch_size 64-128
   - 12GB VRAM: batch_size 32-64
   - 8GB VRAM: batch_size 16-32

2. **Speed**: Larger CLIP models are more accurate but slower
   - `ViT-B/32`: Fastest, good quality
   - `ViT-B/16`: Medium speed, better quality
   - `ViT-L/14`: Slowest, best quality

3. **Storage**: Use `--no_save_scores` to save disk space if you don't need all scores

## Example Workflow

Complete workflow for FFHQ dataset:

```bash
# Step 1: Filter images
python filter_lighting_images.py \
    --dataset_path /path/to/ffhq-dataset \
    --output_dir ./ffhq_filtered \
    --num_images 50000 \
    --batch_size 64

# Step 2: Verify filtering quality
python verify_filtering.py \
    --results ./ffhq_filtered/all_scores.csv \
    --output_dir ./ffhq_verification \
    --cutoff 50000

# Step 3: Analyze results
python analyze_results.py \
    --results_json ./ffhq_filtered/filtered_images.json \
    --output_dir ./ffhq_analysis \
    --create_grid \
    --create_splits

# Step 4: Use filtered images for your task
# The filtered image paths are in ./ffhq_filtered/filtered_images.txt
```

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
python filter_lighting_images.py --batch_size 8 ...
```

### No Images Found

Ensure your dataset path is correct and contains supported image formats (.jpg, .jpeg, .png, .bmp, .webp).

### Slow Processing

- Use a smaller CLIP model (`ViT-B/32`)
- Increase batch size if you have more GPU memory
- Ensure CUDA is available: `torch.cuda.is_available()`

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- 8GB+ GPU memory (recommended)
- Sufficient disk space for dataset and outputs

## Citation

If you use this code in your research, please cite the original CLIP paper:

```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={International conference on machine learning},
  year={2021}
}
```

## License

This project is provided as-is for research and educational purposes.

## Support

For issues or questions, please check:
1. GPU memory availability
2. Dataset path correctness
3. Required dependencies installed
4. CUDA compatibility with PyTorch

---

Happy filtering! 🎨✨

