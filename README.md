# Image Relighting Diffusion Project

Complete pipeline for training instruction-based image editing models and preprocessing/filtering image datasets.

## 📁 Project Structure

```
/mnt/localssd/diffusion/
├── training/         # InstructPix2Pix model training
│   ├── sd1_5/       # Stable Diffusion 1.5 setup
│   ├── sdxl/        # Stable Diffusion XL setup
│   └── flux/        # Flux setup (experimental)
│
└── filter_images/   # Image quality filtering utilities
    └── ...          # Lighting-based filtering scripts
```

## 🎯 Two Main Components

### 1. **Training** (`training/`)

Train models to edit images based on text instructions (InstructPix2Pix):

- **Input**: Original image + text instruction (e.g., "make sky blue")
- **Output**: Edited image following the instruction
- **Models**: SD 1.5 (fast), SDXL (best quality), Flux (experimental)
- **Hardware**: Optimized for 8xA100 GPUs

```bash
cd training
cat START_HERE.txt
```

### 2. **Image Filtering** (`filter_images/`)

Filter and preprocess images based on lighting quality:

- Remove dark or poorly lit images
- Automatic quality scoring
- Batch processing support
- Useful for dataset cleaning before training

```bash
cd filter_images
cat README.md
```

## 🚀 Quick Start

### For Model Training:
```bash
cd training
cat START_HERE.txt  # Read the complete guide
cd sd1_5            # Start with SD 1.5 (fastest)
```

### For Image Filtering:
```bash
cd filter_images
cat README.md       # Read the filtering guide
python filter_lighting_images.py --help
```

## 💡 Typical Workflow

1. **Clean your dataset** (optional)
   ```bash
   cd filter_images
   python filter_lighting_images.py --input_dir /path/to/images
   ```

2. **Prepare training data**
   ```bash
   cd ../training/sd1_5
   python validate_data.py --data_dir /path/to/triplet_data
   python convert_to_hf_dataset.py --data_dir /path/to/triplet_data
   ```

3. **Train the model**
   ```bash
   ./train.sh --data_dir ./data_hf
   ```

4. **Run inference**
   ```bash
   python inference.py --model_path ./output/model --input_image test.jpg
   ```

## 📊 Training Models Comparison

| Model | Quality | Training Time | Resolution | Status |
|-------|---------|---------------|------------|--------|
| **SD 1.5** | Good | ~1.5-2 days | 512×512 | ✅ Ready |
| **SDXL** | Excellent | ~3-5 days | 1024×1024 | ✅ Ready |
| **Flux** | Best? | TBD | 1024×1024 | ⏳ Experimental |

## 📚 Documentation

- **Training**: See `training/README.md` for complete training documentation
- **Image Filtering**: See `filter_images/README.md` for filtering utilities
- **Quick Start**: See respective `START_HERE.txt` files in each folder

## 🎓 What's Next?

**For Training:**
```bash
cd training && cat START_HERE.txt
```

**For Image Filtering:**
```bash
cd filter_images && cat README.md
```

---

**Two powerful tools in one project: Train models and filter datasets!**

