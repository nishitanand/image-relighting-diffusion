# Image Relighting with Diffusion Models

A complete end-to-end pipeline for training image relighting models based on the IC-Light methodology. This project enables you to:

1. **Filter** high-quality images with good lighting from large datasets
2. **Generate** relighting training data (albedo extraction + degradation synthesis)
3. **Train** instruction-based image editing models (InstructPix2Pix)

## 🎯 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMAGE RELIGHTING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STEP 1                    STEP 2                      STEP 3              │
│   ───────                   ───────                     ───────             │
│                                                                             │
│   filter_images/     →     albedo/                  →   training/           │
│                            relightingDataGen-parallel                       │
│                                                                             │
│   ┌─────────────┐          ┌─────────────────┐          ┌──────────────┐   │
│   │ FFHQ 70k    │          │ Filtered Images │          │ Triplet Data │   │
│   │ Images      │    →     │ (12k well-lit)  │    →     │ (input,inst, │   │
│   └─────────────┘          └─────────────────┘          │  output)     │   │
│         │                         │                     └──────────────┘   │
│         ▼                         ▼                            │            │
│   ┌─────────────┐          ┌─────────────────┐                 ▼            │
│   │ CLIP Filter │          │ • SAM3 Segment  │          ┌──────────────┐   │
│   │ Lighting    │          │ • Albedo Extract│          │ Train Model  │   │
│   │ Quality     │          │ • Degradation   │          │ SD1.5 / SDXL │   │
│   └─────────────┘          └─────────────────┘          └──────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
image-relighting-diffusion/
├── filter_images/                    # Step 1: Image filtering
│   ├── filter_lighting_images.py     # CLIP-based lighting filter
│   ├── verify_filtering.py           # Verification tools
│   └── analyze_results.py            # Create train/val/test splits
│
├── albedo/                           # Step 2: Training data generation
│   └── relightingDataGen-parallel/   # Multi-GPU parallel processing
│       ├── scripts/
│       │   └── run_multi_gpu_batched.py  # Main entry point
│       └── src/
│           ├── stages/
│           │   ├── stage_1_segmentation_sam3.py  # SAM3/SAM2 segmentation
│           │   ├── stage_2_albedo.py              # Albedo extraction
│           │   └── stage_3_shadow.py              # Degradation synthesis
│           └── utils/                             # Helper modules
│
└── training/                         # Step 3: Model training
    ├── sd1_5/                        # Stable Diffusion 1.5 (fast)
    ├── sdxl/                         # Stable Diffusion XL (best quality)
    └── flux/                         # Flux (experimental)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (24GB+ VRAM recommended)
- Conda or virtualenv

### Step 1: Filter Images (Optional but Recommended)

Select high-quality, well-lit images from your dataset using CLIP-based filtering.

```bash
cd filter_images

# Install dependencies
pip install -r requirements.txt

# Filter top 12k images with best lighting
python filter_lighting_images.py \
    --dataset_path /path/to/your/images \
    --output_dir ./output \
    --num_images 12000 \
    --batch_size 64

# Create train/val/test splits
python analyze_results.py \
    --results_json ./output/filtered_images.json \
    --output_dir ./output \
    --create_splits
```

**Output**: `train_images.csv`, `val_images.csv`, `test_images.csv`

📖 See [`filter_images/README.md`](filter_images/README.md) for details.

---

### Step 2: Generate Relighting Training Data

Process filtered images through the IC-Light pipeline to create (input, degraded_output) pairs.

```bash
cd albedo/relightingDataGen-parallel

# Create and activate environment
conda create -n sam3 python=3.10 -y
conda activate sam3

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

# Run multi-GPU processing
python scripts/run_multi_gpu_batched.py \
    --config config/mvp_config.yaml \
    --csv /path/to/filter_images/output/train_images.csv \
    --num-gpus 8 \
    --batch-size 8
```

**Pipeline stages**:
1. **SAM3/SAM2 Segmentation** - Extract foreground from background
2. **Albedo Extraction** - Remove lighting using Retinex/LAB methods
3. **Degradation Synthesis** - Apply new lighting (soft shading, hard shadows, specular)

**Output**: Training triplets in `data/outputs/`

📖 See [`albedo/relightingDataGen-parallel/README.md`](albedo/relightingDataGen-parallel/README.md) for details.

---

### Step 3: Train the Model

Train an InstructPix2Pix model on your generated data.

```bash
cd training/sd1_5  # Start with SD 1.5 for fast prototyping

# Install dependencies
pip install -r requirements.txt

# Prepare data
python validate_data.py --data_dir /path/to/triplet_data
python convert_to_hf_dataset.py --data_dir /path/to/triplet_data --output_dir ./data_hf

# Configure accelerate for multi-GPU
./setup_accelerate.sh

# Train! (~1.5-2 days on 8xA100)
./train.sh --data_dir ./data_hf
```

📖 See [`training/README.md`](training/README.md) for details.

---

## 📊 Model Comparison

| Model | Quality | Training Time | Resolution | Status |
|-------|---------|---------------|------------|--------|
| **SD 1.5** | Good ⭐⭐⭐ | ~1.5-2 days | 512×512 | ✅ Ready |
| **SDXL** | Excellent ⭐⭐⭐⭐⭐ | ~3-5 days | 1024×1024 | ✅ Ready |
| **Flux** | Best? ⭐⭐⭐⭐⭐⭐ | TBD | 1024×1024 | ⏳ Experimental |

**Recommendation**: Start with **SD 1.5** for rapid prototyping, then scale to **SDXL** for production.

---

## 💡 Typical End-to-End Workflow

```bash
# ═══════════════════════════════════════════════════════════════
# STEP 1: Filter Images (~1-2 hours)
# ═══════════════════════════════════════════════════════════════
cd filter_images
python filter_lighting_images.py \
    --dataset_path /path/to/ffhq \
    --output_dir ./ffhq_filtered \
    --num_images 12000

python analyze_results.py \
    --results_json ./ffhq_filtered/filtered_images.json \
    --output_dir ./ffhq_filtered \
    --create_splits

# ═══════════════════════════════════════════════════════════════
# STEP 2: Generate Training Data (~2-4 hours for 10k images)
# ═══════════════════════════════════════════════════════════════
cd ../albedo/relightingDataGen-parallel
conda activate sam3

python scripts/run_multi_gpu_batched.py \
    --config config/mvp_config.yaml \
    --csv ../filter_images/ffhq_filtered/train_images.csv \
    --num-gpus 8 \
    --batch-size 8

# ═══════════════════════════════════════════════════════════════
# STEP 3: Train Model (~1.5-2 days for SD1.5)
# ═══════════════════════════════════════════════════════════════
cd ../../training/sd1_5

python convert_to_hf_dataset.py \
    --data_dir /path/to/generated/data \
    --output_dir ./data_hf

./train.sh --data_dir ./data_hf

# ═══════════════════════════════════════════════════════════════
# STEP 4: Inference
# ═══════════════════════════════════════════════════════════════
python inference.py \
    --model_path ./output/instruct-pix2pix-sd15 \
    --input_image test.jpg \
    --instruction "change the lighting to sunset" \
    --output_path result.png
```

---

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 1x 24GB | 8x A100 (80GB) |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 500GB | 2TB+ SSD |

### Per-Component GPU Usage

- **Filter Images**: ~4GB (CLIP model)
- **Data Generation**: ~8-12GB peak (SAM + MiDaS)
- **Training SD 1.5**: ~35-45GB per GPU
- **Training SDXL**: ~60-70GB per GPU

---

## 📚 Documentation

| Component | Documentation |
|-----------|---------------|
| Image Filtering | [`filter_images/README.md`](filter_images/README.md) |
| Data Generation | [`albedo/relightingDataGen-parallel/README.md`](albedo/relightingDataGen-parallel/README.md) |
| Model Training | [`training/README.md`](training/README.md) |
| SD 1.5 Training | [`training/sd1_5/README.md`](training/sd1_5/README.md) |
| SDXL Training | [`training/sdxl/README.md`](training/sdxl/README.md) |

---

## 🔬 Methodology

This pipeline implements the training data generation approach from the **IC-Light paper** (Section 3.1):

1. **Albedo Extraction**: Remove existing lighting from images to get intrinsic reflectance
2. **Degradation Synthesis**: Apply new, varied illumination:
   - **Soft Shading** (40%): Lambertian shading with MiDaS normals
   - **Hard Shadows** (40%): Procedural shadow patterns
   - **Specular Highlights** (20%): Phong specular reflections
3. **Training Pairs**: Create (original, degraded) pairs for instruction-following model training

---

## 📖 References

- **IC-Light Paper**: [ICLR 2024](https://openreview.net/pdf?id=u1cQYxRI1H)
- **InstructPix2Pix**: [arXiv:2211.09800](https://arxiv.org/abs/2211.09800)
- **SAM2/SAM3**: [GitHub](https://github.com/facebookresearch/sam2)
- **CLIP**: [OpenAI](https://github.com/openai/CLIP)
- **HuggingFace Diffusers**: [GitHub](https://github.com/huggingface/diffusers)

---

## 📝 Citation

If you use this pipeline, please cite:

```bibtex
@inproceedings{iclight2024,
  title={IC-Light: Illumination-Conditioned Image Generation},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{brooks2023instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  booktitle={CVPR},
  year={2023}
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is provided for research and educational purposes.

---

**Happy Relighting! 🎨✨**
