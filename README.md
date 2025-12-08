# Image Relighting with Diffusion Models

A complete end-to-end pipeline for training image relighting models based on the IC-Light methodology. This project enables you to:

1. **Filter** high-quality images with good lighting from large datasets
2. **Generate** albedo/degraded images (training pairs)
3. **Caption** images with lighting keywords using VLM
4. **Train** instruction-based image editing models (InstructPix2Pix)

## 🎯 Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           IMAGE RELIGHTING PIPELINE                                  │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  STEP 1              STEP 2                   STEP 3              STEP 4             │
│  ──────              ──────                   ──────              ──────             │
│                                                                                      │
│  filter_images/  →   albedo/                →  edit_keywords/  →   training/         │
│                      relightingDataGen-                                              │
│                      parallel                                                        │
│                                                                                      │
│  ┌────────────┐      ┌──────────────────┐     ┌──────────────┐    ┌─────────────┐   │
│  │ FFHQ 70k   │      │ Filtered Images  │     │ CSV +        │    │ Triplet     │   │
│  │ Images     │  →   │ + Degraded       │  →  │ Keywords     │ →  │ Training    │   │
│  └────────────┘      │ Outputs          │     └──────────────┘    └─────────────┘   │
│        │             └──────────────────┘            │                   │          │
│        ▼                     │                       ▼                   ▼          │
│  ┌────────────┐              ▼               ┌──────────────┐    ┌─────────────┐   │
│  │ CLIP       │      ┌──────────────────┐    │ VLM          │    │ Train       │   │
│  │ Lighting   │      │ • SAM3 Segment   │    │ (Qwen3-VL    │    │ SD1.5/SDXL  │   │
│  │ Filter     │      │ • Albedo Extract │    │  default)    │    │ Model       │   │
│  └────────────┘      │ • Degradation    │    └──────────────┘    └─────────────┘   │
│                      └──────────────────┘                                           │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘

TRAINING DATA MAPPING:
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Training Input  = Degraded Image (flat lighting from Step 2)                       │
│  Instruction     = Lighting Keywords (from Step 3: "sunlight through blinds")       │
│  Training Output = Original Image (real lighting)                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
image-relighting-diffusion/
├── filter_images/                    # Step 1: Image filtering (CLIP-based)
│   ├── filter_lighting_images.py     
│   ├── verify_filtering.py           
│   └── analyze_results.py            
│
├── albedo/                           # Step 2: Training data generation
│   └── relightingDataGen-parallel/   
│       ├── scripts/
│       │   └── run_multi_gpu_batched.py
│       ├── albedo_csv_files/         # Output CSVs saved here
│       └── src/
│           └── stages/               # SAM3, Albedo, Shadow stages
│
├── edit_keywords/                    # Step 3: Lighting keywords generation
│   ├── generate_keywords.py          # VLM-based keyword generation
│   ├── prepare_training_data.py      # Convert to training format
│   └── README.md
│
└── training/                         # Step 4: Model training
    ├── sd1_5/                        # Stable Diffusion 1.5
    ├── sdxl/                         # Stable Diffusion XL
    └── flux/                         # Flux (experimental)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (24GB+ VRAM recommended)
- For Step 3: Either GPU for Qwen3-VL (default, free) or API key for Mistral/OpenAI

---

### Step 1: Filter Images

Select high-quality, well-lit images from your dataset using CLIP-based filtering.

```bash
cd filter_images
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

### Step 2: Generate Albedo/Degraded Images

Process filtered images to create degraded versions (flat lighting) for training pairs.

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
    --csv ../../filter_images/output/train_images.csv \
    --num-gpus 8 \
    --batch-size 8
```

**Output**: 
- Images in `data-train/`
- CSV in `albedo_csv_files/train_images_with_albedo.csv`

📖 See [`albedo/relightingDataGen-parallel/README.md`](albedo/relightingDataGen-parallel/README.md) for details.

---

### Step 3: Generate Lighting Keywords

Use a VLM to generate lighting description keywords for each original image. **Default: Qwen3-VL-30B** (free, runs locally with vLLM).

```bash
cd edit_keywords
pip install -r requirements.txt

# Option 1: Qwen3-VL with vLLM (DEFAULT - free, fast)
python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --batch_size 8

# Option 2: Mistral API
export MISTRAL_API_KEY="your-api-key"
python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --provider mistral
```

**Output**: CSV with 4 columns:
- `image_path` → Original image (becomes training OUTPUT)
- `lighting_score` → CLIP score
- `output_image_path` → Degraded image (becomes training INPUT)
- `lighting_keywords` → Edit instruction (e.g., "sunlight through blinds, indoor")

**Example Keywords Generated**:
| Image | Keywords |
|-------|----------|
| Portrait with window | "sunlight through the blinds, near window blinds" |
| Beach scene | "sunlight from the left side, beach" |
| Forest portrait | "magic golden lit, forest" |
| Night cityscape | "neo punk, city night" |

📖 See [`edit_keywords/README.md`](edit_keywords/README.md) for details.

---

### Step 4: Train the Model

Train an InstructPix2Pix model on your generated data.

```bash
cd training/sd1_5
pip install -r requirements.txt

# Prepare training data
python ../../edit_keywords/prepare_training_data.py \
    --csv ../../edit_keywords/output/train_images_with_albedo_with_keywords.csv \
    --output_dir ./data_triplets

# Convert to HuggingFace dataset
python convert_to_hf_dataset.py --data_dir ./data_triplets --output_dir ./data_hf

# Configure and train
./setup_accelerate.sh
./train.sh --data_dir ./data_hf
```

📖 See [`training/README.md`](training/README.md) for details.

---

## 💡 Complete End-to-End Workflow

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
# STEP 2: Generate Albedo/Degraded Images (~2-4 hours for 10k)
# ═══════════════════════════════════════════════════════════════
cd ../albedo/relightingDataGen-parallel
conda activate sam3

python scripts/run_multi_gpu_batched.py \
    --config config/mvp_config.yaml \
    --csv ../../filter_images/ffhq_filtered/train_images.csv \
    --num-gpus 8 \
    --batch-size 8

# ═══════════════════════════════════════════════════════════════
# STEP 3: Generate Lighting Keywords (~20-30 min with Qwen3-VL)
# ═══════════════════════════════════════════════════════════════
cd ../../edit_keywords

# Default: Qwen3-VL-30B with vLLM (free, fast)
python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --batch_size 8

# Prepare training format
python prepare_training_data.py \
    --csv ./output/train_images_with_albedo_with_keywords.csv \
    --output_dir ../training/sd1_5/data_triplets

# ═══════════════════════════════════════════════════════════════
# STEP 4: Train Model (~1.5-2 days for SD1.5)
# ═══════════════════════════════════════════════════════════════
cd ../training/sd1_5

python convert_to_hf_dataset.py \
    --data_dir ./data_triplets \
    --output_dir ./data_hf

./train.sh --data_dir ./data_hf

# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════
python inference.py \
    --model_path ./output/instruct-pix2pix-sd15 \
    --input_image test.jpg \
    --instruction "sunlight through the blinds, near window" \
    --output_path result.png
```

---

## 📊 Model Comparison

| Model | Quality | Training Time | Resolution | Status |
|-------|---------|---------------|------------|--------|
| **SD 1.5** | Good ⭐⭐⭐ | ~1.5-2 days | 512×512 | ✅ Ready |
| **SDXL** | Excellent ⭐⭐⭐⭐⭐ | ~3-5 days | 1024×1024 | ✅ Ready |
| **Flux** | Best? ⭐⭐⭐⭐⭐⭐ | TBD | 1024×1024 | ⏳ Experimental |

**Recommendation**: Start with **SD 1.5** for rapid prototyping, then scale to **SDXL** for production.

---

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 1x 24GB | 8x A100 (80GB) |
| **RAM** | 32GB | 64GB+ |
| **Storage** | 500GB | 2TB+ SSD |

### Per-Step Resource Usage

| Step | GPU Memory | Time (10k images) |
|------|------------|-------------------|
| 1. Filter Images | ~4GB | ~1-2 hours |
| 2. Generate Albedo | ~8-12GB/GPU | ~2-4 hours (8 GPU) |
| 3. Edit Keywords (Qwen3-VL) | ~40GB (4x24GB TP) | ~20-30 min |
| 4. Training SD1.5 | ~35-45GB/GPU | ~1.5-2 days |

---

## 📚 Documentation

| Component | Documentation |
|-----------|---------------|
| Image Filtering | [`filter_images/README.md`](filter_images/README.md) |
| Albedo Generation | [`albedo/relightingDataGen-parallel/README.md`](albedo/relightingDataGen-parallel/README.md) |
| Keyword Generation | [`edit_keywords/README.md`](edit_keywords/README.md) |
| Model Training | [`training/README.md`](training/README.md) |

---

## 🔬 Methodology

This pipeline implements the training data generation approach from the **IC-Light paper** (Section 3.1):

### Training Data Creation

1. **Original Image** → Has real-world lighting (shadows, highlights, etc.)
2. **Albedo Extraction** → Remove lighting to get flat, uniformly-lit image
3. **Degradation** → Apply synthetic lighting variations
4. **Keywords** → VLM describes the original image's lighting

### Training Objective

The model learns:
> "Given a flat-lit/degraded image + lighting description → Produce realistically lit output"

This is the **inverse** of traditional relighting:
- **Input**: Degraded image (flat lighting)
- **Instruction**: Lighting keywords ("sunlight through blinds")
- **Output**: Original image (with real lighting)

---

## 📖 References

- **IC-Light Paper**: [ICLR 2024](https://openreview.net/pdf?id=u1cQYxRI1H)
- **InstructPix2Pix**: [arXiv:2211.09800](https://arxiv.org/abs/2211.09800)
- **Qwen3-VL**: [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | [GitHub](https://github.com/QwenLM/Qwen3-VL)
- **vLLM**: [Docs](https://docs.vllm.ai/) | [Qwen3-VL Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- **SAM2/SAM3**: [GitHub](https://github.com/facebookresearch/sam2)
- **CLIP**: [OpenAI](https://github.com/openai/CLIP)
- **HuggingFace Diffusers**: [GitHub](https://github.com/huggingface/diffusers)

---

## 📝 Citation

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

## 📄 License

This project is provided for research and educational purposes.

---

**Happy Relighting! 🎨✨**
