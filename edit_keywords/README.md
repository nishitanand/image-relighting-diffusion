# Step 3: Edit Keywords Generation

Generate lighting description keywords for images using Vision-Language Models (VLM).

## Overview

This step takes the CSV output from Step 2 (relightingDataGen-parallel) and generates lighting/environment description keywords for each original image. These keywords will be used as **edit instructions** during training.

### Pipeline Flow

```
Step 2 Output CSV:                    Step 3 Output CSV:
┌─────────────────────────┐           ┌─────────────────────────────────┐
│ image_path              │           │ image_path                      │
│ lighting_score          │    →      │ lighting_score                  │
│ output_image_path       │           │ output_image_path               │
└─────────────────────────┘           │ lighting_keywords  ← NEW!       │
                                      └─────────────────────────────────┘
```

### Training Data Mapping

After Step 3, the data is ready for training with this mapping:

| Training Role | CSV Column | Description |
|---------------|------------|-------------|
| **Input Image** | `output_image_path` | Albedo degraded image (from Step 2) |
| **Instruction** | `lighting_keywords` | VLM-generated lighting description |
| **Output Image** | `image_path` | Original image (with real lighting) |

This teaches the model: "Given a flat-lit image, apply lighting described by the keywords to produce the output."

## VLM Providers

| Provider | Model | Cost | Speed | Quality | GPU Required |
|----------|-------|------|-------|---------|--------------|
| **qwen3vl** (DEFAULT) | Qwen3-VL-30B | Free | ⚡ Fast | ⭐⭐⭐⭐⭐ | Yes (40GB+) |
| **qwen3vl-server** | Qwen3-VL-30B | Free | ⚡⚡ Fastest | ⭐⭐⭐⭐⭐ | Yes (server) |
| **mistral** | Pixtral-large | ~$0.002/img | Fast | ⭐⭐⭐⭐⭐ | No |
| **openai** | GPT-4o | ~$0.003/img | Fast | ⭐⭐⭐⭐⭐ | No |

**Recommendation**: Use **qwen3vl** (default) for best quality at no cost!

## Installation

```bash
cd edit_keywords
pip install -r requirements.txt
```

### For Qwen3-VL (Default)

```bash
# Install vLLM and dependencies
pip install vllm>=0.8.0 qwen-vl-utils>=0.0.14 transformers>=4.57.0

# Optional: Install flash-attention for faster inference
pip install flash-attn --no-build-isolation
```

## Quick Start

### Option 1: Qwen3-VL with vLLM (Recommended - Default)

The simplest and most cost-effective option. Runs locally with vLLM for fast inference.

```bash
# Just run it - Qwen3-VL is the default!
python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --batch_size 8
```

For multi-GPU:
```bash
python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --tensor_parallel_size 4 \
    --batch_size 16
```

### Option 2: vLLM Server Mode (Best for Large Datasets)

Start a vLLM server separately for maximum throughput:

**Terminal 1 - Start Server:**
```bash
# Start vLLM server with Qwen3-VL-30B
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data \
    --gpu-memory-utilization 0.9 \
    --port 8000

# Or use the helper script
./start_vllm_server.sh
```

**Terminal 2 - Run Generation:**
```bash
python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --provider qwen3vl-server \
    --vllm_url http://localhost:8000/v1 \
    --num_workers 8
```

### Option 3: Mistral API

```bash
export MISTRAL_API_KEY="your-mistral-api-key"

python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --provider mistral \
    --num_workers 4
```

### Option 4: OpenAI API

```bash
export OPENAI_API_KEY="your-openai-api-key"

python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --provider openai \
    --model gpt-4o-mini  # or gpt-4o for better quality
```

## Example Keywords Generated

The VLM generates short, descriptive phrases like:

| Image Type | Generated Keywords |
|------------|-------------------|
| Portrait with window light | "sunlight through the blinds, near window blinds" |
| Beach portrait | "sunlight from the left side, beach" |
| Forest scene | "magic golden lit, forest" |
| Night cityscape | "neo punk, city night" |
| Studio portrait | "soft studio lighting, neutral background" |
| Moody portrait | "dramatic side lighting, dark moody" |
| Outdoor daytime | "natural daylight, overcast sky" |
| Cyberpunk style | "neon pink and blue, cyberpunk city" |

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv` | str | *required* | Path to input CSV from Step 2 |
| `--output_dir` | str | `./output` | Directory to save output CSV |
| `--provider` | str | `qwen3vl` | VLM provider: `qwen3vl`, `qwen3vl-server`, `mistral`, `openai` |
| `--model` | str | *auto* | Model name (uses provider default if not specified) |
| `--batch_size` | int | `8` | Batch size for qwen3vl |
| `--num_workers` | int | `4` | Parallel workers for API providers |
| `--vllm_url` | str | - | vLLM server URL (for qwen3vl-server) |
| `--tensor_parallel_size` | int | *auto* | Number of GPUs for tensor parallelism |
| `--no_resume` | flag | - | Don't resume from checkpoint |

## Output

### Output CSV Format

```csv
image_path,lighting_score,output_image_path,lighting_keywords
/path/to/original/001.png,0.342,/path/to/degraded/001_output.png,"soft natural lighting, indoor"
/path/to/original/002.png,0.318,/path/to/degraded/002_output.png,"dramatic sunset, golden hour"
...
```

### Output Location

```
edit_keywords/
└── output/
    ├── train_images_with_albedo_with_keywords.csv
    ├── val_images_with_albedo_with_keywords.csv
    └── test_images_with_albedo_with_keywords.csv
```

## Hardware Requirements

### For Qwen3-VL-30B (Default)

| Configuration | GPU Memory | Speed |
|---------------|------------|-------|
| Single GPU | 40GB+ (A100) | ~2-3 img/s |
| 2x GPU (TP=2) | 2x 24GB | ~4-5 img/s |
| 4x GPU (TP=4) | 4x 24GB | ~8-10 img/s |
| 8x GPU (TP=8) | 8x 24GB | ~15-20 img/s |

### Estimated Processing Time (10k images)

| Provider | Time |
|----------|------|
| qwen3vl (4 GPU) | ~20-30 min |
| qwen3vl-server (4 GPU) | ~15-20 min |
| mistral API | ~30-45 min |
| openai API | ~30-45 min |

## Checkpointing

The script automatically saves checkpoints every 100 images. If interrupted, simply re-run the same command to resume from where it left off.

To start fresh (ignore checkpoint):
```bash
python generate_keywords.py --csv input.csv --no_resume
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use more GPUs:
```bash
python generate_keywords.py --csv input.csv --batch_size 4 --tensor_parallel_size 4
```

### vLLM Server Connection Error

Make sure the server is running:
```bash
# Check if server is up
curl http://localhost:8000/health

# Check server logs for errors
```

### Slow Processing

1. Use vLLM server mode for maximum throughput
2. Increase batch size if you have more GPU memory
3. Use more GPUs with tensor parallelism

## References

- **Qwen3-VL**: [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- **vLLM Qwen3-VL Guide**: [vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- **Qwen3-VL GitHub**: [GitHub](https://github.com/QwenLM/Qwen3-VL)

## Next Steps

After generating keywords, proceed to Step 4 (Training):

```bash
cd ../training/sd1_5

# Prepare data using the keywords CSV
python ../edit_keywords/prepare_training_data.py \
    --csv ../edit_keywords/output/train_images_with_albedo_with_keywords.csv \
    --output_dir ./data_triplets

python convert_to_hf_dataset.py \
    --data_dir ./data_triplets \
    --output_dir ./data_hf
```

See [`training/README.md`](../training/README.md) for detailed training instructions.

---

**Step 3 Complete! Your data now has lighting keywords for training.** 🎨
