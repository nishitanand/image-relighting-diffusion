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

## Installation

```bash
cd edit_keywords
pip install -r requirements.txt

# Set API key for your chosen provider
export MISTRAL_API_KEY="your-key-here"
# OR
export OPENAI_API_KEY="your-key-here"
```

## Quick Start

### Using Mistral (Recommended)

```bash
# Set API key
export MISTRAL_API_KEY="your-mistral-api-key"

# Generate keywords
python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --provider mistral \
    --num_workers 4
```

### Using OpenAI

```bash
export OPENAI_API_KEY="your-openai-api-key"

python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --provider openai \
    --model gpt-4o-mini  # or gpt-4o for better quality
```

### Using Local Model (Qwen-VL)

```bash
# Install additional dependencies
pip install transformers torch qwen-vl-utils accelerate

python generate_keywords.py \
    --csv ../albedo/relightingDataGen-parallel/albedo_csv_files/train_images_with_albedo.csv \
    --output_dir ./output \
    --provider local \
    --model Qwen/Qwen2-VL-7B-Instruct
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
| `--provider` | str | `mistral` | VLM provider: `mistral`, `openai`, or `local` |
| `--model` | str | *auto* | Model name (uses provider default if not specified) |
| `--num_workers` | int | `4` | Parallel workers for API providers |
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

## Provider Comparison

| Provider | Quality | Speed | Cost | GPU Required |
|----------|---------|-------|------|--------------|
| **Mistral (pixtral-large)** | ⭐⭐⭐⭐⭐ | Fast | ~$0.002/image | No |
| **OpenAI (gpt-4o)** | ⭐⭐⭐⭐⭐ | Fast | ~$0.003/image | No |
| **OpenAI (gpt-4o-mini)** | ⭐⭐⭐⭐ | Very Fast | ~$0.0003/image | No |
| **Local (Qwen2-VL-7B)** | ⭐⭐⭐⭐ | Medium | Free | Yes (24GB) |

**Recommendation**: Use Mistral's Pixtral or OpenAI's GPT-4o-mini for best cost/quality ratio.

## Checkpointing

The script automatically saves checkpoints every 100 images. If interrupted, simply re-run the same command to resume from where it left off.

To start fresh (ignore checkpoint):
```bash
python generate_keywords.py --csv input.csv --no_resume
```

## API Key Setup

### Mistral
1. Go to https://console.mistral.ai/
2. Create an API key
3. Set environment variable: `export MISTRAL_API_KEY="your-key"`

### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create an API key
3. Set environment variable: `export OPENAI_API_KEY="your-key"`

## Troubleshooting

### Rate Limiting
If you hit rate limits, reduce `--num_workers`:
```bash
python generate_keywords.py --csv input.csv --num_workers 2
```

### Out of Memory (Local Models)
Use a smaller model or reduce batch size:
```bash
python generate_keywords.py --csv input.csv --provider local --model Qwen/Qwen2-VL-2B-Instruct
```

### API Errors
The script automatically retries failed requests 3 times. Check your API key and network connection if errors persist.

## Next Steps

After generating keywords, proceed to Step 4 (Training):

```bash
cd ../training/sd1_5

# Prepare data using the keywords CSV
python convert_to_hf_dataset.py \
    --csv ../edit_keywords/output/train_images_with_albedo_with_keywords.csv \
    --output_dir ./data_hf
```

See [`training/README.md`](../training/README.md) for detailed training instructions.

---

**Step 3 Complete! Your data now has lighting keywords for training.** 🎨

