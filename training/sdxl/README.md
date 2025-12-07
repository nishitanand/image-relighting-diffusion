# InstructPix2Pix Training on SDXL

**Using Official HuggingFace Diffusers Training Script** ✨

Complete training pipeline for fine-tuning Stable Diffusion XL with InstructPix2Pix on custom image-text-image triplet data, optimized for 8xA100 GPUs.

## 🌟 Why SDXL?

- ✅ **Better Quality**: Significantly improved over SD 1.5
- ✅ **Higher Resolution**: Native 1024×1024 (vs 512×512)
- ✅ **Better Text Understanding**: Improved CLIP encoders
- ✅ **Official Script**: Actively maintained by HuggingFace
- ⚠️ **More Compute**: ~3-5 days training (vs 1.5-2 days for SD 1.5)

## 🚀 Quick Start

### 1. Install & Prepare
```bash
pip install -r requirements.txt
python convert_to_hf_dataset.py --data_dir /path/to/data --output_dir ./data_hf
./setup_accelerate.sh
```

### 2. Train
```bash
./train.sh --data_dir ./data_hf
```

## 📊 Training Details

| Specification | Value |
|---------------|-------|
| **Base Model** | SDXL 1.0 |
| **Hardware** | 8x A100 (80GB) |
| **Training Time** | ~3-5 days (50k samples) |
| **VRAM per GPU** | ~60-70GB |
| **Batch Size** | 32 global (4×8×2 grad accum) |
| **Resolution** | 1024×1024 |
| **Parameters** | ~6.6B |

## 🎯 Inference

```bash
python inference.py \
  --model_path ./output/instruct-pix2pix-sdxl \
  --input_image test.jpg \
  --instruction "turn the sky into sunset" \
  --output_path output.png
```

## 💡 SDXL vs SD 1.5

| Aspect | SD 1.5 | SDXL |
|--------|--------|------|
| Quality | Good | Excellent ⭐ |
| Resolution | 512×512 | 1024×1024 |
| Training Time | ~1.5-2 days | ~3-5 days |
| VRAM/GPU | ~35-45GB | ~60-70GB |
| Batch Size | 8/GPU | 4/GPU |

## 🐛 Troubleshooting

**Out of Memory?**
```bash
./train.sh --train_batch_size 3  # Reduce from 4 to 3
```

**Same data format as SD 1.5** - reuse your converted dataset:
```bash
ln -s ../sd1_5/data_hf ./data_hf
```

See full documentation in SD 1.5 folder - same workflow, just higher quality!

---

**Official Script**: https://github.com/huggingface/diffusers/tree/main/examples/instruct_pix2pix

