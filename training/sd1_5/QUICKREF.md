# InstructPix2Pix Training - Quick Reference

## 🎯 What This Is

Train Stable Diffusion 1.5 with InstructPix2Pix on your custom image editing data (50k triplets: input image + instruction + output image).

**Using**: Official HuggingFace Diffusers training script (actively maintained)
**Hardware**: Optimized for 8xA100 GPUs
**Training Time**: ~1.5-2 days for 50k samples

---

## 🚀 3-Step Quick Start

### 1️⃣ Prepare Your Data

Create `my_data/metadata.jsonl` (one JSON per line):
```jsonl
{"input_image": "inputs/001.jpg", "instruction": "make sky blue", "output_image": "outputs/001.jpg"}
{"input_image": "inputs/002.jpg", "instruction": "add snow", "output_image": "outputs/002.jpg"}
```

### 2️⃣ Convert & Validate

```bash
# Validate format
python validate_data.py --data_dir /path/to/my_data

# Convert to HuggingFace format
python convert_to_hf_dataset.py \
  --data_dir /path/to/my_data \
  --output_dir ./data_hf
```

### 3️⃣ Train!

```bash
# One-time setup
pip install -r requirements.txt
./setup_accelerate.sh

# Launch training
./train.sh --data_dir ./data_hf
```

---

## 📊 Key Info

| Item | Value |
|------|-------|
| **Base Model** | Stable Diffusion 1.5 |
| **Batch Size** | 64 global (8 per GPU × 8 GPUs) |
| **Training Time** | 1.5-2 days (50k samples, 100 epochs) |
| **VRAM per GPU** | ~35-45GB |
| **Resolution** | 512×512 |
| **Precision** | FP16 |

---

## 🎨 After Training: Inference

```bash
python inference.py \
  --model_path ./output/instruct-pix2pix-sd15 \
  --input_image photo.jpg \
  --instruction "turn sky into sunset" \
  --output_path result.png \
  --num_inference_steps 50 \
  --image_guidance_scale 1.5 \
  --guidance_scale 7.5
```

**Tune these for quality:**
- `--image_guidance_scale`: 1.0-2.0 (higher = closer to input)
- `--guidance_scale`: 5.0-10.0 (higher = stronger instruction)
- `--num_inference_steps`: 50-100 (more = better quality)

---

## 📁 Files Explained

| File | Purpose |
|------|---------|
| `train_instruct_pix2pix.py` | ⭐ Official HF training script |
| `convert_to_hf_dataset.py` | Convert your data format |
| `validate_data.py` | Check data before training |
| `inference.py` | Run trained model |
| `train.sh` | Launch training on 8 GPUs |
| `quickstart.sh` | Automated setup + training |
| `setup_accelerate.sh` | Configure distributed training |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |

---

## 🔧 Common Commands

**Resume training:**
```bash
./train.sh --data_dir ./data_hf --resume_from_checkpoint ./output/instruct-pix2pix-sd15/checkpoint-5000
```

**Custom settings:**
```bash
./train.sh \
  --data_dir ./data_hf \
  --train_batch_size 6 \
  --num_epochs 150 \
  --learning_rate 1e-5
```

**Monitor GPUs:**
```bash
watch -n 1 nvidia-smi
```

---

## ⚡ Troubleshooting

**Out of Memory?**
```bash
./train.sh --train_batch_size 4  # Reduce from 8 to 4
```

**Dataset error?**
Check: Did you run `convert_to_hf_dataset.py`? Dataset should be at `./data_hf/`

**No xformers?**
```bash
pip install xformers>=0.0.22
```

---

## 📚 Full Details

See `README.md` for:
- Complete documentation
- Parameter tuning guide
- Data format details
- Advanced options
- Troubleshooting

---

## 🎓 What's Next?

After SD 1.5:
1. **SDXL** - Better quality (more compute)
2. **Flux** - State-of-the-art (newest)
3. **Domain fine-tuning** - Specialize on specific tasks
4. **Inference optimization** - LCM, distillation

---

**Made with ❤️ using HuggingFace Diffusers**

Official script: https://github.com/huggingface/diffusers/tree/main/examples/instruct_pix2pix

