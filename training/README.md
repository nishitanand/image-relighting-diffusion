# Text-to-Image Instruction Tuning (T2I) - Complete Training Suite

Training setups for instruction-based image editing models on custom triplet data (input image, instruction, output image).

## 📁 Project Structure

```
diffusion/
├── sd1_5/          ✅ Stable Diffusion 1.5 (READY - Recommended Start)
├── sdxl/           ✅ Stable Diffusion XL (READY - Best Quality)
└── flux/           ⏳ Flux (Experimental - Not Yet Available)
```

## 🚀 Quick Model Comparison

| Model | Quality | Speed | Training Time | Resolution | Status |
|-------|---------|-------|---------------|------------|--------|
| **SD 1.5** | Good ⭐⭐⭐ | Fast | ~1.5-2 days | 512×512 | ✅ Ready |
| **SDXL** | Excellent ⭐⭐⭐⭐⭐ | Medium | ~3-5 days | 1024×1024 | ✅ Ready |
| **Flux** | Best? ⭐⭐⭐⭐⭐⭐ | Slow | ~4-7 days* | 1024×1024 | ⏳ Not Yet |

*Estimated - official training not yet available

## 🎯 Which Model Should You Use?

### ✅ **Start with SD 1.5** - Fastest Iteration
```bash
cd sd1_5
cat START_HERE.txt
```
- **Best for**: Rapid prototyping, testing your data
- **Training time**: ~1.5-2 days on 8xA100
- **Resolution**: 512×512
- **Quality**: Good
- **Script**: Official HuggingFace ✅

### ✅ **Scale to SDXL** - Production Quality
```bash
cd sdxl
cat START_HERE.txt
```
- **Best for**: Production deployment, high quality needs
- **Training time**: ~3-5 days on 8xA100
- **Resolution**: 1024×1024
- **Quality**: Excellent
- **Script**: Official HuggingFace ✅

### ⏳ **Flux** - Future Option
```bash
cd flux
cat START_HERE.txt
```
- **Status**: Official InstructPix2Pix training not yet available
- **Recommendation**: Use SDXL for now
- **Updates**: Monitor HuggingFace Diffusers repository

## 💡 Recommended Workflow

1. **Prototype with SD 1.5** (~1.5-2 days)
   - Validate your data quality
   - Test training pipeline
   - Quick iteration

2. **Scale to SDXL** (~3-5 days)
   - Production-quality results
   - Higher resolution
   - Better text understanding

3. **Future: Migrate to Flux** (when available)
   - State-of-the-art quality
   - Advanced capabilities

## 📊 Detailed Specifications

### SD 1.5 (Proven & Fast)
- **Base Model**: `runwayml/stable-diffusion-v1-5`
- **Parameters**: ~1B
- **Resolution**: 512×512
- **VRAM/GPU**: ~35-45GB
- **Batch Size**: 8 per GPU
- **Global Batch**: 64 (8×8)
- **Training**: ~1.5-2 days
- **Script**: Official HuggingFace ✅

### SDXL (Best Current Quality)
- **Base Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Parameters**: ~6.6B
- **Resolution**: 1024×1024
- **VRAM/GPU**: ~60-70GB
- **Batch Size**: 4 per GPU
- **Global Batch**: 64 (4×8×2)
- **Training**: ~3-5 days
- **Script**: Official HuggingFace ✅

### Flux (Experimental)
- **Base Model**: `black-forest-labs/FLUX.1-dev`
- **Parameters**: ~12B
- **Resolution**: Up to 1024×1024
- **VRAM/GPU**: ~50-70GB (estimated)
- **Training**: Not yet available
- **Script**: Waiting for official release ⏳

## 💾 Your Data Format (Same for All Models!)

All three models use the **same data format**:

Create `my_data/metadata.jsonl`:
```jsonl
{"input_image": "inputs/001.jpg", "instruction": "make sky blue", "output_image": "outputs/001.jpg"}
{"input_image": "inputs/002.jpg", "instruction": "add snow", "output_image": "outputs/002.jpg"}
```

Convert once, use everywhere:
```bash
# In any folder (sd1_5, sdxl, or flux)
python convert_to_hf_dataset.py --data_dir /path/to/my_data --output_dir ./data_hf
```

## 🎨 Training Workflow

### Step 1: Prepare Data (Once)
```bash
cd sd1_5  # Or sdxl
python validate_data.py --data_dir /path/to/my_data
python convert_to_hf_dataset.py --data_dir /path/to/my_data --output_dir ./data_hf
```

### Step 2: Train SD 1.5 (Fast Prototype)
```bash
cd sd1_5
pip install -r requirements.txt
./setup_accelerate.sh
./train.sh --data_dir ./data_hf
# Wait ~1.5-2 days
```

### Step 3: Evaluate & Iterate
```bash
python inference.py \
  --model_path ./output/instruct-pix2pix-sd15 \
  --input_image test.jpg \
  --instruction "edit instruction" \
  --output_path result.png
```

### Step 4: Scale to SDXL (Production)
```bash
cd ../sdxl
ln -s ../sd1_5/data_hf ./data_hf  # Reuse data!
./train.sh --data_dir ./data_hf
# Wait ~3-5 days
```

## 🔧 Hardware Requirements

**Minimum**: 8x A100 (80GB)
- SD 1.5: Uses ~35-45GB per GPU
- SDXL: Uses ~60-70GB per GPU
- Flux: Would use ~50-70GB per GPU (estimated)

**Can I use fewer GPUs?**
- Yes, but adjust batch size accordingly
- Training time will increase proportionally

## 📚 Documentation Structure

Each folder contains:
- `START_HERE.txt` - Quick entry point
- `README.md` - Complete documentation
- `train.sh` - Launch script
- `inference.py` - Test trained model
- `convert_to_hf_dataset.py` - Data conversion
- `validate_data.py` - Data validation

## 🎓 Learning Path

**Week 1: SD 1.5**
- Learn the workflow
- Validate your data quality
- Understand the training process
- Result: Working model in ~2 days

**Week 2: SDXL**
- Scale to higher quality
- Fine-tune hyperparameters
- Production-ready deployment
- Result: High-quality model in ~5 days

**Future: Flux**
- Monitor for official release
- Migrate when ready
- State-of-the-art results

## 🔗 Resources

### Official Repositories
- **HuggingFace Diffusers**: https://github.com/huggingface/diffusers
- **InstructPix2Pix Paper**: https://arxiv.org/abs/2211.09800
- **SDXL Paper**: https://arxiv.org/abs/2307.01952
- **Flux**: https://github.com/black-forest-labs/flux

### Training Scripts
- **SD 1.5**: Official HuggingFace (actively maintained)
- **SDXL**: Official HuggingFace (actively maintained)
- **Flux**: Waiting for official release

## 📊 Training Costs Estimate

On 8xA100 GPUs:
- **SD 1.5**: ~36-48 GPU-hours (~$200-300 on cloud)
- **SDXL**: ~72-120 GPU-hours (~$400-700 on cloud)
- **Flux**: TBD (estimated higher)

## 🎯 Get Started Now!

```bash
# 1. Choose your model
cd sd1_5          # Fast prototyping
# OR
cd sdxl           # Best current quality

# 2. Read the docs
cat START_HERE.txt

# 3. Prepare your data
python convert_to_hf_dataset.py --data_dir /path/to/data --output_dir ./data_hf

# 4. Train!
./train.sh --data_dir ./data_hf
```

## 📝 Quick Reference

| Task | Command |
|------|---------|
| **Start SD 1.5** | `cd sd1_5 && cat START_HERE.txt` |
| **Start SDXL** | `cd sdxl && cat START_HERE.txt` |
| **Check Flux** | `cd flux && cat START_HERE.txt` |
| **Validate data** | `python validate_data.py --data_dir /path/to/data` |
| **Convert data** | `python convert_to_hf_dataset.py --data_dir /path/to/data` |
| **Monitor GPUs** | `watch -n 1 nvidia-smi` |

---

**Current Recommendation**: Start with **SD 1.5** for prototyping, then scale to **SDXL** for production.

**Future**: Monitor for Flux InstructPix2Pix release for state-of-the-art quality.

Happy Training! 🚀
