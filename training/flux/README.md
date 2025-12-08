# Flux InstructPix2Pix Training (Experimental)

## ⚠️ IMPORTANT: Flux Training Not Yet Available

**Flux is very new** (released August 2024) and **official InstructPix2Pix training scripts are not yet available** from HuggingFace or Black Forest Labs.

## 🚀 RECOMMENDED ALTERNATIVES

### ✅ **Option 1: Use SDXL** (Best Current Option)

SDXL provides excellent quality with proven training:

```bash
cd ../sdxl
./train.sh --data_dir ./data_hf
```

**Why SDXL?**
- ✅ Official training scripts
- ✅ Excellent quality (1024×1024)
- ✅ Battle-tested and reliable
- ✅ Training time: ~3-5 days

### ✅ **Option 2: Use SD 1.5** (Fastest)

Start here for rapid prototyping:

```bash
cd ../sd1_5
./train.sh --data_dir ./data_hf
```

**Why SD 1.5?**
- ✅ Fastest training (~1.5-2 days)
- ✅ Proven approach
- ✅ Good quality (512×512)
- ✅ Easy to iterate

### ⏰ **Option 3: Wait for Official Flux Scripts**

Monitor these repositories:
- https://github.com/huggingface/diffusers
- https://github.com/black-forest-labs/flux

## 🔧 Why No Flux InstructPix2Pix Yet?

Flux uses fundamentally different architecture:

| Aspect | SD 1.5 / SDXL | Flux |
|--------|---------------|------|
| **Architecture** | UNet | Transformer |
| **Training** | Diffusion | Flow Matching |
| **Text Encoder** | CLIP | Dual (T5 + CLIP) |
| **Conditioning** | Cross-attention | Different approach |

**Adapting for InstructPix2Pix requires:**
1. Modifying transformer for conditional input images
2. Flow matching with image conditioning
3. Proper handling of dual text encoders
4. Extensive testing and validation

This is non-trivial and requires official implementation.

## 📊 Current Flux Training Status

| Training Type | Status |
|---------------|--------|
| LoRA Fine-tuning | ✅ Available |
| DreamBooth | ✅ Available |
| Text-to-Image | ✅ Available |
| InstructPix2Pix | ❌ Not yet |
| ControlNet-style | ❌ Not yet |

## 💡 What's in This Folder?

This folder contains:
- **Placeholder scripts** - To reserve the structure
- **Documentation** - Explaining the situation
- **Utility scripts** - Data conversion (same as SD 1.5/SDXL)

When official Flux InstructPix2Pix training becomes available, this folder will be updated with proper training scripts.

## 🎯 Recommendation

**For Production Use:**

1. **Prototype with SD 1.5** (~1.5-2 days)
   - Fast iteration
   - Validate your approach

2. **Scale to SDXL** (~3-5 days)
   - Production-quality results
   - 1024×1024 resolution

3. **Monitor Flux updates**
   - Check repositories monthly
   - Migrate when official support arrives

## 📚 Resources

### Official Repositories
- **Flux Model**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **Diffusers**: https://github.com/huggingface/diffusers
- **Black Forest Labs**: https://github.com/black-forest-labs/flux

### Community Implementations
Check https://github.com/topics/flux-ai for community experiments (use with caution - not official)

## 🔮 Future Updates

This folder will be updated when:
1. Official Flux InstructPix2Pix training is released
2. Community has validated working implementations
3. Proper evaluation shows quality improvements over SDXL

**For now, use SDXL for best results!**

---

## 📝 Quick Commands (When Available)

**Placeholder structure** (not functional yet):

```bash
# Install dependencies
pip install -r requirements.txt

# Convert data (same format!)
python convert_to_hf_dataset.py --data_dir /path/to/data --output_dir ./data_hf

# Train (when available)
./train.sh --data_dir ./data_hf
```

---

**Current Status**: ⏳ Waiting for official implementation  
**Recommended Action**: Use SDXL (cd ../sdxl)  
**Last Updated**: December 2024

