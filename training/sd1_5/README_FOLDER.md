# SD 1.5 InstructPix2Pix Training Setup

Complete setup for training InstructPix2Pix on Stable Diffusion 1.5 with your custom 50k image triplet data.

## What's Inside

This folder contains everything needed to train an instruction-based image editing model:

- ✅ **Official HuggingFace training script** (actively maintained)
- ✅ **Data conversion utilities** (your format → HF format)
- ✅ **8xA100 optimized** (global batch size 64)
- ✅ **Easy inference script** (test your trained model)
- ✅ **Complete documentation**

## Quick Links

- **Start Here**: `QUICKREF.md` - One-page quick reference
- **Full Guide**: `README.md` - Complete documentation
- **Data Format**: `data_format_examples.txt` - Example data structures

## Usage

```bash
# 1. Prepare data (see data_format_examples.txt)
# 2. Convert to HF format
python convert_to_hf_dataset.py --data_dir /path/to/data --output_dir ./data_hf

# 3. Train
./train.sh --data_dir ./data_hf

# 4. Inference
python inference.py --model_path ./output/instruct-pix2pix-sd15 --input_image test.jpg --instruction "edit instruction" --output_path output.png
```

See `QUICKREF.md` for quick reference or `README.md` for detailed instructions.

