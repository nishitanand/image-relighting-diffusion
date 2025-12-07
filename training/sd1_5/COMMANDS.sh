#!/bin/bash
# Complete step-by-step commands to train InstructPix2Pix
# Copy and paste these commands one by one

echo "══════════════════════════════════════════════════════════════"
echo "InstructPix2Pix Training - Complete Command List"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# STEP 1: NAVIGATE TO PROJECT
# ============================================================================
cd /mnt/localssd/diffusion/training/sd1_5

# ============================================================================
# STEP 2: INSTALL DEPENDENCIES (First time only)
# ============================================================================
pip install -r requirements.txt

# ============================================================================
# STEP 3: PREPARE YOUR DATA
# ============================================================================
# Create your data directory with structure:
#   my_data/
#     ├── metadata.jsonl
#     ├── input_images/
#     └── output_images/
#
# metadata.jsonl format (one JSON per line):
# {"input_image": "input_images/001.jpg", "instruction": "make sky blue", "output_image": "output_images/001.jpg"}

# ============================================================================
# STEP 4: VALIDATE YOUR DATA (Optional but recommended)
# ============================================================================
python validate_data.py --data_dir /path/to/your/my_data

# ============================================================================
# STEP 5: CONVERT TO HUGGINGFACE FORMAT
# ============================================================================
python convert_to_hf_dataset.py \
  --data_dir /path/to/your/my_data \
  --output_dir ./data_hf

# ============================================================================
# STEP 6: SETUP ACCELERATE (First time only)
# ============================================================================
./setup_accelerate.sh

# ============================================================================
# STEP 7: LAUNCH TRAINING! 🚀
# ============================================================================

# Option A: Basic training (recommended)
./train.sh --data_dir ./data_hf

# Option B: Custom settings
./train.sh \
  --data_dir ./data_hf \
  --output_dir ./output/my_model \
  --train_batch_size 8 \
  --num_epochs 100 \
  --learning_rate 5e-5

# Option C: Resume from checkpoint
./train.sh \
  --data_dir ./data_hf \
  --resume_from_checkpoint ./output/instruct-pix2pix-sd15/checkpoint-5000

# ============================================================================
# STEP 8: MONITOR TRAINING (In another terminal)
# ============================================================================
watch -n 1 nvidia-smi

# Or check logs
tail -f ./output/instruct-pix2pix-sd15/*.log

# ============================================================================
# STEP 9: AFTER TRAINING - RUN INFERENCE 🎨
# ============================================================================

# Basic inference
python inference.py \
  --model_path ./output/instruct-pix2pix-sd15 \
  --input_image test.jpg \
  --instruction "turn the sky into sunset" \
  --output_path output.png

# Advanced inference with tuning
python inference.py \
  --model_path ./output/instruct-pix2pix-sd15 \
  --input_image test.jpg \
  --instruction "turn the sky into sunset" \
  --output_path output.png \
  --num_inference_steps 100 \
  --image_guidance_scale 1.5 \
  --guidance_scale 7.5 \
  --seed 42

# Generate multiple variations
python inference.py \
  --model_path ./output/instruct-pix2pix-sd15 \
  --input_image test.jpg \
  --instruction "make it look like a painting" \
  --output_path variations.png \
  --num_images 4 \
  --seed 42

# ============================================================================
# USEFUL COMMANDS
# ============================================================================

# Check GPU usage
nvidia-smi

# Count your training samples
python -c "from datasets import load_from_disk; ds = load_from_disk('./data_hf'); print(f'Total samples: {len(ds)}')"

# List checkpoints
ls -lh ./output/instruct-pix2pix-sd15/checkpoint-*/

# Check disk space
df -h

# Monitor training progress with WandB
# Go to: https://wandb.ai/ and check your project

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "✅ All commands ready!"
echo "══════════════════════════════════════════════════════════════"

