# InstructPix2Pix Training on Custom Data (SD 1.5)

**Using Official HuggingFace Diffusers Training Script** ✨

Complete training pipeline for fine-tuning Stable Diffusion 1.5 with InstructPix2Pix on custom image-text-image triplet data, optimized for 8xA100 GPUs.

## 🌟 Why This Setup?

- ✅ **Official HuggingFace Script**: Actively maintained, battle-tested
- ✅ **Production Ready**: Used by thousands of researchers
- ✅ **Latest Features**: Always up-to-date with Diffusers library
- ✅ **8xA100 Optimized**: ~1.5-2 days training time for 50k samples

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Your data should be in triplet format: (input_image, instruction, output_image)

Create a `metadata.jsonl` file (one JSON object per line):

```jsonl
{"input_image": "input_images/img_001.jpg", "instruction": "Make the sky blue", "output_image": "output_images/img_001.jpg"}
{"input_image": "input_images/img_002.jpg", "instruction": "Add snow", "output_image": "output_images/img_002.jpg"}
```

**Directory structure:**
```
my_data/
  ├── metadata.jsonl
  ├── input_images/
  │   ├── img_001.jpg
  │   ├── img_002.jpg
  │   └── ...
  └── output_images/
      ├── img_001.jpg
      ├── img_002.jpg
      └── ...
```

See `data_format_examples.txt` for more examples.

### 3. Validate Your Data (Optional but Recommended)

```bash
python validate_data.py --data_dir /path/to/my_data
```

### 4. Convert to HuggingFace Dataset Format

The official script expects data in HuggingFace Dataset format:

```bash
python convert_to_hf_dataset.py \
  --data_dir /path/to/my_data \
  --output_dir ./data_hf
```

This will convert your metadata.jsonl into the format expected by the training script.

### 5. Setup Accelerate (First Time Only)

```bash
chmod +x setup_accelerate.sh
./setup_accelerate.sh
```

Or manually configure:
```bash
accelerate config
```
Select: Multi-GPU, 8 processes, fp16 mixed precision

### 6. Launch Training! 🚀

```bash
chmod +x train.sh
./train.sh --data_dir ./data_hf
```

Or with custom parameters:
```bash
./train.sh \
  --data_dir ./data_hf \
  --output_dir ./output/my_model \
  --train_batch_size 8 \
  --num_epochs 100 \
  --learning_rate 5e-5
```

## 📊 Training Details

### Hardware & Performance
- **GPUs**: 8x A100 (80GB)
- **VRAM per GPU**: ~35-45GB during training
- **Training Time**: ~1.5-2 days for 50k samples (100 epochs)
- **Global Batch Size**: 64 (8 GPUs × 8 per GPU)

### Model Details
- **Base Model**: Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`)
- **Architecture**: Modified UNet with 8-channel input
  - 4 channels: input image latents
  - 4 channels: noisy output image latents
- **Resolution**: 512×512
- **Mixed Precision**: FP16

### Key Features
- ✅ Official HuggingFace implementation
- ✅ Distributed training with Accelerate
- ✅ Memory-efficient attention (xformers)
- ✅ Gradient checkpointing
- ✅ Automatic checkpoint management
- ✅ WandB logging
- ✅ Validation during training

## 🎯 Inference

After training, use the inference script:

```bash
python inference.py \
  --model_path ./output/instruct-pix2pix-sd15 \
  --input_image test.jpg \
  --instruction "turn the sky into sunset" \
  --output_path output.png
```

### Inference Parameters

```bash
python inference.py \
  --model_path ./output/instruct-pix2pix-sd15 \
  --input_image input.jpg \
  --instruction "your editing instruction" \
  --output_path output.png \
  --num_inference_steps 50 \
  --image_guidance_scale 1.5 \
  --guidance_scale 7.5 \
  --seed 42 \
  --num_images 1
```

**Parameter Guide:**
- `--num_inference_steps`: More steps = better quality (default: 50)
- `--image_guidance_scale`: How much to follow input image (default: 1.5)
  - Lower (1.0-1.3): More creative edits
  - Higher (1.5-2.0): Closer to input image
- `--guidance_scale`: How much to follow instruction (default: 7.5)
  - Lower (5.0-7.0): Subtle changes
  - Higher (7.5-10.0): Stronger instruction following
- `--seed`: For reproducible results
- `--num_images`: Generate multiple variations

## 📁 Project Structure

```
sd1_5/
├── train_instruct_pix2pix.py    # Official HuggingFace training script
├── convert_to_hf_dataset.py     # Convert your data to HF format
├── validate_data.py              # Validate your data format
├── inference.py                  # Run inference on trained model
├── train.sh                      # Launch script for 8 GPUs
├── setup_accelerate.sh           # Accelerate configuration helper
├── requirements.txt              # Python dependencies
├── data_format_examples.txt      # Example data formats
├── README.md                     # This file
├── data_hf/                      # Converted HuggingFace dataset (you create)
└── output/                       # Training outputs (auto-created)
    └── instruct-pix2pix-sd15/
        ├── checkpoint-1000/
        ├── checkpoint-2000/
        └── ...
```

## 🔧 Training Script Options

The `train.sh` script supports these arguments:

```bash
./train.sh \
  --data_dir ./data_hf \                    # HF dataset directory (required)
  --output_dir ./output/my_model \          # Output directory
  --train_batch_size 8 \                     # Batch size per GPU
  --num_epochs 100 \                         # Number of epochs
  --learning_rate 5e-5 \                     # Learning rate
  --resolution 512 \                         # Image resolution
  --checkpointing_steps 1000 \              # Save checkpoint every N steps
  --resume_from_checkpoint ./checkpoint \    # Resume from checkpoint
  --seed 42                                  # Random seed
```

## 📈 Monitoring Training

### WandB (Recommended)

1. Install and login:
```bash
pip install wandb
wandb login
```

2. Training will automatically log to WandB (configured in `train.sh`)

### Check Progress

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

Check training logs:
```bash
tail -f output/instruct-pix2pix-sd15/logs/training.log
```

## 🔄 Resuming Training

If training is interrupted:

```bash
./train.sh \
  --data_dir ./data_hf \
  --resume_from_checkpoint ./output/instruct-pix2pix-sd15/checkpoint-5000
```

Or resume from the latest checkpoint automatically by checking the output directory.

## 💡 Tips & Tricks

### Data Quality
- Ensure input/output images are well-aligned
- Instructions should be clear and specific
- Diverse instruction types lead to better generalization

### Hyperparameter Tuning
- Start with default settings (tested on original InstructPix2Pix)
- If overfitting: reduce `num_epochs` or increase `conditioning_dropout_prob`
- If underfitting: increase `num_epochs` or `learning_rate`

### Memory Optimization
If you encounter OOM:
1. Reduce `--train_batch_size` (e.g., to 4 or 6)
2. Increase `--gradient_accumulation_steps` to maintain effective batch size
3. Already enabled: gradient checkpointing, xformers

### Speed Optimization
- Already optimized with TF32, xformers, fp16
- Ensure `num_workers` in dataloader is optimal (default: 4)
- Use local SSD for dataset (faster I/O)

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
```bash
./train.sh --train_batch_size 4  # Reduce batch size
```

### Error: "Dataset not found"
Make sure you ran:
```bash
python convert_to_hf_dataset.py --data_dir /path/to/data --output_dir ./data_hf
```

### Error: "No module named 'xformers'"
```bash
pip install xformers>=0.0.22
```

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Ensure all 8 GPUs are being used
- Check if data loading is bottleneck (reduce/increase `num_workers`)

## 📚 Data Format Details

### Your Original Format
```jsonl
{"input_image": "path/to/input.jpg", "instruction": "edit instruction", "output_image": "path/to/output.jpg"}
```

### After Conversion (HuggingFace Dataset)
The conversion script (`convert_to_hf_dataset.py`) transforms this into:
```python
{
  'original_image': PIL.Image,      # Your input_image
  'edit_prompt': str,               # Your instruction
  'edited_image': PIL.Image,        # Your output_image
}
```

The official training script expects these exact column names: `original_image`, `edit_prompt`, `edited_image`.

## 🎓 Next Steps

After successfully training on SD 1.5:

1. **Evaluate Results**: Test on held-out images
2. **Fine-tune Further**: Continue training on specific domains
3. **Scale to SDXL**: Higher quality (requires more compute)
4. **Try Flux**: State-of-the-art quality
5. **Optimize Inference**: LCM, distillation for faster inference

## 📖 References

- **Official HF Script**: https://github.com/huggingface/diffusers/tree/main/examples/instruct_pix2pix
- **Diffusers Docs**: https://huggingface.co/docs/diffusers

## 📄 License

- Training code: Apache 2.0 (HuggingFace Diffusers)
- Stable Diffusion 1.5: CreativeML Open RAIL-M
- Your trained models inherit the base model's license

---

**Happy Training! 🚀**

Questions? Check the [HuggingFace Diffusers documentation](https://huggingface.co/docs/diffusers) or the troubleshooting section above.
