#!/bin/bash
# Quick start script - Run this after preparing your data

echo "=================================================="
echo "  InstructPix2Pix Training - Quick Start"
echo "=================================================="
echo ""

# Check if virtual environment exists, if not suggest creating one
if [ -z "$VIRTUAL_ENV" ]; then
    echo "💡 Tip: Consider using a virtual environment:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
fi

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
echo "------------------------------------"
pip install -q -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo "✅ Dependencies installed"
echo ""

# Step 2: Check for data
echo "Step 2: Checking for data..."
echo "------------------------------------"
if [ ! -d "data_hf" ]; then
    echo "⚠️  No converted dataset found at ./data_hf"
    echo ""
    echo "Please prepare your data:"
    echo "  1. Create your metadata.jsonl with triplets"
    echo "  2. Run: python validate_data.py --data_dir /path/to/your/data"
    echo "  3. Run: python convert_to_hf_dataset.py --data_dir /path/to/your/data --output_dir ./data_hf"
    echo ""
    echo "See data_format_examples.txt for the expected format"
    exit 1
fi
echo "✅ Found dataset at ./data_hf"
echo ""

# Step 3: Setup accelerate
echo "Step 3: Configuring accelerate..."
echo "------------------------------------"
if [ ! -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ] && [ ! -f "accelerate_config.yaml" ]; then
    echo "Setting up accelerate for 8 GPUs..."
    ./setup_accelerate.sh
else
    echo "✅ Accelerate already configured"
fi
echo ""

# Step 4: Launch training
echo "Step 4: Launching training..."
echo "------------------------------------"
echo "Starting training on 8xA100 GPUs with:"
echo "  - Batch size: 8 per GPU (64 global)"
echo "  - Epochs: 100"
echo "  - Learning rate: 5e-5"
echo "  - Mixed precision: fp16"
echo ""
echo "Monitor training with: watch -n 1 nvidia-smi"
echo "=================================================="
echo ""

# Make train.sh executable
chmod +x train.sh

# Launch training
./train.sh --data_dir ./data_hf

echo ""
echo "=================================================="
echo "Training complete!"
echo "Find your model at: ./output/instruct-pix2pix-sd15/"
echo ""
echo "Run inference with:"
echo "  python inference.py \\"
echo "    --model_path ./output/instruct-pix2pix-sd15 \\"
echo "    --input_image test.jpg \\"
echo "    --instruction 'your edit instruction' \\"
echo "    --output_path output.png"
echo "=================================================="

