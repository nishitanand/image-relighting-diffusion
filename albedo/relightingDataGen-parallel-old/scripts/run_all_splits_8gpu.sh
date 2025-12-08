#!/bin/bash
# Run multi-GPU pipeline on all splits (train, val, test)

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project directory
cd "$PROJECT_DIR"

echo "======================================================================"
echo "MULTI-GPU PARALLEL PROCESSING - ALL SPLITS"
echo "======================================================================"
echo "Working directory: $PROJECT_DIR"
echo "Using 8 GPUs"
echo "======================================================================"
echo ""

# Configuration
CONFIG="config/mvp_config.yaml"
NUM_GPUS=8
CSV_DIR="/mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random"

# Process training set (10,000 images)
echo "▶ Processing TRAINING set (10,000 images)..."
echo "   Expected time: ~2 hours"
echo ""
python scripts/run_multi_gpu.py \
    --config "$CONFIG" \
    --csv "$CSV_DIR/train_images.csv" \
    --num-gpus "$NUM_GPUS"

if [ $? -ne 0 ]; then
    echo "❌ Training set failed!"
    exit 1
fi

echo ""
echo "✅ Training set complete!"
echo ""
echo "======================================================================"
echo ""

# Process validation set (1,000 images)
echo "▶ Processing VALIDATION set (1,000 images)..."
echo "   Expected time: ~12 minutes"
echo ""
python scripts/run_multi_gpu.py \
    --config "$CONFIG" \
    --csv "$CSV_DIR/val_images.csv" \
    --num-gpus "$NUM_GPUS"

if [ $? -ne 0 ]; then
    echo "❌ Validation set failed!"
    exit 1
fi

echo ""
echo "✅ Validation set complete!"
echo ""
echo "======================================================================"
echo ""

# Process test set (1,000 images)
echo "▶ Processing TEST set (1,000 images)..."
echo "   Expected time: ~12 minutes"
echo ""
python scripts/run_multi_gpu.py \
    --config "$CONFIG" \
    --csv "$CSV_DIR/test_images.csv" \
    --num-gpus "$NUM_GPUS"

if [ $? -ne 0 ]; then
    echo "❌ Test set failed!"
    exit 1
fi

echo ""
echo "✅ Test set complete!"
echo ""
echo "======================================================================"
echo "✅ ALL SPLITS PROCESSED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Output directories:"
echo "  - $PROJECT_DIR/data-train/ (10,000 images)"
echo "  - $PROJECT_DIR/data-val/ (1,000 images)"
echo "  - $PROJECT_DIR/data-test/ (1,000 images)"
echo ""
echo "Updated CSVs with output paths:"
echo "  - $CSV_DIR/train_images_with_relighting_outputs.csv"
echo "  - $CSV_DIR/val_images_with_relighting_outputs.csv"
echo "  - $CSV_DIR/test_images_with_relighting_outputs.csv"
echo ""
echo "Total processing time: ~2.5 hours (vs 70 hours sequential)"
echo "Speedup: ~28x"
echo "======================================================================"


