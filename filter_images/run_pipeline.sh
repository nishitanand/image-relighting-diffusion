#!/bin/bash
# Example: Run the FFHQ filtering pipeline

# Configuration
DATASET_PATH="/mnt/localssd/diffusion/filter_images/ffhq_github/ffhq-dataset/images1024x1024"
OUTPUT_DIR="./ffhq_filtered_output"
NUM_IMAGES=50000
BATCH_SIZE=64

echo "=========================================="
echo "FFHQ Image Filtering with CLIP"
echo "=========================================="
echo ""
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Selecting: $NUM_IMAGES out of 70,000 images"
echo ""

# Step 1: Filter images based on lighting quality
echo "Step 1: Filtering images with CLIP..."
python filter_lighting_images.py \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_images $NUM_IMAGES \
    --batch_size $BATCH_SIZE \
    --model_name ViT-B/32

echo ""
echo "Step 2: Verifying filter quality..."
python verify_filtering.py \
    --results "$OUTPUT_DIR/all_scores.csv" \
    --output_dir "$OUTPUT_DIR/verification" \
    --cutoff $NUM_IMAGES

echo ""
echo "Step 3: Creating analysis and splits..."
python analyze_results.py \
    --results_json "$OUTPUT_DIR/filtered_images.json" \
    --output_dir "$OUTPUT_DIR/analysis" \
    --create_grid \
    --create_splits

echo ""
echo "=========================================="
echo "✓ Pipeline complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - Filtered list: $OUTPUT_DIR/filtered_images.txt"
echo "  - With scores: $OUTPUT_DIR/filtered_images.json"
echo "  - All scores: $OUTPUT_DIR/all_scores.csv"
echo "  - Verification: $OUTPUT_DIR/verification/"
echo "  - Analysis: $OUTPUT_DIR/analysis/"
echo ""

