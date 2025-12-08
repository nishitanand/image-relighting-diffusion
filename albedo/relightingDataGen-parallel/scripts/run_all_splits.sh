#!/bin/bash
# Run pipeline on train/val/test splits with automatic CSV mapping

# Configuration
CONFIG="config/mvp_config.yaml"
BASE_CSV_DIR="/mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random"

# Train CSV
TRAIN_CSV="${BASE_CSV_DIR}/train_images.csv"
# Val CSV  
VAL_CSV="${BASE_CSV_DIR}/val_images.csv"
# Test CSV
TEST_CSV="${BASE_CSV_DIR}/test_images.csv"

# Number of samples (set to null for all)
NUM_SAMPLES_TRAIN=null  # Process all training images
NUM_SAMPLES_VAL=null     # Process all validation images
NUM_SAMPLES_TEST=null    # Process all test images

echo "=========================================="
echo "PARALLEL RELIGHTING PIPELINE"
echo "=========================================="
echo ""

# Function to run pipeline on a split
run_split() {
    local split_name=$1
    local csv_path=$2
    local num_samples=$3
    
    echo "Processing $split_name split..."
    echo "CSV: $csv_path"
    
    if [ "$num_samples" = "null" ]; then
        python scripts/run_pipeline_with_csv_mapping.py \
            --config "$CONFIG" \
            --csv "$csv_path"
    else
        python scripts/run_pipeline_with_csv_mapping.py \
            --config "$CONFIG" \
            --csv "$csv_path" \
            --num-samples "$num_samples"
    fi
    
    echo "$split_name split complete!"
    echo ""
}

# Process each split
run_split "TRAIN" "$TRAIN_CSV" "$NUM_SAMPLES_TRAIN"
run_split "VAL" "$VAL_CSV" "$NUM_SAMPLES_VAL"  
run_split "TEST" "$TEST_CSV" "$NUM_SAMPLES_TEST"

echo "=========================================="
echo "ALL SPLITS PROCESSED!"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  - data-train/"
echo "  - data-val/"
echo "  - data-test/"
echo ""
echo "Updated CSVs (with output_image_path column):"
echo "  - ${BASE_CSV_DIR}/train_images_with_outputs.csv"
echo "  - ${BASE_CSV_DIR}/val_images_with_outputs.csv"
echo "  - ${BASE_CSV_DIR}/test_images_with_outputs.csv"

