#!/bin/bash
# Quick verification script to check if everything is ready

echo "======================================================================"
echo "PRE-FLIGHT CHECKS FOR 8-GPU PIPELINE"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

CHECKS_PASSED=0
CHECKS_FAILED=0

# Check 1: Conda environment
echo "✓ Checking conda environment..."
if command -v conda &> /dev/null; then
    CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    echo "  Current environment: $CURRENT_ENV"
    if [ "$CURRENT_ENV" = "sam3" ]; then
        echo "  ✅ Correct environment (sam3)"
        ((CHECKS_PASSED++))
    else
        echo "  ⚠️  Not in 'sam3' environment. Run: conda activate sam3"
        ((CHECKS_FAILED++))
    fi
else
    echo "  ⚠️  Conda not found"
    ((CHECKS_FAILED++))
fi
echo ""

# Check 2: GPU availability
echo "✓ Checking GPUs..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "  GPUs available: $GPU_COUNT"
    if [ "$GPU_COUNT" -ge 8 ]; then
        echo "  ✅ $GPU_COUNT GPUs detected (need 8)"
        ((CHECKS_PASSED++))
    else
        echo "  ⚠️  Only $GPU_COUNT GPUs detected (need 8)"
        ((CHECKS_FAILED++))
    fi
else
    echo "  ❌ nvidia-smi not found"
    ((CHECKS_FAILED++))
fi
echo ""

# Check 3: Config file
echo "✓ Checking config file..."
if [ -f "config/mvp_config.yaml" ]; then
    echo "  ✅ config/mvp_config.yaml exists"
    ((CHECKS_PASSED++))
else
    echo "  ❌ config/mvp_config.yaml not found"
    ((CHECKS_FAILED++))
fi
echo ""

# Check 4: Input CSVs
echo "✓ Checking input CSVs..."
CSV_DIR="/mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random"
for split in train val test; do
    CSV_FILE="$CSV_DIR/${split}_images.csv"
    if [ -f "$CSV_FILE" ]; then
        LINE_COUNT=$(wc -l < "$CSV_FILE")
        echo "  ✅ ${split}_images.csv exists ($((LINE_COUNT-1)) images)"
        ((CHECKS_PASSED++))
    else
        echo "  ❌ ${split}_images.csv not found"
        ((CHECKS_FAILED++))
    fi
done
echo ""

# Check 5: Python dependencies
echo "✓ Checking Python dependencies..."
python -c "import torch; import PIL; import numpy; import pandas; import yaml" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  ✅ Core Python packages installed"
    ((CHECKS_PASSED++))
else
    echo "  ❌ Missing Python packages"
    ((CHECKS_FAILED++))
fi
echo ""

# Check 6: SAM3 checkpoint
echo "✓ Checking SAM3 checkpoint (optional)..."
if grep -q "checkpoint_path:" config/mvp_config.yaml; then
    CHECKPOINT=$(grep "checkpoint_path:" config/mvp_config.yaml | awk '{print $2}')
    if [ "$CHECKPOINT" != "null" ] && [ -f "$CHECKPOINT" ]; then
        echo "  ✅ SAM3 checkpoint found: $CHECKPOINT"
        ((CHECKS_PASSED++))
    else
        echo "  ⚠️  Will download from HuggingFace (may require authentication)"
        ((CHECKS_PASSED++))
    fi
else
    echo "  ⚠️  No checkpoint path in config"
    ((CHECKS_PASSED++))
fi
echo ""

# Check 7: Disk space
echo "✓ Checking disk space..."
AVAILABLE_GB=$(df -BG "$PROJECT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  Available space: ${AVAILABLE_GB}GB"
if [ "$AVAILABLE_GB" -gt 100 ]; then
    echo "  ✅ Sufficient disk space (need ~50GB for outputs)"
    ((CHECKS_PASSED++))
else
    echo "  ⚠️  Low disk space (may need more)"
    ((CHECKS_FAILED++))
fi
echo ""

# Summary
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo "✅ Checks passed: $CHECKS_PASSED"
echo "❌ Checks failed: $CHECKS_FAILED"
echo ""

if [ "$CHECKS_FAILED" -eq 0 ]; then
    echo "🚀 All checks passed! Ready to run:"
    echo ""
    echo "   cd $PROJECT_DIR"
    echo "   ./scripts/run_all_splits_8gpu.sh"
    echo ""
    echo "Or test with validation set first:"
    echo ""
    echo "   python scripts/run_multi_gpu.py \\"
    echo "       --config config/mvp_config.yaml \\"
    echo "       --csv $CSV_DIR/val_images.csv \\"
    echo "       --num-gpus 8"
    echo ""
else
    echo "⚠️  Some checks failed. Please fix the issues above before running."
    echo ""
fi
echo "======================================================================"

