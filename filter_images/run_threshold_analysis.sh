#!/bin/bash
# Run threshold analysis to find optimal cutoff point

cd /mnt/localssd/diffusion/filter_images

echo "=========================================="
echo "Threshold Analysis"
echo "=========================================="
echo ""
echo "This will show you images at regular intervals"
echo "to help determine the optimal filtering threshold"
echo ""

python threshold_analysis.py \
    --results ./ffhq_output/all_scores.csv \
    --output_dir ./threshold_analysis \
    --interval 5000 \
    --images_per_interval 5

echo ""
echo "✅ Complete! Check the output:"
echo "   ./threshold_analysis/threshold_analysis_5000interval.png"
echo "   ./threshold_analysis/score_ranges_summary.csv"
echo ""

