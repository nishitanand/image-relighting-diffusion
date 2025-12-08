#!/bin/bash
# Analyze images with individual prompt scores and create visualizations

cd /mnt/localssd/diffusion/filter_images

echo "=========================================="
echo "Individual Prompt Analysis"
echo "=========================================="
echo ""

# Extract image paths from existing CSV
echo "Extracting image paths from all_scores.csv..."
tail -n +2 ffhq_output/all_scores.csv | cut -d',' -f1 > temp_image_list.txt

echo "Found $(wc -l < temp_image_list.txt) images"
echo ""

# Run analysis
echo "Computing individual prompt scores..."
echo "This will take a while (similar to original filtering)..."
echo ""

python analyze_individual_prompts.py \
    --image_list temp_image_list.txt \
    --output_scores ./prompt_analysis/individual_prompt_scores.csv \
    --output_visualizations ./prompt_analysis \
    --batch_size 32 \
    --interval 5000 \
    --images_per_interval 5

# Clean up
rm temp_image_list.txt

echo ""
echo "✅ Complete! Check the outputs:"
echo "   ./prompt_analysis/individual_prompt_scores.csv"
echo "   ./prompt_analysis/prompt_beautiful_lighting.png"
echo "   ./prompt_analysis/prompt_good_lighting.png"
echo "   ./prompt_analysis/prompt_well_lit_face.png"
echo "   ./prompt_analysis/prompt_professional_lighting.png"
echo "   ./prompt_analysis/prompt_natural_light.png"
echo "   ./prompt_analysis/prompt_illumination.png"
echo "   ./prompt_analysis/prompt_bright_and_clear_lighting.png"
echo ""

