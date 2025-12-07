# Verification Script Output Guide

## What `verify_filtering.py` Shows

When you run the verification script, it provides comprehensive analysis of your filtering results.

## Console Output

### 1. Top 10 Highest Scores (Best Lighting)
```
📊 TOP 10 HIGHEST SCORES:
  #    1: 0.3856 - 12345.png
  #    2: 0.3821 - 45678.png
  ...
```
These should be professionally lit, bright, clear images.

### 2. Around 50k Cutoff
```
✂️  AT 50K CUTOFF (image #50000):
  #50000: 0.2847 - 49999.png

  Around the 50k cutoff:
  #49995: 0.2851 ✓ SELECTED
  #49996: 0.2850 ✓ SELECTED
  #49997: 0.2849 ✓ SELECTED
  #49998: 0.2848 ✓ SELECTED
  #49999: 0.2847 ✓ SELECTED
  #50000: 0.2847 ✓ SELECTED  ← Last selected
  #50001: 0.2845 ✗ REJECTED  ← First rejected
  #50002: 0.2844 ✗ REJECTED
  #50003: 0.2843 ✗ REJECTED
```

### 3. Bottom 10 Overall (Rejected - Worst Lighting)
```
📉 BOTTOM 10 LOWEST SCORES (ALL DATA):
  #69991: 0.1567 - 69990.png
  #69992: 0.1554 - 69991.png
  ...
  #70000: 0.1234 - 69999.png
```
These should be very dark, underlit, poor quality images.

### 4. Bottom 20 from Filtered Set (NEW!)
```
⚠️  BOTTOM 20 FROM FILTERED SET (Worst of the 50k selected):
  #49981: 0.2852 ✓ SELECTED - 34567.png
  #49982: 0.2851 ✓ SELECTED - 23456.png
  ...
  #49999: 0.2847 ✓ SELECTED - 12345.png
  #50000: 0.2847 ✓ SELECTED - 67890.png
```
**Important**: These are the 20 worst images that MADE IT into your filtered set.
- They should still have acceptable lighting (just not as good as top images)
- If these look terrible, you may want to select fewer images (e.g., 40k instead of 50k)
- If these still look good, your cutoff is conservative

### 5. 20 Random Samples from Filtered Set (NEW!)
```
🎲 20 RANDOM SAMPLES FROM FILTERED SET:
  #  1234: 0.3245 - 01234.png
  #  5678: 0.3123 - 05678.png
  # 12345: 0.3056 - 12345.png
  # 23456: 0.2989 - 23456.png
  # 34567: 0.2912 - 34567.png
  ...
```
**Important**: These show the typical quality of your filtered dataset.
- Gives you a realistic sense of what most images look like
- Should all have decent to good lighting
- More representative than just looking at top 10

## Visual Outputs

### 1. `filtering_verification.png`
Multi-section grid showing:
- Top 10 (green titles - best)
- 20 random samples (blue titles - typical)
- Bottom 20 from filtered (orange titles - worst selected)
- Around cutoff (orange/red - boundary)
- Bottom 10 overall (red titles - rejected)

### 2. `bottom_20_filtered.png` (NEW!)
2 rows × 10 columns grid showing images #49,980 to #50,000
- All 20 worst images that made it into your filtered set
- Orange titles with rank and score
- Critical for quality control!

**What to look for:**
- ✅ Still have visible faces with some lighting
- ✅ Not completely dark or blacked out
- ✅ Acceptable for training data
- ❌ If these look bad, reduce `--num_images` to 40k or 45k

### 3. `random_20_filtered.png` (NEW!)
2 rows × 10 columns grid showing 20 random samples
- Blue titles with rank and score
- Shows typical quality you'll get in training
- Better indicator of overall dataset quality than top/bottom

**What to look for:**
- ✅ Consistent lighting quality across samples
- ✅ Good variety of lighting conditions
- ✅ All images usable for training
- ❌ If quality varies too much, consider more aggressive filtering

### 4. `top_vs_bottom_comparison.png`
2 rows: Top 10 vs Bottom 10 overall
- Direct visual comparison
- Top row: best lighting (green)
- Bottom row: worst lighting (red)

## Score Statistics

```
CUTOFF ANALYSIS at 50000
======================================================================

SELECTED (top 50000):
  Score range: 0.2847 to 0.3856
  Mean score: 0.3201
  Median score: 0.3198

REJECTED (bottom 20000):
  Score range: 0.1234 to 0.2846
  Mean score: 0.2103
  Median score: 0.2099

Score gap at cutoff: 0.0001
  ✓ Clean separation between selected and rejected images
```

## Interpretation Guide

### Scenario 1: Good Filtering ✅
```
Top 10: 0.35-0.40 (bright, professional)
Random 20: 0.28-0.35 (consistently good)
Bottom 20 filtered: 0.27-0.29 (still acceptable)
Bottom 10 overall: 0.12-0.20 (dark, poor)
```
**Action**: Proceed with 50k images

### Scenario 2: Too Aggressive ⚠️
```
Top 10: 0.38-0.42 (excellent)
Random 20: 0.32-0.38 (very good)
Bottom 20 filtered: 0.31-0.33 (still very good)
Bottom 10 overall: 0.20-0.30 (acceptable but rejected)
```
**Action**: You could select more images (60k instead of 50k)

### Scenario 3: Too Lenient ❌
```
Top 10: 0.35-0.40 (good)
Random 20: 0.22-0.35 (highly variable)
Bottom 20 filtered: 0.18-0.22 (poor quality)
Bottom 10 overall: 0.10-0.17 (very poor)
```
**Action**: Select fewer images (40k instead of 50k)

### Scenario 4: Dataset Issue 🚨
```
Top 10: 0.28-0.32 (mediocre)
Random 20: 0.20-0.28 (poor)
Bottom 20 filtered: 0.18-0.20 (very poor)
Bottom 10 overall: 0.10-0.17 (terrible)
```
**Action**: Dataset may not have enough well-lit images, or CLIP prompts need adjustment

## Key Questions to Answer

### After viewing bottom 20 filtered images:
1. Would you be happy training a model on these images?
2. Do they still show clear faces with some lighting?
3. Are they significantly better than the rejected images?

### After viewing random 20 samples:
1. Is the quality consistent across the random samples?
2. Do most images have good lighting quality?
3. Is there good variety in lighting conditions?

### After viewing all outputs:
1. Is there clear visual difference between selected and rejected?
2. Does the cutoff make sense based on visual inspection?
3. Should you adjust the number of selected images?

## Example Usage

```bash
# Run verification
python verify_filtering.py \
    --results ./ffhq_output/all_scores.csv \
    --output_dir ./ffhq_verification \
    --cutoff 50000

# Check the outputs
ls ./ffhq_verification/
# - filtering_verification.png      (multi-section overview)
# - bottom_20_filtered.png          (worst selected - quality check!)
# - random_20_filtered.png           (typical quality - very important!)
# - top_vs_bottom_comparison.png    (extreme comparison)
```

## Pro Tips

1. **Always check bottom 20 filtered** - This is the most important quality check!
2. **Random samples are more informative than top 10** - They show what you'll actually train on
3. **Compare scores to visual quality** - Trust your eyes, not just numbers
4. **Iterate if needed** - Adjust `--num_images` based on verification results
5. **Document your choice** - Save the verification images for future reference

## Adjusting Based on Results

```bash
# If bottom 20 filtered look bad, select fewer:
python filter_lighting_images.py \
    --dataset_path /path/to/images \
    --num_images 40000 \
    --output_dir ./ffhq_40k

# If bottom 20 filtered still look great, select more:
python filter_lighting_images.py \
    --dataset_path /path/to/images \
    --num_images 60000 \
    --output_dir ./ffhq_60k
```

Remember: The goal is to have a high-quality training dataset, not necessarily to hit exactly 50k images!

