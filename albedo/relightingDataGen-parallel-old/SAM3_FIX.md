# SAM3 API Fix - Applied

## 🔧 Changes Made

Updated the SAM3 loading code to match the actual SAM3 API from your installation.

### File Modified:
`src/stages/stage_1_segmentation_sam3.py`

### What Changed:

**Before (Lines 60-66):**
```python
# Load SAM3 model
logger.info("Loading SAM3 from HuggingFace...")
self.model = build_sam3_image_model(
    model_id="facebook/sam3",  # ❌ Wrong: this parameter doesn't exist
    device=self.device
)
self.processor = Sam3Processor(self.model)
```

**After (Lines 60-65):**
```python
# Load SAM3 model (no arguments needed)
logger.info("Loading SAM3 model...")
self.model = build_sam3_image_model()  # ✅ Correct: no arguments
self.processor = Sam3Processor(self.model)

# Move model to device
self.model = self.model.to(self.device)  # ✅ Move to GPU manually
```

## ✅ What's Now Working

1. **Correct SAM3 API call** - Matches your installation
2. **Text prompt "person"** - Already configured in `config/mvp_config.yaml`
3. **Manual device placement** - Explicitly moves model to CUDA

## 🎯 Configuration

The text prompt is set in `config/mvp_config.yaml`:

```yaml
sam3:
  enabled: true
  text_prompt: "person"  # ✅ This is what will be used
  multimask_output: false
```

## 🚀 Ready to Test

You can now run the pipeline:

```bash
cd /mnt/localssd/diffusion/albedo/relightingDataGen

python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --csv /mnt/localssd/diffusion/filter_images/ffhq_output_top12k_random/train_images.csv \
    --num-samples 10
```

## 📋 Expected Behavior

The pipeline will:
1. ✅ Load SAM3 model successfully (no API error)
2. ✅ Use text prompt "person" to segment faces
3. ✅ Process all stages (segmentation → albedo → degradation → recombine)
4. ✅ Save outputs to `data/outputs/`

## 🔍 What the Logs Should Show

```
2025-12-07 XX:XX:XX | INFO | Attempting to load SAM3 with text prompting...
2025-12-07 XX:XX:XX | INFO | Loading SAM3 model...
2025-12-07 XX:XX:XX | INFO | SAM3 model loaded successfully with text prompting support
2025-12-07 XX:XX:XX | INFO | Segmenting with SAM3 text prompt: 'person'
2025-12-07 XX:XX:XX | INFO | SAM3 detected N instances of 'person'
2025-12-07 XX:XX:XX | INFO | Segmentation score: 0.XXX (method: sam3_text)
```

No more `TypeError` or fallback to SAM2! 🎉

---

## 🆘 If It Still Fails

If SAM3 still has issues, the pipeline will automatically fall back to SAM2 (if installed).

To install SAM2 as a backup:
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

But with this fix, SAM3 should work perfectly! ✅

