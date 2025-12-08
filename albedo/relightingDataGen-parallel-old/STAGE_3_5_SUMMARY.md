# Stage 3.5: Background Recombination

## Overview

Created a separate stage (3.5) for background recombination instead of modifying stage 3. This keeps stage 3 focused on degradation synthesis (foreground only) and adds a clean separation of concerns.

## Pipeline Structure

### Before (Mixed approach):
```
Stage 3: Degradation → Output: composite image
```

### After (Clean separation):
```
Stage 3: Degradation → Output: degraded foreground only
Stage 3.5: Recombination → Output: composite (foreground + background)
```

## Implementation

### New File: `src/stages/stage_3_5_recombine.py`

```python
class BackgroundRecombinationStage(BaseStage):
    """
    Recombine degraded foreground with original background.
    """

    def recombine_with_background(
        self,
        degraded_foreground: Image.Image,
        background: Image.Image,
        mask: Image.Image
    ) -> Image.Image:
        """
        Alpha compositing: result = foreground * mask + background * (1 - mask)
        """
        # Convert to numpy
        foreground_np = np.array(degraded_foreground).astype(np.float32)
        background_np = np.array(background).astype(np.float32)
        mask_np = np.array(mask).astype(np.float32) / 255.0

        # Expand mask to 3 channels
        mask_3ch = mask_np[:, :, np.newaxis]

        # Composite
        composite_np = foreground_np * mask_3ch + background_np * (1 - mask_3ch)

        return Image.fromarray(composite_np.astype(np.uint8))

    def process(self, input_data):
        """
        Recombine degraded_image (foreground) with background.
        """
        composite = self.recombine_with_background(
            degraded_foreground=input_data['degraded_image'],
            background=input_data['background'],
            mask=input_data['mask']
        )

        output = input_data.copy()
        output.update({
            'composite_image': composite,
            'degraded_foreground': input_data['degraded_image']
        })

        return output
```

**Features:**
- No model loading needed (pure numpy operations)
- Fast alpha compositing
- Preserves all previous pipeline data
- Clean separation of concerns

---

## Pipeline Runner Updates

### Added Stage 3.5 to Pipeline

**File: `src/pipeline/pipeline_runner.py`**

**Imports:**
```python
from ..stages.stage_3_5_recombine import BackgroundRecombinationStage
```

**Stage initialization:**
```python
self.stages = {
    'segmentation': SAM3SegmentationStage(config, str(self.device)),
    'albedo': AlbedoExtractionStage(config, str(self.device)),
    'shadow': ShadowGenerationStage(config, str(self.device)),
    'recombine': BackgroundRecombinationStage(config, str(self.device)),  # NEW
    'captioning': CaptioningStage(config, str(self.device))
}
```

**Processing flow:**
```python
# Stage 3: Degradation synthesis (foreground only)
data = self.run_stage('shadow', data)
self.save_intermediate('stage_3', data)

# Stage 3.5: Background recombination  # NEW
data = self.run_stage('recombine', data)
self.save_intermediate('stage_3_5', data)
```

---

## Output Structure

### Stage 3 Output (Unchanged):
```
data/stage_3/
├── 00000_degraded.png    # Degraded foreground only
└── 00000_params.json     # Degradation parameters
```

### Stage 3.5 Output (New):
```
data/stage_3_5/
└── 00000_composite.png   # Composite (foreground + background)
```

### Final Output:
```
data/outputs/
├── 00000_input.png         # Original image
├── 00000_output.png        # COMPOSITE from stage 3.5 ⭐
├── 00000_degraded_fg.png   # Degraded foreground from stage 3
├── 00000_albedo.png        # Albedo map
├── 00000_foreground.png    # Segmented foreground
├── 00000_background.png    # Segmented background
└── 00000_metadata.json     # All metadata
```

---

## Data Flow

```
Input Image
    ↓
[Stage 1: Segmentation]
    → foreground, background, mask
    ↓
[Stage 2: Albedo Extraction]
    → albedo (from foreground)
    ↓
[Stage 3: Degradation Synthesis]
    → degraded_image (foreground only with new lighting)
    ↓
[Stage 3.5: Background Recombination] ⭐ NEW
    → composite_image = degraded_foreground × mask + background × (1 - mask)
    ↓
Output: Complete relit image
```

---

## Metadata Changes

### Updated metadata structure:

```json
{
  "image_id": 0,
  "original": "data/raw/00000.png",
  "foreground": "data/stage_1/00000_foreground.png",
  "background": "data/stage_1/00000_background.png",
  "albedo": "data/stage_2/00000_albedo.png",
  "degraded_foreground": "data/stage_3/00000_degraded.png",
  "composite_image": "data/stage_3_5/00000_composite.png",
  "degradation_type": "hard_shadow",
  "degradation_metadata": {
    "degradation_type": "hard_shadow",
    "opacity": 0.35,
    "pattern_type": "geometric"
  }
}
```

---

## Shadow Softness Improvements (Still Applied)

Stage 3 still has the softer shadows:
- **Blur size:** 31-111 pixels (soft, gradual edges)
- **Opacity:** 20-50% (subtle shadows)

These improvements are preserved in stage 3 and flow through to the composite in stage 3.5.

---

## Files Modified

1. **`src/stages/stage_3_shadow.py`**
   - Reverted to output `degraded_image` (foreground only)
   - Removed recombination logic
   - Kept shadow softness improvements

2. **`src/stages/stage_3_5_recombine.py`** ⭐ NEW
   - Background recombination stage
   - Alpha compositing
   - No model loading needed

3. **`src/pipeline/pipeline_runner.py`**
   - Added stage_3_5 to initialization
   - Updated directory setup
   - Updated save_intermediate for stage_3 and stage_3_5
   - Updated save_final_output to use composite from stage_3_5
   - Updated metadata generation

4. **`config/mvp_config.yaml`**
   - Shadow softness improvements preserved
   - No new config needed for stage 3.5

---

## Benefits

1. **Clean separation of concerns**
   - Stage 3: Focus on lighting/degradation synthesis
   - Stage 3.5: Focus on compositing

2. **Flexibility**
   - Can use degraded foreground only (stage 3 output)
   - Can use complete composite (stage 3.5 output)
   - Easy to disable stage 3.5 if needed

3. **Debugging**
   - Easier to debug degradation issues (stage 3 output)
   - Easier to debug compositing issues (stage 3.5 output)

4. **Modularity**
   - Stage 3.5 can be reused for other pipelines
   - Stage 3 remains focused on relighting

---

## Testing

```bash
# Run pipeline with stage 3.5
python scripts/run_pipeline.py --config config/mvp_config.yaml --num-samples 1

# Check stage 3 output (foreground only)
ls -la data/stage_3/
# Should see: 00000_degraded.png

# Check stage 3.5 output (composite)
ls -la data/stage_3_5/
# Should see: 00000_composite.png

# Check final output
ls -la data/outputs/
# 00000_output.png should be the composite image
```

---

## Summary

✅ **Stage 3:** Degradation synthesis (foreground only) with soft shadows
✅ **Stage 3.5:** Background recombination (alpha compositing)
✅ **Final output:** Complete relit image (degraded foreground + original background)
✅ **Clean architecture:** Separation of concerns, easy debugging, modular design

**Status:** Ready for testing!
