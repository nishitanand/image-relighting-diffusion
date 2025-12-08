# Soft Shadow Focus: Much Lighter & More Natural

## Summary

Based on your feedback that **only soft shadows are acceptable**, I've:

1. ✅ Made soft shadows **MUCH LIGHTER** (60-85% ambient instead of 10-30%)
2. ✅ Added **very subtle patterns** (venetian blinds, trees) with 5-15% opacity
3. ✅ **DISABLED** all other shadow methods (hard, depth-aware, advanced - all too dark)
4. ✅ Set soft shading to **80% weight** (primary method)
5. ✅ Removed specular highlights (too strong)

---

## Key Changes

### 1. **MUCH Lighter Ambient Light** 💡

**Before:**
```python
ambient_range: [0.1, 0.3]  # 10-30% ambient
# Result: Dark shadows, strong contrast
```

**After:**
```python
ambient_range: [0.6, 0.85]  # 60-85% ambient
# Result: VERY LIGHT shadows, barely visible
```

**Impact:**
- Shadows are now **barely visible** - just subtle shading
- Face is well-lit even in "shadow" areas
- Natural, soft appearance perfect for relighting

### 2. **Very Subtle Shadow Patterns** (Optional)

Added **extremely subtle** pattern overlay (5-15% opacity):

```python
add_subtle_patterns: true
pattern_opacity: [0.05, 0.15]  # Very light (5-15%)
pattern_blur: [71, 91, 111, 131, 151]  # Heavy blur for subtlety
```

**Pattern Types:**
- Tree foliage (30%) - Organic, natural
- Venetian blinds (25%) - Slat shadows
- Window frames (20%) - Grid patterns
- Voronoi (15%) - Irregular organic
- Lattice (10%) - Architectural

**All patterns are:**
- ✅ **Very blurred** (71-151 pixel blur)
- ✅ **Very light** (5-15% opacity)
- ✅ **Barely noticeable** - just adds subtle texture

### 3. **Disabled All Dark Shadow Methods**

```yaml
soft_shading:
  weight: 0.8  # 80% - PRIMARY METHOD

hard_shadow:
  weight: 0.0  # DISABLED (too dark)

advanced_shading:
  weight: 0.0  # DISABLED (too complex/dark)

specular:
  weight: 0.0  # DISABLED (too strong)
```

**Now using ONLY soft shading** - the method you approved!

---

## Technical Details

### **Modified: `src/utils/shading_synthesis.py`**

#### **1. Increased Default Ambient:**

```python
def compute_lambertian_shading(
    normal_map,
    light_direction,
    ambient: float = 0.5  # Was 0.2 - now MUCH higher
):
    # VERY HIGH AMBIENT = VERY LIGHT SHADOWS
    shading = ambient + (1 - ambient) * diffuse
```

#### **2. Updated Random Soft Shading:**

```python
def generate_random_soft_shading(albedo, normal_map, config):
    # VERY HIGH ambient
    ambient = random.uniform(0.6, 0.85)  # Was 0.1-0.3

    # NO specular (too strong)
    add_specular = False

    # Generate base shading (very light)
    degraded, metadata = generate_soft_shading_degradation(...)

    # Add VERY SUBTLE pattern overlay
    if config.get('add_subtle_patterns', False):
        pattern, pattern_meta = generate_random_shadow_pattern((h, w))

        # Heavy blur (71-151 pixels)
        blur_size = random.choice([71, 91, 111, 131, 151])
        pattern = cv2.GaussianBlur(pattern, (blur_size, blur_size), 0)

        # Very low opacity (5-15%)
        pattern_opacity = random.uniform(0.05, 0.15)

        # Apply pattern
        degraded_np = degraded_np * (1 - pattern * pattern_opacity)
```

### **Modified: `config/mvp_config.yaml`**

```yaml
degradation:
  soft_shading:
    weight: 0.8  # PRIMARY METHOD (80%)
    ambient_range: [0.6, 0.85]  # VERY HIGH = VERY LIGHT
    specular_intensity_range: [0.05, 0.2]  # Lower
    add_subtle_patterns: true  # Very subtle overlay
    pattern_opacity: [0.05, 0.15]  # 5-15% opacity

  hard_shadow:
    weight: 0.0  # DISABLED

  specular:
    weight: 0.0  # DISABLED

  advanced_shading:
    weight: 0.0  # DISABLED
```

---

## Visual Examples

### Shadow Intensity Comparison:

```
Before (ambient 0.1-0.3):
Shadow areas: ████████░░  (Very Dark)
Face coverage: Large dark regions
Problem: Too harsh, unnatural

After (ambient 0.6-0.85):
Shadow areas: █░░░░░░░░░  (Barely Visible)
Face coverage: Subtle shading only
Result: Natural, very light
```

### Pattern Overlay (Optional):

```
Base Soft Shading:
[Smooth, very light shading across face]

+ Subtle Pattern (5-15% opacity):
[Barely visible tree/window pattern texture]
(So subtle you have to look closely to see it)

= Final Result:
Natural lighting with optional very subtle texture
```

---

## Configuration Options

### **Disable Patterns** (if still too noticeable):

```yaml
# In config/mvp_config.yaml
degradation:
  soft_shading:
    add_subtle_patterns: false  # No patterns at all
```

### **Even Lighter Shadows** (if still too dark):

```yaml
degradation:
  soft_shading:
    ambient_range: [0.75, 0.95]  # Even higher ambient
    # 75-95% = almost no shadows at all
```

### **Adjust Pattern Strength** (if patterns visible):

```yaml
degradation:
  soft_shading:
    pattern_opacity: [0.02, 0.08]  # Even lighter (2-8%)
    # Or disable entirely with add_subtle_patterns: false
```

---

## What Was Removed

To simplify and focus only on what works:

### ❌ **Removed/Disabled:**

1. **Hard Shadows** - Too dark, covered large parts of face
2. **Depth-Aware Shadows** - Rendering issues, too complex
3. **Advanced Shading** - AO too dark, multi-light too complex
4. **Specular Highlights** - Too strong, distracting
5. **Strong Patterns** - Weird patterns, too noticeable

### ✅ **Kept Only:**

1. **Soft Shading** - Works well, natural
2. **Very Light Ambient** - 60-85% (barely any shadows)
3. **Optional Subtle Patterns** - 5-15% opacity, heavily blurred
4. **No Specular** - Clean, simple shading

---

## Expected Results

When you run the pipeline now:

✅ **Shadows are VERY light** - barely noticeable
✅ **Face is well-lit** - no large dark regions
✅ **Natural appearance** - subtle shading only
✅ **Optional patterns** - extremely subtle (5-15%), heavily blurred
✅ **No weird artifacts** - clean, simple soft shading
✅ **Consistent quality** - 80% of images use this method

### Metadata Example:

```json
{
  "degradation_type": "soft_shading",
  "light_direction": [0.5, 0.7, 0.5],
  "ambient": 0.73,  # Very high!
  "has_specular": false,
  "has_pattern": true,
  "pattern_type": "tree_foliage",
  "pattern_opacity": 0.12,  # Very low!
  "pattern_blur": 111  # Heavy blur
}
```

---

## Testing

```bash
# Run with new soft shadow settings
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --num-samples 5

# Check outputs
ls data/stage_3/
cat data/stage_3/00000_params.json

# You should see:
# - degradation_type: "soft_shading"
# - ambient: 0.6-0.85 (very high)
# - has_pattern: true/false
# - If has_pattern: pattern_opacity ~0.05-0.15 (very low)
```

### What To Look For:

✅ **Very light shadows** - face should be well-lit
✅ **Subtle shading** - just gentle gradients
✅ **Minimal face coverage** - no large dark regions
✅ **Natural appearance** - looks like soft, diffuse lighting
✅ **Optional subtle texture** - barely visible pattern (if enabled)

---

## Troubleshooting

### If shadows still too dark:

1. **Increase ambient further:**
   ```yaml
   ambient_range: [0.8, 0.95]  # Even lighter
   ```

2. **Verify config is loaded:**
   ```bash
   grep "ambient_range" config/mvp_config.yaml
   # Should show: [0.6, 0.85]
   ```

### If patterns too noticeable:

1. **Reduce opacity:**
   ```yaml
   pattern_opacity: [0.02, 0.08]  # 2-8% instead of 5-15%
   ```

2. **Or disable entirely:**
   ```yaml
   add_subtle_patterns: false
   ```

### If patterns not showing:

1. **Check import:**
   ```bash
   python -c "from src.utils.shadow_patterns import generate_random_shadow_pattern; print('OK')"
   ```

2. **Check config:**
   ```yaml
   add_subtle_patterns: true  # Must be true
   ```

---

## Files Modified

1. **`src/utils/shading_synthesis.py`**
   - Increased default ambient: 0.5 (was 0.2)
   - Updated `generate_random_soft_shading()`:
     - ambient_range: [0.6, 0.85] (was [0.1, 0.3])
     - Disabled specular
     - Added optional subtle pattern overlay
   - Added import for shadow_patterns

2. **`config/mvp_config.yaml`**
   - soft_shading weight: 0.8 (was 0.2)
   - ambient_range: [0.6, 0.85] (was [0.1, 0.3])
   - Added: add_subtle_patterns, pattern_opacity
   - Disabled: hard_shadow (0.0), specular (0.0), advanced_shading (0.0)

---

## Quick Reference

### Current Settings:

```yaml
# config/mvp_config.yaml
degradation:
  soft_shading:
    weight: 0.8               # Use 80% of time
    ambient_range: [0.6, 0.85]  # VERY LIGHT shadows
    add_subtle_patterns: true   # Optional patterns
    pattern_opacity: [0.05, 0.15]  # Barely visible

  # All others disabled (weight: 0.0)
```

### Ambient Levels Guide:

```python
# Shadow Strength by Ambient Level:

ambient: 0.1-0.3   # Dark shadows (old - too harsh)
ambient: 0.4-0.6   # Medium shadows
ambient: 0.6-0.85  # Light shadows ⭐ CURRENT
ambient: 0.8-0.95  # Almost no shadows (if needed)
```

---

## Summary

### Before Your Feedback:
- ❌ Multiple shadow methods (hard, depth, advanced)
- ❌ All too dark and covered large parts of face
- ❌ Weird patterns, unnatural appearance
- ❌ Depth-aware not rendering

### After This Update:
- ✅ **ONLY soft shading** (80% weight)
- ✅ **VERY light** (60-85% ambient)
- ✅ **Optional subtle patterns** (5-15% opacity, heavily blurred)
- ✅ **Face well-lit** - minimal shadow coverage
- ✅ **Natural appearance** - gentle, diffuse shading

**Focus:** Keep it simple, light, and natural - exactly what works!

---

**Status:** ✅ Complete - Ready for testing!

```bash
# Test now:
python scripts/run_pipeline.py --config config/mvp_config.yaml --num-samples 1
```

**You should see MUCH lighter, more natural soft shadows now!** 🌤️
