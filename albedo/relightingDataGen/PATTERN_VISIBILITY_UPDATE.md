# Pattern Visibility Update

## Summary

Made shadow patterns **more visible** while keeping shadows light and natural.

---

## Changes Made

### **Increased Pattern Opacity**

| Setting | Before | After | Change |
|---------|--------|-------|--------|
| **Pattern Opacity** | 5-15% | **15-35%** | 2-3x more visible |
| **Pattern Blur** | 71-151px | **31-71px** | Sharper, more defined |

### **What This Means:**

✅ **Patterns now visible** - You can actually see venetian blinds, tree leaves, window frames
✅ **Still natural** - Not overwhelming, just noticeable
✅ **Shadows still light** - Ambient 60-85% unchanged
✅ **Better texture** - Adds visual interest without being harsh

---

## Updated Settings

### **config/mvp_config.yaml:**

```yaml
soft_shading:
  weight: 0.8  # Still primary method
  ambient_range: [0.6, 0.85]  # Still very light
  add_subtle_patterns: true

  # NEW SETTINGS - More Visible:
  pattern_opacity: [0.15, 0.35]  # Was [0.05, 0.15]
  pattern_blur_range: [31, 71]   # Was [71, 151]
```

### **src/utils/shading_synthesis.py:**

```python
# Pattern blur - less blur = more visible
blur_size = random.choice([31, 41, 51, 61, 71])  # Was [71, 91, 111, 131, 151]

# Pattern opacity - higher = more visible
pattern_opacity = random.uniform(0.15, 0.35)  # Was 0.05-0.15
```

---

## Visual Comparison

### Before (5-15% opacity, 71-151px blur):
```
Base: [Very light soft shading]
Pattern: [..................] (basically invisible)
Result: Clean but no pattern visible
```

### After (15-35% opacity, 31-71px blur):
```
Base: [Very light soft shading]
Pattern: [░░█░░█░░█░░█░░] (clearly visible)
Result: Natural shading with visible pattern texture
```

---

## Pattern Examples

You should now see:

### **Venetian Blinds** (25% of patterns):
```
═══════  (visible horizontal lines)
░░░░░░░
═══════
░░░░░░░
═══════
```

### **Tree Foliage** (30% of patterns):
```
░█░░█░  (organic dappled shadow)
░░█░░█
█░░█░░
```

### **Window Frames** (20% of patterns):
```
┌─┬─┐  (visible grid)
│ │ │
├─┼─┤
│ │ │
```

---

## Testing

```bash
# Run with new pattern settings
python scripts/run_pipeline.py \
    --config config/mvp_config.yaml \
    --num-samples 5

# Check metadata
cat data/stage_3/00000_params.json

# Should show:
# - pattern_opacity: ~0.15-0.35 (higher)
# - pattern_blur: ~31-71 (lower)
# - pattern_type: tree_foliage / venetian_blind / etc
```

---

## Fine-Tuning Options

### If patterns too strong:

```yaml
pattern_opacity: [0.10, 0.25]  # Reduce opacity
pattern_blur_range: [41, 81]    # More blur
```

### If patterns still too subtle:

```yaml
pattern_opacity: [0.25, 0.45]  # Increase opacity
pattern_blur_range: [21, 51]    # Less blur (sharper)
```

### If you want specific pattern types:

Edit `src/utils/shadow_patterns.py` line ~421:
```python
# Current weights:
pattern_types = [
    'venetian_blind',  # 25%
    'window_frame',    # 20%
    'tree_foliage',    # 30%
    'voronoi',         # 15%
    'lattice'          # 10%
]
weights = [0.25, 0.20, 0.30, 0.15, 0.10]

# To use only tree patterns:
pattern_types = ['tree_foliage']
weights = [1.0]
```

---

## Expected Results

### Metadata Example:

```json
{
  "degradation_type": "soft_shading",
  "ambient": 0.74,
  "has_pattern": true,
  "pattern_type": "tree_foliage",
  "pattern_opacity": 0.28,  # Now visible (15-35%)
  "pattern_blur": 51        # Less blur (31-71px)
}
```

### Visual Quality:

✅ **Soft shadows** - Still very light (60-85% ambient)
✅ **Visible patterns** - Can see venetian blinds, trees, windows
✅ **Natural look** - Patterns add texture without being harsh
✅ **Good for relighting** - Adds variety to training data

---

## Summary

| Aspect | Setting |
|--------|---------|
| **Shadows** | Very light (ambient 60-85%) ✅ |
| **Pattern Opacity** | 15-35% (was 5-15%) ✅ |
| **Pattern Blur** | 31-71px (was 71-151px) ✅ |
| **Pattern Visibility** | Now clearly visible ✅ |
| **Overall Look** | Light shadows + visible patterns ✅ |

**Patterns should now be visible while keeping the soft, light shadows you liked!** 🎨
