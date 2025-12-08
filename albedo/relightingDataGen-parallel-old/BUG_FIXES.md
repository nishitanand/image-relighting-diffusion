# Bug Fixes - Shadow Generation

## Critical Bug Found and Fixed ✅

### The Actual Problem

**Error Message:**
```
matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)
```

**Root Cause:**
The error was **NOT** in `shading_synthesis.py` (that fix was correct but not the issue).

The real bug was in `src/utils/shadow_generation.py` at **line 213** in the `apply_shadow_transform()` function.

### The Bug

**File:** `src/utils/shadow_generation.py`
**Function:** `apply_shadow_transform()`
**Line:** 213

**Broken Code:**
```python
if scale:
    scale_x = random.uniform(0.7, 1.3)
    scale_y = random.uniform(0.7, 1.3)
    M_scale = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
    M = M @ np.vstack([M_scale, [0, 0, 1]])[:2]  # ❌ WRONG!
```

**Problem:**
- `M` is shape `(2, 3)` - affine transformation matrix
- `M_scale` is shape `(2, 3)` - scale matrix
- `np.vstack([M_scale, [0, 0, 1]])` creates `(3, 3)` matrix
- `[:2]` slices it to... `(3, 2)` ❌ WRONG AXIS!
- Matrix multiplication `(2, 3) @ (3, 2)` expects second operand shape `(3, n)` not `(n, 2)`

### The Fix

**Fixed Code:**
```python
if scale:
    scale_x = random.uniform(0.7, 1.3)
    scale_y = random.uniform(0.7, 1.3)
    M_scale = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
    # Combine affine transformations: M_combined = M @ M_scale_3x3
    # Convert both to 3x3, multiply, then take top 2 rows
    M_3x3 = np.vstack([M, [0, 0, 1]])
    M_scale_3x3 = np.vstack([M_scale, [0, 0, 1]])
    M = (M_3x3 @ M_scale_3x3)[:2, :]  # ✅ CORRECT!
```

**Why This Works:**
- Convert both matrices to proper 3x3 homogeneous coordinates
- Multiply: `(3, 3) @ (3, 3) = (3, 3)`
- Take top 2 rows: `[:2, :]` gives `(2, 3)` - correct shape for cv2.warpAffine!

### Testing

**Test Result:**
```bash
$ python -c "from src.utils.shadow_generation import generate_random_hard_shadow; ..."
✅ Hard shadow generation succeeded!
Metadata: {'degradation_type': 'hard_shadow', 'opacity': 0.652, ...}
Degraded image size: (512, 512)
```

---

## Secondary Issue: Missing `timm` Dependency

### Problem

MiDaS requires the `timm` package, which is not currently installed in the conda environment.

**Error:**
```
ModuleNotFoundError: No module named 'timm'
```

**Impact:**
- Soft shading degradation cannot run (requires MiDaS for normal estimation)
- Specular degradation cannot run (also requires normals)
- Hard shadow degradation still works (doesn't need MiDaS)

### Solution

**Install timm:**
```bash
conda activate relighting  # or your environment name
pip install timm>=0.9.0
```

**Already in requirements.txt:**
```txt
timm>=0.9.0  # for MiDaS
```

Just needs to be installed.

---

## What Now Works

### ✅ Hard Shadow Generation
- Fully functional
- Uses procedural shadow patterns
- No dependencies on MiDaS

### ⏳ Soft Shading & Specular (after installing timm)
- Will work once `timm` is installed
- Requires MiDaS for normal estimation
- Should work without further code changes

---

## Files Modified

1. **`src/utils/shadow_generation.py`**
   - Line 213-217: Fixed matrix multiplication for affine transformations
   - Changed from incorrect `[:2]` slicing to proper `[:2, :]` slicing

2. **`src/utils/shading_synthesis.py`** (previous fix - not the main issue)
   - Line 103, 145-147: Added reshape for broadcasting (still correct, just not the cause)

---

## Testing Instructions

### Test 1: Hard Shadow Only (Works Now!)

```bash
python scripts/run_pipeline.py --config config/mvp_config.yaml --num-samples 1

# Check output
cat data/stage_3/00000_params.json
# Should show: "degradation_type": "hard_shadow"
```

### Test 2: After Installing timm

```bash
# First install
pip install timm>=0.9.0

# Then run
python scripts/run_pipeline.py --config config/mvp_config.yaml --num-samples 1

# Check output
cat data/stage_3/00000_params.json
# Should show one of: "soft_shading", "hard_shadow", or "specular"
```

### Test 3: Verify All Degradation Types

Run multiple samples to see variety:
```bash
python scripts/run_pipeline.py --config config/mvp_config.yaml --num-samples 10

# Check variety
grep "degradation_type" data/stage_3/*_params.json
```

---

## Summary

### What Was Wrong
1. **Matrix multiplication bug** in shadow transform (line 213 of shadow_generation.py)
2. **Missing timm dependency** preventing MiDaS from loading

### What Was Fixed
1. ✅ Shadow generation matrix math corrected
2. ✅ Hard shadow degradation now works
3. ⏳ Soft shading/specular will work after `pip install timm`

### Action Items for User
```bash
# 1. Install timm
pip install timm>=0.9.0

# 2. Re-run pipeline
python scripts/run_pipeline.py --config config/mvp_config.yaml --num-samples 1

# 3. Verify success
cat data/stage_3/00000_params.json
ls -la data/outputs/
```

That's it! The core bug is fixed. The pipeline should work now.
