# Pipeline Summary - Understanding the Relighting Data Generation Code

## ✅ Your Understanding is MOSTLY CORRECT! 

Here's what the pipeline actually does:

---

## 📋 Pipeline Overview

The code generates training data for image relighting models.

### Full Pipeline Flow:

```
Input Image (e.g., face photo)
    ↓
[Stage 1] SAM3/SAM2 Segmentation
    → Segments "person" from background
    → Outputs: foreground, background, mask
    ↓
[Stage 2] Albedo Extraction  
    → Removes ALL lighting from foreground
    → Outputs: albedo (intrinsic reflectance, no shadows/highlights)
    ↓
[Stage 3] Degradation Synthesis
    → Applies NEW lighting/shadows to albedo
    → 3 methods: soft shading (80%), hard shadow (0%), specular (0%)
    → Outputs: degraded foreground (albedo + new lighting)
    ↓
[Stage 3.5] Background Recombination
    → Composites degraded foreground BACK onto original background
    → Uses mask for alpha blending
    → Outputs: final composite image
    ↓
Final Output: Image with SAME person, SAME background, but DIFFERENT lighting
```

---

## 🔍 Detailed Stage Breakdown

### **Stage 1: Segmentation (SAM3/SAM2)**

**What it does:**
- Uses **text prompt "person"** with SAM3 to segment the person
- Falls back to SAM2 if SAM3 unavailable
- Separates image into:
  - **Foreground**: Just the person (with transparent/black background)
  - **Background**: Original background without person
  - **Mask**: Binary mask (white=person, black=background)

**Your understanding:** ✅ **CORRECT** - "segments foreground using sam3 with prompt 'person'"

**Key file:** `src/stages/stage_1_segmentation_sam3.py`

---

### **Stage 2: Albedo Extraction**

**What it does:**
- Takes the **foreground person** from Stage 1
- **Removes ALL existing lighting** (shadows, highlights, shading)
- Extracts **intrinsic albedo** (pure material color/reflectance)
- Uses 3 methods with fallback:
  1. **IntrinsicAnything** (disabled by default, requires checkpoint)
  2. **Multi-Scale Retinex** (primary method, 80% enabled)
  3. **LAB-based** (simple fallback, always enabled)

**Your understanding:** ✅ **CORRECT** - "extracts albedo from this foreground"

**Key file:** `src/stages/stage_2_albedo.py`

**Important:** The albedo is **NOT** pasted on black/gray background yet. It still retains the original image dimensions with the person extracted.

---

### **Stage 3: Degradation Synthesis**

**What it does:**
- Takes the **albedo** from Stage 2
- **Applies NEW lighting** to create "degradation image"
- 3 degradation methods (randomly selected based on weights):

  **A. Soft Shading (80% probability - PRIMARY METHOD)**
  - Estimates normal map using MiDaS depth model
  - Applies Lambertian shading: `I = albedo × max(0, N·L)`
  - Random light direction from hemisphere
  - **High ambient light** (0.6-0.85) = very subtle shadows
  - Optionally adds subtle shadow patterns
  
  **B. Hard Shadow (0% - DISABLED)**
  - Generates procedural shadow patterns
  - Uses Perlin noise, geometric shapes, blob shadows
  
  **C. Specular Reflection (0% - DISABLED)**
  - Soft shading + Phong highlights
  - `I_spec = (R·V)^shininess`

**Your understanding:** ⚠️ **PARTIALLY CORRECT** - "adds degradations to it"
- You said "adds degradations" which is correct
- But "degradation" here means **new lighting**, not corruption/noise
- "Degradation" here means "altered illumination"

**Key file:** `src/stages/stage_3_shadow.py`

**Output:** Degraded foreground with new lighting applied

---

### **Stage 3.5: Background Recombination** ⭐ KEY STAGE

**What it does:**
- Takes the **degraded foreground** from Stage 3
- Takes the **original background** from Stage 1
- Uses the **mask** from Stage 1
- **Composites them together** using alpha blending:
  ```
  composite = degraded_foreground × mask + background × (1 - mask)
  ```

**Your understanding:** ❌ **INCORRECT** - "pastes back foreground to gray or black background"
- It does **NOT** use a gray/black background
- It uses the **ORIGINAL background** from the input image
- The final image looks like the original scene, but with different lighting on the person

**Key file:** `src/stages/stage_3_5_recombine.py`

**Output:** Complete image with person having new lighting + original background

---

## 🎯 What Makes This Different From Your Understanding

### What You Thought:
> "pasts back this foreground to a gray or black background of same resolution as input image"

### What Actually Happens:
The degraded foreground is composited back onto the **ORIGINAL BACKGROUND**, not a gray/black background.

**Why?**
- The goal is to create realistic training pairs for relighting
- Original background provides context and realism
- Only the person's lighting changes, not the scene

**Example:**
```
Input:  Person in a room with furniture
        ↓
Output: SAME person in SAME room, but person has different lighting/shadows
        (background unchanged)
```

---

## 📊 Current Configuration (mvp_config.yaml)

### Segmentation:
- **SAM3** enabled with text prompt "person"
- Falls back to **SAM2** if unavailable

### Albedo Extraction:
- **Retinex** (primary): Multi-scale with scales [15, 80, 250]
- **LAB** (fallback): Always enabled
- **IntrinsicAnything** (disabled): Requires checkpoint
- **Blending**: 15-25% of original mixed in to reduce over-brightening

### Degradation:
- **Soft Shading**: 80% probability (MAIN METHOD)
  - Very high ambient light (0.6-0.85) for subtle shadows
  - Adds subtle shadow patterns
- **Hard Shadow**: 0% (DISABLED)
- **Specular**: 0% (DISABLED)
- **Advanced Shading**: 0% (DISABLED)

---

## 🚀 How to Run

### Quick Test (1 image):
```bash
cd /mnt/localssd/diffusion/albedo/relightingDataGen
conda activate relighting  # or create environment first

# Process 1 image
python scripts/run_pipeline.py --config config/mvp_config.yaml --num-samples 1
```

### Full Run:
```bash
# Process all images in data/raw/
python scripts/run_pipeline.py --config config/mvp_config.yaml
```

### Using Your Filtered FFHQ Images:
You'll need to:
1. Copy your top 12k filtered images to `data/raw/`
2. Or modify the config to point to your image paths

---

## 📁 Expected Output Structure

```
data/
├── raw/                          # Input images
│   └── 00000.png
├── stage_1/                      # Segmentation outputs
│   ├── 00000_foreground.png      # Person only
│   ├── 00000_background.png      # Background only
│   └── 00000_mask.png            # Binary mask
├── stage_2/                      # Albedo extraction
│   └── 00000_albedo.png          # Albedo (no lighting)
├── stage_3/                      # Degradation synthesis
│   ├── 00000_degraded.png        # Degraded foreground
│   └── 00000_params.json         # Lighting parameters
├── stage_3_5/                    # Background recombination
│   └── 00000_composite.png       # Final composite
└── outputs/                      # Final organized outputs
    ├── 00000_input.png           # Original input
    ├── 00000_albedo.png          # Albedo
    ├── 00000_output.png          # Final output (composite)
    ├── 00000_foreground.png      # Foreground
    └── 00000_metadata.json       # All metadata
```

---

## 🎓 Key Concepts

### What are "Degradation Images"?

**NOT** corrupted/noisy images, but:
- Same object with same albedo
- Completely altered illumination
- Used as training pairs for relighting models

### Process Summary:
1. **Extract albedo** → Remove all existing lighting
2. **Apply new lighting** → Synthesize realistic shading/shadows
3. **Preserve intrinsics** → Maintain object's reflectance

This creates diverse training data for learning lighting-invariant representations.

---

## 📝 Summary of Your Understanding

| Your Statement | Accuracy | Notes |
|----------------|----------|-------|
| "segments foreground using sam3 with prompt 'person'" | ✅ Correct | |
| "extracts albedo from this foreground" | ✅ Correct | |
| "adds degradations to it" | ⚠️ Partially | "Degradation" = new lighting, not corruption |
| "pastes back to gray or black background" | ❌ Incorrect | Uses **original background**, not gray/black |

---

## 🔧 Next Steps

1. **Check dependencies:**
   ```bash
   cat requirements.txt
   pip list
   ```

2. **Test on 1 image:**
   ```bash
   python scripts/run_pipeline.py --config config/mvp_config.yaml --num-samples 1
   ```

3. **Inspect outputs:**
   - Check `data/stage_3_5/` for final composites
   - View side-by-side: input vs output

4. **Scale to your 12k images:**
   - Prepare your filtered FFHQ images
   - Update paths in config
   - Run full pipeline

---

## 📚 Key Files to Review

1. **`README.md`** - Full documentation
2. **`QUICKSTART.md`** - Quick setup guide
3. **`config/mvp_config.yaml`** - Current configuration
4. **`src/stages/stage_3_5_recombine.py`** - Background compositing (KEY!)
5. **`src/stages/stage_3_shadow.py`** - Degradation methods

---

## ❓ Questions to Consider

1. Do you want to use the **original background** or a **neutral background**?
   - Current code: original background
   - If you want gray/black: need to modify Stage 3.5

2. Do you need captions for your training data?
   - Stage 4 (captioning) exists but is disabled in MVP config

3. Which albedo method works best for your faces?
   - Test Retinex vs LAB on a few samples

4. Are the shadows too subtle or too strong?
   - Adjust `ambient_range` in soft_shading config

---

**Status:** Ready to run! The code is production-ready with proper error handling and fallbacks. 🚀

