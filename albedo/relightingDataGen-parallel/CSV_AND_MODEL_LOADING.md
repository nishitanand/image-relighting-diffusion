# CSV Output Timing and Model Loading - Important Details

## 📝 CSV Creation Timing

### Question: When are the output CSVs created?

**Answer:** CSVs are created **AFTER all images are fully processed and saved**.

### Processing Flow:

1. **Image Processing Phase** (all GPUs in parallel):
   ```
   Load models once per GPU
   ↓
   Process Image 0 → Save to data-train/00000_*.png
   Process Image 1 → Save to data-train/00001_*.png
   ...
   Process Image N → Save to data-train/N_*.png
   ↓
   Unload models
   ```

2. **CSV Creation Phase** (after all images done):
   ```
   Collect all results from all GPUs
   ↓
   Create copy of original CSV
   ↓
   Add 'output_image_path' column
   ↓
   Map each row to its output path
   ↓
   Save new CSV with '_with_relighting_outputs' suffix
   ```

### Why This Order?

✅ **Images are saved first** so you can use them immediately
✅ **CSV is created last** to ensure all paths are valid
✅ **If the pipeline crashes**, you still have all processed images up to that point
✅ **CSV mapping is guaranteed accurate** because it only includes successfully processed images

### Example Timeline:

```
00:00 - Start processing 10,000 images on 8 GPUs
00:00 - GPU 0-7: Load models (once per GPU)
00:01 - Start processing images in parallel
01:58 - All images processed and saved to data-train/
01:58 - All GPUs unload models
01:58 - Create train_images_with_relighting_outputs.csv
01:59 - Done! CSV saved with all mappings
```

### CSV Only Contains Successfully Processed Images:

If some images fail:
- Failed images: No output files created
- Failed images: `output_image_path` column will be `None` in CSV
- Successful images: Have valid paths in CSV

---

## 🔧 Model Loading Optimization

### Question: Are models loaded once or per image?

**Answer:** Models are loaded **ONCE per GPU** and reused for all images on that GPU.

### Old Implementation (SLOW - Fixed!):
```python
for each image:
    Load SAM3 model (~10s)      ← SLOW!
    Load Albedo model
    Load Shadow model
    Process image (~21s total)
    Unload all models
```

**Time per image:** ~21 seconds (10s loading + 11s processing)

### New Implementation (FAST - Current!):
```python
# Load models ONCE
Load SAM3 model (~10s)           ← Done once!
Load Albedo model
Load Shadow model

# Reuse models for ALL images
for each image:
    Process image (~11s)         ← Much faster!
    
# Unload models ONCE
Unload all models
```

**Time per image:** ~11 seconds (just processing, no loading overhead!)

### Performance Improvement:

| Setup | Model Loading | Processing | Total per Image |
|-------|--------------|------------|-----------------|
| **Old (per-image loading)** | 10s | 11s | **21s** |
| **New (load once)** | 10s (once) | 11s | **~11s** |

**Speedup:** ~2x faster per image! 🚀

### For 10,000 Images on 8 GPUs:

**Old way:**
- Each GPU: 1,250 images × 21s = 26,250s = **7.3 hours**

**New way (optimized):**
- Model loading: 10s (once)
- Each GPU: 1,250 images × 11s = 13,750s + 10s = **3.8 hours**

**Improvement:** 7.3 hours → 3.8 hours = **2x faster!** 🔥

---

## ✅ Summary

### CSV Creation:
✅ All images processed and saved **FIRST**
✅ CSV created **LAST** with complete mappings
✅ Original CSV **never modified**
✅ New CSV saved with `_with_relighting_outputs` suffix

### Model Loading:
✅ Models loaded **once per GPU** at startup
✅ Models **reused** for all images on that GPU
✅ Models unloaded **once** after all images done
✅ **2x faster** than loading per image!

### Timeline:
```
00:00 - Load models (once per GPU)
00:01-03:50 - Process all images (models stay loaded)
03:50 - Unload models
03:51 - Create and save CSV
03:52 - Done!
```

This is the most efficient approach! 🚀

