"""
Prepare training data from keywords CSV.

This script converts the Step 3 CSV output into the format required by the training scripts.

Training triplet mapping:
- Input Image: output_image_path (albedo degraded image)
- Instruction: lighting_keywords (VLM-generated description)  
- Output Image: image_path (original image with real lighting)

Usage:
    python prepare_training_data.py \
        --csv output/train_images_with_albedo_with_keywords.csv \
        --output_dir ../training/sd1_5/data_triplets
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import json
import shutil
from tqdm import tqdm


def prepare_training_data(
    csv_path: str,
    output_dir: str,
    copy_images: bool = False,
    validate: bool = True
):
    """
    Prepare training data from keywords CSV.
    
    Args:
        csv_path: Path to CSV with keywords
        output_dir: Output directory for training data
        copy_images: Whether to copy images to output directory
        validate: Validate that all image paths exist
    """
    # Load CSV
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['image_path', 'output_image_path', 'lighting_keywords']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    
    # Filter out rows with errors or missing data
    df_valid = df[
        df['output_image_path'].notna() & 
        df['lighting_keywords'].notna() &
        ~df['lighting_keywords'].str.startswith('ERROR:', na=False)
    ].copy()
    
    print(f"Total rows: {len(df)}")
    print(f"Valid rows: {len(df_valid)}")
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if copy_images:
        (output_dir / "inputs").mkdir(exist_ok=True)
        (output_dir / "outputs").mkdir(exist_ok=True)
    
    # Validate paths if requested
    if validate:
        print("Validating image paths...")
        valid_mask = []
        for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid)):
            input_exists = Path(row['output_image_path']).exists()
            output_exists = Path(row['image_path']).exists()
            valid_mask.append(input_exists and output_exists)
        
        df_valid = df_valid[valid_mask]
        print(f"After validation: {len(df_valid)} valid rows")
    
    # Create metadata.jsonl
    metadata = []
    
    for idx, (_, row) in enumerate(tqdm(df_valid.iterrows(), desc="Preparing data")):
        # Source paths
        input_src = row['output_image_path']  # Degraded image is INPUT for training
        output_src = row['image_path']         # Original image is OUTPUT for training
        instruction = row['lighting_keywords']
        
        if copy_images:
            # Copy images with consistent naming
            input_ext = Path(input_src).suffix
            output_ext = Path(output_src).suffix
            
            input_dst = output_dir / "inputs" / f"{idx:06d}{input_ext}"
            output_dst = output_dir / "outputs" / f"{idx:06d}{output_ext}"
            
            shutil.copy2(input_src, input_dst)
            shutil.copy2(output_src, output_dst)
            
            # Relative paths for metadata
            input_path = f"inputs/{idx:06d}{input_ext}"
            output_path = f"outputs/{idx:06d}{output_ext}"
        else:
            # Use absolute paths
            input_path = str(input_src)
            output_path = str(output_src)
        
        metadata.append({
            "input_image": input_path,
            "instruction": instruction,
            "output_image": output_path
        })
    
    # Save metadata.jsonl
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
    
    # Also save as JSON for easy inspection
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save statistics
    stats = {
        "total_samples": len(metadata),
        "source_csv": str(csv_path),
        "copy_images": copy_images,
        "columns_used": {
            "input_image": "output_image_path (degraded)",
            "instruction": "lighting_keywords",
            "output_image": "image_path (original)"
        }
    }
    
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ TRAINING DATA PREPARED")
    print(f"{'='*60}")
    print(f"Total samples: {len(metadata)}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - metadata.jsonl ({len(metadata)} entries)")
    print(f"  - metadata.json")
    print(f"  - stats.json")
    if copy_images:
        print(f"  - inputs/ ({len(metadata)} images)")
        print(f"  - outputs/ ({len(metadata)} images)")
    print(f"\n📋 Next Step: Convert to HuggingFace dataset")
    print(f"   cd ../training/sd1_5")
    print(f"   python convert_to_hf_dataset.py --data_dir {output_dir} --output_dir ./data_hf")
    print(f"{'='*60}\n")
    
    # Print sample entries
    print("Sample entries:")
    for i, entry in enumerate(metadata[:3]):
        print(f"\n  [{i}]")
        print(f"    Input: {entry['input_image'][:60]}...")
        print(f"    Instruction: {entry['instruction']}")
        print(f"    Output: {entry['output_image'][:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from keywords CSV"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV with keywords")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for training data")
    parser.add_argument("--copy_images", action="store_true",
                        help="Copy images to output directory")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip path validation")
    
    args = parser.parse_args()
    
    prepare_training_data(
        csv_path=args.csv,
        output_dir=args.output_dir,
        copy_images=args.copy_images,
        validate=not args.no_validate
    )


if __name__ == "__main__":
    main()

