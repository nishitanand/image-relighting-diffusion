"""
Optimized parallel pipeline with proper GPU utilization and CSV mapping.
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import setup_logging
from src.pipeline.pipeline_runner import RelightingPipeline


def get_split_name_from_csv(csv_path: str) -> str:
    """Extract split name (train/val/test) from CSV filename."""
    csv_name = Path(csv_path).stem.lower()
    
    if 'train' in csv_name:
        return 'train'
    elif 'val' in csv_name:
        return 'val'
    elif 'test' in csv_name:
        return 'test'
    else:
        return 'data'


def main():
    parser = argparse.ArgumentParser(
        description="Run relighting pipeline with CSV output mapping"
    )
    parser.add_argument("--config", type=str, default="config/mvp_config.yaml")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with image paths")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save updated CSV")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(project_root) / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load CSV
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    
    if 'image_path' not in df.columns:
        print("ERROR: CSV must contain 'image_path' column")
        sys.exit(1)
    
    # Determine split and output directory
    split_name = get_split_name_from_csv(args.csv)
    output_root = Path(project_root) / f"data-{split_name}"
    
    # Update config with new output path
    config['paths']['output_root'] = str(output_root)
    config['paths']['data_root'] = str(output_root.parent / f"data_{split_name}_intermediate")
    
    print(f"Split: {split_name}")
    print(f"Output directory: {output_root}")
    
    # Initialize pipeline
    pipeline = RelightingPipeline(config)
    
    # Get images to process
    image_paths = df['image_path'].tolist()
    if args.num_samples:
        image_paths = image_paths[:args.num_samples]
    
    print(f"Processing {len(image_paths)} images...")
    
    # Process images
    results = []
    output_paths_list = []
    
    for idx, image_path in enumerate(tqdm(image_paths, desc="Processing")):
        try:
            result = pipeline.process_image(str(image_path), idx)
            if result:
                output_path = output_root / f"{idx:05d}_output.png"
                results.append(result)
                output_paths_list.append(str(output_path))
            else:
                output_paths_list.append(None)
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            output_paths_list.append(None)
    
    # Update CSV with output paths
    df_updated = df.copy()
    df_updated['output_image_path'] = None
    
    for i in range(len(output_paths_list)):
        if i < len(df_updated):
            df_updated.loc[i, 'output_image_path'] = output_paths_list[i]
    
    # Save updated CSV
    if args.output_csv:
        output_csv_path = args.output_csv
    else:
        csv_dir = Path(args.csv).parent
        csv_name = Path(args.csv).stem
        output_csv_path = csv_dir / f"{csv_name}_with_outputs.csv"
    
    df_updated.to_csv(output_csv_path, index=False)
    
    print(f"\n✅ Processing complete!")
    print(f"Output directory: {output_root}")
    print(f"Updated CSV: {output_csv_path}")
    print(f"Successful: {len(results)}/{len(output_paths_list)}")


if __name__ == "__main__":
    main()

