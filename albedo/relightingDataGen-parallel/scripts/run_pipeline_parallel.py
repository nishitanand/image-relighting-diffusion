"""
Parallel pipeline runner with CSV output mapping.

Features:
- Multi-GPU/multi-process parallelization
- Dynamic output folder naming (data-train, data-test, data-val)
- Automatic CSV update with output image paths
- Efficient resource utilization
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import pandas as pd
from multiprocessing import Pool, cpu_count
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import setup_logging
from src.pipeline.pipeline_runner import RelightingPipeline


def get_split_name_from_csv(csv_path: str) -> str:
    """
    Extract split name (train/val/test) from CSV filename.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Split name (train, val, or test)
    """
    csv_name = Path(csv_path).stem.lower()
    
    if 'train' in csv_name:
        return 'train'
    elif 'val' in csv_name:
        return 'val'
    elif 'test' in csv_name:
        return 'test'
    else:
        return 'unknown'


def process_single_image(args_tuple):
    """
    Process a single image (for parallel processing).
    
    Args:
        args_tuple: (image_idx, image_path, config, output_root)
        
    Returns:
        Dict with results or None if failed
    """
    image_idx, image_path, config, output_root = args_tuple
    
    try:
        # Create a pipeline instance for this process
        from PIL import Image
        from src.stages.stage_1_segmentation_sam3 import SAM3SegmentationStage
        from src.stages.stage_2_albedo import AlbedoExtractionStage
        from src.stages.stage_3_shadow import DegradationSynthesisStage
        from src.stages.stage_3_5_recombine import BackgroundRecombinationStage
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Initialize stages
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        
        seg_stage = SAM3SegmentationStage(config, device)
        albedo_stage = AlbedoExtractionStage(config, device)
        degrad_stage = DegradationSynthesisStage(config, device)
        recomb_stage = BackgroundRecombinationStage(config, device)
        
        # Process through all stages
        data = {'image': image, 'image_id': image_idx, 'original_image': image}
        
        # Stage 1: Segmentation
        seg_stage.load_model()
        data = seg_stage.process(data)
        seg_stage.unload_model()
        
        # Stage 2: Albedo
        albedo_stage.load_model()
        data = albedo_stage.process(data)
        albedo_stage.unload_model()
        
        # Stage 3: Degradation
        degrad_stage.load_model()
        data = degrad_stage.process(data)
        degrad_stage.unload_model()
        
        # Stage 3.5: Recombine
        recomb_stage.load_model()
        data = recomb_stage.process(data)
        recomb_stage.unload_model()
        
        # Save outputs
        output_path = Path(output_root) / f"{image_idx:05d}_output.png"
        data['composite_image'].save(output_path)
        
        # Also save other outputs
        data['original_image'].save(Path(output_root) / f"{image_idx:05d}_input.png")
        data['albedo'].save(Path(output_root) / f"{image_idx:05d}_albedo.png")
        data['degraded_image'].save(Path(output_root) / f"{image_idx:05d}_degraded_fg.png")
        data['foreground'].save(Path(output_root) / f"{image_idx:05d}_foreground.png")
        data['background'].save(Path(output_root) / f"{image_idx:05d}_background.png")
        
        return {
            'index': image_idx,
            'image_path': str(image_path),
            'output_path': str(output_path),
            'success': True
        }
        
    except Exception as e:
        print(f"Error processing image {image_idx}: {e}")
        return {
            'index': image_idx,
            'image_path': str(image_path),
            'output_path': None,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run the relighting pipeline in parallel with CSV mapping"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/mvp_config.yaml",
        help="Path to pipeline configuration file"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with image paths"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save updated CSV with output paths (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(project_root) / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load CSV
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    
    if 'image_path' not in df.columns:
        print(f"ERROR: CSV must contain 'image_path' column")
        sys.exit(1)
    
    # Determine split name
    split_name = get_split_name_from_csv(args.csv)
    print(f"Detected split: {split_name}")
    
    # Update output paths
    output_root = Path(project_root) / f"data-{split_name}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_root}")
    
    # Get images to process
    image_paths = df['image_path'].tolist()[:args.num_samples]
    
    print(f"Processing {len(image_paths)} images with {args.num_workers} workers...")
    
    # Prepare arguments for parallel processing
    process_args = [
        (idx, img_path, config, str(output_root))
        for idx, img_path in enumerate(image_paths)
    ]
    
    # Process in parallel
    results = []
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap(process_single_image, process_args), 
                          total=len(process_args),
                          desc="Processing images"):
            results.append(result)
    
    # Update CSV with output paths
    output_paths = [None] * len(df)
    for result in results:
        if result['success']:
            output_paths[result['index']] = result['output_path']
    
    # Add output column to first N rows
    df_updated = df.copy()
    df_updated['output_image_path'] = None
    for i, path in enumerate(output_paths[:len(results)]):
        df_updated.loc[i, 'output_image_path'] = path
    
    # Save updated CSV
    if args.output_csv:
        output_csv_path = args.output_csv
    else:
        csv_dir = Path(args.csv).parent
        csv_name = Path(args.csv).stem
        output_csv_path = csv_dir / f"{csv_name}_with_outputs.csv"
    
    df_updated.to_csv(output_csv_path, index=False)
    print(f"\n✅ Updated CSV saved to: {output_csv_path}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_root}")
    print(f"Updated CSV: {output_csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

