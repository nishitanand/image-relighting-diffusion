"""
Enhanced pipeline runner with CSV mapping and dynamic output directories.
Processes train/val/test splits and creates updated CSVs with output paths.
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import pandas as pd

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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML files."""
    with open(config_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)

    # Load model config if exists
    model_config_path = Path(config_path).parent / 'model_config.yaml'
    if model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        pipeline_config.update(model_config)

    return pipeline_config


def main():
    parser = argparse.ArgumentParser(
        description="Run relighting pipeline with CSV mapping"
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
        help="Path to CSV file with image paths (must have 'image_path' column)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all available)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save updated CSV with output paths (default: auto-generated)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )

    args = parser.parse_args()

    # Validate CSV exists
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    # Load CSV
    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    
    if 'image_path' not in df.columns:
        print(f"Error: CSV must contain 'image_path' column. Found: {df.columns.tolist()}")
        sys.exit(1)

    # Determine split name and create output directory
    split_name = get_split_name_from_csv(args.csv)
    print(f"Detected split: {split_name}")

    # Load configuration
    config_path = Path(project_root) / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Update paths for this split
    output_root = Path(project_root) / f"data-{split_name}"
    data_root = Path(project_root) / f"data_{split_name}_intermediate"
    
    config['paths']['output_root'] = str(output_root)
    config['paths']['data_root'] = str(data_root)
    
    # Create log directory
    log_dir = Path(project_root) / "logs" / split_name
    log_dir.mkdir(parents=True, exist_ok=True)
    config['paths']['logs_root'] = str(log_dir)

    # Update num_samples if specified
    if args.num_samples is not None:
        config['pipeline']['num_samples'] = args.num_samples

    # Setup logging
    log_level = getattr(__import__('logging'), args.log_level)
    logger = setup_logging(
        log_dir=str(log_dir),
        log_level=log_level,
        log_to_file=True,
        log_to_console=True
    )

    logger.info("="*80)
    logger.info(f"RELIGHTING PIPELINE - {split_name.upper()} SPLIT")
    logger.info("="*80)
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Input CSV: {args.csv}")
    logger.info(f"Output directory: {output_root}")
    logger.info(f"Number of samples: {args.num_samples or 'all'}")

    # Initialize and run pipeline
    try:
        logger.info("\nInitializing pipeline...")
        pipeline = RelightingPipeline(config)

        logger.info("\nRunning pipeline...")
        results = pipeline.run_pipeline(
            num_samples=config['pipeline'].get('num_samples'),
            csv_path=args.csv
        )

        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION COMPLETED")
        logger.info("="*80)
        logger.info(f"Processed {len(results)} images successfully")
        logger.info(f"Results saved to: {output_root}")

        # Update CSV with output paths
        logger.info("\nUpdating CSV with output image paths...")
        df_updated = df.copy()
        df_updated['output_image_path'] = None
        
        # Map output paths based on processing order
        num_processed = min(len(results), len(df))
        for idx in range(num_processed):
            output_path = output_root / f"{idx:05d}_output.png"
            if output_path.exists():
                df_updated.loc[idx, 'output_image_path'] = str(output_path)
        
        # Determine output CSV path
        if args.output_csv:
            output_csv_path = args.output_csv
        else:
            csv_dir = Path(args.csv).parent
            csv_name = Path(args.csv).stem
            output_csv_path = csv_dir / f"{csv_name}_with_outputs.csv"
        
        # Save updated CSV
        df_updated.to_csv(output_csv_path, index=False)
        logger.info(f"Updated CSV saved to: {output_csv_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Split: {split_name}")
        print(f"Successful: {len(results)} / {len(df)}")
        print(f"Output directory: {output_root}")
        print(f"Updated CSV: {output_csv_path}")
        print(f"{'='*80}\n")

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

