"""
Main script to run the relighting pipeline end-to-end.
"""

import os
import sys
from pathlib import Path
import argparse
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import setup_logging
from src.pipeline.pipeline_runner import RelightingPipeline


def load_config(config_path: str) -> dict:
    """Load configuration from YAML files."""
    with open(config_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)

    # Load model config
    model_config_path = Path(config_path).parent / 'model_config.yaml'
    if model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)

        # Merge configs
        pipeline_config.update(model_config)

    return pipeline_config


def main():
    parser = argparse.ArgumentParser(
        description="Run the relighting pipeline to generate dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline configuration file"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file with image paths (must have 'image_path' column)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all available)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(project_root) / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Update num_samples if specified
    if args.num_samples is not None:
        config['pipeline']['num_samples'] = args.num_samples

    # Setup logging
    log_dir = config.get('paths', {}).get('logs_root', 'logs')
    log_level = getattr(__import__('logging'), args.log_level)

    logger = setup_logging(
        log_dir=log_dir,
        log_level=log_level,
        log_to_file=True,
        log_to_console=True
    )

    logger.info("="*80)
    logger.info("RELIGHTING PIPELINE - Starting execution")
    logger.info("="*80)
    logger.info(f"Configuration loaded from: {config_path}")
    logger.info(f"Number of samples: {config['pipeline'].get('num_samples', 'all')}")
    
    if args.csv:
        logger.info(f"CSV file: {args.csv}")
    else:
        logger.info(f"Data root: {config['paths']['data_root']}")
    
    logger.info(f"Models root: {config['paths']['models_root']}")
    logger.info(f"Output root: {config['paths']['output_root']}")

    # Check if data exists (only if not using CSV)
    if not args.csv:
        data_root = Path(config['paths']['data_root'])
        raw_dir = data_root / 'raw'

        if not raw_dir.exists() or not list(raw_dir.glob('*.png')):
            logger.error(f"No images found in {raw_dir}")
            logger.error("Please provide --csv argument or run: python scripts/download_ffhq.py")
            sys.exit(1)
    else:
        # Validate CSV file exists
        if not Path(args.csv).exists():
            logger.error(f"CSV file not found: {args.csv}")
            sys.exit(1)

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
        logger.info(f"Results saved to: {config['paths']['output_root']}")

        # Print sample caption
        if results:
            logger.info("\nSample output:")
            logger.info(f"Image ID: {results[0]['image_id']}")
            logger.info(f"Caption: {results[0].get('caption', 'N/A')[:200]}...")

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
