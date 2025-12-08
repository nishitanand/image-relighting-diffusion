"""
Test the pipeline on a single image for debugging and validation.
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
from src.utils.visualization import visualize_pipeline_stages


def main():
    parser = argparse.ArgumentParser(
        description="Test pipeline on a single image"
    )
    parser.add_argument(
        "--image-id",
        type=int,
        default=0,
        help="Image ID to process (default: 0)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline configuration"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization after processing"
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

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model config
    model_config_path = config_path.parent / 'model_config.yaml'
    if model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        config.update(model_config)

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
    logger.info(f"TESTING PIPELINE ON SINGLE IMAGE (ID: {args.image_id})")
    logger.info("="*80)

    # Check if image exists
    data_root = Path(config['paths']['data_root'])
    raw_dir = data_root / 'raw'
    image_path = raw_dir / f"{args.image_id:05d}.png"

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        logger.error("Please run: python scripts/download_ffhq.py")
        sys.exit(1)

    logger.info(f"Processing image: {image_path}")

    try:
        # Initialize pipeline
        logger.info("\nInitializing pipeline...")
        pipeline = RelightingPipeline(config)

        # Process single image
        logger.info("\nProcessing image...")
        result = pipeline.process_image(str(image_path), args.image_id)

        if result is None:
            logger.error("Processing failed!")
            sys.exit(1)

        logger.info("\n" + "="*80)
        logger.info("PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        # Print result summary
        logger.info(f"\nResults:")
        logger.info(f"  Image ID: {result['image_id']}")
        logger.info(f"  Segmentation score: {result.get('segmentation_score', 'N/A')}")
        logger.info(f"  Mask coverage: {result.get('mask_coverage', 'N/A'):.1%}")
        logger.info(f"  Light direction: {result.get('light_direction', 'N/A')}")
        logger.info(f"  Caption length: {result.get('caption_length', 'N/A')} chars")

        logger.info(f"\nCaption:")
        logger.info(f"  {result.get('caption', 'N/A')}")

        logger.info(f"\nOutputs saved to:")
        logger.info(f"  Stage 1: {data_root / 'stage_1'}")
        logger.info(f"  Stage 2: {data_root / 'stage_2'}")
        logger.info(f"  Stage 3: {data_root / 'stage_3'}")
        logger.info(f"  Stage 4: {data_root / 'stage_4'}")
        logger.info(f"  Final: {config['paths']['output_root']}")

        # Visualize if requested
        if args.visualize:
            logger.info("\nGenerating visualization...")
            try:
                output_path = data_root / 'outputs' / f"visualization_{args.image_id:05d}.png"
                visualize_pipeline_stages(
                    args.image_id,
                    data_root=str(data_root),
                    output_path=str(output_path),
                    show=True
                )
                logger.info(f"Visualization saved to: {output_path}")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nProcessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
