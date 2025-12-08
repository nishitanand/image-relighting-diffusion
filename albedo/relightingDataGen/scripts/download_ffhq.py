"""
Download FFHQ dataset subset from HuggingFace.
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
import yaml


def download_ffhq(
    output_dir: str,
    num_samples: int = 100,
    dataset_name: str = "bitmind/ffhq-256",
    start_idx: int = 0
):
    """
    Download FFHQ dataset subset.

    Args:
        output_dir: Directory to save images
        num_samples: Number of images to download
        dataset_name: HuggingFace dataset name
        start_idx: Starting index in dataset
    """
    print(f"Downloading {num_samples} images from {dataset_name}...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset from HuggingFace...")
    end_idx = start_idx + num_samples
    dataset = load_dataset(
        dataset_name,
        split=f"train[{start_idx}:{end_idx}]",
        trust_remote_code=True
    )

    print(f"Downloaded {len(dataset)} samples")

    # Save images
    print(f"Saving images to {output_path}...")
    for idx, sample in enumerate(tqdm(dataset, desc="Saving images")):
        image = sample['image']

        # Save with zero-padded filename
        image_id = start_idx + idx
        filename = f"{image_id:05d}.png"
        filepath = output_path / filename

        image.save(filepath)

    print(f"Successfully saved {len(dataset)} images to {output_path}")
    print(f"Image IDs: {start_idx:05d} to {start_idx + num_samples - 1:05d}")


def main():
    parser = argparse.ArgumentParser(description="Download FFHQ dataset subset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for images"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to download"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="bitmind/ffhq-256",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline config file"
    )

    args = parser.parse_args()

    # Load config if using defaults
    if args.output_dir == "data/raw" and args.config:
        config_path = Path(project_root) / args.config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Override with config values if not explicitly set
            if 'dataset' in config:
                dataset_config = config['dataset']
                args.output_dir = config['paths']['data_root'] + '/raw'
                args.num_samples = dataset_config.get('num_download', args.num_samples)
                args.dataset_name = dataset_config.get('name', args.dataset_name)

    download_ffhq(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        dataset_name=args.dataset_name,
        start_idx=args.start_idx
    )


if __name__ == "__main__":
    main()
