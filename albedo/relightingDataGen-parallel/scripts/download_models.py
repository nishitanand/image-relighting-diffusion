"""
Download all model checkpoints for the relighting pipeline.
"""

import os
import sys
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download, hf_hub_download
import yaml


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_sam2(models_dir: str):
    """
    Download SAM2 model checkpoint.

    Args:
        models_dir: Base models directory
    """
    print("\n" + "="*80)
    print("Downloading SAM2 model...")
    print("="*80)

    sam2_dir = Path(models_dir) / "sam2"
    sam2_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download SAM2 checkpoint from HuggingFace
        checkpoint = hf_hub_download(
            repo_id="facebook/sam2.1-hiera-large",
            filename="sam2_hiera_l.pt",
            local_dir=str(sam2_dir),
            local_dir_use_symlinks=False
        )
        print(f"SAM2 checkpoint downloaded to: {checkpoint}")
    except Exception as e:
        print(f"Error downloading SAM2: {e}")
        print("Please download manually from: https://github.com/facebookresearch/sam2")


def download_intrinsic_anything(models_dir: str):
    """
    Download IntrinsicAnything model.

    Args:
        models_dir: Base models directory
    """
    print("\n" + "="*80)
    print("Downloading IntrinsicAnything model...")
    print("="*80)

    intrinsic_dir = Path(models_dir) / "intrinsic_anything"
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the full IntrinsicAnything repository
        snapshot_download(
            repo_id="zju3dv/IntrinsicAnything",
            local_dir=str(intrinsic_dir),
            local_dir_use_symlinks=False
        )
        print(f"IntrinsicAnything model downloaded to: {intrinsic_dir}")
    except Exception as e:
        print(f"Error downloading IntrinsicAnything: {e}")
        print("Please download manually from: https://huggingface.co/zju3dv/IntrinsicAnything")


def download_ic_light(models_dir: str):
    """
    Download IC-Light model.

    Args:
        models_dir: Base models directory
    """
    print("\n" + "="*80)
    print("Downloading IC-Light model...")
    print("="*80)

    ic_light_dir = Path(models_dir) / "ic_light"
    ic_light_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download IC-Light checkpoint
        snapshot_download(
            repo_id="lllyasviel/IC-Light",
            local_dir=str(ic_light_dir),
            local_dir_use_symlinks=False
        )
        print(f"IC-Light model downloaded to: {ic_light_dir}")
    except Exception as e:
        print(f"Error downloading IC-Light: {e}")
        print("Please download manually from: https://huggingface.co/lllyasviel/IC-Light")


def download_qwen_vl(models_dir: str):
    """
    Download Qwen2.5-VL model.

    Args:
        models_dir: Base models directory
    """
    print("\n" + "="*80)
    print("Downloading Qwen2.5-VL-7B model...")
    print("="*80)
    print("Note: This is a large model (~16GB). It will be downloaded on first use.")
    print("To pre-download, uncomment the code below.")

    qwen_dir = Path(models_dir) / "qwen2.5_vl"
    qwen_dir.mkdir(parents=True, exist_ok=True)

    # Pre-download (optional - can be slow)
    # Uncomment to download now instead of on first use
    # try:
    #     from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    #
    #     print("Downloading Qwen2.5-VL-7B-Instruct...")
    #     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #         "Qwen/Qwen2.5-VL-7B-Instruct",
    #         torch_dtype="auto",
    #         device_map="cpu",
    #         cache_dir=str(qwen_dir)
    #     )
    #     processor = AutoProcessor.from_pretrained(
    #         "Qwen/Qwen2.5-VL-7B-Instruct",
    #         cache_dir=str(qwen_dir)
    #     )
    #     print(f"Qwen2.5-VL model downloaded to: {qwen_dir}")
    # except Exception as e:
    #     print(f"Error downloading Qwen2.5-VL: {e}")

    print(f"Qwen2.5-VL will be cached at: {qwen_dir}")


def download_sd_base_model(models_dir: str):
    """
    Download Stable Diffusion 1.5 base model (for IC-Light).

    Args:
        models_dir: Base models directory
    """
    print("\n" + "="*80)
    print("Downloading Stable Diffusion 1.5 base model...")
    print("="*80)
    print("Note: This will be downloaded automatically when IC-Light runs.")
    print("No action needed unless you want to pre-download.")


def main():
    parser = argparse.ArgumentParser(description="Download all model checkpoints")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sam2", "intrinsic", "iclight", "qwen", "all"],
        default="all",
        help="Which model to download (default: all)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline config file"
    )

    args = parser.parse_args()

    # Load config if using defaults
    if args.models_dir == "models" and args.config:
        config_path = Path(project_root) / args.config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            args.models_dir = config['paths']['models_root']

    print(f"Models will be downloaded to: {args.models_dir}")
    print(f"This may take some time depending on your internet connection.")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Download requested models
    if args.model == "all" or args.model == "sam2":
        download_sam2(str(models_dir))

    if args.model == "all" or args.model == "intrinsic":
        download_intrinsic_anything(str(models_dir))

    if args.model == "all" or args.model == "iclight":
        download_ic_light(str(models_dir))

    if args.model == "all" or args.model == "qwen":
        download_qwen_vl(str(models_dir))

    if args.model == "all":
        download_sd_base_model(str(models_dir))

    print("\n" + "="*80)
    print("Model download process completed!")
    print("="*80)
    print("\nNote: Some models may be downloaded automatically on first use.")
    print("Check the logs above for any manual download instructions.")


if __name__ == "__main__":
    main()
