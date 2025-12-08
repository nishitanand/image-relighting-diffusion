"""
Visualization utilities for the relighting pipeline.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import json
from typing import Optional, List
import numpy as np


def visualize_pipeline_stages(
    image_id: int,
    data_root: str = "data",
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize all pipeline stages for a single image.

    Args:
        image_id: Image ID to visualize
        data_root: Root directory containing data
        output_path: Path to save visualization (optional)
        show: Whether to display the plot
    """
    data_root = Path(data_root)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Pipeline Stages - Image {image_id:05d}", fontsize=16, fontweight='bold')

    # Helper function to load and display image
    def show_image(ax, image_path, title):
        if image_path.exists():
            img = Image.open(image_path)
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
            ax.set_title(title, fontsize=12)
            ax.axis('off')

    # Row 1: Original, Foreground, Background
    show_image(axes[0, 0], data_root / 'raw' / f"{image_id:05d}.png", "1. Original Image")
    show_image(axes[0, 1], data_root / 'stage_1' / f"{image_id:05d}_foreground.png", "2. Foreground (SAM3)")
    show_image(axes[0, 2], data_root / 'stage_1' / f"{image_id:05d}_background.png", "3. Background")

    # Row 2: Mask, Albedo, Specular
    show_image(axes[1, 0], data_root / 'stage_1' / f"{image_id:05d}_mask.png", "4. Segmentation Mask")
    show_image(axes[1, 1], data_root / 'stage_2' / f"{image_id:05d}_albedo.png", "5. Albedo (IntrinsicAnything)")

    specular_path = data_root / 'stage_2' / f"{image_id:05d}_specular.png"
    if specular_path.exists():
        show_image(axes[1, 2], specular_path, "6. Specular")
    else:
        axes[1, 2].axis('off')
        axes[1, 2].text(0.5, 0.5, 'No Specular', ha='center', va='center')
        axes[1, 2].set_title("6. Specular (Optional)", fontsize=12)

    # Row 3: Shadow Image, Final Output, Caption
    show_image(axes[2, 0], data_root / 'stage_3' / f"{image_id:05d}_shadow.png", "7. Shadow Image (IC-Light)")
    show_image(axes[2, 1], data_root / 'outputs' / f"{image_id:05d}_output.png", "8. Final Output")

    # Show caption and parameters
    caption_path = data_root / 'stage_4' / f"{image_id:05d}_caption.txt"
    params_path = data_root / 'stage_3' / f"{image_id:05d}_params.json"

    caption_text = ""
    if caption_path.exists():
        with open(caption_path, 'r') as f:
            caption = f.read()
        caption_text = f"Caption:\n{caption[:200]}..."

    if params_path.exists():
        with open(params_path, 'r') as f:
            params = json.load(f)
        caption_text += f"\n\nLight Direction: {params.get('light_direction')}"
        caption_text += f"\nIntensity: {params.get('shadow_intensity', 'N/A')}"

    axes[2, 2].text(0.05, 0.95, caption_text, fontsize=9, va='top', wrap=True)
    axes[2, 2].set_title("9. Caption & Parameters", fontsize=12)
    axes[2, 2].axis('off')

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch(
    image_ids: List[int],
    data_root: str = "data",
    output_dir: Optional[str] = None
):
    """
    Visualize multiple images in a grid.

    Args:
        image_ids: List of image IDs to visualize
        data_root: Root directory containing data
        output_dir: Directory to save visualizations (optional)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        output_path = output_dir / f"visualization_{image_id:05d}.png" if output_dir else None
        visualize_pipeline_stages(image_id, data_root, output_path, show=False)

    print(f"Generated {len(image_ids)} visualizations")


def compare_stages(
    image_id: int,
    stages: List[str] = ['original', 'albedo', 'shadow'],
    data_root: str = "data",
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Compare specific stages side-by-side.

    Args:
        image_id: Image ID to visualize
        stages: List of stages to compare
        data_root: Root directory
        output_path: Path to save comparison
        show: Whether to display
    """
    data_root = Path(data_root)

    # Map stage names to file paths
    stage_paths = {
        'original': data_root / 'raw' / f"{image_id:05d}.png",
        'foreground': data_root / 'stage_1' / f"{image_id:05d}_foreground.png",
        'mask': data_root / 'stage_1' / f"{image_id:05d}_mask.png",
        'albedo': data_root / 'stage_2' / f"{image_id:05d}_albedo.png",
        'shadow': data_root / 'stage_3' / f"{image_id:05d}_shadow.png",
        'output': data_root / 'outputs' / f"{image_id:05d}_output.png"
    }

    # Create figure
    fig, axes = plt.subplots(1, len(stages), figsize=(5*len(stages), 5))
    if len(stages) == 1:
        axes = [axes]

    fig.suptitle(f"Stage Comparison - Image {image_id:05d}", fontsize=14, fontweight='bold')

    for i, stage in enumerate(stages):
        if stage in stage_paths:
            path = stage_paths[stage]
            if path.exists():
                img = Image.open(path)
                axes[i].imshow(img)
                axes[i].set_title(stage.capitalize(), fontsize=12)
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'{stage}\nNot Found', ha='center', va='center')
                axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'Unknown Stage:\n{stage}', ha='center', va='center')
            axes[i].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_lighting_distribution(data_root: str = "data", output_path: Optional[str] = None):
    """
    Plot distribution of light directions used.

    Args:
        data_root: Root directory
        output_path: Path to save plot
    """
    data_root = Path(data_root)
    params_dir = data_root / 'stage_3'

    light_directions = []

    # Collect all light directions
    for params_file in params_dir.glob('*_params.json'):
        with open(params_file, 'r') as f:
            params = json.load(f)
            light_dir = params.get('light_direction')
            if light_dir:
                light_directions.append(light_dir)

    if not light_directions:
        print("No lighting parameters found")
        return

    light_directions = np.array(light_directions)

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(light_directions[:, 0], light_directions[:, 1], light_directions[:, 2],
               c='blue', marker='o', s=100, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Light Direction Distribution (n={len(light_directions)})', fontsize=14)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Visualize pipeline outputs")
    parser.add_argument("--image-id", type=int, default=0, help="Image ID to visualize")
    parser.add_argument("--data-root", type=str, default="data", help="Data root directory")
    parser.add_argument("--output", type=str, default=None, help="Output path")

    args = parser.parse_args()

    visualize_pipeline_stages(args.image_id, args.data_root, args.output)
