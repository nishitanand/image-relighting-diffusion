"""
Utility functions for image filtering and analysis.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_filtered_results(json_path: str) -> List[Dict]:
    """
    Load filtered image results from JSON file.
    
    Args:
        json_path: Path to the filtered_images.json file
        
    Returns:
        List of dictionaries containing image paths and scores
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def visualize_score_distribution(scores: List[float], output_path: str = None):
    """
    Visualize the distribution of lighting scores.
    
    Args:
        scores: List of lighting scores
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Lighting Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lighting Scores')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_image_grid(image_paths: List[str], grid_size: tuple = (4, 4), 
                     output_path: str = None, scores: List[float] = None):
    """
    Create a grid visualization of images.
    
    Args:
        image_paths: List of image file paths
        grid_size: Tuple of (rows, cols) for the grid
        output_path: Path to save the grid image (optional)
        scores: Optional list of scores to display with images
    """
    rows, cols = grid_size
    num_images = min(len(image_paths), rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx in range(rows * cols):
        ax = axes[idx]
        
        if idx < num_images:
            try:
                img = Image.open(image_paths[idx])
                ax.imshow(img)
                
                if scores:
                    title = f"Score: {scores[idx]:.3f}"
                    ax.set_title(title, fontsize=10)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\n{Path(image_paths[idx]).name}", 
                       ha='center', va='center')
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def get_statistics(scores: List[float]) -> Dict:
    """
    Compute statistics for lighting scores.
    
    Args:
        scores: List of lighting scores
        
    Returns:
        Dictionary of statistics
    """
    scores_array = np.array(scores)
    
    return {
        'count': len(scores),
        'mean': float(np.mean(scores_array)),
        'std': float(np.std(scores_array)),
        'min': float(np.min(scores_array)),
        'max': float(np.max(scores_array)),
        'median': float(np.median(scores_array)),
        'q25': float(np.percentile(scores_array, 25)),
        'q75': float(np.percentile(scores_array, 75))
    }


def split_dataset(image_paths: List[str], split_ratios: Dict[str, float] = None,
                 output_dir: str = None) -> Dict[str, List[str]]:
    """
    Split filtered images into train/val/test sets.
    
    Args:
        image_paths: List of image file paths
        split_ratios: Dictionary with 'train', 'val', 'test' ratios (default: 0.8/0.1/0.1)
        output_dir: Directory to save split files (optional)
        
    Returns:
        Dictionary with split names as keys and image path lists as values
    """
    if split_ratios is None:
        split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    
    # Validate ratios
    assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    # Shuffle images
    np.random.shuffle(image_paths)
    
    # Calculate split indices
    n_total = len(image_paths)
    n_train = int(n_total * split_ratios['train'])
    n_val = int(n_total * split_ratios['val'])
    
    splits = {
        'train': image_paths[:n_train],
        'val': image_paths[n_train:n_train + n_val],
        'test': image_paths[n_train + n_val:]
    }
    
    # Save to files if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, paths in splits.items():
            split_file = output_dir / f"{split_name}.txt"
            with open(split_file, 'w') as f:
                for path in paths:
                    f.write(f"{path}\n")
            print(f"Saved {split_name} split ({len(paths)} images) to: {split_file}")
    
    return splits


def verify_images(image_paths: List[str]) -> tuple:
    """
    Verify that all images can be loaded properly.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Tuple of (valid_paths, invalid_paths)
    """
    valid = []
    invalid = []
    
    for path in image_paths:
        try:
            img = Image.open(path)
            img.verify()
            valid.append(path)
        except Exception as e:
            print(f"Invalid image {path}: {e}")
            invalid.append(path)
    
    return valid, invalid

