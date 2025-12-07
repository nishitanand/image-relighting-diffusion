"""
Script to analyze and visualize filtering results.
"""

import argparse
import json
from pathlib import Path
from utils import (
    load_filtered_results,
    visualize_score_distribution,
    create_image_grid,
    get_statistics,
    split_dataset
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize image filtering results"
    )
    parser.add_argument(
        "--results_json",
        type=str,
        required=True,
        help="Path to filtered_images.json file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis",
        help="Directory to save analysis outputs"
    )
    parser.add_argument(
        "--create_grid",
        action="store_true",
        help="Create visualization grid of top images"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        default=[4, 4],
        help="Grid size as rows cols (default: 4 4)"
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Create train/val/test splits"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_json}...")
    results = load_filtered_results(args.results_json)
    
    image_paths = [r['image_path'] for r in results]
    scores = [r['lighting_score'] for r in results]
    
    print(f"Loaded {len(results)} filtered images")
    
    # Compute and save statistics
    print("\nComputing statistics...")
    stats = get_statistics(scores)
    
    print("\nLighting Score Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to: {stats_file}")
    
    # Create distribution plot
    print("\nCreating score distribution plot...")
    dist_plot_path = output_dir / "score_distribution.png"
    visualize_score_distribution(scores, str(dist_plot_path))
    
    # Create image grid if requested
    if args.create_grid:
        print("\nCreating image grid...")
        grid_path = output_dir / "top_images_grid.png"
        grid_size = tuple(args.grid_size)
        num_images = grid_size[0] * grid_size[1]
        
        create_image_grid(
            image_paths[:num_images],
            grid_size=grid_size,
            output_path=str(grid_path),
            scores=scores[:num_images]
        )
    
    # Create splits if requested
    if args.create_splits:
        print("\nCreating dataset splits...")
        splits_dir = output_dir / "splits"
        split_dataset(image_paths, output_dir=str(splits_dir))
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

