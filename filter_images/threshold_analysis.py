"""
Script to visualize images at regular intervals across the score distribution.
Helps identify the optimal filtering threshold by showing quality degradation.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np


def visualize_threshold_analysis(
    results_file: str,
    output_dir: str,
    interval: int = 5000,
    images_per_interval: int = 5
):
    """
    Visualize images at regular intervals to identify filtering threshold.
    
    Args:
        results_file: Path to all_scores.csv or filtered_images.json
        output_dir: Directory to save visualizations
        interval: Spacing between sample groups (default: 5000)
        images_per_interval: Number of images to show per interval (default: 5)
    """
    # Load results
    print(f"Loading results from {results_file}...")
    
    if results_file.endswith('.json'):
        with open(results_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif results_file.endswith('.csv'):
        df = pd.read_csv(results_file)
    else:
        raise ValueError("Input must be .json or .csv file")
    
    # Sort by score (descending)
    df = df.sort_values('lighting_score', ascending=False).reset_index(drop=True)
    
    print(f"Total images: {len(df)}")
    print(f"Score range: {df['lighting_score'].min():.4f} to {df['lighting_score'].max():.4f}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine sample points
    sample_groups = []
    
    # First group (1-5)
    sample_groups.append((0, images_per_interval, "Top 1-5"))
    
    # Regular intervals
    current_pos = interval
    while current_pos < len(df):
        end_pos = min(current_pos + images_per_interval, len(df))
        label = f"#{current_pos+1}-{end_pos}"
        sample_groups.append((current_pos, end_pos, label))
        current_pos += interval
    
    print(f"\nCreating {len(sample_groups)} sample groups at {interval} image intervals")
    
    # Calculate grid dimensions
    n_groups = len(sample_groups)
    n_cols = images_per_interval
    n_rows = n_groups
    
    # Create large visualization
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.5))
    
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS - Scores at Regular Intervals")
    print("="*80)
    
    for group_idx, (start_idx, end_idx, label) in enumerate(sample_groups):
        print(f"\n{label}:")
        
        for img_offset in range(end_idx - start_idx):
            img_idx = start_idx + img_offset
            
            if img_idx >= len(df):
                break
            
            row = df.iloc[img_idx]
            
            # Calculate subplot position
            ax_idx = group_idx * n_cols + img_offset + 1
            ax = plt.subplot(n_rows, n_cols, ax_idx)
            
            try:
                # Load and display image
                img = Image.open(row['image_path'])
                ax.imshow(img)
                
                # Determine color based on position
                rank = img_idx + 1
                score = row['lighting_score']
                
                if rank <= 5000:
                    color = 'green'
                    quality = "EXCELLENT"
                elif rank <= 10000:
                    color = 'lightgreen'
                    quality = "VERY GOOD"
                elif rank <= 20000:
                    color = 'yellowgreen'
                    quality = "GOOD"
                elif rank <= 30000:
                    color = 'yellow'
                    quality = "DECENT"
                elif rank <= 40000:
                    color = 'orange'
                    quality = "FAIR"
                elif rank <= 50000:
                    color = 'darkorange'
                    quality = "ACCEPTABLE"
                elif rank <= 60000:
                    color = 'orangered'
                    quality = "POOR"
                else:
                    color = 'red'
                    quality = "VERY POOR"
                
                # Title with rank and score
                title_text = f"#{rank}\n{score:.4f}\n{quality}"
                ax.set_title(title_text, fontsize=10, fontweight='bold', color=color)
                
                # Print to console
                print(f"  #{rank:5d}: {score:.4f} ({quality}) - {Path(row['image_path']).name}")
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error\nloading", 
                       ha='center', va='center', fontsize=8)
                print(f"  Error loading: {row['image_path']}")
            
            ax.axis('off')
        
        # Add group label on the left
        if img_offset >= 0:
            ax_first = plt.subplot(n_rows, n_cols, group_idx * n_cols + 1)
            ax_first.text(-0.3, 0.5, label, 
                         transform=ax_first.transAxes,
                         fontsize=12, fontweight='bold',
                         rotation=90, va='center', ha='right')
    
    # Add main title
    fig.suptitle(f'Threshold Analysis: Images Every {interval} Ranks (Decreasing Similarity)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Add color legend
    legend_text = (
        "Color Guide:\n"
        "Green = Excellent (1-5k)\n"
        "Light Green = Very Good (5-10k)\n"
        "Yellow-Green = Good (10-20k)\n"
        "Yellow = Decent (20-30k)\n"
        "Orange = Fair (30-40k)\n"
        "Dark Orange = Acceptable (40-50k)\n"
        "Red-Orange = Poor (50-60k)\n"
        "Red = Very Poor (60-70k)"
    )
    fig.text(0.02, 0.02, legend_text, fontsize=10, 
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"threshold_analysis_{interval}interval.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved threshold analysis to: {output_path}")
    
    plt.close()
    
    # Create score summary
    print("\n" + "="*80)
    print("SCORE SUMMARY BY RANGES")
    print("="*80)
    
    ranges = [
        (0, 5000, "Top 5k (EXCELLENT)"),
        (5000, 10000, "5k-10k (VERY GOOD)"),
        (10000, 20000, "10k-20k (GOOD)"),
        (20000, 30000, "20k-30k (DECENT)"),
        (30000, 40000, "30k-40k (FAIR)"),
        (40000, 50000, "40k-50k (ACCEPTABLE)"),
        (50000, 60000, "50k-60k (POOR)"),
        (60000, 70000, "60k-70k (VERY POOR)"),
    ]
    
    summary_data = []
    
    for start, end, label in ranges:
        if end <= len(df):
            subset = df.iloc[start:end]
            summary_data.append({
                'Range': label,
                'Count': len(subset),
                'Min Score': subset['lighting_score'].min(),
                'Max Score': subset['lighting_score'].max(),
                'Mean Score': subset['lighting_score'].mean(),
                'Median Score': subset['lighting_score'].median(),
            })
            
            print(f"\n{label}:")
            print(f"  Count: {len(subset)}")
            print(f"  Score range: {subset['lighting_score'].min():.4f} to {subset['lighting_score'].max():.4f}")
            print(f"  Mean: {subset['lighting_score'].mean():.4f}")
            print(f"  Median: {subset['lighting_score'].median():.4f}")
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "score_ranges_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Saved score summary to: {summary_path}")
    
    # Suggest threshold
    print("\n" + "="*80)
    print("RECOMMENDED THRESHOLDS")
    print("="*80)
    
    thresholds = [
        (30000, "Conservative: Top 30k (Fair+ quality)"),
        (40000, "Moderate: Top 40k (Acceptable+ quality)"),
        (50000, "Lenient: Top 50k (includes some lower quality)"),
        (60000, "Very Lenient: Top 60k (includes poor quality)"),
    ]
    
    for threshold, description in thresholds:
        if threshold <= len(df):
            score_at_threshold = df.iloc[threshold-1]['lighting_score']
            print(f"\n{description}")
            print(f"  Threshold: {threshold} images")
            print(f"  Cutoff score: {score_at_threshold:.4f}")
            print(f"  Images above this score: {threshold:,}")
            print(f"  Images below this score: {len(df) - threshold:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze threshold by visualizing images at regular intervals"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to all_scores.csv or filtered_images.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./threshold_analysis",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5000,
        help="Spacing between sample groups (default: 5000)"
    )
    parser.add_argument(
        "--images_per_interval",
        type=int,
        default=5,
        help="Number of images to show per interval (default: 5)"
    )
    
    args = parser.parse_args()
    
    visualize_threshold_analysis(
        args.results,
        args.output_dir,
        args.interval,
        args.images_per_interval
    )
    
    print("\n✅ Threshold analysis complete!")
    print("\nReview the visualization to determine your optimal cutoff point.")
    print("Look for where image quality noticeably degrades.")


if __name__ == "__main__":
    main()

