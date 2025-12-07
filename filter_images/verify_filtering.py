"""
Script to verify filtering results by visualizing high and low scoring images.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np


def visualize_score_comparison(results_json: str, output_dir: str = None):
    """
    Visualize images at different score levels to verify filtering quality.
    
    Args:
        results_json: Path to filtered_images.json or all_scores.csv
        output_dir: Directory to save visualization (optional)
    """
    # Load results
    print(f"Loading results from {results_json}...")
    
    if results_json.endswith('.json'):
        with open(results_json, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif results_json.endswith('.csv'):
        df = pd.read_csv(results_json)
    else:
        raise ValueError("Input must be .json or .csv file")
    
    # Sort by score
    df = df.sort_values('lighting_score', ascending=False).reset_index(drop=True)
    
    print(f"Total images: {len(df)}")
    print(f"Score range: {df['lighting_score'].min():.4f} to {df['lighting_score'].max():.4f}")
    print(f"Mean score: {df['lighting_score'].mean():.4f}")
    
    # Get different score ranges
    print("\n" + "="*70)
    print("SCORE ANALYSIS")
    print("="*70)
    
    # Top 10
    print("\n📊 TOP 10 HIGHEST SCORES:")
    for idx in range(min(10, len(df))):
        print(f"  #{idx+1:5d}: {df.loc[idx, 'lighting_score']:.4f} - {Path(df.loc[idx, 'image_path']).name}")
    
    # Around 50k mark (49999th image)
    if len(df) >= 50000:
        cutoff_idx = 49999
        print(f"\n✂️  AT 50K CUTOFF (image #{cutoff_idx+1}):")
        print(f"  #{cutoff_idx+1:5d}: {df.loc[cutoff_idx, 'lighting_score']:.4f} - {Path(df.loc[cutoff_idx, 'image_path']).name}")
        
        # Show a few around the cutoff
        print(f"\n  Around the 50k cutoff:")
        for offset in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
            idx = cutoff_idx + offset
            if 0 <= idx < len(df):
                status = "✓ SELECTED" if idx < 50000 else "✗ REJECTED"
                print(f"  #{idx+1:5d}: {df.loc[idx, 'lighting_score']:.4f} {status}")
    
    # Bottom 10 overall
    print(f"\n📉 BOTTOM 10 LOWEST SCORES (ALL DATA):")
    for i in range(min(10, len(df))):
        idx = len(df) - 1 - i
        print(f"  #{idx+1:5d}: {df.loc[idx, 'lighting_score']:.4f} - {Path(df.loc[idx, 'image_path']).name}")
    
    # Bottom 20 from filtered set (if we have 50k+ images)
    if len(df) >= 50000:
        print(f"\n⚠️  BOTTOM 20 FROM FILTERED SET (Worst of the 50k selected):")
        for i in range(20):
            idx = 50000 - 20 + i
            status = "✓ SELECTED" if idx < 50000 else "✗ REJECTED"
            print(f"  #{idx+1:5d}: {df.loc[idx, 'lighting_score']:.4f} {status} - {Path(df.loc[idx, 'image_path']).name}")
    
    # Random samples from filtered set
    if len(df) >= 50000:
        print(f"\n🎲 20 RANDOM SAMPLES FROM FILTERED SET:")
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.choice(50000, size=min(20, 50000), replace=False)
        random_indices = sorted(random_indices)
        for idx in random_indices:
            print(f"  #{idx+1:5d}: {df.loc[idx, 'lighting_score']:.4f} - {Path(df.loc[idx, 'image_path']).name}")
    
    print("\n" + "="*70)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Define what to show
    sections = []
    
    # Top 10
    if len(df) >= 10:
        sections.append(('TOP 10 (Highest Scores - BEST Lighting)', df.head(10)))
    
    # Random 20 from filtered set
    if len(df) >= 50000:
        np.random.seed(42)
        random_indices = np.random.choice(50000, size=min(20, 50000), replace=False)
        random_indices = sorted(random_indices)
        random_df = df.iloc[random_indices].head(10)  # Show first 10 of the 20 random
        sections.append(('RANDOM SAMPLES FROM FILTERED SET', random_df))
    
    # Bottom 20 from filtered set (worst of selected)
    if len(df) >= 50000:
        bottom_filtered_start = 50000 - 20
        bottom_filtered_df = df.iloc[bottom_filtered_start:50000].head(10)
        sections.append(('BOTTOM 20 OF FILTERED SET (Worst Selected)', bottom_filtered_df))
    
    # Around 50k cutoff
    if len(df) >= 50000:
        cutoff_start = max(0, 49995)
        cutoff_end = min(len(df), 50005)
        sections.append((f'AROUND 50K CUTOFF (#{cutoff_start+1}-#{cutoff_end})', 
                        df.iloc[cutoff_start:cutoff_end].head(10)))
    
    # Bottom 10 overall (rejected)
    if len(df) >= 10:
        sections.append(('BOTTOM 10 OVERALL (Rejected - WORST Lighting)', df.tail(10)))
    
    # Create grid
    n_sections = len(sections)
    n_cols = 5
    n_rows_per_section = 2
    
    for section_idx, (title, section_df) in enumerate(sections):
        print(f"\n{title}:")
        
        for img_idx, (_, row) in enumerate(section_df.iterrows()):
            if img_idx >= 10:  # Show max 10 per section
                break
                
            ax_idx = section_idx * (n_rows_per_section * n_cols) + img_idx + 1
            ax = plt.subplot(n_sections * n_rows_per_section, n_cols, ax_idx)
            
            try:
                img = Image.open(row['image_path'])
                ax.imshow(img)
                
                # Title with rank and score
                rank = df.index[df['image_path'] == row['image_path']].tolist()[0] + 1
                score = row['lighting_score']
                
                if section_idx == 0:  # Top scores - green
                    color = 'green'
                    status = "✓ TOP"
                elif section_idx == len(sections) - 1:  # Bottom scores - red
                    color = 'red'
                    status = "✗ LOW"
                else:  # Middle - orange
                    color = 'orange'
                    if rank <= 50000:
                        status = "✓ IN"
                    else:
                        status = "✗ OUT"
                
                title_text = f"#{rank}: {score:.4f}\n{status}"
                ax.set_title(title_text, fontsize=10, fontweight='bold', color=color)
                
                print(f"  #{rank:5d}: {score:.4f} - {Path(row['image_path']).name}")
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\nimage", 
                       ha='center', va='center', fontsize=8)
                print(f"  Error loading: {row['image_path']}")
            
            ax.axis('off')
        
        # Add section title
        if img_idx >= 0:
            first_ax_in_section = section_idx * (n_rows_per_section * n_cols) + 1
            fig.text(0.5, 1 - (section_idx * (n_rows_per_section / (n_sections * n_rows_per_section))) - 0.02,
                    title, ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "filtering_verification.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved verification plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Create additional visualizations for bottom 20 of filtered set
    if len(df) >= 50000:
        print("\n" + "="*70)
        print("Creating detailed visualization: Bottom 20 from Filtered Set")
        print("="*70)
        
        fig, axes = plt.subplots(2, 10, figsize=(25, 6))
        bottom_20_indices = range(49980, 50000)
        
        for i, idx in enumerate(bottom_20_indices):
            row_idx = i // 10
            col_idx = i % 10
            ax = axes[row_idx, col_idx]
            
            try:
                img = Image.open(df.iloc[idx]['image_path'])
                ax.imshow(img)
                ax.set_title(f"#{idx+1}\n{df.iloc[idx]['lighting_score']:.4f}", 
                           fontsize=10, color='orange', fontweight='bold')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')
        
        fig.suptitle('Bottom 20 Images from Filtered Set (Worst of 50k Selected)', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if output_dir:
            bottom20_path = output_dir / "bottom_20_filtered.png"
            plt.savefig(bottom20_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved bottom 20 filtered plot to: {bottom20_path}")
        else:
            plt.show()
        
        plt.close()
    
    # Create visualization for random samples
    if len(df) >= 50000:
        print("\n" + "="*70)
        print("Creating detailed visualization: 20 Random Samples from Filtered Set")
        print("="*70)
        
        fig, axes = plt.subplots(2, 10, figsize=(25, 6))
        np.random.seed(42)
        random_indices = np.random.choice(50000, size=20, replace=False)
        random_indices = sorted(random_indices)
        
        for i, idx in enumerate(random_indices):
            row_idx = i // 10
            col_idx = i % 10
            ax = axes[row_idx, col_idx]
            
            try:
                img = Image.open(df.iloc[idx]['image_path'])
                ax.imshow(img)
                ax.set_title(f"#{idx+1}\n{df.iloc[idx]['lighting_score']:.4f}", 
                           fontsize=10, color='blue', fontweight='bold')
                print(f"  Random sample #{idx+1}: {df.iloc[idx]['lighting_score']:.4f}")
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')
        
        fig.suptitle('20 Random Samples from Filtered Set (Typical Quality)', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if output_dir:
            random_path = output_dir / "random_20_filtered.png"
            plt.savefig(random_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved random samples plot to: {random_path}")
        else:
            plt.show()
        
        plt.close()
    
    # Create a detailed comparison plot
    if len(df) >= 10:
        fig, axes = plt.subplots(2, 10, figsize=(25, 6))
        
        # Top row: highest scores
        for i in range(10):
            ax = axes[0, i]
            try:
                img = Image.open(df.iloc[i]['image_path'])
                ax.imshow(img)
                ax.set_title(f"#{i+1}\n{df.iloc[i]['lighting_score']:.4f}", 
                           fontsize=10, color='green', fontweight='bold')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')
        
        # Bottom row: lowest scores
        for i in range(10):
            ax = axes[1, i]
            idx = len(df) - 10 + i
            try:
                img = Image.open(df.iloc[idx]['image_path'])
                ax.imshow(img)
                ax.set_title(f"#{idx+1}\n{df.iloc[idx]['lighting_score']:.4f}", 
                           fontsize=10, color='red', fontweight='bold')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')
        
        # Add row labels
        fig.text(0.02, 0.75, 'HIGHEST SCORES\n(BEST Lighting)', 
                fontsize=12, fontweight='bold', rotation=90, va='center', color='green')
        fig.text(0.02, 0.25, 'LOWEST SCORES\n(WORST Lighting)', 
                fontsize=12, fontweight='bold', rotation=90, va='center', color='red')
        
        plt.tight_layout()
        
        if output_dir:
            comparison_path = output_dir / "top_vs_bottom_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved comparison plot to: {comparison_path}")
        else:
            plt.show()
        
        plt.close()


def show_cutoff_analysis(results_json: str, cutoff: int = 50000):
    """
    Show detailed analysis around the cutoff point.
    
    Args:
        results_json: Path to results file
        cutoff: Cutoff index (default: 50000)
    """
    # Load results
    if results_json.endswith('.json'):
        with open(results_json, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(results_json)
    
    df = df.sort_values('lighting_score', ascending=False).reset_index(drop=True)
    
    if len(df) < cutoff:
        print(f"⚠️  Only {len(df)} images available, less than cutoff of {cutoff}")
        cutoff = len(df)
    
    print("\n" + "="*70)
    print(f"CUTOFF ANALYSIS at {cutoff}")
    print("="*70)
    
    # Show statistics
    if cutoff > 0:
        selected = df.head(cutoff)
        rejected = df.iloc[cutoff:]
        
        print(f"\nSELECTED (top {cutoff}):")
        print(f"  Score range: {selected['lighting_score'].min():.4f} to {selected['lighting_score'].max():.4f}")
        print(f"  Mean score: {selected['lighting_score'].mean():.4f}")
        print(f"  Median score: {selected['lighting_score'].median():.4f}")
        
        if len(rejected) > 0:
            print(f"\nREJECTED (bottom {len(rejected)}):")
            print(f"  Score range: {rejected['lighting_score'].min():.4f} to {rejected['lighting_score'].max():.4f}")
            print(f"  Mean score: {rejected['lighting_score'].mean():.4f}")
            print(f"  Median score: {rejected['lighting_score'].median():.4f}")
        
        # Gap analysis
        if cutoff < len(df):
            gap = selected['lighting_score'].min() - rejected['lighting_score'].max()
            print(f"\nScore gap at cutoff: {gap:.4f}")
            
            if gap < 0:
                print("  ⚠️  Warning: Overlap detected! Some rejected images have higher scores than selected ones.")
            elif gap < 0.001:
                print("  ℹ️  Very small gap - cutoff is in a dense region of scores")
            else:
                print("  ✓ Clean separation between selected and rejected images")


def main():
    parser = argparse.ArgumentParser(
        description="Verify image filtering results by visualizing score distribution"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to filtered_images.json or all_scores.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (optional, shows plot if not provided)"
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=50000,
        help="Cutoff index for analysis (default: 50000)"
    )
    
    args = parser.parse_args()
    
    # Run cutoff analysis
    show_cutoff_analysis(args.results, args.cutoff)
    
    # Create visualizations
    print("\n" + "="*70)
    print("Creating visualizations...")
    print("="*70)
    visualize_score_comparison(args.results, args.output_dir)
    
    print("\n✅ Verification complete!")
    print("\nInterpretation:")
    print("  - Top scores should show images with good/beautiful lighting")
    print("  - Bottom scores should show dark/poorly lit images")
    print("  - Images near 50k cutoff should still have reasonable lighting")
    print("  - Large score gap indicates clear quality separation")


if __name__ == "__main__":
    main()

