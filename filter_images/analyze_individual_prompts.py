"""
Re-analyze images with individual prompt scores saved separately.
This allows you to see which specific prompts are most effective.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

try:
    import open_clip
    USE_OPEN_CLIP = True
except ImportError:
    USE_OPEN_CLIP = False
    from transformers import CLIPProcessor, CLIPModel


def compute_individual_prompt_scores(
    image_paths: List[str],
    output_file: str,
    model_name: str = "ViT-B/32",
    batch_size: int = 32,
    device=None
):
    """
    Compute and save scores for each prompt individually.
    
    Args:
        image_paths: List of image paths to analyze
        output_file: Path to save results CSV
        model_name: CLIP model to use
        batch_size: Batch size for processing
        device: torch device
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load CLIP model
    if USE_OPEN_CLIP:
        print(f"Loading OpenCLIP model: {model_name}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='openai'
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        print(f"Loading HuggingFace CLIP model")
        model = CLIPModel.from_pretrained(f"openai/clip-{model_name.lower().replace('/', '-')}")
        processor = CLIPProcessor.from_pretrained(f"openai/clip-{model_name.lower().replace('/', '-')}")
        preprocess = None
    
    model = model.to(device)
    model.eval()
    
    # Define prompts
    prompts = [
        "beautiful lighting",
        "good lighting",
        "well lit face",
        "professional lighting",
        "natural light",
        "illumination",
        "bright and clear lighting",
    ]
    
    print(f"Encoding {len(prompts)} text prompts...")
    
    # Encode text prompts
    with torch.no_grad():
        if USE_OPEN_CLIP:
            text = tokenizer(prompts).to(device)
            text_features = model.encode_text(text)
        else:
            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True
            ).to(device)
            text_features = model.get_text_features(**inputs)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Process images
    results = []
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        # Load images
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                batch_images.append(image)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if not batch_images:
            continue
        
        # Process batch
        with torch.no_grad():
            if USE_OPEN_CLIP:
                image_inputs = torch.stack([
                    preprocess(img) for img in batch_images
                ]).to(device)
                image_features = model.encode_image(image_inputs)
            else:
                inputs = processor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                image_features = model.get_image_features(**inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity with each prompt individually
            similarity = image_features @ text_features.T  # Shape: [batch_size, 7]
            
            # Convert to numpy
            similarity_np = similarity.cpu().numpy()
            
            # Save individual scores for each image
            for idx, path in enumerate(valid_paths):
                scores_dict = {
                    'image_path': path,
                    'avg_score': float(similarity_np[idx].mean()),
                }
                
                # Add individual prompt scores
                for prompt_idx, prompt in enumerate(prompts):
                    prompt_key = prompt.replace(' ', '_')
                    scores_dict[prompt_key] = float(similarity_np[idx, prompt_idx])
                
                results.append(scores_dict)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by average score
    df = df.sort_values('avg_score', ascending=False).reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved individual prompt scores to: {output_file}")
    
    # Print statistics
    print("\n" + "="*80)
    print("SCORE STATISTICS BY PROMPT")
    print("="*80)
    
    for prompt in prompts:
        prompt_key = prompt.replace(' ', '_')
        scores = df[prompt_key]
        print(f"\n{prompt}:")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std:  {scores.std():.4f}")
        print(f"  Min:  {scores.min():.4f}")
        print(f"  Max:  {scores.max():.4f}")
    
    return df


def visualize_prompt_comparison(
    scores_file: str,
    output_dir: str,
    interval: int = 5000,
    images_per_interval: int = 5
):
    """
    Create separate visualizations for each prompt.
    
    Args:
        scores_file: CSV file with individual prompt scores
        output_dir: Directory to save visualizations
        interval: Spacing between samples
        images_per_interval: Images to show per interval
    """
    print(f"\nLoading scores from {scores_file}...")
    df = pd.read_csv(scores_file)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = [
        "beautiful_lighting",
        "good_lighting",
        "well_lit_face",
        "professional_lighting",
        "natural_light",
        "illumination",
        "bright_and_clear_lighting",
    ]
    
    print(f"\nCreating visualizations for each of {len(prompts)} prompts...")
    
    for prompt_key in prompts:
        print(f"\nProcessing: {prompt_key.replace('_', ' ')}")
        
        # Sort by this specific prompt's score
        df_sorted = df.sort_values(prompt_key, ascending=False).reset_index(drop=True)
        
        # Determine sample points
        sample_groups = []
        sample_groups.append((0, images_per_interval, "Top 1-5"))
        
        current_pos = interval
        while current_pos < len(df_sorted):
            end_pos = min(current_pos + images_per_interval, len(df_sorted))
            label = f"#{current_pos+1}-{end_pos}"
            sample_groups.append((current_pos, end_pos, label))
            current_pos += interval
        
        # Create grid
        n_groups = len(sample_groups)
        n_cols = images_per_interval
        n_rows = n_groups
        
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.5))
        
        for group_idx, (start_idx, end_idx, label) in enumerate(sample_groups):
            for img_offset in range(end_idx - start_idx):
                img_idx = start_idx + img_offset
                
                if img_idx >= len(df_sorted):
                    break
                
                row = df_sorted.iloc[img_idx]
                
                # Calculate subplot position
                ax_idx = group_idx * n_cols + img_offset + 1
                ax = plt.subplot(n_rows, n_cols, ax_idx)
                
                try:
                    # Load and display image
                    img = Image.open(row['image_path'])
                    ax.imshow(img)
                    
                    # Get scores
                    rank = img_idx + 1
                    prompt_score = row[prompt_key]
                    avg_score = row['avg_score']
                    
                    # Color coding
                    if rank <= 10000:
                        color = 'green'
                    elif rank <= 30000:
                        color = 'orange'
                    elif rank <= 50000:
                        color = 'darkorange'
                    else:
                        color = 'red'
                    
                    # Title with both scores
                    title_text = f"#{rank}\nPrompt: {prompt_score:.4f}\nAvg: {avg_score:.4f}"
                    ax.set_title(title_text, fontsize=9, fontweight='bold', color=color)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error", ha='center', va='center', fontsize=8)
                
                ax.axis('off')
            
            # Add group label
            if img_offset >= 0:
                ax_first = plt.subplot(n_rows, n_cols, group_idx * n_cols + 1)
                ax_first.text(-0.3, 0.5, label, 
                             transform=ax_first.transAxes,
                             fontsize=12, fontweight='bold',
                             rotation=90, va='center', ha='right')
        
        # Add title
        prompt_display = prompt_key.replace('_', ' ').title()
        fig.suptitle(f'Prompt Analysis: "{prompt_display}" (Every {interval} images)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"prompt_{prompt_key}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        
        plt.close()
    
    print(f"\n✅ Created visualizations for all {len(prompts)} prompts!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze images with individual prompt scores"
    )
    parser.add_argument(
        "--image_list",
        type=str,
        required=True,
        help="Path to text file with image paths (one per line)"
    )
    parser.add_argument(
        "--output_scores",
        type=str,
        default="./individual_prompt_scores.csv",
        help="Path to save individual prompt scores CSV"
    )
    parser.add_argument(
        "--output_visualizations",
        type=str,
        default="./prompt_analysis",
        help="Directory to save prompt visualizations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP model to use"
    )
    parser.add_argument(
        "--skip_compute",
        action="store_true",
        help="Skip computing scores, just visualize existing scores file"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5000,
        help="Interval between sample groups"
    )
    parser.add_argument(
        "--images_per_interval",
        type=int,
        default=5,
        help="Images to show per interval"
    )
    
    args = parser.parse_args()
    
    if not args.skip_compute:
        # Load image paths
        print(f"Loading image paths from {args.image_list}...")
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(image_paths)} image paths")
        
        # Compute individual scores
        print("\n" + "="*80)
        print("COMPUTING INDIVIDUAL PROMPT SCORES")
        print("="*80)
        
        df = compute_individual_prompt_scores(
            image_paths,
            args.output_scores,
            args.model_name,
            args.batch_size
        )
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING PROMPT VISUALIZATIONS")
    print("="*80)
    
    visualize_prompt_comparison(
        args.output_scores if not args.skip_compute else args.image_list,
        args.output_visualizations,
        args.interval,
        args.images_per_interval
    )
    
    print("\n✅ Complete!")
    print(f"\nOutputs:")
    print(f"  - Individual scores: {args.output_scores}")
    print(f"  - Visualizations: {args.output_visualizations}/")


if __name__ == "__main__":
    main()

