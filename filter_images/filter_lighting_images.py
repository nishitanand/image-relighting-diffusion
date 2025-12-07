"""
CLIP-based Image Filtering for Lighting Quality
Filters images from a dataset based on their lighting quality using CLIP similarity scores.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd

try:
    import open_clip
    USE_OPEN_CLIP = True
except ImportError:
    USE_OPEN_CLIP = False
    from transformers import CLIPProcessor, CLIPModel


class LightingImageFilter:
    """Filter images based on lighting quality using CLIP."""
    
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Initialize the CLIP model for filtering.
        
        Args:
            model_name: CLIP model variant to use
            device: torch device (defaults to cuda if available)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        if USE_OPEN_CLIP:
            print(f"Loading OpenCLIP model: {model_name}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained='openai'
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            print(f"Loading HuggingFace CLIP model: {model_name}")
            self.model = CLIPModel.from_pretrained(f"openai/clip-{model_name.lower().replace('/', '-')}")
            self.processor = CLIPProcessor.from_pretrained(f"openai/clip-{model_name.lower().replace('/', '-')}")
            self.preprocess = None
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Lighting-related prompts (similar to the paper)
        self.lighting_prompts = [
            "beautiful lighting",
            "good lighting",
            "well lit face",
            "professional lighting",
            "natural light",
            "illumination",
            "bright and clear lighting",
        ]
        
        # Encode text prompts
        self.text_features = self._encode_text_prompts()
        
    def _encode_text_prompts(self) -> torch.Tensor:
        """Encode all lighting-related text prompts."""
        print(f"Encoding {len(self.lighting_prompts)} text prompts...")
        
        with torch.no_grad():
            if USE_OPEN_CLIP:
                text = self.tokenizer(self.lighting_prompts).to(self.device)
                text_features = self.model.encode_text(text)
            else:
                inputs = self.processor(
                    text=self.lighting_prompts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                text_features = self.model.get_text_features(**inputs)
            
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_lighting_score(self, image_path: str) -> float:
        """
        Compute lighting quality score for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Average similarity score across all lighting prompts
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                if USE_OPEN_CLIP:
                    image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    image_features = self.model.encode_image(image_input)
                else:
                    inputs = self.processor(
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    image_features = self.model.get_image_features(**inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity with all text prompts
                similarity = (image_features @ self.text_features.T).squeeze(0)
                
                # Return average similarity score
                avg_score = similarity.mean().item()
            
            return avg_score
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return -1.0
    
    def compute_batch_scores(self, image_paths: List[str], batch_size: int = 32) -> List[float]:
        """
        Compute lighting scores for a batch of images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once
            
        Returns:
            List of average similarity scores
        """
        scores = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            # Load images
            for idx, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    scores.append(-1.0)
            
            if not batch_images:
                continue
            
            # Process batch
            with torch.no_grad():
                if USE_OPEN_CLIP:
                    image_inputs = torch.stack([
                        self.preprocess(img) for img in batch_images
                    ]).to(self.device)
                    image_features = self.model.encode_image(image_inputs)
                else:
                    inputs = self.processor(
                        images=batch_images,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    image_features = self.model.get_image_features(**inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity with all text prompts
                similarity = image_features @ self.text_features.T
                
                # Average across all prompts
                avg_scores = similarity.mean(dim=-1).cpu().numpy()
            
            # Add scores for valid images
            batch_scores = [-1.0] * len(batch_paths)
            for idx, score in zip(valid_indices, avg_scores):
                batch_scores[idx] = float(score)
            
            scores.extend(batch_scores)
        
        return scores


def find_images(dataset_path: str, extensions: List[str] = None) -> List[str]:
    """
    Recursively find all images in the dataset directory.
    
    Args:
        dataset_path: Root directory of the dataset
        extensions: List of valid image extensions
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    image_paths = []
    dataset_path = Path(dataset_path)
    
    print(f"Searching for images in {dataset_path}...")
    
    for ext in extensions:
        image_paths.extend(dataset_path.rglob(f"*{ext}"))
        image_paths.extend(dataset_path.rglob(f"*{ext.upper()}"))
    
    image_paths = [str(p) for p in image_paths]
    print(f"Found {len(image_paths)} images")
    
    return image_paths


def filter_images(
    dataset_path: str,
    output_dir: str,
    num_images: int = 50000,
    batch_size: int = 32,
    model_name: str = "ViT-B/32",
    save_scores: bool = True,
    copy_images: bool = False
) -> None:
    """
    Filter images based on lighting quality.
    
    Args:
        dataset_path: Path to the dataset directory
        output_dir: Directory to save filtered results
        num_images: Number of top images to select
        batch_size: Batch size for processing
        model_name: CLIP model variant
        save_scores: Whether to save all scores
        copy_images: Whether to copy filtered images to output directory
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_paths = find_images(dataset_path)
    
    if len(image_paths) == 0:
        print("No images found in the dataset path!")
        return
    
    print(f"Total images found: {len(image_paths)}")
    
    # Initialize filter
    filter_model = LightingImageFilter(model_name=model_name)
    
    # Compute scores
    print("\nComputing lighting scores...")
    scores = filter_model.compute_batch_scores(image_paths, batch_size=batch_size)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'lighting_score': scores
    })
    
    # Remove failed images (score = -1.0)
    results_df = results_df[results_df['lighting_score'] >= 0]
    print(f"\nSuccessfully processed {len(results_df)} images")
    
    # Sort by score and select top N
    results_df = results_df.sort_values('lighting_score', ascending=False)
    
    # Select top num_images
    num_to_select = min(num_images, len(results_df))
    filtered_df = results_df.head(num_to_select)
    
    print(f"\nSelected top {num_to_select} images")
    print(f"Score range: {filtered_df['lighting_score'].min():.4f} to {filtered_df['lighting_score'].max():.4f}")
    print(f"Mean score: {filtered_df['lighting_score'].mean():.4f}")
    
    # Save filtered list
    filtered_list_path = output_dir / "filtered_images.txt"
    with open(filtered_list_path, 'w') as f:
        for path in filtered_df['image_path']:
            f.write(f"{path}\n")
    print(f"\nSaved filtered image list to: {filtered_list_path}")
    
    # Save as JSON with scores
    filtered_json_path = output_dir / "filtered_images.json"
    filtered_data = filtered_df.to_dict('records')
    with open(filtered_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    print(f"Saved filtered image data to: {filtered_json_path}")
    
    # Save all scores if requested
    if save_scores:
        all_scores_path = output_dir / "all_scores.csv"
        results_df.to_csv(all_scores_path, index=False)
        print(f"Saved all scores to: {all_scores_path}")
    
    # Copy images if requested
    if copy_images:
        print("\nCopying filtered images...")
        filtered_images_dir = output_dir / "filtered_images"
        filtered_images_dir.mkdir(exist_ok=True)
        
        for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
            src_path = Path(row['image_path'])
            # Use original filename with index to avoid collisions
            dst_path = filtered_images_dir / f"{idx:06d}_{src_path.name}"
            
            try:
                import shutil
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        
        print(f"Copied images to: {filtered_images_dir}")
    
    print("\n✓ Filtering complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Filter images based on lighting quality using CLIP"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the image dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./filtered_output",
        help="Directory to save filtered results"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50000,
        help="Number of top images to select (default: 50000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model variant to use (default: ViT-B/32)"
    )
    parser.add_argument(
        "--no_save_scores",
        action="store_true",
        help="Don't save all scores to CSV"
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy filtered images to output directory"
    )
    
    args = parser.parse_args()
    
    filter_images(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        model_name=args.model_name,
        save_scores=not args.no_save_scores,
        copy_images=args.copy_images
    )


if __name__ == "__main__":
    main()

