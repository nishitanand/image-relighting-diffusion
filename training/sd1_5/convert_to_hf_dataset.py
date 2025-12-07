"""
Convert your custom triplet data format to HuggingFace Dataset format
This script converts metadata.jsonl to the format expected by the official training script
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image
from datasets import Dataset, Features, Image as ImageFeature, Value


def convert_to_hf_dataset(data_dir: str, output_dir: str, metadata_file: str = None):
    """
    Convert custom format to HuggingFace Dataset
    
    Input format (metadata.jsonl):
        {"input_image": "path/to/input.jpg", "instruction": "text", "output_image": "path/to/output.jpg"}
    
    Output format (HuggingFace Dataset):
        {"input_image": PIL.Image, "edit_prompt": "text", "edited_image": PIL.Image}
    
    Note: The training script can map these column names using:
        --original_image_column=input_image
        --edit_prompt_column=edit_prompt
        --edited_image_column=edited_image
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find metadata file
    if metadata_file is None:
        possible_files = [
            data_dir / "metadata.jsonl",
            data_dir / "metadata.json",
            data_dir / "instructions.jsonl",
            data_dir / "instructions.json",
            data_dir / "data.jsonl",
            data_dir / "data.json",
        ]
        
        for f in possible_files:
            if f.exists():
                metadata_file = f
                break
        
        if metadata_file is None:
            raise FileNotFoundError(f"Could not find metadata file in {data_dir}")
    else:
        metadata_file = Path(metadata_file)
    
    print(f"Loading metadata from: {metadata_file}")
    
    # Load samples
    samples = []
    if str(metadata_file).endswith('.jsonl'):
        with open(metadata_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:  # .json
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and 'data' in data:
                samples = data['data']
    
    print(f"Found {len(samples)} samples")
    
    # Convert to HuggingFace format
    print("Converting to HuggingFace Dataset format...")
    
    hf_samples = []
    for i, sample in enumerate(samples):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(samples)}...")
        
        try:
            # Load images
            input_image_path = data_dir / sample['input_image']
            output_image_path = data_dir / sample['output_image']
            
            input_image = Image.open(input_image_path).convert('RGB')
            output_image = Image.open(output_image_path).convert('RGB')
            
            hf_samples.append({
                'input_image': input_image,          # Matches your metadata.jsonl
                'edit_prompt': sample['instruction'],
                'edited_image': output_image,
            })
        except Exception as e:
            print(f"Warning: Skipping sample {i} due to error: {e}")
            continue
    
    print(f"Successfully converted {len(hf_samples)} samples")
    
    # Create HuggingFace Dataset
    print("Creating HuggingFace Dataset...")
    dataset = Dataset.from_dict({
        'input_image': [s['input_image'] for s in hf_samples],
        'edit_prompt': [s['edit_prompt'] for s in hf_samples],
        'edited_image': [s['edited_image'] for s in hf_samples],
    })
    
    # Cast to proper types
    dataset = dataset.cast_column('input_image', ImageFeature())
    dataset = dataset.cast_column('edited_image', ImageFeature())
    
    # Save dataset
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))
    
    print(f"✅ Dataset saved successfully!")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Location: {output_dir}")
    print(f"   Columns: {dataset.column_names}")
    print(f"\nYou can now use this with the training script:")
    print(f"   ./train.sh --data_dir {output_dir}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom triplet data to HuggingFace Dataset format"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing your original data (with metadata.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data_hf",
        help="Directory to save the converted HuggingFace dataset",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Path to metadata file (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    convert_to_hf_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
    )


if __name__ == "__main__":
    main()

