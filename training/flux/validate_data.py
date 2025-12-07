"""
Quick validation script to check your data format before training
"""

import argparse
import json
from pathlib import Path
from PIL import Image


def validate_metadata_file(data_dir: Path, metadata_file: Path):
    """Validate metadata file format"""
    print(f"\n📄 Checking metadata file: {metadata_file}")
    
    samples = []
    
    # Load metadata
    if str(metadata_file).endswith('.jsonl'):
        with open(metadata_file, 'r') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing line {i}: {e}")
                        return False
    else:  # .json
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict) and 'data' in data:
                    samples = data['data']
                else:
                    print(f"❌ Unexpected JSON format. Expected list or dict with 'data' key")
                    return False
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            return False
    
    print(f"✅ Found {len(samples)} samples")
    
    # Validate samples
    required_keys = {'input_image', 'instruction', 'output_image'}
    missing_images = []
    invalid_samples = []
    
    for i, sample in enumerate(samples[:10]):  # Check first 10 in detail
        # Check required keys
        if not all(key in sample for key in required_keys):
            invalid_samples.append((i, sample))
            continue
        
        # Check if image files exist
        input_path = data_dir / sample['input_image']
        output_path = data_dir / sample['output_image']
        
        if not input_path.exists():
            missing_images.append(('input', i, input_path))
        if not output_path.exists():
            missing_images.append(('output', i, output_path))
        
        # Try to load images
        try:
            if input_path.exists():
                img = Image.open(input_path)
                img.verify()
        except Exception as e:
            print(f"⚠️  Sample {i}: Cannot open input image: {e}")
        
        try:
            if output_path.exists():
                img = Image.open(output_path)
                img.verify()
        except Exception as e:
            print(f"⚠️  Sample {i}: Cannot open output image: {e}")
    
    # Report issues
    if invalid_samples:
        print(f"\n❌ Found {len(invalid_samples)} samples with missing required keys:")
        for i, sample in invalid_samples[:3]:
            print(f"   Sample {i}: {set(sample.keys())}")
        return False
    
    if missing_images:
        print(f"\n⚠️  Warning: Found {len(missing_images)} missing image files (showing first 5):")
        for img_type, idx, path in missing_images[:5]:
            print(f"   Sample {idx} ({img_type}): {path}")
    
    # Show example samples
    print(f"\n📋 Example samples:")
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n  Sample {i}:")
        print(f"    Input: {sample['input_image']}")
        print(f"    Instruction: {sample['instruction'][:80]}{'...' if len(sample['instruction']) > 80 else ''}")
        print(f"    Output: {sample['output_image']}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate training data format")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=None,
        help="Metadata file name (default: auto-detect)",
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print(f"🔍 Validating data in: {data_dir}")
    
    # Find metadata file
    if args.metadata_file:
        metadata_file = data_dir / args.metadata_file
    else:
        possible_files = [
            data_dir / "metadata.jsonl",
            data_dir / "metadata.json",
            data_dir / "instructions.jsonl",
            data_dir / "instructions.json",
            data_dir / "data.jsonl",
            data_dir / "data.json",
        ]
        
        metadata_file = None
        for f in possible_files:
            if f.exists():
                metadata_file = f
                break
        
        if metadata_file is None:
            print("❌ Could not find metadata file. Looking for:")
            for f in possible_files:
                print(f"   - {f.name}")
            return
    
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return
    
    # Validate
    success = validate_metadata_file(data_dir, metadata_file)
    
    if success:
        print("\n✅ Data validation passed! Ready for training.")
    else:
        print("\n❌ Data validation failed. Please fix the issues above.")


if __name__ == "__main__":
    main()

