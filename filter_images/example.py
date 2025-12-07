#!/usr/bin/env python3
"""
Quick example script demonstrating how to use the image filter.
"""

from filter_lighting_images import filter_images

# Example configuration
config = {
    'dataset_path': '/path/to/your/ffhq-dataset/images1024x1024',
    'output_dir': './output',
    'num_images': 50000,
    'batch_size': 32,
    'model_name': 'ViT-B/32',
    'save_scores': True,
    'copy_images': False
}

if __name__ == "__main__":
    print("="*60)
    print("CLIP-Based Image Filtering Example")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n" + "="*60)
    
    # Update the dataset_path to your actual path
    print("\n⚠️  Please update 'dataset_path' in this script to your actual dataset path!")
    print("    Then run: python example.py\n")
    
    # Uncomment the line below after setting the correct dataset_path
    # filter_images(**config)

