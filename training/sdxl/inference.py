"""
Inference script for InstructPix2Pix SDXL trained model
Supports single image inference and batch processing
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from diffusers import StableDiffusionXLInstructPix2PixPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run InstructPix2Pix SDXL inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (e.g., ./output/instruct-pix2pix-sdxl)",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Text instruction for editing",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save output image",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps (default: 50, more steps = better quality but slower)",
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="Image guidance scale (how much to follow input image). Higher = closer to input (default: 1.5)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Text guidance scale (how much to follow instruction). Higher = stronger instruction following (default: 7.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of output images to generate",
    )
    args = parser.parse_args()
    return args


def load_pipeline(model_path: str, device: str = "cuda"):
    """Load the InstructPix2Pix SDXL pipeline"""
    print(f"Loading SDXL model from {model_path}...")
    
    pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    )
    
    pipeline = pipeline.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except Exception as e:
            print(f"Could not enable xformers: {e}")
        
        # Enable VAE slicing for memory efficiency
        pipeline.enable_vae_slicing()
    
    print("Model loaded successfully!")
    return pipeline


def run_inference(
    pipeline,
    input_image_path: str,
    instruction: str,
    num_inference_steps: int = 50,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    num_images: int = 1,
):
    """Run inference with the InstructPix2Pix SDXL model"""
    
    # Load input image
    input_image = Image.open(input_image_path).convert("RGB")
    print(f"Loaded input image: {input_image.size}")
    
    # Set seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        print(f"Using seed: {seed}")
    
    print(f"\nRunning SDXL inference...")
    print(f"  Instruction: {instruction}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Image guidance scale: {image_guidance_scale}")
    print(f"  Text guidance scale: {guidance_scale}")
    print(f"  Number of images: {num_images}")
    
    # Run inference
    output = pipeline(
        prompt=instruction,
        image=input_image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_images,
    )
    
    return output.images


def main():
    args = parse_args()
    
    # Load pipeline
    pipeline = load_pipeline(args.model_path, args.device)
    
    # Run inference
    output_images = run_inference(
        pipeline=pipeline,
        input_image_path=args.input_image,
        instruction=args.instruction,
        num_inference_steps=args.num_inference_steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_images=args.num_images,
    )
    
    # Save output images
    if args.num_images == 1:
        output_images[0].save(args.output_path)
        print(f"\nSaved output image to: {args.output_path}")
    else:
        output_dir = Path(args.output_path).parent
        output_name = Path(args.output_path).stem
        output_ext = Path(args.output_path).suffix
        
        for i, img in enumerate(output_images):
            save_path = output_dir / f"{output_name}_{i}{output_ext}"
            img.save(save_path)
            print(f"Saved output image {i+1} to: {save_path}")
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()

