"""
Batched multi-GPU pipeline for maximum throughput.
Uses batched SAM3 inference to fully utilize GPU memory.
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import pandas as pd
import torch
from multiprocessing import Process, Queue
from tqdm import tqdm
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_split_name_from_csv(csv_path: str) -> str:
    """Extract split name (train/val/test) from CSV filename."""
    csv_name = Path(csv_path).stem.lower()
    
    if 'train' in csv_name:
        return 'train'
    elif 'val' in csv_name:
        return 'val'
    elif 'test' in csv_name:
        return 'test'
    else:
        return 'data'


def worker_process_batched(gpu_id, image_indices, image_paths, config, output_root, batch_size, result_queue):
    """
    Worker process with BATCHED SAM3 inference for maximum GPU utilization.
    
    Args:
        gpu_id: GPU device ID (0-7)
        image_indices: List of image indices to process
        image_paths: List of image paths to process
        config: Pipeline configuration
        output_root: Output directory path
        batch_size: Number of images to process in each SAM3 batch
        result_queue: Queue to store results
    """
    # Import required modules first
    import os
    import warnings
    import torch
    from PIL import Image
    import logging
    import json
    import numpy as np
    
    # Set GPU for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TORCH_HOME'] = '/mnt/localssd/.cache/torch'
    
    # Suppress warnings and verbose output
    warnings.filterwarnings('ignore')
    logging.getLogger('torch.hub').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('timm').setLevel(logging.WARNING)
    
    # Setup logging for this worker
    logging.basicConfig(
        level=logging.INFO,
        format=f'GPU {gpu_id} | %(levelname)s | %(message)s'
    )
    logger = logging.getLogger(f"GPU_{gpu_id}")
    
    # Import SAM3 from HuggingFace transformers (supports TRUE batching!)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from transformers import Sam3Processor, Sam3Model
        logger.info("Using HuggingFace transformers Sam3Model for batched inference")
    except ImportError as e:
        logger.error(f"Failed to import transformers: {e}")
        logger.error("Install with: pip install transformers")
        raise
    
    from src.stages.stage_2_albedo import AlbedoExtractionStage
    from src.stages.stage_3_shadow import ShadowGenerationStage
    from src.stages.stage_3_5_recombine import BackgroundRecombinationStage
    
    # Override config paths for this worker
    worker_config = config.copy()
    worker_config['paths']['output_root'] = str(output_root)
    worker_config['paths']['data_root'] = str(Path(output_root).parent / f"data_intermediate_gpu{gpu_id}")
    
    # Create output directory
    Path(output_root).mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.hub.set_dir('/mnt/localssd/.cache/torch/hub')
    
    try:
        logger.info(f"Initializing stages with BATCHED inference (batch_size={batch_size})...")
        
        # Load SAM3 model and processor from HuggingFace for batched inference
        logger.info(f"Loading SAM3 from HuggingFace for batched inference...")
        sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
        sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        sam3_model.eval()
        logger.info(f"✅ SAM3 model and processor loaded for TRUE batched inference")
        
        # Initialize other stages (non-batched)
        stages = {
            'albedo': AlbedoExtractionStage(worker_config, str(device)),
            'shadow': ShadowGenerationStage(worker_config, str(device)),
            'recombine': BackgroundRecombinationStage(worker_config, str(device))
        }
        
        logger.info(f"Loading albedo model...")
        stages['albedo'].load_model()
        
        logger.info(f"Loading shadow model (MiDaS)...")
        stages['shadow'].load_model()
        
        logger.info(f"Loading recombine stage...")
        stages['recombine'].load_model()
        
        logger.info(f"✅ All models loaded! Processing {len(image_indices)} images in batches of {batch_size}...")
        
        # Process images in batches
        num_batches = (len(image_indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(image_indices))
            batch_indices = image_indices[batch_start:batch_end]
            batch_paths = image_paths[batch_start:batch_end]
            
            try:
                # Load batch of images
                batch_images = []
                for img_path in batch_paths:
                    img = Image.open(str(img_path)).convert('RGB')
                    batch_images.append(img)
                
                # BATCHED SAM3 INFERENCE (TRUE BATCHING!)
                # Process entire batch at once using HuggingFace API
                text_prompts = ["person"] * len(batch_images)
                
                # Prepare inputs for batch
                inputs = sam3_processor(
                    images=batch_images,
                    text=text_prompts,
                    return_tensors="pt"
                ).to(device)
                
                # Run batched inference
                with torch.no_grad():
                    outputs = sam3_model(**inputs)
                
                # Post-process results for all images in batch
                sam3_results = sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )
                
                # Process each image individually (albedo, shadow, recombine)
                for i, (idx, img, sam3_result) in enumerate(zip(batch_indices, batch_images, sam3_results)):
                    try:
                        # Extract mask from SAM3 result
                        if 'masks' in sam3_result and len(sam3_result['masks']) > 0:
                            # Get the mask with highest score
                            best_mask_idx = torch.argmax(sam3_result['scores']).item()
                            mask = sam3_result['masks'][best_mask_idx]
                            
                            # Convert to numpy if needed
                            if torch.is_tensor(mask):
                                mask = mask.cpu().numpy()
                            
                            # Ensure mask is 2D
                            if len(mask.shape) > 2:
                                mask = np.squeeze(mask)
                            if len(mask.shape) == 1:
                                # Reshape to image dimensions
                                h, w = img.size[1], img.size[0]
                                mask = mask.reshape(h, w)
                            
                            # Convert mask to PIL Image for consistency
                            mask_uint8 = (mask * 255).astype(np.uint8)
                            mask_pil = Image.fromarray(mask_uint8)
                            
                            # Create data dictionary
                            data = {
                                'image': img,
                                'image_id': idx,
                                'original_image': img,
                                'mask': mask_pil,  # Store as PIL Image
                                'foreground': None,  # Will be computed in stages
                                'background': None
                            }
                            
                            # Extract foreground/background using mask
                            img_array = np.array(img)
                            mask_3ch = np.stack([mask] * 3, axis=2)
                            foreground = (img_array * mask_3ch).astype(np.uint8)
                            background = (img_array * (1 - mask_3ch)).astype(np.uint8)
                            
                            data['foreground'] = Image.fromarray(foreground)
                            data['background'] = Image.fromarray(background)
                            
                            # Stage 2: Albedo (not batched)
                            data = stages['albedo'].process(data)
                            
                            # Stage 3: Shadow/Degradation (not batched)
                            data = stages['shadow'].process(data)
                            
                            # Stage 3.5: Recombine
                            data = stages['recombine'].process(data)
                            
                            # Save outputs
                            output_path = Path(output_root) / f"{idx:05d}_output.png"
                            data['composite_image'].save(output_path)
                            data['original_image'].save(Path(output_root) / f"{idx:05d}_input.png")
                            data['albedo'].save(Path(output_root) / f"{idx:05d}_albedo.png")
                            data['degraded_image'].save(Path(output_root) / f"{idx:05d}_degraded_fg.png")
                            data['foreground'].save(Path(output_root) / f"{idx:05d}_foreground.png")
                            data['background'].save(Path(output_root) / f"{idx:05d}_background.png")
                            
                            # Save metadata
                            if 'degradation_metadata' in data:
                                with open(Path(output_root) / f"{idx:05d}_metadata.json", 'w') as f:
                                    json.dump(data['degradation_metadata'], f, indent=2)
                            
                            result_queue.put({
                                'gpu_id': gpu_id,
                                'index': idx,
                                'image_path': str(batch_paths[i]),
                                'output_path': str(output_path),
                                'success': True
                            })
                        else:
                            # No person detected
                            logger.warning(f"No person detected in image {idx}")
                            result_queue.put({
                                'gpu_id': gpu_id,
                                'index': idx,
                                'image_path': str(batch_paths[i]),
                                'output_path': None,
                                'success': False,
                                'error': 'No person detected'
                            })
                            
                    except Exception as e:
                        logger.error(f"Failed to process image {idx}: {e}")
                        result_queue.put({
                            'gpu_id': gpu_id,
                            'index': idx,
                            'image_path': str(batch_paths[i]),
                            'output_path': None,
                            'success': False,
                            'error': str(e)
                        })
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_end}/{len(image_indices)} images ({batch_idx+1}/{num_batches} batches)")
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
                # Mark all images in batch as failed
                for i, idx in enumerate(batch_indices):
                    result_queue.put({
                        'gpu_id': gpu_id,
                        'index': idx,
                        'image_path': str(batch_paths[i]),
                        'output_path': None,
                        'success': False,
                        'error': f"Batch processing failed: {str(e)}"
                    })
        
        # UNLOAD MODELS AFTER ALL IMAGES ARE PROCESSED
        logger.info(f"Unloading models...")
        del sam3_model, sam3_processor
        for stage_name, stage in stages.items():
            stage.unload_model()
        
        torch.cuda.empty_cache()
        logger.info(f"✅ Completed all {len(image_indices)} images!")
                
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Run relighting pipeline with BATCHED inference across 8 GPUs"
    )
    parser.add_argument("--config", type=str, default="config/mvp_config.yaml")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with image paths")
    parser.add_argument("--num-samples", type=int, default=None, help="Number to process (default: all)")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for SAM3 inference (default: 8)")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(project_root) / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load CSV
    print(f"Loading CSV: {args.csv}")
    df_original = pd.read_csv(args.csv)
    
    if 'image_path' not in df_original.columns:
        print("ERROR: CSV must contain 'image_path' column")
        sys.exit(1)
    
    # Determine split and output directory
    split_name = get_split_name_from_csv(args.csv)
    output_root = Path(project_root) / f"data-{split_name}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"BATCHED MULTI-GPU PARALLEL PROCESSING")
    print(f"{'='*60}")
    print(f"Split: {split_name}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Batch size: {args.batch_size} images per GPU")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}\n")
    
    # Get images to process
    image_paths = df_original['image_path'].tolist()
    if args.num_samples:
        image_paths = image_paths[:args.num_samples]
    
    total_images = len(image_paths)
    print(f"Processing {total_images} images across {args.num_gpus} GPUs...")
    print(f"Images per GPU: ~{total_images // args.num_gpus}")
    print(f"Expected GPU memory usage: ~15-20GB (batched inference)\n")
    
    # Distribute images across GPUs
    images_per_gpu = total_images // args.num_gpus
    remainder = total_images % args.num_gpus
    
    gpu_assignments = []
    start_idx = 0
    
    for gpu_id in range(args.num_gpus):
        count = images_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + count
        
        if count > 0:
            gpu_assignments.append({
                'gpu_id': gpu_id,
                'indices': list(range(start_idx, end_idx)),
                'paths': image_paths[start_idx:end_idx]
            })
        
        start_idx = end_idx
    
    # Print assignment
    for assignment in gpu_assignments:
        print(f"GPU {assignment['gpu_id']}: {len(assignment['indices'])} images (indices {assignment['indices'][0]}-{assignment['indices'][-1]})")
    
    print(f"\n🚀 Starting batched parallel processing...\n")
    
    # Create result queue
    result_queue = Queue()
    
    # Start worker processes
    processes = []
    for assignment in gpu_assignments:
        p = Process(
            target=worker_process_batched,
            args=(
                assignment['gpu_id'],
                assignment['indices'],
                assignment['paths'],
                config,
                output_root,
                args.batch_size,
                result_queue
            )
        )
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger starts
    
    # Collect results with progress bar
    results = []
    with tqdm(total=total_images, desc="Total progress") as pbar:
        while len(results) < total_images:
            if not result_queue.empty():
                result = result_queue.get()
                results.append(result)
                pbar.update(1)
            else:
                time.sleep(0.1)
            
            if all(not p.is_alive() for p in processes):
                while not result_queue.empty():
                    result = result_queue.get()
                    results.append(result)
                    pbar.update(1)
                break
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Create updated CSV
    print(f"\n📝 Creating updated CSV...")
    df_updated = df_original.copy()
    df_updated['output_image_path'] = None
    
    successful = 0
    for result in results:
        if result['success']:
            idx = result['index']
            if idx < len(df_updated):
                df_updated.loc[idx, 'output_image_path'] = result['output_path']
                successful += 1
    
    # Save updated CSV to albedo_csv_files folder
    albedo_csv_dir = Path(project_root) / "albedo_csv_files"
    albedo_csv_dir.mkdir(parents=True, exist_ok=True)
    
    csv_name = Path(args.csv).stem
    output_csv_path = albedo_csv_dir / f"{csv_name}_with_albedo.csv"
    df_updated.to_csv(output_csv_path, index=False)
    
    # Also save a copy in the original location for backwards compatibility
    legacy_csv_path = Path(args.csv).parent / f"{csv_name}_with_relighting_outputs.csv"
    df_updated.to_csv(legacy_csv_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ BATCHED MULTI-GPU PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Split: {split_name}")
    print(f"Total images: {total_images}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_images - successful}")
    print(f"GPUs used: {args.num_gpus}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nOutputs:")
    print(f"  Images: {output_root}/")
    print(f"  CSV (primary): {output_csv_path}")
    print(f"  CSV (legacy): {legacy_csv_path}")
    print(f"\n📋 Next Step: Run edit_keywords to generate lighting descriptions")
    print(f"   python ../../edit_keywords/generate_keywords.py --csv {output_csv_path}")
    print(f"{'='*60}\n")
    
    # Show per-GPU stats
    gpu_stats = {}
    for result in results:
        gpu_id = result['gpu_id']
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = {'success': 0, 'failed': 0}
        if result['success']:
            gpu_stats[gpu_id]['success'] += 1
        else:
            gpu_stats[gpu_id]['failed'] += 1
    
    print("Per-GPU Statistics:")
    for gpu_id in sorted(gpu_stats.keys()):
        stats = gpu_stats[gpu_id]
        print(f"  GPU {gpu_id}: {stats['success']} successful, {stats['failed']} failed")


if __name__ == "__main__":
    main()

