"""
Multi-GPU parallel pipeline for 8 GPUs.
Each GPU processes images independently for maximum throughput.
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


def worker_process(gpu_id, image_indices, image_paths, config, output_root, result_queue):
    """
    Worker process for a single GPU.
    
    IMPORTANT: Models are loaded ONCE per GPU at startup and reused for all images.
    This is much faster than loading/unloading for each image.
    
    Args:
        gpu_id: GPU device ID (0-7)
        image_indices: List of image indices to process
        image_paths: List of image paths to process
        config: Pipeline configuration
        output_root: Output directory path
        result_queue: Queue to store results
    """
    # Import required modules first
    import os
    import warnings
    import torch
    from PIL import Image
    import logging
    
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
    
    # Import stages directly for persistent model loading
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.stages.stage_1_segmentation_sam3 import SAM3SegmentationStage
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
        logger.info(f"GPU {gpu_id}: Initializing stages and loading models...")
        
        # Initialize stages
        stages = {
            'segmentation': SAM3SegmentationStage(worker_config, str(device)),
            'albedo': AlbedoExtractionStage(worker_config, str(device)),
            'shadow': ShadowGenerationStage(worker_config, str(device)),
            'recombine': BackgroundRecombinationStage(worker_config, str(device))
        }
        
        # LOAD ALL MODELS ONCE AT STARTUP
        logger.info(f"GPU {gpu_id}: Loading segmentation model (SAM3)...")
        stages['segmentation'].load_model()
        assert stages['segmentation'].model is not None, "SAM3 model failed to load"
        
        logger.info(f"GPU {gpu_id}: Loading albedo model (Retinex)...")
        stages['albedo'].load_model()
        
        logger.info(f"GPU {gpu_id}: Loading shadow model (MiDaS)...")
        stages['shadow'].load_model()
        assert stages['shadow'].normal_estimator is not None, "MiDaS model failed to load"
        assert stages['shadow'].normal_estimator.model is not None, "MiDaS model is None after load_model"
        
        logger.info(f"GPU {gpu_id}: Loading recombine stage...")
        stages['recombine'].load_model()
        
        logger.info(f"✅ All models loaded and verified! Processing {len(image_indices)} images...")
        logger.info(f"Models will be REUSED for all images (no reloading per image)")
        
        # Track model object IDs to verify they don't change
        sam3_model_id = id(stages['segmentation'].model)
        midas_model_id = id(stages['shadow'].normal_estimator.model)
        
        # Process assigned images (MODELS ALREADY LOADED, REUSED FOR ALL IMAGES)
        for i, (idx, img_path) in enumerate(zip(image_indices, image_paths)):
            try:
                # VERIFY models haven't been reloaded (object ID should be same)
                if i % 100 == 0 and i > 0:
                    current_sam3_id = id(stages['segmentation'].model)
                    current_midas_id = id(stages['shadow'].normal_estimator.model)
                    if current_sam3_id != sam3_model_id or current_midas_id != midas_model_id:
                        logger.warning(f"⚠️ Model object changed! SAM3: {sam3_model_id != current_sam3_id}, MiDaS: {midas_model_id != current_midas_id}")
                    else:
                        logger.info(f"✓ Models still same objects (not reloaded) - {i} images processed")
                
                # Load image
                image = Image.open(str(img_path)).convert('RGB')
                data = {
                    'image': image,
                    'image_id': idx,
                    'original_image': image
                }
                
                # Stage 1: Segmentation (model already loaded)
                data = stages['segmentation'].process(data)
                
                # Stage 2: Albedo (model already loaded)
                data = stages['albedo'].process(data)
                
                # Stage 3: Shadow/Degradation (model already loaded)
                data = stages['shadow'].process(data)
                
                # Stage 3.5: Recombine (model already loaded)
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
                    import json
                    with open(Path(output_root) / f"{idx:05d}_metadata.json", 'w') as f:
                        json.dump(data['degradation_metadata'], f, indent=2)
                
                result_queue.put({
                    'gpu_id': gpu_id,
                    'index': idx,
                    'image_path': str(img_path),
                    'output_path': str(output_path),
                    'success': True
                })
                
                if (i + 1) % 100 == 0:
                    logger.info(f"GPU {gpu_id}: Processed {i+1}/{len(image_indices)} images")
                    
            except Exception as e:
                logger.error(f"GPU {gpu_id}: Failed to process image {idx}: {e}")
                result_queue.put({
                    'gpu_id': gpu_id,
                    'index': idx,
                    'image_path': str(img_path),
                    'output_path': None,
                    'success': False,
                    'error': str(e)
                })
        
        # UNLOAD MODELS AFTER ALL IMAGES ARE PROCESSED
        logger.info(f"GPU {gpu_id}: Unloading models...")
        for stage_name, stage in stages.items():
            stage.unload_model()
        
        logger.info(f"GPU {gpu_id}: Completed all {len(image_indices)} images!")
                
    except Exception as e:
        logger.error(f"GPU {gpu_id} worker failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Run relighting pipeline in parallel across 8 GPUs"
    )
    parser.add_argument("--config", type=str, default="config/mvp_config.yaml")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with image paths")
    parser.add_argument("--num-samples", type=int, default=None, help="Number to process (default: all)")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    
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
    print(f"MULTI-GPU PARALLEL PROCESSING")
    print(f"{'='*60}")
    print(f"Split: {split_name}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}\n")
    
    # Get images to process
    image_paths = df_original['image_path'].tolist()
    if args.num_samples:
        image_paths = image_paths[:args.num_samples]
    
    total_images = len(image_paths)
    print(f"Processing {total_images} images across {args.num_gpus} GPUs...")
    print(f"Images per GPU: ~{total_images // args.num_gpus}\n")
    
    # Distribute images across GPUs
    images_per_gpu = total_images // args.num_gpus
    remainder = total_images % args.num_gpus
    
    gpu_assignments = []
    start_idx = 0
    
    for gpu_id in range(args.num_gpus):
        # Distribute remainder across first few GPUs
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
    
    print(f"\n🚀 Starting parallel processing...\n")
    
    # Create result queue
    result_queue = Queue()
    
    # Start worker processes
    processes = []
    for assignment in gpu_assignments:
        p = Process(
            target=worker_process,
            args=(
                assignment['gpu_id'],
                assignment['indices'],
                assignment['paths'],
                config,
                output_root,
                result_queue
            )
        )
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger starts to avoid race conditions
    
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
            
            # Check if all processes are done
            if all(not p.is_alive() for p in processes):
                # Drain remaining results
                while not result_queue.empty():
                    result = result_queue.get()
                    results.append(result)
                    pbar.update(1)
                break
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Create updated CSV (copy of original with output paths)
    print(f"\n📝 Creating updated CSV...")
    df_updated = df_original.copy()
    df_updated['output_image_path'] = None
    
    # Map results
    successful = 0
    for result in results:
        if result['success']:
            idx = result['index']
            if idx < len(df_updated):
                df_updated.loc[idx, 'output_image_path'] = result['output_path']
                successful += 1
    
    # Save updated CSV (new file, don't overwrite original)
    csv_dir = Path(args.csv).parent
    csv_name = Path(args.csv).stem
    output_csv_path = csv_dir / f"{csv_name}_with_relighting_outputs.csv"
    
    df_updated.to_csv(output_csv_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ MULTI-GPU PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Split: {split_name}")
    print(f"Total images: {total_images}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_images - successful}")
    print(f"GPUs used: {args.num_gpus}")
    print(f"\nOutputs:")
    print(f"  Images: {output_root}/")
    print(f"  CSV: {output_csv_path}")
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

