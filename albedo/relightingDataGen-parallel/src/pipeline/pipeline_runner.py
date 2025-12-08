"""
Main pipeline runner for the relighting dataset generation.
Orchestrates all 4 stages with sequential model loading/unloading.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
import json
from tqdm import tqdm
from PIL import Image
import pandas as pd

from .memory_manager import MemoryManager
from ..stages.stage_1_segmentation_sam3 import SAM3SegmentationStage
from ..stages.stage_2_albedo import AlbedoExtractionStage
from ..stages.stage_3_shadow import ShadowGenerationStage
from ..stages.stage_3_5_recombine import BackgroundRecombinationStage
from ..stages.stage_4_captioning import CaptioningStage

logger = logging.getLogger(__name__)


class RelightingPipeline:
    """
    End-to-end pipeline for generating relighting dataset.

    Processes images through 4 sequential stages:
    1. SAM2 segmentation (foreground/background separation)
    2. IntrinsicAnything albedo extraction
    3. Shadow generation
    4. Qwen2.5-VL captioning
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initializing RelightingPipeline on device: {self.device}")

        # Initialize memory manager
        max_memory_gb = config.get('memory', {}).get('max_gpu_memory_gb', 22)
        self.memory_manager = MemoryManager(max_memory_gb=max_memory_gb)

        # Initialize stages (don't load models yet)
        logger.info("Initializing pipeline stages...")
        self.stages = {
            'segmentation': SAM3SegmentationStage(config, str(self.device)),
            'albedo': AlbedoExtractionStage(config, str(self.device)),
            'shadow': ShadowGenerationStage(config, str(self.device)),
            'recombine': BackgroundRecombinationStage(config, str(self.device)),
            'captioning': CaptioningStage(config, str(self.device))
        }

        # Setup paths
        self.paths = config.get('paths', {})
        self.data_root = Path(self.paths.get('data_root', 'data'))
        self.output_root = Path(self.paths.get('output_root', 'data/outputs'))

        # Create output directories
        self._setup_directories()

        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'stage_times': {
                'segmentation': [],
                'albedo': [],
                'shadow': [],
                'recombine': [],
                'captioning': []
            },
            'peak_memory': []
        }

        logger.info("Pipeline initialized successfully")

    def _setup_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.data_root / 'stage_1',
            self.data_root / 'stage_2',
            self.data_root / 'stage_3',
            self.data_root / 'stage_3_5',
            self.data_root / 'stage_4',
            self.output_root
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directories created at: {self.output_root}")

    def run_stage(self, stage_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single stage with memory management.

        Args:
            stage_name: Name of the stage to run
            input_data: Input data dictionary

        Returns:
            Output data dictionary
        """
        stage = self.stages[stage_name]

        logger.info(f"\n{'='*80}")
        logger.info(f"Running stage: {stage_name}")
        logger.info(f"{'='*80}")

        # Reset peak memory stats
        self.memory_manager.reset_peak_stats()

        # Load model
        start_time = time.time()
        logger.info(f"Loading {stage_name} model...")
        stage.load_model()

        # Log memory after loading
        self.memory_manager.log_memory_stats(f"After loading {stage_name}: ")

        # Process
        logger.info(f"Processing with {stage_name}...")
        output = stage.process(input_data)

        # Record processing time
        elapsed = time.time() - start_time
        self.stats['stage_times'][stage_name].append(elapsed)
        logger.info(f"{stage_name} completed in {elapsed:.2f}s")

        # Unload model
        logger.info(f"Unloading {stage_name} model...")
        stage.unload_model()

        # Clear cache
        if self.config.get('memory', {}).get('clear_cache_between_stages', True):
            self.memory_manager.clear_cache()

        # Log peak memory
        mem_stats = self.memory_manager.get_memory_stats()
        peak_memory = mem_stats['peak']
        self.stats['peak_memory'].append(peak_memory)
        logger.info(f"Peak GPU memory for {stage_name}: {peak_memory:.2f}GB")

        return output

    def save_intermediate(self, stage_name: str, data: Dict[str, Any]):
        """
        Save intermediate outputs from a stage.

        Args:
            stage_name: Name of the stage
            data: Data dictionary with outputs
        """
        image_id = data['image_id']
        stage_dir = self.data_root / stage_name

        # Determine what to save based on stage
        if stage_name == 'stage_1':
            # Save foreground, background, mask
            if 'foreground' in data:
                data['foreground'].save(stage_dir / f"{image_id:05d}_foreground.png")
            if 'background' in data:
                data['background'].save(stage_dir / f"{image_id:05d}_background.png")
            if 'mask' in data:
                data['mask'].save(stage_dir / f"{image_id:05d}_mask.png")

        elif stage_name == 'stage_2':
            # Save albedo (and specular if available)
            if 'albedo' in data:
                data['albedo'].save(stage_dir / f"{image_id:05d}_albedo.png")
            if 'specular' in data and data['specular'] is not None:
                data['specular'].save(stage_dir / f"{image_id:05d}_specular.png")

        elif stage_name == 'stage_3':
            # Save degraded image (foreground only)
            if 'degraded_image' in data:
                data['degraded_image'].save(stage_dir / f"{image_id:05d}_degraded.png")

            # Save parameters as JSON
            params = data.get('degradation_metadata', {})
            with open(stage_dir / f"{image_id:05d}_params.json", 'w') as f:
                json.dump(params, f, indent=2)

        elif stage_name == 'stage_3_5':
            # Save composite image (foreground + background)
            if 'composite_image' in data:
                data['composite_image'].save(stage_dir / f"{image_id:05d}_composite.png")

        elif stage_name == 'stage_4':
            # Save caption
            if 'caption' in data:
                with open(stage_dir / f"{image_id:05d}_caption.txt", 'w') as f:
                    f.write(data['caption'])

    def save_final_output(self, data: Dict[str, Any]):
        """
        Save final output (input, composite, albedo).

        Args:
            data: Complete data dictionary
        """
        image_id = data['image_id']

        # Save original image
        if 'original_image' in data:
            data['original_image'].save(
                self.output_root / f"{image_id:05d}_input.png"
            )

        # Save composite image (MAIN OUTPUT from stage 3.5)
        if 'composite_image' in data:
            data['composite_image'].save(
                self.output_root / f"{image_id:05d}_output.png"
            )

        # Save degraded foreground only (from stage 3)
        if 'degraded_image' in data:
            data['degraded_image'].save(
                self.output_root / f"{image_id:05d}_degraded_fg.png"
            )

        # Save albedo
        if 'albedo' in data:
            data['albedo'].save(
                self.output_root / f"{image_id:05d}_albedo.png"
            )

        # Save foreground
        if 'foreground' in data:
            data['foreground'].save(
                self.output_root / f"{image_id:05d}_foreground.png"
            )

        # Save background
        if 'background' in data:
            data['background'].save(
                self.output_root / f"{image_id:05d}_background.png"
            )

        # Save degradation metadata
        if 'degradation_metadata' in data:
            with open(self.output_root / f"{image_id:05d}_metadata.json", 'w') as f:
                json.dump(data['degradation_metadata'], f, indent=2)

    def process_image(self, image_path: str, image_id: int) -> Optional[Dict[str, Any]]:
        """
        Process a single image through all stages.

        Args:
            image_path: Path to input image
            image_id: Numeric ID for the image

        Returns:
            Final output dictionary, or None if processing failed
        """
        try:
            logger.info(f"\n\n{'#'*80}")
            logger.info(f"# Processing image {image_id}: {image_path}")
            logger.info(f"{'#'*80}\n")

            # Load image
            image = Image.open(image_path).convert('RGB')
            data = {
                'image': image,
                'image_id': image_id,
                'original_image': image
            }

            # Stage 1: Segmentation
            data = self.run_stage('segmentation', data)
            self.save_intermediate('stage_1', data)

            # Stage 2: Albedo extraction
            data = self.run_stage('albedo', data)
            self.save_intermediate('stage_2', data)

            # Stage 3: Degradation synthesis (foreground only)
            data = self.run_stage('shadow', data)
            self.save_intermediate('stage_3', data)

            # Stage 3.5: Background recombination
            data = self.run_stage('recombine', data)
            self.save_intermediate('stage_3_5', data)

            # Stage 4: Captioning (DISABLED for MVP)
            # data = self.run_stage('captioning', data)
            # self.save_intermediate('stage_4', data)

            # Save final output
            self.save_final_output(data)

            logger.info(f"\n{'='*80}")
            logger.info(f"Successfully processed image {image_id}")
            logger.info(f"{'='*80}\n")

            self.stats['successful'] += 1
            return data

        except Exception as e:
            logger.error(f"Failed to process image {image_id}: {e}", exc_info=True)
            self.stats['failed'] += 1
            return None

    def run_pipeline(self, num_samples: Optional[int] = None, csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run pipeline on dataset.

        Args:
            num_samples: Number of samples to process (None = all)
            csv_path: Path to CSV file with image paths (optional, overrides raw_dir)

        Returns:
            List of processed data dictionaries
        """
        logger.info("Starting pipeline execution")

        # Get list of images
        if csv_path:
            # Load from CSV
            logger.info(f"Loading image paths from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Check for 'image_path' column
            if 'image_path' not in df.columns:
                logger.error(f"CSV file must contain 'image_path' column. Found columns: {df.columns.tolist()}")
                return []
            
            image_paths = df['image_path'].tolist()
            image_paths = [Path(p) for p in image_paths]
            logger.info(f"Loaded {len(image_paths)} image paths from CSV")
        else:
            # Load from raw directory (old behavior)
            raw_dir = self.data_root / 'raw'
            image_paths = sorted(raw_dir.glob('*.png'))
            
            if not image_paths:
                logger.error(f"No images found in {raw_dir}")
                logger.error("Please run: python scripts/download_ffhq.py")
                return []

        # Limit number of samples
        if num_samples is not None:
            image_paths = image_paths[:num_samples]

        logger.info(f"Found {len(image_paths)} images to process")

        # Process all images
        results = []
        self.stats['total_images'] = len(image_paths)

        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            # Use index as image ID
            image_id = idx

            result = self.process_image(str(image_path), image_id)

            if result is not None:
                results.append(result)

        # Save metadata
        self._save_metadata(results)

        # Print summary
        self._print_summary()

        return results

    def _save_metadata(self, results: List[Dict[str, Any]]):
        """
        Save pipeline metadata and results.

        Args:
            results: List of processed results
        """
        metadata = {
            'dataset_name': self.config.get('pipeline', {}).get('name', 'ffhq_relighting_poc'),
            'num_samples': len(results),
            'total_attempted': self.stats['total_images'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'samples': []
        }

        for result in results:
            sample_meta = {
                'image_id': result['image_id'],
                'original': str(self.data_root / 'raw' / f"{result['image_id']:05d}.png"),
                'foreground': str(self.data_root / 'stage_1' / f"{result['image_id']:05d}_foreground.png"),
                'background': str(self.data_root / 'stage_1' / f"{result['image_id']:05d}_background.png"),
                'albedo': str(self.data_root / 'stage_2' / f"{result['image_id']:05d}_albedo.png"),
                'albedo_method': result.get('albedo_method', 'unknown'),
                'degraded_foreground': str(self.data_root / 'stage_3' / f"{result['image_id']:05d}_degraded.png"),
                'composite_image': str(self.data_root / 'stage_3_5' / f"{result['image_id']:05d}_composite.png"),
                'degradation_type': result.get('degradation_metadata', {}).get('degradation_type', 'unknown'),
                'degradation_metadata': result.get('degradation_metadata', {}),
            }
            metadata['samples'].append(sample_meta)

        # Save metadata
        metadata_path = self.output_root / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to: {metadata_path}")

    def _print_summary(self):
        """Print pipeline execution summary."""
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)

        logger.info(f"Total images: {self.stats['total_images']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")

        logger.info("\nAverage processing times:")
        for stage, times in self.stats['stage_times'].items():
            if times:
                avg_time = sum(times) / len(times)
                logger.info(f"  {stage}: {avg_time:.2f}s")

        if self.stats['peak_memory']:
            max_mem = max(self.stats['peak_memory'])
            avg_mem = sum(self.stats['peak_memory']) / len(self.stats['peak_memory'])
            logger.info(f"\nPeak GPU memory: {max_mem:.2f}GB (avg: {avg_mem:.2f}GB)")

        logger.info("="*80)
