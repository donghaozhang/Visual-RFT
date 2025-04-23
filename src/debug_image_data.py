#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
from PIL import Image
import logging
from transformers import AutoProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_data_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image_data_debug")

def inspect_image_data():
    """Inspect image data in training pipeline to identify zero-size issues"""
    # Path to the model processor
    model_path = os.path.join(os.getcwd(), "share_models/Qwen2-VL-2B-Instruct")
    
    try:
        # Load processor
        logger.info("Loading Qwen2-VL processor...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Processor loaded successfully")
        
        # Test case 1: Empty image (1x1 pixel)
        logger.info("Testing with empty 1x1 image")
        empty_image = Image.new('RGB', (1, 1), color='white')
        
        # Process the empty image
        empty_result = processor(
            text="Test empty image",
            images=empty_image,
            return_tensors="pt"
        )
        
        # Examine the results
        logger.info(f"Empty image processing results:")
        logger.info(f"Keys in output: {empty_result.keys()}")
        if 'image_grid_thw' in empty_result:
            logger.info(f"image_grid_thw: {empty_result['image_grid_thw']}")
            logger.info(f"shape: {empty_result['image_grid_thw'].shape}")
            logger.info(f"sum: {sum(empty_result['image_grid_thw'])}")
        if 'pixel_values' in empty_result:
            logger.info(f"pixel_values shape: {empty_result['pixel_values'].shape}")
            logger.info(f"non-zero elements: {torch.count_nonzero(empty_result['pixel_values']).item()}")
        
        # Test case 2: Normal image
        logger.info("\nTesting with normal image")
        try:
            # Try to load a test image from assets directory
            image_paths = [
                "assets/test_image.png",
                "assets/test_image.jpg",
                "../assets/test_image.png",
                "../assets/test_image.jpg"
            ]
            
            test_image = None
            for path in image_paths:
                try:
                    if os.path.exists(path):
                        test_image = Image.open(path)
                        logger.info(f"Loaded test image from {path}")
                        break
                except Exception:
                    continue
            
            # If no test image found, create one
            if test_image is None:
                logger.info("No test image found, creating a sample image")
                # Create a gradient test image
                width, height = 224, 224
                test_image = Image.new('RGB', (width, height))
                pixels = test_image.load()
                
                for i in range(width):
                    for j in range(height):
                        r = int(255 * i / width)
                        g = int(255 * j / height)
                        b = int(255 * (i + j) / (width + height))
                        pixels[i, j] = (r, g, b)
            
            # Process the test image
            normal_result = processor(
                text="Test normal image",
                images=test_image,
                return_tensors="pt"
            )
            
            # Examine the results
            logger.info(f"Normal image processing results:")
            logger.info(f"Keys in output: {normal_result.keys()}")
            if 'image_grid_thw' in normal_result:
                logger.info(f"image_grid_thw: {normal_result['image_grid_thw']}")
                logger.info(f"shape: {normal_result['image_grid_thw'].shape}")
                logger.info(f"sum: {sum(normal_result['image_grid_thw'])}")
            if 'pixel_values' in normal_result:
                logger.info(f"pixel_values shape: {normal_result['pixel_values'].shape}")
                logger.info(f"non-zero elements: {torch.count_nonzero(normal_result['pixel_values']).item()}")
            
            # Compare against normal case
            logger.info("\nComparison between empty and normal image:")
            if 'image_grid_thw' in empty_result and 'image_grid_thw' in normal_result:
                logger.info(f"Empty image_grid_thw: {empty_result['image_grid_thw']}")
                logger.info(f"Normal image_grid_thw: {normal_result['image_grid_thw']}")
                
            # Test simulate zero values to reproduce error
            logger.info("\nSimulating zero image_grid_thw to reproduce the error:")
            zero_grid = torch.zeros_like(normal_result['image_grid_thw'])
            logger.info(f"Zero grid: {zero_grid}")
            
            logger.info("\nAnalysis:")
            logger.info("The issue occurs when image_grid_thw contains all zeros [0, 0, 0, 0]")
            logger.info("This happens with empty images or when image processing fails")
            logger.info("In normal training, this should be checked and either:")
            logger.info("1. Skip batches with zero-sized images, or")
            logger.info("2. Replace zero values with valid ones that sum to the expected dimension size")
            
        except Exception as e:
            logger.error(f"Error testing normal image: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"Error in image data inspection: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting image data inspection")
    inspect_image_data()
    logger.info("Inspection complete") 