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
        logger.info("Testing with empty 1x1 image (This will fail with factor error)")
        empty_image = Image.new('RGB', (1, 1), color='white')
        try:
            # This will fail with "height:1 and width:3 must be larger than factor:28"
            empty_result = processor(
                text="Test empty image",
                images=empty_image,
                return_tensors="pt"
            )
            logger.info(f"Empty image processing results (unexpected success):")
            logger.info(f"Keys in output: {empty_result.keys()}")
        except ValueError as e:
            logger.info(f"Expected error with tiny image: {e}")
        
        # Test case 1b: Small but valid image (28x28 pixels - minimum size)
        logger.info("\nTesting with minimum valid size image (28x28)")
        min_valid_image = Image.new('RGB', (28, 28), color='blue')
        try:
            min_result = processor(
                text="Test minimum size image",
                images=min_valid_image,
                return_tensors="pt"
            )
            logger.info(f"Minimum size image processing results:")
            logger.info(f"Keys in output: {min_result.keys()}")
            if 'image_grid_thw' in min_result:
                logger.info(f"image_grid_thw: {min_result['image_grid_thw']}")
                logger.info(f"shape: {min_result['image_grid_thw'].shape}")
                logger.info(f"sum: {sum(min_result['image_grid_thw'])}")
            if 'pixel_values' in min_result:
                logger.info(f"pixel_values shape: {min_result['pixel_values'].shape}")
                logger.info(f"non-zero elements: {torch.count_nonzero(min_result['pixel_values']).item()}")
        except Exception as e:
            logger.error(f"Unexpected error with minimum size image: {e}")
        
        # Test case 2: Normal image (224x224 - recommended size)
        logger.info("\nTesting with normal 224x224 image")
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
        
        try:
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
            
            # Test our fix: Let's simulate what happens when we fix a tiny image
            logger.info("\nTesting our fix by creating a valid-sized placeholder for tiny images")
            tiny_image = Image.new('RGB', (5, 10), color='red')
            logger.info(f"Original tiny image size: {tiny_image.width}x{tiny_image.height}")
            
            # Apply our fix
            MIN_IMAGE_SIZE = 224
            placeholder = Image.new('RGB', (MIN_IMAGE_SIZE, MIN_IMAGE_SIZE), color='gray')
            # Paste the tiny image in the center if possible
            if tiny_image.width > 0 and tiny_image.height > 0:
                paste_x = (MIN_IMAGE_SIZE - tiny_image.width) // 2
                paste_y = (MIN_IMAGE_SIZE - tiny_image.height) // 2
                placeholder.paste(tiny_image, (paste_x, paste_y))
            
            # Process the fixed image
            fixed_result = processor(
                text="Test fixed image",
                images=placeholder,
                return_tensors="pt"
            )
            
            logger.info(f"Fixed image processing results:")
            logger.info(f"image_grid_thw: {fixed_result['image_grid_thw']}")
            logger.info(f"sum: {sum(fixed_result['image_grid_thw'])}")
            
            logger.info("\nAnalysis:")
            logger.info("1. Tiny images (smaller than 28x28) fail with error: height and width must be larger than factor:28")
            logger.info("2. This causes image_grid_thw to be all zeros [0,0,0,0] later in the pipeline")
            logger.info("3. When torch.split() is called with these zeros, it causes the RuntimeError")
            logger.info("4. Our fix creates properly sized placeholder images before processing, ensuring valid values")
            
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