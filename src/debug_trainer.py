#!/usr/bin/env python
import os
import sys
import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoProcessor, AutoModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_trainer")

def patch_unwrap_model_for_generation():
    """Patch the unwrap_model_for_generation function to add debugging"""
    from virft.src.open_r1.trainer.grpo_trainer import unwrap_model_for_generation
    
    original_unwrap = unwrap_model_for_generation
    
    def patched_unwrap(model, accelerator):
        logger.info("Unwrapping model for generation")
        return original_unwrap(model, accelerator)
    
    # Replace the function with our patched version
    sys.modules['virft.src.open_r1.trainer.grpo_trainer'].unwrap_model_for_generation = patched_unwrap
    logger.info("Patched unwrap_model_for_generation function")

def patch_model_generate():
    """Monkey patch the model's generate method to add debugging"""
    def patch_generate(original_generate):
        def wrapped_generate(self, **kwargs):
            logger.info("Starting model.generate")
            
            # Log the inputs to generate
            for key, value in kwargs.items():
                if key == 'pixel_values' and isinstance(value, torch.Tensor):
                    logger.info(f"pixel_values shape: {value.shape}")
                elif key == 'image_grid_thw':
                    logger.info(f"image_grid_thw value: {value}")
                    logger.info(f"image_grid_thw sum: {sum(value)}")
                elif isinstance(value, torch.Tensor):
                    logger.info(f"{key} shape: {value.shape}")
                else:
                    logger.info(f"{key} type: {type(value)}")
            
            try:
                return original_generate(self, **kwargs)
            except Exception as e:
                logger.error(f"Error in generate: {e}")
                logger.error(traceback.format_exc())
                
                # Add extra debug info for split_with_sizes error
                if "split_with_sizes" in str(e):
                    logger.error("Found split_with_sizes error, investigating...")
                    
                    # Check image_grid_thw values
                    if 'image_grid_thw' in kwargs:
                        image_grid_thw = kwargs['image_grid_thw']
                        logger.error(f"image_grid_thw: {image_grid_thw}")
                        logger.error(f"Sum of image_grid_thw: {sum(image_grid_thw)}")
                        logger.error(f"Contains zeros: {0 in image_grid_thw}")
                
                raise e
                
        return wrapped_generate

    # Find and patch any Qwen2-VL models in the codebase
    logger.info("Attempting to patch model.generate method")
    
    # This will be applied when a model is loaded
    original_from_pretrained = AutoModel.from_pretrained
    
    def patched_from_pretrained(*args, **kwargs):
        model = original_from_pretrained(*args, **kwargs)
        if hasattr(model, 'generate'):
            logger.info("Patching model.generate method")
            model.generate = patch_generate(model.generate)
        return model
    
    AutoModel.from_pretrained = patched_from_pretrained
    logger.info("Set up model.generate patching")

def patch_trainer_compute_loss():
    """Patch the compute_loss method of the GRPOTrainer to add debugging"""
    try:
        from virft.src.open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
        
        original_compute_loss = Qwen2VLGRPOTrainer.compute_loss
        
        def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            logger.info("Starting compute_loss")
            
            try:
                # Log relevant input data
                if isinstance(inputs, list) and len(inputs) > 0:
                    logger.info(f"Input batch size: {len(inputs)}")
                    
                    # Check if 'prompt' and 'image' exist in inputs
                    if 'prompt' in inputs[0] and 'image' in inputs[0]:
                        logger.info(f"Input contains 'prompt' and 'image' fields")
                        
                        # Sample from the first item
                        prompt_example = inputs[0]['prompt']
                        logger.info(f"Prompt type: {type(prompt_example)}")
                        
                        image_example = inputs[0]['image']
                        logger.info(f"Image type: {type(image_example)}")
                        
                        if hasattr(image_example, 'size'):
                            logger.info(f"Image size: {image_example.size}")
                
                # Continue with original method
                return original_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
                
            except Exception as e:
                logger.error(f"Error in compute_loss: {e}")
                logger.error(traceback.format_exc())
                
                if "split_with_sizes" in str(e):
                    logger.error("Detected split_with_sizes error in compute_loss")
                    
                    # Try to extract image information
                    if hasattr(self, 'processing_class'):
                        logger.error(f"Processing class: {type(self.processing_class)}")
                    
                raise e
        
        # Replace the method
        Qwen2VLGRPOTrainer.compute_loss = patched_compute_loss
        logger.info("Patched compute_loss method")
        
    except Exception as e:
        logger.error(f"Failed to patch compute_loss: {e}")
        logger.error(traceback.format_exc())

def debug_image_preprocessing():
    """Test the image preprocessing components"""
    logger.info("Testing image preprocessing")
    
    try:
        # Path to the model
        model_path = os.path.join(os.getcwd(), "share_models/Qwen2-VL-2B-Instruct")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Test with a sample image
        from PIL import Image
        import requests
        from io import BytesIO
        
        # Get a sample image
        response = requests.get("https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/imgs/transformers_logo_name.png")
        test_image = Image.open(BytesIO(response.content))
        
        # Process different image configurations
        logger.info("Testing single image processing")
        single_result = processor(
            text="Test single image",
            images=test_image,
            return_tensors="pt"
        )
        
        logger.info(f"Single image result keys: {single_result.keys()}")
        logger.info(f"Single image grid_thw: {single_result.get('image_grid_thw')}")
        
        # Test with multiple images
        logger.info("Testing multiple images processing")
        multi_result = processor(
            text="Test multiple images",
            images=[test_image, test_image],
            return_tensors="pt"
        )
        
        logger.info(f"Multiple images result keys: {multi_result.keys()}")
        logger.info(f"Multiple images grid_thw: {multi_result.get('image_grid_thw')}")
        
        # Test with empty image (to simulate the error condition)
        logger.info("Testing empty image processing")
        try:
            # Create a 1x1 transparent image
            empty_image = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            empty_result = processor(
                text="Test empty image",
                images=empty_image,
                return_tensors="pt"
            )
            
            logger.info(f"Empty image result keys: {empty_result.keys()}")
            logger.info(f"Empty image grid_thw: {empty_result.get('image_grid_thw')}")
        except Exception as e:
            logger.error(f"Error processing empty image: {e}")
            
    except Exception as e:
        logger.error(f"Error in image preprocessing test: {e}")
        logger.error(traceback.format_exc())

def main():
    logger.info("Starting trainer debugging")
    
    # Set up patches to monitor code execution
    patch_unwrap_model_for_generation()
    patch_model_generate()
    patch_trainer_compute_loss()
    
    # Test image preprocessing
    debug_image_preprocessing()
    
    logger.info("Debug patches installed. The next training run will include detailed logging.")
    logger.info("Run your normal training command now, and check debug_trainer.log for detailed information.")
    
if __name__ == "__main__":
    main() 