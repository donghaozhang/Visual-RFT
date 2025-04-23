#!/usr/bin/env python
import os
import torch
import argparse
import logging
import transformers
from transformers import AutoTokenizer, AutoModel, AutoProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_visual_rft")

def debug_model_loading():
    """Debug the model loading process and examine any errors"""
    logger.info("Starting model debugging")
    
    # Path to the model
    model_path = os.path.join(os.getcwd(), "share_models/Qwen2-VL-2B-Instruct")
    
    try:
        logger.info(f"Loading Qwen2-VL model from {model_path}")
        
        # Step 1: Load processor and tokenizer
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Processor loaded successfully")
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")
        
        # Step 2: Load model
        logger.info("Loading model (this may take some time)...")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info("Model loaded successfully")
        
        # Step 3: Trace internal components
        logger.info("Examining model architecture")
        model_info = {
            "model_type": model.config.model_type,
            "architectures": model.config.architectures if hasattr(model.config, "architectures") else "Not specified",
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size
        }
        logger.info(f"Model information: {model_info}")
        
        # Step 4: Check image processor components
        logger.info("Checking image processing components")
        processor_config = processor.image_processor.config if hasattr(processor, "image_processor") else "No image processor found"
        logger.info(f"Image processor config: {processor_config}")
        
        # Step 5: Run a simple test with a debug image
        logger.info("Preparing a test with a sample image")
        from PIL import Image
        import requests
        from io import BytesIO
        
        # Get a sample image - using a placeholder
        try:
            response = requests.get("https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/imgs/transformers_logo_name.png")
            test_image = Image.open(BytesIO(response.content))
            
            # Process text and image
            prompts = "What's in this image?"
            
            logger.info("Processing inputs...")
            input_data = processor(
                text=prompts,
                images=test_image,
                return_tensors="pt",
                padding=True
            )
            
            # Check input data shapes
            logger.info(f"Input IDs shape: {input_data['input_ids'].shape}")
            logger.info(f"Attention mask shape: {input_data['attention_mask'].shape}")
            logger.info(f"Pixel values shape: {input_data['pixel_values'].shape}")
            logger.info(f"Image grid thw: {input_data['image_grid_thw']}")
            
            # Extract key components
            image_nums = input_data['image_grid_thw']
            logger.info(f"Image nums: {image_nums}")
            logger.info(f"Image nums sum: {sum(image_nums)}")
            
            # Generate test output
            logger.info("Attempting to generate output...")
            try:
                with torch.no_grad():
                    outputs = model.generate(**input_data, max_new_tokens=30)
                    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info(f"Generated output: {decoded_output}")
            except Exception as e:
                logger.error(f"Generation error: {e}")
                # Inspect the error in detail
                import traceback
                logger.error(traceback.format_exc())
                
                # Try to isolate the split_with_sizes error
                if "split_with_sizes" in str(e):
                    logger.info("Detected split_with_sizes error, investigating...")
                    # Look for the relevant variables
                    if hasattr(model, "_expand_dict_for_generation_visual"):
                        logger.info("Found _expand_dict_for_generation_visual method")
                    else:
                        logger.info("Model doesn't have _expand_dict_for_generation_visual method directly exposed")
            
        except Exception as img_err:
            logger.error(f"Error in image processing: {img_err}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    debug_model_loading() 