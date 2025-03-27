import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import functools
import time

# Load environment variables
load_dotenv()

class HandwritingRecognitionModel:
    """
    Class for handwriting recognition using TrOCR model from Transformers
    """
    
    def __init__(self, model_name="microsoft/trocr-small-handwritten", model_path=None):
        """
        Initialize the handwriting recognition model
        
        Args:
            model_name: Name of the pre-trained model to use
            model_path: Path to the model weights, or None to use the pre-trained model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Start timer for model loading
        start_time = time.time()
        
        if model_path and os.path.exists(model_path):
            # Load from local path if specified and exists
            print(f"Loading model from {model_path}")
            self.processor = TrOCRProcessor.from_pretrained(model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        else:
            # Otherwise use pre-trained model
            print(f"Loading pre-trained model: {model_name}")
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            
            # Save model path for future reference
            if model_path:
                print(f"Saving model to {model_path}")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.processor.save_pretrained(model_path)
                self.model.save_pretrained(model_path)
        
        # Set model to evaluation mode for inference
        self.model.eval()
        
        # Report loading time
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        
        # Cache for results to avoid reprocessing identical images
        self.cache = {}
        self.max_cache_size = 50
        self.last_result = None  # Track last result to avoid duplicates
    
    @functools.lru_cache(maxsize=32)
    def recognize_single_char(self, image_hash):
        """
        Special handling for single character recognition
        This complements the main recognition when we detect a possible single character
        
        Args:
            image_hash: Hash of the image for caching
            
        Returns:
            Most likely character
        """
        # We would implement character-specific recognition here
        # For now, we'll rely on the main model but with special flags
        return None
        
    def clear_cache(self):
        """Clear all caches to force fresh processing"""
        self.cache = {}
        self.last_result = None
        # Clear the LRU cache for recognize_single_char
        self.recognize_single_char.cache_clear()
        print("Model cache cleared")
    
    def recognize(self, image, force_new_processing=False):
        """
        Recognize text in the given image
        
        Args:
            image: PIL Image or numpy array containing handwritten text
            force_new_processing: If True, ignore cache and reprocess the image
            
        Returns:
            Recognized text as string
        """
        if image is None:
            return ""
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Generate a hash that includes a timestamp component to avoid cache collisions
        current_time = time.time()
        try:
            # Add timestamp to reduce chance of collisions
            hash_input = image.tobytes() + str(current_time % 60).encode('utf-8')
            image_hash = hash(hash_input)
            
            # Check cache first, but only if not forcing new processing
            if not force_new_processing and image_hash in self.cache:
                print(f"Using cached recognition result for hash: {image_hash % 10000}")
                return self.cache[image_hash]
        except Exception as e:
            print(f"Error generating hash: {e}")
            image_hash = None  # Continue without caching
        
        # Special handling for very small amount of content (likely single characters)
        try:
            img_array = np.array(image)
            non_white_pixels = np.sum(np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array < 250)
            is_small_content = non_white_pixels < 500  # Threshold can be adjusted
        except Exception:
            is_small_content = False
        
        # Process the image for the model
        with torch.no_grad():  # Disable gradient calculation for inference
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Set special settings for small content
            if is_small_content:
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=2,  # Short sequence for single characters
                    num_beams=5,   # More careful beam search
                    temperature=0.7  # Slightly lower temperature for more focused generation
                )
            else:
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=64,  # Standard for text
                    num_beams=4,
                    temperature=1.0
                )
            
            # Decode the generated IDs to text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Try special handling for single characters if needed
        if is_small_content and (not generated_text or len(generated_text.strip()) > 1):
            # Create a hash for the single character recognition
            try:
                sc_hash = hash(image.tobytes())
                single_char = self.recognize_single_char(sc_hash)
                if single_char:
                    generated_text = single_char
            except Exception as e:
                print(f"Error in single character recognition: {e}")
        
        # Check if this result is the same as the last one and it shouldn't be
        if self.last_result == generated_text and not force_new_processing:
            # Try again with special parameters for potentially better recognition
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            try:
                # Use different generation parameters
                alt_generated_ids = self.model.generate(
                    pixel_values,
                    max_length=8,
                    num_beams=5,
                    temperature=0.9  # Higher temperature for more variability
                )
                alt_text = self.processor.batch_decode(alt_generated_ids, skip_special_tokens=True)[0]
                
                # If the alternative is different and not empty, use it
                if alt_text and alt_text != generated_text:
                    generated_text = alt_text
                    print(f"Used alternative recognition: '{alt_text}'")
            except Exception as e:
                print(f"Error in alternative recognition: {e}")
        
        # Store in cache
        if image_hash is not None:
            if len(self.cache) >= self.max_cache_size:
                # Remove a random item if cache is full
                self.cache.pop(next(iter(self.cache)))
            self.cache[image_hash] = generated_text
        
        self.last_result = generated_text
        
        return generated_text

# Singleton pattern with caching
_model_instance = None

def get_model(model_name=None, model_path=None):
    """
    Get the handwriting recognition model instance
    
    Args:
        model_name: Optional name of pre-trained model to use
        model_path: Optional path to model weights
        
    Returns:
        HandwritingRecognitionModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        # Get model path from environment variable if not provided
        if model_path is None:
            model_path = os.environ.get("MODEL_WEIGHTS_PATH", "models/model_weights/trocr_handwritten")
        
        # Use provided model name or default
        if model_name is None:
            model_name = os.environ.get("MODEL_NAME", "microsoft/trocr-small-handwritten")
        
        _model_instance = HandwritingRecognitionModel(model_name, model_path)
    
    return _model_instance

def clear_model_cache():
    """Clear the model cache if it exists"""
    global _model_instance
    
    if _model_instance is not None:
        _model_instance.clear_cache()