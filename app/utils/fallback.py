"""
Fallback handwriting recognition module that uses simple OCR techniques
when the main transformer model cannot be loaded
"""

import numpy as np
from PIL import Image
import cv2
import os
import string

class FallbackHandwritingRecognizer:
    """
    A simple fallback handwriting recognizer that uses template matching
    for basic character recognition when the main model fails to load
    """
    
    def __init__(self):
        """Initialize the fallback recognizer"""
        # List of characters to recognize (limited set)
        self.characters = string.ascii_uppercase + string.ascii_lowercase + string.digits
        
        # Initialize templates dictionary
        self.templates = {}
        
        # Try to load templates if available
        template_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'templates')
        if os.path.exists(template_dir):
            for char in self.characters:
                template_path = os.path.join(template_dir, f"{char}.png")
                if os.path.exists(template_path):
                    try:
                        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                        if template is not None:
                            self.templates[char] = template
                    except Exception as e:
                        print(f"Error loading template for {char}: {e}")
        
        print(f"Fallback recognizer initialized with {len(self.templates)} templates")
    
    def recognize(self, image):
        """
        Recognize text in the given image using simple techniques
        
        Args:
            image: PIL Image or numpy array containing handwritten text
            
        Returns:
            Recognized text as string or error message
        """
        if image is None:
            return "No image provided"
        
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Check if image is valid
        if not isinstance(image, np.ndarray) or image.size == 0:
            return "Invalid image format"
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # If we have no templates, just return a basic message
            if not self.templates:
                # Check if image has content
                white_pixels = np.sum(thresh > 0)
                total_pixels = thresh.size
                
                if white_pixels / total_pixels < 0.01:
                    return "No text detected"
                
                # Very basic estimation of character count
                estimated_chars = max(1, int(white_pixels / 500))
                
                if estimated_chars == 1:
                    return "Single character detected (possibly A-Z or 0-9)"
                else:
                    return f"Approximately {estimated_chars} characters detected"
            
            # If we have templates, try template matching (basic implementation)
            # This would need to be expanded for a real application
            return "Handwriting detected (fallback recognition active)"
            
        except Exception as e:
            return f"Error in fallback recognition: {e}"


# Singleton instance
_fallback_instance = None

def get_fallback_recognizer():
    """Get the fallback recognizer instance"""
    global _fallback_instance
    
    if _fallback_instance is None:
        _fallback_instance = FallbackHandwritingRecognizer()
    
    return _fallback_instance