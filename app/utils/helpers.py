import numpy as np
from PIL import Image
import cv2

def preprocess_image(image):
    """
    Preprocess image for handwriting recognition
    
    Args:
        image: PIL Image, numpy array or dictionary representing the drawn image
        
    Returns:
        Preprocessed image ready for model input
    """
    if image is None:
        return None
    
    # Handle dictionary format from newer Gradio versions
    if isinstance(image, dict):
        # Check if 'image' key exists (from gr.Paint)
        if 'image' in image:
            image = image['image']
        # For other formats, try common keys
        elif 'value' in image:
            image = image['value']
        else:
            # Return first value if we can't identify the right key
            for key in image:
                if isinstance(image[key], (np.ndarray, Image.Image)):
                    image = image[key]
                    break
    
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Make sure we actually have an image to process
    if not isinstance(image, np.ndarray):
        print(f"Unexpected image type: {type(image)}")
        return None
    
    # Check if image is already grayscale or RGB
    if len(image.shape) == 2:
        # Image is already grayscale
        gray = image
    else:
        # Convert to grayscale
        if image.shape[2] == 4:  # RGBA
            # Create white background
            white_background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
            
            # Extract alpha channel
            alpha = image[:, :, 3:4] / 255.0
            
            # Blend with white background
            rgb = image[:, :, :3]
            image = (rgb * alpha + white_background * (1 - alpha)).astype(np.uint8)
            
        # Convert to grayscale
        if len(image.shape) == 3:  # RGB or RGBA converted to RGB
            try:
                import cv2
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            except ImportError:
                # Fallback if OpenCV is not available
                gray = np.mean(image, axis=2).astype(np.uint8)
    
    # Apply thresholding to make the handwriting clearer
    try:
        import cv2
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    except ImportError:
        # Simple thresholding fallback
        thresh = np.where(gray < 200, 255, 0).astype(np.uint8)
    
    # TODO: Add any additional preprocessing steps needed for the OCR model
    
    return thresh

def postprocess_result(result):
    """
    Postprocess recognition results
    
    Args:
        result: Raw output from the handwriting recognition model
        
    Returns:
        Cleaned and formatted text output
    """
    if result is None:
        return ""
    
    # TODO: Implement actual result postprocessing
    # For now, just return the input string
    
    # Remove any extra whitespace
    result = ' '.join(result.split())
    
    return result

def save_user_data(user_data):
    """
    Save user data and handwriting sample to database or file
    
    Args:
        user_data: Dictionary containing user information and analysis
        
    Returns:
        Success status
    """
    # TODO: Implement data saving functionality
    
    return True