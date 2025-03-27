import numpy as np
from PIL import Image
import cv2
import io
import hashlib
import time

# Cache for processed images to avoid redundant processing
_preprocess_cache = {}
_max_cache_size = 20
_last_image_hash = None  # Track the last processed image hash

def extract_image_from_gradio(image_data):
    """
    Extract image from various Gradio formats
    
    Args:
        image_data: Input from Gradio canvas in various possible formats
        
    Returns:
        Extracted image as numpy array or None if extraction fails
    """
    try:
        if image_data is None:
            return None
            
        print(f"Received image data type: {type(image_data)}")
        
        # Handle dictionary format from newer Gradio versions
        if isinstance(image_data, dict):
            # Try known keys
            for key in ['image', 'value', 'img', 'data']:
                if key in image_data and image_data[key] is not None:
                    return extract_image_from_gradio(image_data[key])
            
            # Try to find any value that might be an image
            for key, value in image_data.items():
                if isinstance(value, (np.ndarray, Image.Image, bytes, str)):
                    return extract_image_from_gradio(value)
                    
            print(f"Unable to extract image from dict: {list(image_data.keys())}")
            
            # Last resort: if there's only one key, try that
            if len(image_data) == 1:
                only_value = next(iter(image_data.values()))
                return extract_image_from_gradio(only_value)
                
            return None
        
        # Handle PIL Image
        if isinstance(image_data, Image.Image):
            return np.array(image_data)
        
        # Handle numpy array
        if isinstance(image_data, np.ndarray):
            # Check if image has content (not all black or white)
            if image_data.size > 0:
                # For RGBA images in numpy
                if len(image_data.shape) == 3 and image_data.shape[2] == 4:
                    # Create white background
                    white_bg = np.ones((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8) * 255
                    # Extract RGB and alpha
                    rgb = image_data[:, :, :3]
                    alpha = image_data[:, :, 3:4] / 255.0
                    # Composite
                    return (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            return image_data
        
        # Handle bytes (e.g., from file upload)
        if isinstance(image_data, bytes):
            try:
                img = Image.open(io.BytesIO(image_data))
                return np.array(img)
            except Exception as e:
                print(f"Error converting bytes to image: {e}")
                return None
        
        # Handle base64 string or file path
        if isinstance(image_data, str):
            try:
                # Try to open as a file path
                img = Image.open(image_data)
                return np.array(img)
            except Exception:
                # Could be a base64 string, but this isn't commonly used in Gradio
                print(f"Unable to process string input: {image_data[:30]}...")
                return None
        
        # Handle list/tuple (might be a triple of RGB values)
        if isinstance(image_data, (list, tuple)) and len(image_data) > 0:
            # Try first element
            return extract_image_from_gradio(image_data[0])
        
        print(f"Unhandled image type: {type(image_data)}")
        return None
    except Exception as e:
        print(f"Error in image extraction: {e}")
        return None

def preprocess_image(image_data, force_new_processing=False):
    """
    Preprocess image for handwriting recognition
    
    Args:
        image_data: Input from Gradio canvas in various possible formats
        force_new_processing: If True, ignore cache and reprocess the image
        
    Returns:
        Preprocessed image as PIL Image ready for model input
    """
    global _preprocess_cache, _last_image_hash
    
    start_time = time.time()
    
    # Extract image data regardless of format
    image = extract_image_from_gradio(image_data)
    
    if image is None:
        print("Failed to extract image data")
        return None
    
    # Check if this is an empty/cleared canvas
    is_empty = False
    if isinstance(image, np.ndarray):
        # Check if image is mostly empty/blank
        if image.size > 0:
            if len(image.shape) == 3:
                # For color images, check if it's mostly one color
                is_empty = np.mean(np.std(image, axis=(0, 1))) < 2.0
            else:
                # For grayscale, check if it's mostly one value
                is_empty = np.std(image) < 2.0
    
    if is_empty:
        print("Canvas appears to be empty/cleared")
        _last_image_hash = None  # Reset last image hash
        _preprocess_cache = {}   # Clear cache on empty canvas
        return None
    
    # Generate a hash for caching
    try:
        # Add a timestamp component to avoid identical hashes for similar-looking content
        timestamp = str(time.time())
        hash_input = image.tobytes() + timestamp.encode('utf-8')
        image_hash = hashlib.md5(hash_input).hexdigest()
        
        # For debugging
        print(f"Generated image hash: {image_hash[:8]}, force_new={force_new_processing}")
        
        # Don't use cache if forcing new processing
        if not force_new_processing and image_hash in _preprocess_cache:
            # Extra check: don't return the same result twice in a row unless it's truly identical
            if image_hash != _last_image_hash or np.array_equal(image, _preprocess_cache.get('raw_image', None)):
                print(f"Using cached preprocessing result for {image_hash[:8]}")
                _last_image_hash = image_hash
                return _preprocess_cache[image_hash]
            else:
                print("Avoiding duplicate result, reprocessing image")
        
        # Store the raw image for verification
        _preprocess_cache['raw_image'] = image.copy() if isinstance(image, np.ndarray) else None
        
    except Exception as e:
        # If hashing fails, just continue without caching
        print(f"Hashing error: {e}")
        image_hash = None
    
    # Check if image is grayscale or color
    if len(image.shape) == 2:
        # Image is already grayscale
        gray = image
    else:
        # Convert to grayscale
        if image.shape[2] == 4:  # RGBA
            # Create white background
            white_background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
            
            # Extract alpha channel and ensure it's in the right range
            alpha = np.clip(image[:, :, 3:4] / 255.0, 0, 1)
            
            # Blend with white background
            rgb = image[:, :, :3]
            image = (rgb * alpha + white_background * (1 - alpha)).astype(np.uint8)
        
        # Convert to grayscale
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except Exception:
            # Fallback to simple grayscale conversion
            gray = np.mean(image, axis=2).astype(np.uint8)
    
    # Check if the image is mostly empty
    non_white_pixels = np.sum(gray < 240)
    is_mostly_empty = non_white_pixels < 100
    
    if is_mostly_empty:
        print("Warning: Image appears to be mostly empty")
    
    # Apply adaptive thresholding for better results with varying stroke widths
    try:
        # First, invert the image if it appears to be white text on dark background
        if np.mean(gray) < 127:
            gray = 255 - gray
            
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # If mostly empty, try a simple threshold as backup
        if is_mostly_empty:
            _, simple_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            # Choose the one with more detectable content
            if np.sum(simple_thresh) > np.sum(thresh):
                thresh = simple_thresh
                
    except Exception:
        # Fallback to basic thresholding
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Cleanup: remove small noise and fill small holes
    try:
        # Create kernel for morphological operations
        kernel = np.ones((2, 2), np.uint8)
        
        # Remove small noise (opening operation)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Close small gaps (closing operation)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    except Exception:
        # Continue without these enhancements if they fail
        pass
    
    # Convert to RGB for the model
    try:
        rgb_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    except Exception:
        # Fallback to stacking
        rgb_image = np.stack([thresh, thresh, thresh], axis=2)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Add padding around the image to improve recognition
    padded_image = Image.new('RGB', (pil_image.width + 40, pil_image.height + 40), color=(255, 255, 255))
    padded_image.paste(pil_image, (20, 20))
    
    # Cache the result if we have a hash
    if image_hash:
        # Clear one item if cache is full
        if len(_preprocess_cache) >= _max_cache_size:
            # Don't remove 'raw_image' key
            keys = [k for k in _preprocess_cache.keys() if k != 'raw_image']
            if keys:
                _preprocess_cache.pop(keys[0])
        
        _preprocess_cache[image_hash] = padded_image
        _last_image_hash = image_hash  # Update last processed hash
    
    print(f"Preprocessing completed in {time.time() - start_time:.3f}s")
    return padded_image


def clear_preprocessing_cache():
    """Clear all preprocessing caches to force fresh processing"""
    global _preprocess_cache, _last_image_hash
    _preprocess_cache = {}
    _last_image_hash = None
    print("Preprocessing cache cleared")


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