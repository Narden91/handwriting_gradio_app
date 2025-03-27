import pytest
from PIL import Image
import numpy as np
from app.utils.helpers import preprocess_image, postprocess_result

def test_preprocess_image():
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='white')
    processed = preprocess_image(test_image)
    assert processed is not None

def test_postprocess_result():
    test_result = "test"
    processed = postprocess_result(test_result)
    assert processed == test_result
