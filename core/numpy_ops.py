# Alternative NumPy implementations for Gaussian, Sobel, and Median filters
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def gaussian_blur_np(img_matrix):
    """
    Applies a 3x3 Gaussian blur filter using compiled NumPy vector operations.
    """
    blur_kernel = np.array([
        [0.0625, 0.1250, 0.0625],
        [0.1250, 0.2500, 0.1250],
        [0.0625, 0.1250, 0.0625]
    ], dtype=np.float32)
    
    pad_img = np.pad(img_matrix, 1, mode='edge')
    output_res = np.zeros_like(img_matrix, dtype=np.float32)
    
    # Overlapping 3x3 array slicing
    for r in range(3):
        for c in range(3):
            output_res += pad_img[r:r+img_matrix.shape[0], c:c+img_matrix.shape[1]] * blur_kernel[r, c]
            
    return output_res.astype(np.uint8)


def sobel_edge_np(img_matrix):
    """
    Applies a 3x3 Sobel filter for edge detection combining X and Y gradients.
    """
    weights_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    weights_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    
    # Padding with constant zero creates a black border for edges context
    pad_img = np.pad(img_matrix, 1, mode='constant', constant_values=0)
    
    gradient_x = np.zeros_like(img_matrix, dtype=np.float32)
    gradient_y = np.zeros_like(img_matrix, dtype=np.float32)
    
    for r in range(3):
        for c in range(3):
            overlay = pad_img[r:r+img_matrix.shape[0], c:c+img_matrix.shape[1]]
            gradient_x += overlay * weights_x[r, c]
            gradient_y += overlay * weights_y[r, c]
            
    # Calculate magnitude correctly and limit to uint8
    grad_magnitude = np.hypot(gradient_x, gradient_y) # hypot is an alternative to sqrt(x^2 + y^2)
    return np.clip(grad_magnitude, 0, 255).astype(np.uint8)


def median_filter_np(img_matrix):
    """
    Applies a 3x3 Median noise reduction filter mapping windows efficiently.
    """
    padded = np.pad(img_matrix, 1, mode='edge')
    blocks = sliding_window_view(padded, (3, 3))
    return np.median(blocks, axis=(2, 3)).astype(np.uint8)
