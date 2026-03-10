# Alternative Python implementations for Gaussian, Sobel, and Median filters
import math

def gaussian_blur_py(img_matrix):
    """
    Applies a 3x3 Gaussian blur filter using pure Python loops.
    """
    rows = len(img_matrix)
    cols = len(img_matrix[0]) if rows > 0 else 0
    
    # 3x3 weights for Gaussian
    filter_weights = [
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]
    
    result_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            val = 0.0
            for m in range(3):
                for n in range(3):
                    val += img_matrix[r - 1 + m][c - 1 + n] * filter_weights[m][n]
            result_img[r][c] = int(val)
            
    # Handle the edges by directly passing original pixels
    for r in range(rows):
        result_img[r][0] = img_matrix[r][0]
        result_img[r][cols - 1] = img_matrix[r][cols - 1]
    for c in range(cols):
        result_img[0][c] = img_matrix[0][c]
        result_img[rows - 1][c] = img_matrix[rows - 1][c]
        
    return result_img


def sobel_edge_py(img_matrix):
    """
    Applies a 3x3 Sobel edge detection filter using pure Python loops.
    """
    rows = len(img_matrix)
    cols = len(img_matrix[0]) if rows > 0 else 0
    
    gx_weights = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    
    gy_weights = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]
    
    result_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            grad_x = 0.0
            grad_y = 0.0
            for m in range(3):
                for n in range(3):
                    pixel = img_matrix[r - 1 + m][c - 1 + n]
                    grad_x += pixel * gx_weights[m][n]
                    grad_y += pixel * gy_weights[m][n]
            
            mag = math.sqrt(grad_x*grad_x + grad_y*grad_y)
            if mag > 255:
                mag = 255
                
            result_img[r][c] = int(mag)
            
    return result_img


def median_filter_py(img_matrix):
    """
    Applies a 3x3 Median filter for noise reduction using pure Python.
    """
    rows = len(img_matrix)
    cols = len(img_matrix[0]) if rows > 0 else 0
    
    result_img = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Gather surrounding 9 pixels
            block = []
            for m in range(3):
                for n in range(3):
                    block.append(img_matrix[r - 1 + m][c - 1 + n])
            
            # Sort the block array to find the median value
            block.sort()
            result_img[r][c] = block[4] 
            
    # Edges copying
    for r in range(rows):
        result_img[r][0] = img_matrix[r][0]
        result_img[r][cols - 1] = img_matrix[r][cols - 1]
    for c in range(cols):
        result_img[0][c] = img_matrix[0][c]
        result_img[rows - 1][c] = img_matrix[rows - 1][c]
        
    return result_img
