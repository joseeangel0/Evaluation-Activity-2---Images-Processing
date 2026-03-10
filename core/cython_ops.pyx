# Alternative Cython implementations optimized for extreme speed mapping arrays
import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport hypot

@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_blur_cy(cnp.ndarray[cnp.uint8_t, ndim=2] img_matrix):
    cdef int rows = img_matrix.shape[0]
    cdef int cols = img_matrix.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] result = np.zeros((rows, cols), dtype=np.uint8)
    
    cdef float kernel[3][3] 
    kernel[0][0] = 0.0625; kernel[0][1] = 0.1250; kernel[0][2] = 0.0625
    kernel[1][0] = 0.1250; kernel[1][1] = 0.2500; kernel[1][2] = 0.1250
    kernel[2][0] = 0.0625; kernel[2][1] = 0.1250; kernel[2][2] = 0.0625

    cdef int r, c, m, n
    cdef float current_val
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            current_val = 0.0
            for m in range(3):
                for n in range(3):
                    current_val += img_matrix[r - 1 + m, c - 1 + n] * kernel[m][n]
            # cast to 8-bit int directly
            result[r, c] = <cnp.uint8_t>current_val
            
    # Edges cloning
    for r in range(rows):
        result[r, 0] = img_matrix[r, 0]
        result[r, cols - 1] = img_matrix[r, cols - 1]
    for c in range(cols):
        result[0, c] = img_matrix[0, c]
        result[rows - 1, c] = img_matrix[rows - 1, c]
        
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def sobel_edge_cy(cnp.ndarray[cnp.uint8_t, ndim=2] img_matrix):
    cdef int rows = img_matrix.shape[0]
    cdef int cols = img_matrix.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] result = np.zeros((rows, cols), dtype=np.uint8)
    
    cdef float x_mask[3][3]
    x_mask[0][0] = -1; x_mask[0][1] = 0; x_mask[0][2] = 1
    x_mask[1][0] = -2; x_mask[1][1] = 0; x_mask[1][2] = 2
    x_mask[2][0] = -1; x_mask[2][1] = 0; x_mask[2][2] = 1
    
    cdef float y_mask[3][3]
    y_mask[0][0] = -1; y_mask[0][1] = -2; y_mask[0][2] = -1
    y_mask[1][0] = 0;  y_mask[1][1] = 0;  y_mask[1][2] = 0
    y_mask[2][0] = 1;  y_mask[2][1] = 2;  y_mask[2][2] = 1

    cdef int r, c, m, n
    cdef float dx, dy, target, total_mag
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            dx = 0.0
            dy = 0.0
            for m in range(3):
                for n in range(3):
                    target = img_matrix[r - 1 + m, c - 1 + n]
                    dx += target * x_mask[m][n]
                    dy += target * y_mask[m][n]
            
            # Use hypot from libc instead of sqrt syntax logic
            total_mag = hypot(dx, dy)
            if total_mag > 255.0:
                total_mag = 255.0
                
            result[r, c] = <cnp.uint8_t>total_mag
            
    return result


# External C sorting algorithm for medians
cdef void bubble_sort(cnp.uint8_t block[9]) nogil:
    cdef int k, l
    cdef cnp.uint8_t tmp
    for k in range(8):
        for l in range(8 - k):
            if block[l] > block[l + 1]:
                tmp = block[l]
                block[l] = block[l + 1]
                block[l + 1] = tmp

@cython.boundscheck(False)
@cython.wraparound(False)
def median_filter_cy(cnp.ndarray[cnp.uint8_t, ndim=2] img_matrix):
    cdef int rows = img_matrix.shape[0]
    cdef int cols = img_matrix.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] result = np.zeros((rows, cols), dtype=np.uint8)
    
    cdef int r, c, m, n, track_idx
    cdef cnp.uint8_t memory_obj[9]
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            track_idx = 0
            for m in range(3):
                for n in range(3):
                    memory_obj[track_idx] = img_matrix[r - 1 + m, c - 1 + n]
                    track_idx += 1
            
            bubble_sort(memory_obj)
            result[r, c] = memory_obj[4]
            
    for r in range(rows):
        result[r, 0] = img_matrix[r, 0]
        result[r, cols - 1] = img_matrix[r, cols - 1]
    for c in range(cols):
        result[0, c] = img_matrix[0, c]
        result[rows - 1, c] = img_matrix[rows - 1, c]
        
    return result
