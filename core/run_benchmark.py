import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import Variant B Filters
from python_ops import gaussian_blur_py, sobel_edge_py, median_filter_py
from numpy_ops import gaussian_blur_np, sobel_edge_np, median_filter_np
from cython_ops import gaussian_blur_cy, sobel_edge_cy, median_filter_cy

def load_source_img(source_dir):
    """Loads the first valid image from data/source."""
    os.makedirs(source_dir, exist_ok=True)
    valid_exts = ('.jpg', '.png', '.jpeg', '.webp', '.bmp')
    
    for fname in os.listdir(source_dir):
        if fname.lower().endswith(valid_exts):
            full_path = os.path.join(source_dir, fname)
            print(f"[*] Found image to process: {fname}")
            return cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            
    # Dummy fallback
    print("[!] No images found in data/source/. Generating generic 250x250 placeholder...")
    placeholder_path = os.path.join(source_dir, 'placeholder.jpg')
    generic_img = np.random.randint(0, 256, (250, 250), dtype=np.uint8)
    cv2.imwrite(placeholder_path, generic_img)
    return generic_img

def render_visuals(base_img, processed_imgs, technique_name, dest_dir):
    """Outputs a 1x4 matplotlib comparison grid."""
    plt.figure(figsize=(16, 4))
    headers = ['Source (Original)', 'Gaussian (Blur)', 'Sobel (Edges)', 'Median (Noise)']
    grid_items = [base_img] + processed_imgs
    
    for x in range(4):
        plt.subplot(1, 4, x + 1)
        plt.imshow(grid_items[x], cmap='gray', vmin=0, vmax=255)
        plt.title(headers[x])
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(dest_dir, f"{technique_name}_Output.png"))
    plt.close()

def build_execution_chart(t_py, t_np, t_cy, dest_dir):
    """Plots algorithmic performance into a log-scaled bar graph."""
    filter_names = ['Gaussian', 'Sobel', 'Median']
    indices = np.arange(len(filter_names))
    bar_width = 0.25
    
    fig, axis = plt.subplots(figsize=(10, 6))
    
    bars_1 = axis.bar(indices - bar_width, t_py, bar_width, label='Python (Nested)', color='#FF9999')
    bars_2 = axis.bar(indices, t_np, bar_width, label='NumPy (Vectorized)', color='#99CCFF')
    bars_3 = axis.bar(indices + bar_width, t_cy, bar_width, label='Cython (Compiled)', color='#99FF99')
    
    axis.set_ylabel('Execution Duration (s) [Logarithmic]')
    axis.set_title('Algorithmic Benchmark: Filter Computations')
    axis.set_xticks(indices)
    axis.set_xticklabels(filter_names)
    axis.set_yscale('log')
    axis.legend(loc='upper right')
    
    def annotate_bars(rects):
        for r in rects:
            y_val = r.get_height()
            axis.annotate(f"{y_val:.4f}",
                          xy=(r.get_x() + r.get_width() / 2, y_val),
                          xytext=(0, 4),  
                          textcoords="offset points",
                          ha='center', va='bottom', rotation=90, fontsize=8)
                          
    annotate_bars(bars_1)
    annotate_bars(bars_2)
    annotate_bars(bars_3)
    
    fig.tight_layout()
    plt.savefig(os.path.join(dest_dir, 'Algorithm_Benchmark_Chart.png'))
    plt.close()

def generate_dynamic_docs(shape_data, t_py, t_np, t_cy, doc_path):
    """Writes the markdown documentation dynamically using mapped relative paths."""
    
    chart_link = "../data/processed/Algorithm_Benchmark_Chart.png"
    p_link = "../data/processed/Python_Output.png"
    n_link = "../data/processed/NumPy_Output.png"
    c_link = "../data/processed/Cython_Output.png"

    md_content = f"""# Performance Analysis: Image Processing Evaluation

**Context:** The target image processed measured {shape_data[1]}x{shape_data[0]} pixels in geometry.

## 1. Code Implementation Framework

This repository isolates computational logic distinctively:
- **Python Framework (`core/python_ops.py`)**: Abstracted algorithms exclusively relying on core python structures (Lists, Standard loops).
- **NumPy Framework (`core/numpy_ops.py`)**: Overlapping operations utilizing multi-threaded C routines under the NumPy wrapper.
- **Cython Framework (`core/cython_ops.pyx`)**: Ahead-of-time C compiled algorithms skipping all runtime Python bounds checking overhead.

## 2. Performance Analysis

### 2.1 Execution Timeline

| Filter Type | Python (Native) | NumPy (Vectors) | Cython (C-Compiled) |
|-------------|-----------------|-----------------|---------------------|
| **Gaussian**| {t_py[0]:<15.4f}s| {t_np[0]:<15.4f}s| {t_cy[0]:<19.4f}s|
| **Sobel**   | {t_py[1]:<15.4f}s| {t_np[1]:<15.4f}s| {t_cy[1]:<19.4f}s|
| **Median**  | {t_py[2]:<15.4f}s| {t_np[2]:<15.4f}s| {t_cy[2]:<19.4f}s|

**Logarithmic Benchmark Visualization:**
![Performance Benchmark Chart]({chart_link})

### 2.2 Critical Insights
The transition from pure python interpretation to machine-code translation is staggering. NumPy introduces instant benefits simply by replacing manual iterations with sliding matrices (vectorization). However, Cython takes the lead during nested neighborhood iterations like Convolutions where pre-compiling explicit `uint8_t` memory addressing completely nullifies dynamic runtime bottlenecks native to generic Python data types. 

## 3. Visual Results

The matrices operations output consistent data across the three computation strategies.

- **Gaussian**: Applies spatial uniformity to noise fields.
- **Sobel**: Emphasizes geometry tracking purely calculating X/Y derivatives.
- **Median**: Destroys localized high-variance noise (salt & pepper) effectively without wiping out structural boundaries like blurring does.

### Python Deliverable
![Python Result]({p_link})

### NumPy Deliverable
![NumPy Result]({n_link})

### Cython Deliverable
![Cython Result]({c_link})
"""
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, 'w', encoding='utf-8') as file:
        file.write(md_content)
    print(f"\n[+] Analysis successfully exported to {doc_path}")

def run_benchmarks():
    source_dir = os.path.join('..', 'data', 'source')
    dest_dir = os.path.join('..', 'data', 'processed')
    doc_path = os.path.join('..', 'docs', 'Performance_Analysis.md')
    os.makedirs(dest_dir, exist_ok=True)
    
    img_matrix = load_source_img(source_dir)
    print(f"[*] Base Image loaded. Size: {img_matrix.shape}")
    
    # Save base target
    base_file = os.path.join(dest_dir, 'source_baseline.png')
    cv2.imwrite(base_file, img_matrix)
    print(base_file)
    
    py_list = img_matrix.tolist()
    
    # 1. PURE PYTHON
    print("\n[>] Baching Pure Python Execution...")
    start = time.time()
    res_g_py = gaussian_blur_py(py_list)
    t1_py = time.time() - start
    
    start = time.time()
    res_s_py = sobel_edge_py(py_list)
    t2_py = time.time() - start
    
    start = time.time()
    res_m_py = median_filter_py(py_list)
    t3_py = time.time() - start
    
    # 2. NUMPY
    print("[>] Baching NumPy Execution...")
    start = time.time()
    res_g_np = gaussian_blur_np(img_matrix)
    t1_np = time.time() - start
    
    start = time.time()
    res_s_np = sobel_edge_np(img_matrix)
    t2_np = time.time() - start
    
    start = time.time()
    res_m_np = median_filter_np(img_matrix)
    t3_np = time.time() - start
    
    # 3. CYTHON
    print("[>] Baching Cython Execution...")
    start = time.time()
    res_g_cy = gaussian_blur_cy(img_matrix)
    t1_cy = time.time() - start
    
    start = time.time()
    res_s_cy = sobel_edge_cy(img_matrix)
    t2_cy = time.time() - start
    
    start = time.time()
    res_m_cy = median_filter_cy(img_matrix)
    t3_cy = time.time() - start
    
    # Compile renderers and docs
    print("\n[*] Processing Graph and Visual Plotting...")
    render_visuals(img_matrix, [np.array(res_g_py, dtype=np.uint8), np.array(res_s_py, dtype=np.uint8), np.array(res_m_py, dtype=np.uint8)], 'Python', dest_dir)
    render_visuals(img_matrix, [res_g_np, res_s_np, res_m_np], 'NumPy', dest_dir)
    render_visuals(img_matrix, [res_g_cy, res_s_cy, res_m_cy], 'Cython', dest_dir)
    
    times_py = (t1_py, t2_py, t3_py)
    times_np = (t1_np, t2_np, t3_np)
    times_cy = (t1_cy, t2_cy, t3_cy)
    
    build_execution_chart(times_py, times_np, times_cy, dest_dir)
    generate_dynamic_docs(img_matrix.shape, times_py, times_np, times_cy, doc_path)

if __name__ == '__main__':
    run_benchmarks()
