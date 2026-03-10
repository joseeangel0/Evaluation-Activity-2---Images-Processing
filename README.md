# Alternative Image Processing Evaluation

This repository houses the computational benchmarking of three common image processing algorithms under native Python loops vs Vectorized wrappers (NumPy) vs Pre-compiled C instructions (Cython).

## Overview

1.  **Gaussian Filter**
2.  **Sobel Filter**
3.  **Median Filter**

### Project Layout

- `/core/`: Script definitions and compilation triggers.
- `/data/source/`: Base images evaluated.
- `/data/processed/`: Evaluation outputs (Images and graphs).
- `/docs/`: Explicit logic documentation and benchmark report data.

## Getting Started

1. Set up the local environment packages defined:

```bash
pip install -r reqs.txt
```

2. Generate the Cython build binaries (Required for execution):

```bash
cd core
python compile_ext.py build_ext --inplace
```

3. Insert your target image into `data/source/` (otherwise a random placeholder will be used), and execute the pipeline:

```bash
python run_benchmark.py
```

Check the `./docs/` folder afterwards for the generated PDF-ready analysis.
