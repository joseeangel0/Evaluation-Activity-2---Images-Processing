from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# A slightly different setup script structure
ext_modules = [
    Extension(
        name="cython_ops",
        sources=["cython_ops.pyx"],
        include_dirs=[np.get_include()],
        language="c"
    )
]

setup(
    name="Cythonized Image Ops",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)
