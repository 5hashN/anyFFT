"""
anyFFT
Copyright (C) 2026 5hashN
All Rights Reserved.
Demonstration only. No license granted.
"""

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

def get_c2c_dtype_str(real_dtype_str):
    if real_dtype_str == "float64": return "complex128"
    elif real_dtype_str == "float32": return "complex64"
    elif real_dtype_str == "complex128": return "complex128"
    elif real_dtype_str == "complex64": return "complex64"
    else: raise ValueError(f"Unknown precision: {real_dtype_str}")

def get_numpy_types(dtype_str):
    if dtype_str in ["float32", "complex64"]: return np.float32, np.complex64
    elif dtype_str in ["float64", "complex128"]: return np.float64, np.complex128
    else: raise ValueError(f"Unsupported dtype: {dtype_str}")

def get_cupy_types(dtype_str):
    if not HAS_CUPY: raise ImportError("CuPy is not installed.")
    if dtype_str in ["float32", "complex64"]: return cp.float32, cp.complex64
    elif dtype_str in ["float64", "complex128"]: return cp.float64, cp.complex128
    else: raise ValueError(f"Unsupported dtype: {dtype_str}")

def generate_data(shape, dtype, start_indices=None, use_gpu=False):
    xp = cp if use_gpu else np
    if start_indices is None:
        start_indices = [0] * len(shape)

    ranges = [xp.arange(start_indices[i], start_indices[i] + shape[i]) for i in range(len(shape))]
    grids = xp.meshgrid(*ranges, indexing="ij")
    data = xp.zeros(shape, dtype=dtype)
    is_complex = np.dtype(dtype).kind == "c"

    if is_complex:
        data += xp.sin(grids[0]) + 1j * xp.cos(grids[0])
        if len(grids) > 1: data += xp.cos(grids[1]) + 1j * xp.sin(grids[1])
        if len(grids) > 2: data += xp.sin(grids[2]) + 1j * xp.cos(grids[2])
        if len(grids) > 3: data += xp.cos(grids[3]) + 1j * xp.sin(grids[3])
    else:
        data += xp.sin(grids[0])
        if len(grids) > 1: data += xp.cos(grids[1])
        if len(grids) > 2: data += xp.sin(grids[2])
        if len(grids) > 3: data += xp.cos(grids[3])

    return data
