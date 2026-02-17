#!/usr/bin/env python
"""
Test if complex division works in kernels vs cuda.compile()
"""

from numba import cuda
import numpy as np

print("Testing complex division in different contexts")
print("=" * 80)

# Test 1: Complex division in a kernel (should work)
print("\n1. Complex division in a kernel:")
try:

    @cuda.jit
    def complex_div_kernel(arr_in, arr_out):
        i = cuda.grid(1)
        if i < arr_out.size:
            arr_out[i] = arr_in[i] / 2.0

    # Test it
    arr_in = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    arr_out = np.zeros_like(arr_in)

    d_in = cuda.to_device(arr_in)
    d_out = cuda.to_device(arr_out)

    complex_div_kernel[1, 2](d_in, d_out)
    result = d_out.copy_to_host()

    print(f"✓ SUCCESS: {arr_in[0]} / 2 = {result[0]}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: Complex division with cuda.compile() - device function
print("\n2. Complex division with cuda.compile() - regular device function:")
try:

    def div_by_2(x):
        return x / 2.0

    from numba.cuda.types import complex128

    sig = complex128(complex128)
    func = cuda.compile(div_by_2, sig, device=True)
    print(f"✓ SUCCESS")
    print(f"  Type: {type(func)}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 3: Simple arithmetic with cuda.compile()
print("\n3. Simple addition with cuda.compile():")
try:

    def simple_add(x):
        return x + (1.0 + 0j)

    from numba.cuda.types import complex128

    sig = complex128(complex128)
    func = cuda.compile(simple_add, sig, device=True)
    print(f"✓ SUCCESS")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 80)
