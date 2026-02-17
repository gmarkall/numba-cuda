#!/usr/bin/env python
import traceback
from numba import cuda
from numba.types import int32

try:

    def func_with_range(n):
        for i in range(n):
            pass
        return n

    sig = int32(int32)
    func = cuda.compile(func_with_range, sig, device=True, abi="c")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
