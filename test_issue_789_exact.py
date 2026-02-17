#!/usr/bin/env python
"""
Test the exact reproducers from issue #789.
"""

import numba.cuda
from numba.cuda.types import complex128

print("Testing exact issue #789 reproducers")
print("=" * 80)

# Reproducer from issue #789: Complex division with C ABI
print("\nTest 1: Complex division with C ABI (exact reproducer from #789)")
try:

    def div_by_2(x):
        return x / 2

    sig = complex128(complex128)
    func = numba.cuda.compile(div_by_2, sig, device=True, abi="c")
    print(f"✓ PASSED: Successfully compiled complex division with C ABI")
    print(f"  Function: {func}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Additional test: Range loop with C ABI
print("\nTest 2: Range loop with C ABI")
try:

    def func_with_range(n):
        for i in range(n):
            pass
        return n

    from numba.types import int32

    sig = int32(int32)
    func = numba.cuda.compile(func_with_range, sig, device=True, abi="c")
    print(f"✓ PASSED: Successfully compiled range loop with C ABI")
    print(f"  Function: {func}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Control test: Simple C ABI function
print("\nTest 3: Simple C ABI function (control)")
try:

    def simple_add(x):
        return x + 1

    from numba.types import int32

    sig = int32(int32)
    func = numba.cuda.compile(simple_add, sig, device=True, abi="c")
    print(f"✓ PASSED: Successfully compiled simple C ABI function")
    print(f"  Function: {func}")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 80)
print("All tests completed!")
