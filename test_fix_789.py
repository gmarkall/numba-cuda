#!/usr/bin/env python
"""
Test script to verify the fix for issue #789.
This tests that C ABI entry points can call device functions that use
exception handling (like range() which validates step != 0).
"""

from numba import cuda, int32, complex128
from numba.cuda import compile_ptx

print(
    "Testing issue #789 fix - C ABI with device functions using exception handling"
)
print("=" * 80)

# Test 1: C ABI function calling a device function with range()
print("\nTest 1: C ABI function with range() loop")
try:

    def device_func_with_range(n):
        total = 0
        for i in range(n):
            total += i
        return total

    ptx, resty = compile_ptx(
        device_func_with_range, (int32,), device=True, abi="c"
    )
    print("✓ PASSED: Successfully compiled C ABI function with range()")
    print(f"  Return type: {resty}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: C ABI function with complex division
print("\nTest 2: C ABI function with complex division")
try:

    def complex_div(x):
        return x / 2

    ptx, resty = compile_ptx(complex_div, (complex128,), device=True, abi="c")
    print(
        "✓ PASSED: Successfully compiled C ABI function with complex division"
    )
    print(f"  Return type: {resty}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 3: C ABI function calling another device function
print("\nTest 3: C ABI function calling device function with range()")
try:

    @cuda.jit(device=True)
    def helper_with_range(n):
        total = 0
        for i in range(n):
            total += i
        return total

    def entry_point(n):
        return helper_with_range(n)

    ptx, resty = compile_ptx(entry_point, (int32,), device=True, abi="c")
    print(
        "✓ PASSED: Successfully compiled C ABI entry point calling device function with range()"
    )
    print(f"  Return type: {resty}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 4: Verify regular device functions still work (should always work)
print("\nTest 4: Regular device function with range() (control test)")
try:

    def regular_device_func(n):
        total = 0
        for i in range(n):
            total += i
        return total

    ptx, resty = compile_ptx(regular_device_func, (int32,), device=True)
    print(
        "✓ PASSED: Successfully compiled regular device function with range()"
    )
    print(f"  Return type: {resty}")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 80)
print("All tests completed!")
