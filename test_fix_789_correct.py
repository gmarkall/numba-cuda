#!/usr/bin/env python
"""
Test script to verify the fix for issue #789.
This tests that C ABI entry points can call device functions that use
exception handling (like range() which validates step != 0).

The key is that the ENTRY POINT with C ABI should NOT use exception handling,
but it should be able to CALL device functions that DO use exception handling.
"""

from numba import cuda, int32, complex128
from numba.cuda import compile_ptx

print(
    "Testing issue #789 fix - C ABI entry points calling device functions with exception handling"
)
print("=" * 80)

# Test 1: C ABI entry point (simple, no exceptions) calling device function with range()
print("\nTest 1: C ABI entry point calling device function with range()")
try:

    @cuda.jit(device=True)
    def helper_with_range(n):
        """Device function that uses range() - requires exception handling"""
        total = 0
        for i in range(n):
            total += i
        return total

    def c_abi_entry_point(n):
        """C ABI entry point that just calls the helper"""
        return helper_with_range(n)

    ptx, resty = compile_ptx(c_abi_entry_point, (int32,), device=True, abi="c")
    print("✓ PASSED: C ABI entry point can call device function with range()")
    print(f"  Return type: {resty}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 2: C ABI entry point calling device function with complex division
print(
    "\nTest 2: C ABI entry point calling device function with complex division"
)
try:

    @cuda.jit(device=True)
    def complex_divide_helper(x):
        """Device function that does complex division - requires exception handling"""
        return x / (2.0 + 0j)

    def c_abi_complex_entry(x):
        """C ABI entry point that calls complex division helper"""
        return complex_divide_helper(x)

    ptx, resty = compile_ptx(
        c_abi_complex_entry, (complex128,), device=True, abi="c"
    )
    print(
        "✓ PASSED: C ABI entry point can call device function with complex division"
    )
    print(f"  Return type: {resty}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 3: Control - regular device function with range() (should always work)
print("\nTest 3: Regular device function with range() (control test)")
try:

    def regular_device_func(n):
        total = 0
        for i in range(n):
            total += i
        return total

    ptx, resty = compile_ptx(regular_device_func, (int32,), device=True)
    print("✓ PASSED: Regular device function with range()")
    print(f"  Return type: {resty}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 4: C ABI entry point with NO exception handling (should always work)
print("\nTest 4: C ABI entry point without exception handling (control test)")
try:

    def simple_c_abi_func(x):
        """Simple function with no exception handling"""
        return x + 1

    ptx, resty = compile_ptx(simple_c_abi_func, (int32,), device=True, abi="c")
    print("✓ PASSED: Simple C ABI function without exception handling")
    print(f"  Return type: {resty}")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 80)
print("All tests completed!")
