#!/usr/bin/env python
"""
Debug script to examine the LLVM IR and understand the calling convention mismatch.
"""

import numba.cuda
from numba.cuda.types import complex128

print("Debugging complex division with C ABI")
print("=" * 80)

# First, let's see what works - a regular (non-C ABI) complex division
print("\n1. Regular complex division (should work):")
try:

    def div_by_2_regular(x):
        return x / 2

    sig = complex128(complex128)
    func = numba.cuda.compile(div_by_2_regular, sig, device=True)
    print(f"✓ SUCCESS")
    # Print some of the PTX to see the calling convention
    ptx = func[0]
    # Look for the complex_div function signature in the PTX
    for line in ptx.split("\n"):
        if "complex_div" in line.lower() or "func" in line:
            print(f"  {line}")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 80)
print("\n2. C ABI complex division (currently failing):")
try:

    def div_by_2_cabi(x):
        return x / 2

    sig = complex128(complex128)
    func = numba.cuda.compile(div_by_2_cabi, sig, device=True, abi="c")
    print(f"✓ SUCCESS")
    ptx = func[0]
    # Look for the complex_div function signature in the PTX
    for line in ptx.split("\n"):
        if "complex_div" in line.lower() or "func" in line:
            print(f"  {line}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("\n3. Let's also try to see the LLVM IR before NVVM:")
try:

    def div_by_2_debug(x):
        return x / 2

    sig = complex128(complex128)

    # Try to compile and capture intermediate representations
    from numba import cuda
    from numba.cuda.compiler import compile_cuda
    from numba.core import targetconfig

    cres = compile_cuda(
        div_by_2_debug,
        None,
        sig.args,
        sig.return_type,
        device=True,
        abi="c",
        debug=True,
    )
    print("Compilation succeeded!")
    print(f"Library: {cres.library}")

    # Try to get the LLVM IR
    try:
        llvm_ir = str(cres.library.get_llvm_str())
        print("\nLLVM IR (first 2000 chars):")
        print(llvm_ir[:2000])
    except:
        print("Could not retrieve LLVM IR")

except Exception as e:
    print(f"Debug compilation failed: {e}")
    import traceback

    traceback.print_exc()
