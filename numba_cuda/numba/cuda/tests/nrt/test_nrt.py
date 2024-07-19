import gc
import numpy as np
import unittest
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin

from numba import cuda

from numba.core import errors, types
from numba.core.extending import overload
from numba.np.arrayobj import (_check_const_str_dtype, is_nonelike,
                               ty_parse_dtype, ty_parse_shape, numpy_empty_nd)

def cuda_empty(shape, dtype):
    pass


@overload(cuda_empty)
def ol_cuda_empty(shape, dtype):
    _check_const_str_dtype("empty", dtype)
    if (dtype is float or
        (isinstance(dtype, types.Function) and dtype.typing_key is float) or
            is_nonelike(dtype)): #default
        nb_dtype = types.double
    else:
        nb_dtype = ty_parse_dtype(dtype)

    ndim = ty_parse_shape(shape)
    if nb_dtype is not None and ndim is not None:
        retty = types.Array(dtype=nb_dtype, ndim=ndim, layout='C')

        def impl(shape, dtype):
            return numpy_empty_nd(shape, dtype, retty)
        return impl
    else:
        msg = f"Cannot parse input types to function np.empty({shape}, {dtype})"
        raise errors.TypingError(msg)


class TestNrtRefCt(EnableNRTStatsMixin, TestCase):

    def setUp(self):
        # Clean up any NRT-backed objects hanging in a dead reference cycle
        gc.collect()
        super(TestNrtRefCt, self).setUp()

    def test_no_return(self):
        """
        Test issue #1291
        """

        @cuda.jit
        def kernel():
            for i in range(10):
                temp = cuda_empty(2, np.int64)
            return None

        init_stats = rtsys.get_allocation_stats()
        kernel[1,1]()
        cur_stats = rtsys.get_allocation_stats()
        self.assertEqual(cur_stats.alloc - init_stats.alloc, n)
        self.assertEqual(cur_stats.free - init_stats.free, n)

if __name__ == '__main__':
    unittest.main()
