
from abc import ABCMeta, abstractmethod
import sys
import weakref
import unittest
from unittest import TestCase as TC
import numpy as np

import sortednp as snp

class Base(metaclass=ABCMeta):
    """
    Define general test cases for the issubset method. Sub-classes need to
    overwrite the dtype method.
    """

    def assertListAlmostEqual(self, a, b, *args, **kwds):
        """
        Check that the given lists are almost equal.
        """
        for A, B in zip(a, b):
            self.assertAlmostEqual(A, B, *args, **kwds)

    def test_assertListAlmostEqual_pass(self):
        """
        Check that assertListAlmostEqual raises no exception, if the given
        values are almost equal.
        """
        a = [0, 1, 2 + 1e-9, 10]
        b = [0, 1, 2       , 10]

        self.assertListAlmostEqual(a, b)

    def test_assertListAlmostEqual_fail(self):
        """
        Check that assertListAlmostEqual raises an exception, if the given
        values differ.
        """
        a = [0, 1, 2 + 1e-3, 10]
        b = [0, 1, 2       , 10]

        self.assertRaises(AssertionError, self.assertListAlmostEqual, a, b)
                

    @abstractmethod
    def get_kwds(self):
        """
        Additional keywords passed to the issubset method. By overwriting
        this, test cases can change the search algorithm.
        """
        pass

    @abstractmethod
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass

    def test_too_small(self):
        """Check isitem returns false if value too small."""
        a = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        c = snp.isitem(-1, a, **self.get_kwds())
        self.assertFalse(c)

    def test_too_large(self):
        """Check isitem returns false if value too large."""
        a = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        c = snp.isitem(100, a, **self.get_kwds())
        self.assertFalse(c)

    def test_miss(self):
        """Check isitem returns false if is not contained."""
        a = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        c = snp.isitem(7, a, **self.get_kwds())
        self.assertFalse(c)

    def test_empty(self):
        """Check isitem returns false if the array is empty."""
        a = np.array([], dtype=self.get_dtype())

        c = snp.isitem(7, a, **self.get_kwds())
        self.assertFalse(c)

    def test_hit(self):
        """Check isitem returns true if value is contained."""
        a = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        c = snp.isitem(5, a, **self.get_kwds())
        self.assertTrue(c)


class ITC_Double:
    def get_dtype(self):
        return 'float64'

class ITC_Float:
    def get_dtype(self):
        return 'float32'

class ITC_Int8:
    def get_dtype(self):
        return 'int8'

class ITC_Int16:
    def get_dtype(self):
        return 'int16'

class ITC_Int32:
    def get_dtype(self):
        return 'int32'

class ITC_Int64:
    def get_dtype(self):
        return 'int64'
 
class ITC_UInt8:
    def get_dtype(self):
        return 'uint8'

class ITC_UInt16:
    def get_dtype(self):
        return 'uint16'

class ITC_UInt32:
    def get_dtype(self):
        return 'uint32'

class ITC_UInt64:
    def get_dtype(self):
        return 'uint64'

class ITC_Default:
    def get_kwds(self):
        """
        Use default value of search algorithm.
        """
        return {}

class ITC_Simple:
    def get_kwds(self):
        """
        Use simple search.
        """
        return {"algorithm": snp.SIMPLE_SEARCH}

class ITC_Binary:
    def get_kwds(self):
        """
        Use binary search.
        """
        return {"algorithm": snp.BINARY_SEARCH}

class ITC_Galloping:
    def get_kwds(self):
        """
        Use galloping search.
        """
        return {"algorithm": snp.GALLOPING_SEARCH}

class ITC_Default_Double(ITC_Default, ITC_Double, TC, Base): pass
class ITC_Default_Float(ITC_Default, ITC_Float, TC, Base): pass
class ITC_Default_Int8(ITC_Default, ITC_Int8, TC, Base): pass
class ITC_Default_Int16(ITC_Default, ITC_Int16, TC, Base): pass
class ITC_Default_Int32(ITC_Default, ITC_Int32, TC, Base): pass
class ITC_Default_Int64(ITC_Default, ITC_Int64, TC, Base): pass
class ITC_Default_UInt8(ITC_Default, ITC_UInt8, TC, Base): pass
class ITC_Default_UInt16(ITC_Default, ITC_UInt16, TC, Base): pass
class ITC_Default_UInt32(ITC_Default, ITC_UInt32, TC, Base): pass
class ITC_Default_UInt64(ITC_Default, ITC_UInt64, TC, Base): pass

class ITC_Simple_Double(ITC_Simple, ITC_Double, TC, Base): pass
class ITC_Simple_Float(ITC_Simple, ITC_Float, TC, Base): pass
class ITC_Simple_Int8(ITC_Simple, ITC_Int8, TC, Base): pass
class ITC_Simple_Int16(ITC_Simple, ITC_Int16, TC, Base): pass
class ITC_Simple_Int32(ITC_Simple, ITC_Int32, TC, Base): pass
class ITC_Simple_Int64(ITC_Simple, ITC_Int64, TC, Base): pass
class ITC_Simple_UInt8(ITC_Simple, ITC_UInt8, TC, Base): pass
class ITC_Simple_UInt16(ITC_Simple, ITC_UInt16, TC, Base): pass
class ITC_Simple_UInt32(ITC_Simple, ITC_UInt32, TC, Base): pass
class ITC_Simple_UInt64(ITC_Simple, ITC_UInt64, TC, Base): pass

class ITC_Binary_Double(ITC_Binary, ITC_Double, TC, Base): pass
class ITC_Binary_Float(ITC_Binary, ITC_Float, TC, Base): pass
class ITC_Binary_Int8(ITC_Binary, ITC_Int8, TC, Base): pass
class ITC_Binary_Int16(ITC_Binary, ITC_Int16, TC, Base): pass
class ITC_Binary_Int32(ITC_Binary, ITC_Int32, TC, Base): pass
class ITC_Binary_Int64(ITC_Binary, ITC_Int64, TC, Base): pass
class ITC_Binary_UInt8(ITC_Binary, ITC_UInt8, TC, Base): pass
class ITC_Binary_UInt16(ITC_Binary, ITC_UInt16, TC, Base): pass
class ITC_Binary_UInt32(ITC_Binary, ITC_UInt32, TC, Base): pass
class ITC_Binary_UInt64(ITC_Binary, ITC_UInt64, TC, Base): pass

class ITC_Galloping_Double(ITC_Galloping, ITC_Double, TC, Base): pass
class ITC_Galloping_Float(ITC_Galloping, ITC_Float, TC, Base): pass
class ITC_Galloping_Int8(ITC_Galloping, ITC_Int8, TC, Base): pass
class ITC_Galloping_Int16(ITC_Galloping, ITC_Int16, TC, Base): pass
class ITC_Galloping_Int32(ITC_Galloping, ITC_Int32, TC, Base): pass
class ITC_Galloping_Int64(ITC_Galloping, ITC_Int64, TC, Base): pass
class ITC_Galloping_UInt8(ITC_Galloping, ITC_UInt8, TC, Base): pass
class ITC_Galloping_UInt16(ITC_Galloping, ITC_UInt16, TC, Base): pass
class ITC_Galloping_UInt32(ITC_Galloping, ITC_UInt32, TC, Base): pass
class ITC_Galloping_UInt64(ITC_Galloping, ITC_UInt64, TC, Base): pass
