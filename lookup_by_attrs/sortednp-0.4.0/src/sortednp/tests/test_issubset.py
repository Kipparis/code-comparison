
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

    def test_readme(self):
        """
        Use example from README.
        """
        a = np.array([2, 4, 5, 10], dtype=self.get_dtype())
        b = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        issubset = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(issubset)

        issubset = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(issubset)

    def test_first_lower(self):
        """
        Check that issubset returns False if all entries in one array are
        smaller than the other.
        """
        a = np.array([2, 4, 5], dtype=self.get_dtype())
        b = np.array([10, 11, 12], dtype=self.get_dtype())

        issubset = snp.issubset(a, b, **self.get_kwds())
        self.assertFalse(issubset)

        issubset = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(issubset)

    def test_span_miss(self):
        """
        Check that issubset returns False if all entries are in the range of
        the other array but has some missing values.
        """
        a = np.array([2, 7, 10], dtype=self.get_dtype())
        b = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        issubset = snp.issubset(a, b, **self.get_kwds())
        self.assertFalse(issubset)

        issubset = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(issubset)

    def test_span_hit(self):
        """
        Check that issubset returns True if all entries are in the range of
        the other array and all items are present.
        """
        a = np.array([2, 6, 10], dtype=self.get_dtype())
        b = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        issubset = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(issubset)

        issubset = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(issubset)

    def test_last_share(self):
        """
        Check that issubset returns True if both arrays share the same last
        item and it's a subset.
        """
        a = np.array([2, 6, 11], dtype=self.get_dtype())
        b = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        issubset = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(issubset)

        issubset = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(issubset)

    def test_first_share(self):
        """
        Check that issubset returns True if both arrays share the same first
        item and it's a subset.
        """
        a = np.array([1, 6, 10], dtype=self.get_dtype())
        b = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        issubset = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(issubset)

        issubset = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(issubset)

    def test_both_share(self):
        """
        Check that issubset returns True if both arrays share the first and
        last item and it's a subset.
        """
        a = np.array([1, 6, 11], dtype=self.get_dtype())
        b = np.array([1, 2, 3, 4, 5, 6, 10, 11], dtype=self.get_dtype())

        issubset = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(issubset)

        issubset = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(issubset)

    def test_overlap(self):
        """
        Check that issubset returns False if the arrays have a partial
        overlap.
        """
        a = np.array([2, 3, 4, 5], dtype=self.get_dtype())
        b = np.array([4, 5, 6, 10, 11], dtype=self.get_dtype())

        issubset = snp.issubset(a, b, **self.get_kwds())
        self.assertFalse(issubset)

        issubset = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(issubset)

    def test_empty_input_single(self):
        """
        Check that issubset returns True (False) if the empty are is the first
        (last) argument.
        """
        a = np.array([], dtype=self.get_dtype())
        b = np.array([2, 4, 7, 8, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)


    def test_empty_input_both(self):
        """
        Check that issubset return True if both arrays are empty.
        """
        a = np.array([], dtype=self.get_dtype())
        b = np.array([], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

    def test_identical(self):
        """
        Check that issubset returns True if both arguments are identical.
        """
        a = np.array([3, 4, 6, 8], dtype=self.get_dtype())

        i = snp.issubset(a, a, **self.get_kwds())
        self.assertTrue(i)

    def test_raise_multi_dim(self):
        """
        Check that passing in a multi dimensional array raises an exception.
        """
        a = np.zeros((10, 2), dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(ValueError, snp.issubset, a, b, **self.get_kwds())
        self.assertRaises(ValueError, snp.issubset, b, a, **self.get_kwds())
        self.assertRaises(ValueError, snp.issubset, a, a, **self.get_kwds())
        
    def test_raise_non_array(self):
        """
        Check that passing in a non-numpy-array raises an exception.
        """
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(TypeError, snp.issubset, 3, b, **self.get_kwds())
        self.assertRaises(TypeError, snp.issubset, b, 2, **self.get_kwds())
        self.assertRaises(TypeError, snp.issubset, 3, "a", **self.get_kwds())

    def test_reference_counting(self):
        """
        Check that the reference counting is done correctly.
        """

        # Create inputs
        a = np.arange(10, dtype=self.get_dtype()) * 3
        b = np.arange(10, dtype=self.get_dtype()) * 2 + 5

        # Check ref count for input. Numpy arrays have two references.
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        # Create weak refs for inputs
        weak_a = weakref.ref(a)
        weak_b = weakref.ref(b)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())

        ## Intersect
        i = snp.issubset(a, b, **self.get_kwds())

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())

        # Delete a
        del a
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNone(weak_a())
        self.assertIsNotNone(weak_b())

        # Delete b
        del b
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())

    def test_reference_counting_early_exit_type(self):
        """
        Check that the reference counts of the input array does not change
        even when the method exists premature due to incompatible inputs
        types.
        """
        a = np.array(10)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(TypeError, snp.issubset, a, [1, 2], **self.get_kwds())
        self.assertEqual(sys.getrefcount(a), 2)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(TypeError, snp.issubset, [1, 2], a, **self.get_kwds())
        self.assertEqual(sys.getrefcount(a), 2)

    def test_reference_counting_early_exit_dim(self):
        """
        Check that the reference counts of the input array does not change
        even when the method exists premature due multidimensional input
        arrays.
        """
        a = np.zeros((10, 2))
        b = np.arange(10)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertRaises(ValueError, snp.issubset, a, b, **self.get_kwds())
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(ValueError, snp.issubset, b, a, **self.get_kwds())
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

    def test_dup_ignore_3_3_start(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 3 items at the start.
        """
        a = np.array([2, 2, 2, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 2, 2, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_ignore_3_2_start(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 2 items at the start.
        """
        a = np.array([2, 2, 2, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 2, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_ignore_3_1_start(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 1 items at the start.
        """
        a = np.array([2, 2, 2, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)


    def test_dup_ignore_3_3_mid(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 2 items in the middle.
        """
        a = np.array([2, 4, 4, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 4, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_ignore_3_2_mid(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 2 items in the middle.
        """
        a = np.array([2, 4, 4, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_ignore_3_1_mid(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 1 items in the middle.
        """
        a = np.array([2, 4, 4, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_ignore_3_3_end(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 3 items at the end.
        """
        a = np.array([2, 4, 20, 20, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 20, 20, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_ignore_3_2_end(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 2 items at the end.
        """
        a = np.array([2, 4, 20, 20, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 20, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_ignore_3_1_end(self):
        """
        Check that duplicate strategy IGNORE yields True if on array contains
        3 and the other 1 items at the end.
        """
        a = np.array([2, 4, 8, 20, 20, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 8, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_ignore_3_1_end_fail(self):
        """
        Check that a mismatch after duplicates leads to a fail.
        """
        a = np.array([2, 4, 8, 20, 20, 20, 21], dtype=self.get_dtype())
        b = np.array([2, 4, 8, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertFalse(i)

        i = snp.issubset(b, a, duplicates=snp.IGNORE, **self.get_kwds())
        self.assertTrue(i)


    def test_dup_repeat_3_3_start(self):
        """
        Check that duplicate strategy REPEAT yields True if on array contains
        3 and the other 3 items at the start.
        """
        a = np.array([2, 2, 2, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 2, 2, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_repeat_3_2_start(self):
        """
        Check that duplicate strategy REPEAT yields False if on array contains
        3 and the other 2 items at the start.
        """
        a = np.array([2, 2, 2, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 2, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertFalse(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_repeat_3_1_start(self):
        """
        Check that duplicate strategy REPEAT yields False if on array contains
        3 and the other 1 items at the start.
        """
        a = np.array([2, 2, 2, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertFalse(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)


    def test_dup_repeat_3_3_mid(self):
        """
        Check that duplicate strategy REPEAT yields True if on array contains
        3 and the other 2 items in the middle.
        """
        a = np.array([2, 4, 4, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 4, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_repeat_3_2_mid(self):
        """
        Check that duplicate strategy REPEAT yields False if on array contains
        3 and the other 2 items in the middle.
        """
        a = np.array([2, 4, 4, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertFalse(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_repeat_3_1_mid(self):
        """
        Check that duplicate strategy REPEAT yields False if on array contains
        3 and the other 1 items in the middle.
        """
        a = np.array([2, 4, 4, 4, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertFalse(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_repeat_3_3_end(self):
        """
        Check that duplicate strategy REPEAT yields True if on array contains
        3 and the other 3 items at the end.
        """
        a = np.array([2, 4, 20, 20, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 20, 20, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_repeat_3_2_end(self):
        """
        Check that duplicate strategy REPEAT yields False if on array contains
        3 and the other 2 items at the end.
        """
        a = np.array([2, 4, 20, 20, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 20, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertFalse(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_repeat_3_1_end(self):
        """
        Check that duplicate strategy REPEAT yields False if on array contains
        3 and the other 1 items at the end.
        """
        a = np.array([2, 4, 8, 20, 20, 20], dtype=self.get_dtype())
        b = np.array([2, 4, 8, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertFalse(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

    def test_dup_repeat_3_1_end_fail(self):
        """
        Check that a mismatch after duplicates leads to a fail.
        """
        a = np.array([2, 4, 8, 20, 20, 20, 21], dtype=self.get_dtype())
        b = np.array([2, 4, 8, 20], dtype=self.get_dtype())

        i = snp.issubset(a, b, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertFalse(i)

        i = snp.issubset(b, a, duplicates=snp.REPEAT, **self.get_kwds())
        self.assertTrue(i)

class ITC_Double:
    def get_dtype(self):
        return 'float64'

    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([-1.3e300, -2.3e-200, 3.14e20], dtype=self.get_dtype())
        b = np.array([-1.3e300, -1.1e300, -2.3e-200, 3.14e20], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)

class ITC_Float:
    def get_dtype(self):
        return 'float32'

    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([-1.3e30, -2.3e-20, 3.14e20], dtype=self.get_dtype())
        b = np.array([-1.3e30, -1.1e30, -2.3e-20, 3.14e20], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)


class ITC_Int8:
    def get_dtype(self):
        return 'int8'
    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([-128, 3, 127], dtype=self.get_dtype())
        b = np.array([-128, 3, 2, 127], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)

class ITC_Int16:
    def get_dtype(self):
        return 'int16'
    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([-32768, 3, 32767], dtype=self.get_dtype())
        b = np.array([-32768, 3, 2, 32767], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)

class ITC_Int32:
    def get_dtype(self):
        return 'int32'
    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([-2147483647, 3, 2147483647], dtype=self.get_dtype())
        b = np.array([-2147483647, 3, 2, 2147483647], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)


class ITC_Int64:
    def get_dtype(self):
        return 'int64'
    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([-9223372036854775807, 3, 9223372036854775807], dtype=self.get_dtype())
        b = np.array([-9223372036854775807, 3, 2, 9223372036854775807], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)
 
 
class ITC_UInt8:
    def get_dtype(self):
        return 'uint8'
        a = np.array([0, 3, 255], dtype=self.get_dtype())
        b = np.array([0, 3, 2, 255], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)

class ITC_UInt16:
    def get_dtype(self):
        return 'uint16'
    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([0, 3, 65535], dtype=self.get_dtype())
        b = np.array([0, 3, 2, 65535], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)

class ITC_UInt32:
    def get_dtype(self):
        return 'uint32'
    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([0, 3, 4294967295], dtype=self.get_dtype())
        b = np.array([0, 3, 2, 4294967295], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)

class ITC_UInt64:
    def get_dtype(self):
        return 'uint64'
    def test_type_limites(self):
        """
        Ensure that issubset works with numbers specific to this data type.
        """
        a = np.array([0, 3, 18446744073709551615], dtype=self.get_dtype())
        b = np.array([0, 3, 2, 18446744073709551615], dtype=self.get_dtype())

        i = snp.issubset(a, b, **self.get_kwds())
        self.assertTrue(i)

        i = snp.issubset(b, a, **self.get_kwds())
        self.assertFalse(i)

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

class ITC_TypeError:
    def test_invalid_type(self):
        """
        Ensure that issubset raises an exception, if it is called with an
        unsupported type.
        """
        a = np.array([1, 3, 7], dtype='complex')
        b = np.array([2, 5, 6], dtype='complex')

        self.assertRaises(ValueError, snp.issubset, a, b, **self.get_kwds())

    def test_different_types(self):
        """
        Ensure that issubset raises an exception, if it is called with two
        different types.
        """
        a = np.array([1, 3, 7], dtype='float32')
        b = np.array([2, 5, 6], dtype='float64')

        self.assertRaises(ValueError, snp.issubset, a, b, **self.get_kwds())


class ITC_Default_TypeError(ITC_Default, ITC_TypeError, TC): pass
class ITC_Simple_TypeError(ITC_Simple, ITC_TypeError, TC): pass
class ITC_Binary_TypeError(ITC_Binary, ITC_TypeError, TC): pass
class ITC_Galloping_TypeError(ITC_Galloping, ITC_TypeError, TC): pass

class IntersectNonCContiguousTestCase(unittest.TestCase):
    """
    Check that issubset works correctly with the issues of non-c-contiguous
    arrays. See Issue 22,
    https://gitlab.sauerburger.com/frank/sortednp/issues/22.
    """

    def test_non_cc_second(self):
        """
        Check that using a non-c-contiguous array as the second argument
        returns the correct value.

        Repeat the test 1000 times because memory issues might be flaky.
        """
        for i in range(1000):
            a = np.array([[1, 2, 3], [3, 4, 1], [7, 9, 10]]) 
            nonzero_row, nonzero_col = np.nonzero(a)
            
            x = np.array([0, 1])
            y = nonzero_col[0:3]
            
            try:
                self.assertTrue(snp.issubset(x, y))
            except ValueError:
                pass
                # Test case not suiteable for 32-bit

    def test_non_cc_first(self):
        """
        Check that using a non-c-contiguous array as the first argument
        returns the correct value.

        Repeat the test 1000 times because memory issues might be flaky.
        """
        for i in range(1000):
            a = np.array([[1, 2, 3], [3, 4, 5], [7, 9, 10]]) 
            nonzero_row, nonzero_col = np.nonzero(a)
            
            x = nonzero_col[0:3]
            y = np.array([0, 1, 2, 3])
            
            try:
                self.assertTrue(snp.issubset(x, y))
            except ValueError:
                pass
                # Test case not suiteable for 32-bit

    def test_non_cc_both(self):
        """
        Check that using a non-c-contiguous array as the both arguments
        returns the correct value.

        Repeat the test 1000 times because memory issues might be flaky.
        """
        for i in range(1000):
            a = np.array([[1, 2, 3], [3, 4, 5], [7, 9, 10]]) 
            nonzero_row, nonzero_col = np.nonzero(a)
            
            x = nonzero_col[0:3]
            y = nonzero_col[0:3]
            
            try:
                self.assertTrue(snp.issubset(x, y))
            except ValueError:
                pass
                # Test case not suiteable for 32-bit
