
from abc import ABCMeta, abstractmethod
import os
import unittest
from unittest import TestCase as TC
from parameterized import parameterized
import numpy as np

import sortednp as snp

N = int(os.environ.get("N_RANDOM_TESTS", 1))
tests = [(s, ) for s in range(N)]

def get_random_inputs(seed, dtype):
    seed += sum(ord(_) * 2**i for i, _ in enumerate(dtype))
    np.random.seed(seed)
    
    n_a = int(10**(np.random.random() * 6))
    n_b = int(10**(np.random.random() * 6))

    r_a = int(10**(3 + np.random.random() * 6))
    r_b = int(10**(3 + np.random.random() * 6))

    try:
        r_a = min(np.iinfo(np.typeDict[dtype]).max, r_a)
        r_b = min(np.iinfo(np.typeDict[dtype]).max, r_b)
    except ValueError:
        r_a = min(np.finfo(np.typeDict[dtype]).max, r_a)
        r_b = min(np.finfo(np.typeDict[dtype]).max, r_b)

    a = (np.random.random(n_a) * r_a).astype(dtype=dtype)
    b = (np.random.random(n_b) * r_b).astype(dtype=dtype)

    return np.sort(a), np.sort(b)

class I_Base(metaclass=ABCMeta):
    """Implement random tests for the intersect method."""

    @abstractmethod
    def get_kwds(self):
        """
        Additional keywords passed to the intersect method. By overwriting
        this, test cases can change the search algorithm.
        """
        pass

    @abstractmethod
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass

    @parameterized.expand(tests)
    def test_interect_drop(self, seed):
        """Check that DROP strategy is identical to numpy.intersect1d"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        si = snp.intersect(a, b, duplicates=snp.DROP, **self.get_kwds())

        ni = np.intersect1d(a, b)

        self.assertEqual(list(si), list(ni))
        
    @parameterized.expand(tests)
    def test_interect_indices(self, seed):
        """Check that the indices point to the location in the input"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        i, (i_a, i_b) = snp.intersect(a, b, indices=True, **self.get_kwds())

        self.assertEqual([a[_] for _ in i_a], list(i))
        self.assertEqual([b[_] for _ in i_b], list(i))

    @parameterized.expand(tests)
    def test_interect_indices_drop(self, seed):
        """Check that the indices point to the input with DROP"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        i, (i_a, i_b) = snp.intersect(a, b, indices=True, duplicates=snp.DROP,
                                      **self.get_kwds())

        self.assertEqual([a[_] for _ in i_a], list(i))
        self.assertEqual([b[_] for _ in i_b], list(i))

    @parameterized.expand(tests)
    def test_interect_indices_min(self, seed):
        """Check that the indices point to the input with KEEP_MIN_N"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        i, (i_a, i_b) = snp.intersect(a, b, indices=True,
                                      duplicates=snp.KEEP_MIN_N,
                                      **self.get_kwds())

        self.assertEqual([a[_] for _ in i_a], list(i))
        self.assertEqual([b[_] for _ in i_b], list(i))

    @parameterized.expand(tests)
    def test_interect_indices_max(self, seed):
        """Check that the indices point to the input with KEEP_MAX_N"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        i, (i_a, i_b) = snp.intersect(a, b, indices=True,
                                      duplicates=snp.KEEP_MAX_N,
                                      **self.get_kwds())

        self.assertEqual([a[_] for _ in i_a], list(i))
        self.assertEqual([b[_] for _ in i_b], list(i))

class M_Base(metaclass=ABCMeta):
    """Implement random tests for the merge method."""

    @abstractmethod
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass

    @parameterized.expand(tests)
    def test_merge_drop(self, seed):
        """Check that DROP strategy is identical to concat and unique"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        sm = snp.merge(a, b, duplicates=snp.DROP)

        concat = np.concatenate([a, b])
        nm = np.unique(concat)
        nm = np.sort(nm)

        self.assertEqual(list(sm), list(nm))

    @parameterized.expand(tests)
    def test_merge_keep(self, seed):
        """Check that KEEP strategy is identical to concat and sort"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        sm = snp.merge(a, b, duplicates=snp.KEEP)

        concat = np.concatenate([a, b])
        nm = np.sort(concat)

        self.assertEqual(list(sm), list(nm))

    @parameterized.expand(tests)
    def test_merge_drop_in_input(self, seed):
        """Check that DROP_IN_INPUT strategy is identical to concat and sort"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        sm = snp.merge(a, b, duplicates=snp.DROP_IN_INPUT)

        a = np.unique(a)
        b = np.unique(b)
        concat = np.concatenate([a, b])
        nm = np.sort(concat)

        self.assertEqual(list(sm), list(nm))

    @parameterized.expand(tests)
    def test_merge_drop(self, seed):
        """Check that indices show where inputs are in the output"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        m, (i_a, i_b) = snp.merge(a, b, duplicates=snp.DROP, indices=True)

        self.assertEqual([m[_] for _ in i_a], list(a))
        self.assertEqual([m[_] for _ in i_b], list(b))

    @parameterized.expand(tests)
    def test_merge_drop_in_input(self, seed):
        """Check that indices show where inputs are in the output"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        m, (i_a, i_b) = snp.merge(a, b, duplicates=snp.DROP_IN_INPUT,
                                  indices=True)

        self.assertEqual([m[_] for _ in i_a], list(a))
        self.assertEqual([m[_] for _ in i_b], list(b))

    @parameterized.expand(tests)
    def test_merge_keep(self, seed):
        """Check that indices show where inputs are in the output"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        m, (i_a, i_b) = snp.merge(a, b, duplicates=snp.KEEP,
                                  indices=True)

        self.assertEqual([m[_] for _ in i_a], list(a))
        self.assertEqual([m[_] for _ in i_b], list(b))


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
        return {}

class ITC_Simple:
    def get_kwds(self):
        return {"algorithm": snp.SIMPLE_SEARCH}

class ITC_Binary:
    def get_kwds(self):
        return {"algorithm": snp.BINARY_SEARCH}

class ITC_Galloping:
    def get_kwds(self):
        return {"algorithm": snp.GALLOPING_SEARCH}

class ITC_Default_Double(ITC_Default, ITC_Double, TC, I_Base): pass
class ITC_Default_Float(ITC_Default, ITC_Float, TC, I_Base): pass
class ITC_Default_Int8(ITC_Default, ITC_Int8, TC, I_Base): pass
class ITC_Default_Int16(ITC_Default, ITC_Int16, TC, I_Base): pass
class ITC_Default_Int32(ITC_Default, ITC_Int32, TC, I_Base): pass
class ITC_Default_Int64(ITC_Default, ITC_Int64, TC, I_Base): pass
class ITC_Default_UInt8(ITC_Default, ITC_UInt8, TC, I_Base): pass
class ITC_Default_UInt16(ITC_Default, ITC_UInt16, TC, I_Base): pass
class ITC_Default_UInt32(ITC_Default, ITC_UInt32, TC, I_Base): pass
class ITC_Default_UInt64(ITC_Default, ITC_UInt64, TC, I_Base): pass

class ITC_Simple_Double(ITC_Simple, ITC_Double, TC, I_Base): pass
class ITC_Simple_Float(ITC_Simple, ITC_Float, TC, I_Base): pass
class ITC_Simple_Int8(ITC_Simple, ITC_Int8, TC, I_Base): pass
class ITC_Simple_Int16(ITC_Simple, ITC_Int16, TC, I_Base): pass
class ITC_Simple_Int32(ITC_Simple, ITC_Int32, TC, I_Base): pass
class ITC_Simple_Int64(ITC_Simple, ITC_Int64, TC, I_Base): pass
class ITC_Simple_UInt8(ITC_Simple, ITC_UInt8, TC, I_Base): pass
class ITC_Simple_UInt16(ITC_Simple, ITC_UInt16, TC, I_Base): pass
class ITC_Simple_UInt32(ITC_Simple, ITC_UInt32, TC, I_Base): pass
class ITC_Simple_UInt64(ITC_Simple, ITC_UInt64, TC, I_Base): pass

class ITC_Binary_Double(ITC_Binary, ITC_Double, TC, I_Base): pass
class ITC_Binary_Float(ITC_Binary, ITC_Float, TC, I_Base): pass
class ITC_Binary_Int8(ITC_Binary, ITC_Int8, TC, I_Base): pass
class ITC_Binary_Int16(ITC_Binary, ITC_Int16, TC, I_Base): pass
class ITC_Binary_Int32(ITC_Binary, ITC_Int32, TC, I_Base): pass
class ITC_Binary_Int64(ITC_Binary, ITC_Int64, TC, I_Base): pass
class ITC_Binary_UInt8(ITC_Binary, ITC_UInt8, TC, I_Base): pass
class ITC_Binary_UInt16(ITC_Binary, ITC_UInt16, TC, I_Base): pass
class ITC_Binary_UInt32(ITC_Binary, ITC_UInt32, TC, I_Base): pass
class ITC_Binary_UInt64(ITC_Binary, ITC_UInt64, TC, I_Base): pass

class ITC_Galloping_Double(ITC_Galloping, ITC_Double, TC, I_Base): pass
class ITC_Galloping_Float(ITC_Galloping, ITC_Float, TC, I_Base): pass
class ITC_Galloping_Int8(ITC_Galloping, ITC_Int8, TC, I_Base): pass
class ITC_Galloping_Int16(ITC_Galloping, ITC_Int16, TC, I_Base): pass
class ITC_Galloping_Int32(ITC_Galloping, ITC_Int32, TC, I_Base): pass
class ITC_Galloping_Int64(ITC_Galloping, ITC_Int64, TC, I_Base): pass
class ITC_Galloping_UInt8(ITC_Galloping, ITC_UInt8, TC, I_Base): pass
class ITC_Galloping_UInt16(ITC_Galloping, ITC_UInt16, TC, I_Base): pass
class ITC_Galloping_UInt32(ITC_Galloping, ITC_UInt32, TC, I_Base): pass
class ITC_Galloping_UInt64(ITC_Galloping, ITC_UInt64, TC, I_Base): pass

class MergeTestCase_Double(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'float64'

class MergeTestCase_Float(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'float32'

class MergeTestCase_Int8(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'int8'

class MergeTestCase_Int16(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'int16'

class MergeTestCase_Int32(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'int32'

class MergeTestCase_Int64(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'int64'
 
class MergeTestCase_UInt8(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'uint8'

class MergeTestCase_UInt16(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'uint16'

class MergeTestCase_UInt32(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'uint32'

class MergeTestCase_UInt64(M_Base, unittest.TestCase):
    def get_dtype(self):
        return 'uint64'

class S_Base(metaclass=ABCMeta):
    """Implement random tests for the issubset method."""

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

    @parameterized.expand(tests)
    def test_issubset(self, seed):
        """Check that DROP strategy is identical to numpy.intersect1d"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        issubset = snp.issubset(a, b, **self.get_kwds())

        ni = np.intersect1d(a, b)
        uniq = np.unique(a)

        self.assertEqual(len(uniq) == len(ni), issubset)

class STC_Double:
    def get_dtype(self):
        return 'float64'

class STC_Float:
    def get_dtype(self):
        return 'float32'

class STC_Int8:
    def get_dtype(self):
        return 'int8'

class STC_Int16:
    def get_dtype(self):
        return 'int16'

class STC_Int32:
    def get_dtype(self):
        return 'int32'

class STC_Int64:
    def get_dtype(self):
        return 'int64'
 
class STC_UInt8:
    def get_dtype(self):
        return 'uint8'

class STC_UInt16:
    def get_dtype(self):
        return 'uint16'

class STC_UInt32:
    def get_dtype(self):
        return 'uint32'

class STC_UInt64:
    def get_dtype(self):
        return 'uint64'

class STC_Default:
    def get_kwds(self):
        return {}

class STC_Simple:
    def get_kwds(self):
        return {"algorithm": snp.SIMPLE_SEARCH}

class STC_Binary:
    def get_kwds(self):
        return {"algorithm": snp.BINARY_SEARCH}

class STC_Galloping:
    def get_kwds(self):
        return {"algorithm": snp.GALLOPING_SEARCH}

class STC_Default_Double(STC_Default, STC_Double, TC, S_Base): pass
class STC_Default_Float(STC_Default, STC_Float, TC, S_Base): pass
class STC_Default_Int8(STC_Default, STC_Int8, TC, S_Base): pass
class STC_Default_Int16(STC_Default, STC_Int16, TC, S_Base): pass
class STC_Default_Int32(STC_Default, STC_Int32, TC, S_Base): pass
class STC_Default_Int64(STC_Default, STC_Int64, TC, S_Base): pass
class STC_Default_UInt8(STC_Default, STC_UInt8, TC, S_Base): pass
class STC_Default_UInt16(STC_Default, STC_UInt16, TC, S_Base): pass
class STC_Default_UInt32(STC_Default, STC_UInt32, TC, S_Base): pass
class STC_Default_UInt64(STC_Default, STC_UInt64, TC, S_Base): pass

class STC_Simple_Double(STC_Simple, STC_Double, TC, S_Base): pass
class STC_Simple_Float(STC_Simple, STC_Float, TC, S_Base): pass
class STC_Simple_Int8(STC_Simple, STC_Int8, TC, S_Base): pass
class STC_Simple_Int16(STC_Simple, STC_Int16, TC, S_Base): pass
class STC_Simple_Int32(STC_Simple, STC_Int32, TC, S_Base): pass
class STC_Simple_Int64(STC_Simple, STC_Int64, TC, S_Base): pass
class STC_Simple_UInt8(STC_Simple, STC_UInt8, TC, S_Base): pass
class STC_Simple_UInt16(STC_Simple, STC_UInt16, TC, S_Base): pass
class STC_Simple_UInt32(STC_Simple, STC_UInt32, TC, S_Base): pass
class STC_Simple_UInt64(STC_Simple, STC_UInt64, TC, S_Base): pass

class STC_Binary_Double(STC_Binary, STC_Double, TC, S_Base): pass
class STC_Binary_Float(STC_Binary, STC_Float, TC, S_Base): pass
class STC_Binary_Int8(STC_Binary, STC_Int8, TC, S_Base): pass
class STC_Binary_Int16(STC_Binary, STC_Int16, TC, S_Base): pass
class STC_Binary_Int32(STC_Binary, STC_Int32, TC, S_Base): pass
class STC_Binary_Int64(STC_Binary, STC_Int64, TC, S_Base): pass
class STC_Binary_UInt8(STC_Binary, STC_UInt8, TC, S_Base): pass
class STC_Binary_UInt16(STC_Binary, STC_UInt16, TC, S_Base): pass
class STC_Binary_UInt32(STC_Binary, STC_UInt32, TC, S_Base): pass
class STC_Binary_UInt64(STC_Binary, STC_UInt64, TC, S_Base): pass

class STC_Galloping_Double(STC_Galloping, STC_Double, TC, S_Base): pass
class STC_Galloping_Float(STC_Galloping, STC_Float, TC, S_Base): pass
class STC_Galloping_Int8(STC_Galloping, STC_Int8, TC, S_Base): pass
class STC_Galloping_Int16(STC_Galloping, STC_Int16, TC, S_Base): pass
class STC_Galloping_Int32(STC_Galloping, STC_Int32, TC, S_Base): pass
class STC_Galloping_Int64(STC_Galloping, STC_Int64, TC, S_Base): pass
class STC_Galloping_UInt8(STC_Galloping, STC_UInt8, TC, S_Base): pass
class STC_Galloping_UInt16(STC_Galloping, STC_UInt16, TC, S_Base): pass
class STC_Galloping_UInt32(STC_Galloping, STC_UInt32, TC, S_Base): pass
class STC_Galloping_UInt64(STC_Galloping, STC_UInt64, TC, S_Base): pass

class C_Base(metaclass=ABCMeta):
    """Implement random tests for the isitem method."""

    @abstractmethod
    def get_kwds(self):
        """
        Additional keywords passed to the isitem method. By overwriting
        this, test cases can change the search algorithm.
        """
        pass

    @abstractmethod
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass

    @parameterized.expand(tests)
    def test_isitem(self, seed):
        """isitem returns true if the element is contained"""
        a, b = get_random_inputs(seed, dtype=self.get_dtype())
        value = np.random.choice(a, 1)[0]

        isitem = snp.isitem(value, b, **self.get_kwds())
        self.assertEqual(value in b, isitem)

class CTC_Double:
    def get_dtype(self):
        return 'float64'

class CTC_Float:
    def get_dtype(self):
        return 'float32'

class CTC_Int8:
    def get_dtype(self):
        return 'int8'

class CTC_Int16:
    def get_dtype(self):
        return 'int16'

class CTC_Int32:
    def get_dtype(self):
        return 'int32'

class CTC_Int64:
    def get_dtype(self):
        return 'int64'

class CTC_UInt8:
    def get_dtype(self):
        return 'uint8'

class CTC_UInt16:
    def get_dtype(self):
        return 'uint16'

class CTC_UInt32:
    def get_dtype(self):
        return 'uint32'

class CTC_UInt64:
    def get_dtype(self):
        return 'uint64'

class CTC_Default:
    def get_kwds(self):
        return {}

class CTC_Simple:
    def get_kwds(self):
        return {"algorithm": snp.SIMPLE_SEARCH}

class CTC_Binary:
    def get_kwds(self):
        return {"algorithm": snp.BINARY_SEARCH}

class CTC_Galloping:
    def get_kwds(self):
        return {"algorithm": snp.GALLOPING_SEARCH}

class CTC_Default_Double(CTC_Default, CTC_Double, TC, C_Base): pass
class CTC_Default_Float(CTC_Default, CTC_Float, TC, C_Base): pass
class CTC_Default_Int8(CTC_Default, CTC_Int8, TC, C_Base): pass
class CTC_Default_Int16(CTC_Default, CTC_Int16, TC, C_Base): pass
class CTC_Default_Int32(CTC_Default, CTC_Int32, TC, C_Base): pass
class CTC_Default_Int64(CTC_Default, CTC_Int64, TC, C_Base): pass
class CTC_Default_UInt8(CTC_Default, CTC_UInt8, TC, C_Base): pass
class CTC_Default_UInt16(CTC_Default, CTC_UInt16, TC, C_Base): pass
class CTC_Default_UInt32(CTC_Default, CTC_UInt32, TC, C_Base): pass
class CTC_Default_UInt64(CTC_Default, CTC_UInt64, TC, C_Base): pass

class CTC_Simple_Double(CTC_Simple, CTC_Double, TC, C_Base): pass
class CTC_Simple_Float(CTC_Simple, CTC_Float, TC, C_Base): pass
class CTC_Simple_Int8(CTC_Simple, CTC_Int8, TC, C_Base): pass
class CTC_Simple_Int16(CTC_Simple, CTC_Int16, TC, C_Base): pass
class CTC_Simple_Int32(CTC_Simple, CTC_Int32, TC, C_Base): pass
class CTC_Simple_Int64(CTC_Simple, CTC_Int64, TC, C_Base): pass
class CTC_Simple_UInt8(CTC_Simple, CTC_UInt8, TC, C_Base): pass
class CTC_Simple_UInt16(CTC_Simple, CTC_UInt16, TC, C_Base): pass
class CTC_Simple_UInt32(CTC_Simple, CTC_UInt32, TC, C_Base): pass
class CTC_Simple_UInt64(CTC_Simple, CTC_UInt64, TC, C_Base): pass

class CTC_Binary_Double(CTC_Binary, CTC_Double, TC, C_Base): pass
class CTC_Binary_Float(CTC_Binary, CTC_Float, TC, C_Base): pass
class CTC_Binary_Int8(CTC_Binary, CTC_Int8, TC, C_Base): pass
class CTC_Binary_Int16(CTC_Binary, CTC_Int16, TC, C_Base): pass
class CTC_Binary_Int32(CTC_Binary, CTC_Int32, TC, C_Base): pass
class CTC_Binary_Int64(CTC_Binary, CTC_Int64, TC, C_Base): pass
class CTC_Binary_UInt8(CTC_Binary, CTC_UInt8, TC, C_Base): pass
class CTC_Binary_UInt16(CTC_Binary, CTC_UInt16, TC, C_Base): pass
class CTC_Binary_UInt32(CTC_Binary, CTC_UInt32, TC, C_Base): pass
class CTC_Binary_UInt64(CTC_Binary, CTC_UInt64, TC, C_Base): pass

class CTC_Galloping_Double(CTC_Galloping, CTC_Double, TC, C_Base): pass
class CTC_Galloping_Float(CTC_Galloping, CTC_Float, TC, C_Base): pass
class CTC_Galloping_Int8(CTC_Galloping, CTC_Int8, TC, C_Base): pass
class CTC_Galloping_Int16(CTC_Galloping, CTC_Int16, TC, C_Base): pass
class CTC_Galloping_Int32(CTC_Galloping, CTC_Int32, TC, C_Base): pass
class CTC_Galloping_Int64(CTC_Galloping, CTC_Int64, TC, C_Base): pass
class CTC_Galloping_UInt8(CTC_Galloping, CTC_UInt8, TC, C_Base): pass
class CTC_Galloping_UInt16(CTC_Galloping, CTC_UInt16, TC, C_Base): pass
class CTC_Galloping_UInt32(CTC_Galloping, CTC_UInt32, TC, C_Base): pass
class CTC_Galloping_UInt64(CTC_Galloping, CTC_UInt64, TC, C_Base): pass
