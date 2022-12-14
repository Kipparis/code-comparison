# Sortednp

[![Pipeline](https://gitlab.sauerburger.com/frank/sortednp/badges/master/pipeline.svg)](https://gitlab.sauerburger.com/frank/sortednp/-/pipelines)
[![Pylint](https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/pylint.svg?job=pylint)](https://gitlab.sauerburger.com/frank/sortednp)
[![C++ lint](https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/cxxlint.svg?job=cpplint)](https://gitlab.sauerburger.com/frank/sortednp)
[![License](https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/license.svg?job=badges)](https://gitlab.sauerburger.com/frank/sortednp/-/blob/master/LICENSE)
[![PyPI](https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/pypi.svg?job=badges)](https://pypi.org/project/sortednp/)

Numpy and Numpy arrays are a really great tool. However, intersecting and
merging multiple sorted numpy arrays is rather less performant. The current numpy
implementation concatenates the two arrays and sorts the combination. If you
want to merge or intersect multiple numpy arrays, there is a much faster way,
by using the property, that the resulting array is sorted.

Sortednp (sorted numpy) operates on sorted numpy arrays to calculate the
intersection or the union of two numpy arrays in an efficient way. The
resulting array is again a sorted numpy array, which can be merged or
intersected with the next array. The intended use case is that sorted numpy
arrays are sorted as the basic data structure and merged or intersected at
request. Typical applications include information retrieval and search engines
in particular.

It is also possible to implement a k-way merging or intersecting algorithm,
which operates on an arbitrary number of arrays at the same time. This package
is intended to deal with arrays with $`10^6`$ or $`10^{10}`$ items. Usually, these
arrays are too large to keep more than two of them in memory at the same
time. This package implements methods to merge and intersect multiple arrays,
which can be loaded on-demand.

## Installation
There are two different methods to install `sortednp`.

### Using `pip` (recommended)

You can install the package directly from PyPI using `pip` (here `pip3`). There are
pre-compiled wheels for `linux` 32- and 64bit.

```bash
$ pip3 install sortednp
```

### Using `setuptools`

Alternatively, you can clone the git repository and run the
setup script.

```bash
$ git clone https://gitlab.sauerburger.com/frank/sortednp.git
$ cd sortednp
$ python3 setup.py install
```
### Numpy Dependency
The installation fails in some cases, because of a build-time dependency on
numpy. Usually, the problem can be solved by manually installing a recent numpy
version via `pip3 install -U numpy`.

## Usage

The package provides two different kinds of methods. The first class is intended
to operate on two arrays. The second class operates on two or more arrays and
calls the first class of methods internally.

### Two-way methods

Two numpy sorted arrays can be merged with the `merge` method, which takes two
numpy arrays and returns the sorted union of the two arrays.

<!-- write merge.py -->
```python
## merge.py
import numpy as np
import sortednp as snp

a = np.array([0, 3, 4, 6, 7])
b = np.array([1, 2, 3, 5, 7, 9])

m = snp.merge(a, b)
print(m)
```

If you run this, you should see the union of both arrays as a sorted numpy
array.
<!-- console_output -->
```python
$ python3 merge.py
[0 1 2 3 3 4 5 6 7 7 9]
```

Two sorted numpy arrays can be intersected with the `intersect` method, which takes two
numpy arrays and returns the sorted intersection of the two arrays.

<!-- write intersect.py -->
```python
## intersect.py
import numpy as np
import sortednp as snp

a = np.array([0, 3, 4, 6, 7])
b = np.array([1, 2, 3, 5, 7, 9])

i = snp.intersect(a, b)
print(i)
```

If you run this, you should see the intersection of both arrays as a sorted numpy
array.
<!-- console_output -->
```python
$ python3 intersect.py
[3 7]
```

Since version 0.4.0, the library provides the `issubset(a, b)` method which
checks if the array `a` is a subset of `b`, and the `isitem(v, a)` method which
checks if `value` is contained in array `a`.

<!-- write set.py -->
```python
## set.py
import numpy as np
import sortednp as snp

a = np.array([2, 4, 5, 10])
b = np.array([1, 2, 3, 4, 5, 6, 10, 11])

print(snp.issubset(a, b))  # a is subset of b
print(snp.issubset(b, a))  # b is not a subset of a

print(snp.isitem(4, a))  # 4 is an item of a
print(snp.isitem(3, a))  # 3 is not an item of a

```

If you execute this example, you get the expected result: `a` is a subset ob
`b`, `4` is a member of `a`.

<!-- console_output -->
```python
$ python3 set.py
True
False
True
False
```


### Returning array indices
The `intersect` method takes an optional argument `indices` which is `False`
by default. If this is set to `True`, the return value consists of the
intersection array and a tuple with the indices of the common values for both
arrays. The index arrays have the length of the output. The indices show the
position in the input from which the value was copied.

<!-- write intersect_indices.py -->
```python
## intersect_indices.py
import numpy as np
import sortednp as snp

a = np.array([2,4,6,8,10])
b = np.array([1,2,3,4])

intersection, indices = snp.intersect(a,b, indices=True)

print(intersection)
print(indices)
```

The above example gives:
<!-- console_output -->
```python
$ python3 intersect_indices.py
[2 4]
(array([0, 1]), array([1, 3]))
```

The first line shows the intersection of the two arrays. The second line
prints a tuple with the indices where the common values appeared in the input
arrays. For example, the value `4` is at position `1` in array `a` and at position
`3` in array `b`. 


Since version 0.3.0, the `merge` has to `indices` argument too. The returned
indices have the length of the inputs. The indices show the position in the
output to which an input value was copied.

<!-- write merge_indices.py -->
```python
## merge_indices.py
import numpy as np
import sortednp as snp

a = np.array([2,4])
b = np.array([3,4,5])

merged, indices = snp.merge(a,b, indices=True)

print(merged)
print(indices)
```

The above example gives:
<!-- console_output -->
```python
$ python3 merge_indices.py
[2 3 4 4 5]
(array([0, 2]), array([1, 3, 4]))
```

The first line shows that the two arrays have been merged. The second line
prints a tuple with the indices. For example, the value `3` from array `b` can
be found at position `1` in the output.

### Duplicate treatment

Since version 0.3.0, sortednp supported multiple different strategies to deal
with duplicated entries.

#### Duplicates during intersecting

There are three different duplicate treatments for the intersect method:

 - `sortednp.DROP`: Ignore any duplicated entries. The output will 
   contain only unique values.

 - `sortednp.KEEP_MIN_N`: If an entry occurs `n` times in one input array and `m`
   times in the other input array, the output will contain the entry `min(n, m)`
   times.

 - `sortednp.KEEP_MAX_N`: If an entry occurs `n` times in one input array and `m`
   times in the other input array, the output will contain the entry `max(n, m)`
   times (assuming the entry occurs at least once in both arrays, i.e.
   `n > 0` and `m > 0`).


The strategy can be selected with the optional `duplicates` argument of
`intersect`. The default is `sortednp.KEEP_MIN_N`. Consider the following example.

<!-- write intersect_duplicates.py -->
```python
## intersect_duplicates.py
import numpy as np
import sortednp as snp

a = np.array([2, 4, 4, 5])    # Twice
b = np.array([3, 4, 4, 4, 5]) # Three times

intersect_drop = snp.intersect(a, b, duplicates=snp.DROP)
print(intersect_drop)  # Contains a single 4

intersect_min = snp.intersect(a, b, duplicates=snp.KEEP_MIN_N)
print(intersect_min)  # Contains 4 twice

intersect_max = snp.intersect(a, b, duplicates=snp.KEEP_MAX_N)
print(intersect_max)  # Contains 4 three times
```

The above example gives:
<!-- console_output -->
```python
$ python3 intersect_duplicates.py
[4 5]
[4 4 5]
[4 4 4 5]
```


#### Duplicates during merging

The `merge` method offers three different duplicates treatment strategies:

 - `sortednp.DROP`: Ignore any duplicated entries. The output will 
   contain only unique values.

 - `sortednp.DROP_IN_INPUT`: Ignores duplicated entries in the input arrays
   separately. This is the same as ensuring that each input array unique values.
   The output contains every value at most twice.

 - `sortednp.KEEP`: Keep all duplicated entries. If an item occurs `n` times in
   one input array and `m` times in the other input array, the output contains
   the item `n + m` times.

The strategy can be selected with the optional `duplicates`.
The default is `sortednp.KEEP`. Consider the following example.

<!-- write merge_duplicates.py -->
```python
## merge_duplicates.py
import numpy as np
import sortednp as snp

a = np.array([2, 4, 4, 5])    # Twice
b = np.array([3, 4, 4, 4, 5]) # Three times

merge_drop = snp.merge(a, b, duplicates=snp.DROP)
print(merge_drop)  # Contains a single 4

merge_dii = snp.merge(a, b, duplicates=snp.DROP_IN_INPUT)
print(merge_dii)  # Contains 4 twice

merge_keep = snp.merge(a, b, duplicates=snp.KEEP)
print(merge_keep)  # Contains 4 five times
```

The above example gives:
<!-- console_output -->
```python
$ python3 merge_duplicates.py
[2 3 4 5]
[2 3 4 4 5 5]
[2 3 4 4 4 4 4 5 5]
```

#### Duplicates during subset checks

The `issubset` method offers two different duplicates treatment strategies:

 - `sortednp.IGNORE`: Ignore any duplications. The method returns True if each
   value in the first array is contained at least once in the second array.
   Duplicated entries in the first array do not change the return value.

 - `sortednp.REPEAT`: For each duplicated item in the first array, require at
   least as many items in the second array. If for one value the first array
   contains more duplicated entries than the second array, the method returns
   False.

The strategy can be selected with the optional `duplicates`.
The default is `sortednp.IGNORE`. Consider the following example.

<!-- write subset_duplicates.py -->
```python
## subset_duplicates.py
import numpy as np
import sortednp as snp

a = np.array([3, 4, 4, 5])    # Twice
b = np.array([3, 4, 4, 4, 5]) # Three times

# Number of occurances ignored
print(snp.issubset(a, b, duplicates=snp.IGNORE))  # is subset
print(snp.issubset(b, a, duplicates=snp.IGNORE))  # is subset

# Number of in subset must be smaller or equal
print(snp.issubset(a, b, duplicates=snp.REPEAT))  # is subset

# three 4s not subset of two 4s
print(snp.issubset(b, a, duplicates=snp.REPEAT))
```

The above example gives:
<!-- console_output -->
```python
$ python3 subset_duplicates.py
True
True
True
False
```

#### Index tracking and duplicates

Tracking indices with the `indices=True` argument is possible while selecting a
non-default duplicate treatment strategy. For merging the indices point to the
position in the output array. If the input has duplicates that were skipped, the
index is simply repeated. For example with `snp.DROP`, if the input is `[9, 9,
9, 9]`, the index array for this input contains four times the position where
`9` is found in the output.

Similarly, with `snp.KEEP_MAX_N` and `intersect`, the index of the last item in
the array with less occurrences is duplicates.

<!-- write duplicates_index.py -->
```python
## duplicates_index.py
import numpy as np
import sortednp as snp

a = np.array([2, 4, 4, 5])    # Twice
b = np.array([3, 4, 4, 4, 5]) # Three times

# Merge
merge_drop, (index_a, index_b) = snp.merge(a, b,
                                           duplicates=snp.DROP,
                                           indices=True)
print(index_b)

# Intersect
intersect_max, (index_a, index_b) = snp.intersect(a, b,
                                                  duplicates=snp.KEEP_MAX_N,
                                                  indices=True)
print(index_a)
```

The above example gives:
<!-- console_output -->
```python
$ python3 duplicates_index.py
[1 2 2 2 3]
[1 2 2 3]
```

For merging, this means that the three `4`s from the input all appear at same position
in the output, namely position `2`.

For the intersect, this means that the second and third occurrence of `4` in the
output, both came from item at position `2` in the input.

### k-way methods
Similarly, the k-way intersect and merge methods take two or more arrays and
perform the merge or intersect operation on its arguments.

<!-- write kway_intersect.py -->
```python
## kway_intersect.py
import numpy as np
import sortednp as snp

a = np.array([0, 3, 4, 6, 7])
b = np.array([0, 3, 5, 7, 9])
c = np.array([1, 2, 3, 5, 7, 9])
d = np.array([2, 3, 6, 7, 8])

i = snp.kway_intersect(a, b, c, d)
print(i)
```

If you run this, you should see the intersection of all four arrays as a sorted numpy
array.
<!-- console_output -->
```python
$ python3 kway_intersect.py
[3 7]
```

The k-way merger `sortednp.kway_merge` works analogously. However, the native
`numpy` implementation is faster compared to the merge provided by this package.
The k-way merger has been added for completeness. The package `heapq` provides
efficient methods to merge multiple arrays simultaneously.

The methods `kway_merge` and `kway_intersect` accept the optional keyword
argument `assume_sorted`. By default, it is set to `True`. If it is set to `False`,
the method calls `sort()` on the input arrays before performing the operation.
The default should be kept if the arrays are already sorted to save the time it
takes to sort the arrays.

Since the arrays might be too large to keep all of them in memory at the same
time, it is possible to pass a `callable` instead of an array to the methods.
The `callable` is expected to return the actual array. It is called immediately
before the array is required. This reduces the memory consumption.

### Algorithms
Intersections are calculated by iterating both arrays. For a given element in
one array, the method needs to search the other and check if the element is
contained. In order to make this more efficient, we can use the fact that the
arrays are sorted. There are three search methods, which can be selected via the
optional keyword argument `algorithm`.

 * `sortednp.SIMPLE_SEARCH`: Search for an element by linearly iterating over the
   array element-by-element.
   [More Information](https://en.wikipedia.org/wiki/Linear_search).
 * `sortednp.BINARY_SEARCH`: Slice the remainder of the array in halves and
   repeat the procedure on the slice which contains the searched element.
   [More Information](https://en.wikipedia.org/wiki/Binary_search_algorithm).
 * `sortednp.GALLOPING_SEARCH`: First, search for an element linearly, doubling
   the step size after each step. If a step goes beyond the search element,
   perform a binary search between the last two positions.
   [More Information](https://en.wikipedia.org/wiki/Exponential_search).

The default is `sortednp.GALLOPING_SEARCH`. The performance of all three
algorithms is compared in the next section. The methods `issubset()` and
`isitem()` also support the algorithm keyword.

## Performance
The performance of the package can be compared with the default implementation
of numpy, the intersect1d` method. The ratio of the execution time between sortednp and numpy is
shown for various different benchmark tests.

The merge or intersect time can be estimated under two different assumptions. If
the arrays, which are merged or intersected, are already sorted, one should not
consider the time it takes to sort the random arrays in the benchmark. On the
other hand, if one considers a scenario in which the arrays are not sorted, one
should take the sorting time into account. The benchmarks here on this page,
assume that the arrays are already sorted. If you would like to benchmark the
package and include the sorting time, have a look at the methods defined in
`ci/benchmark.py`.

The random scattering of the points indicates the uncertainty caused by random
load fluctuations on the benchmark machine (Spikes of serveral orders of
magnitude usualy mean that there was a shortage of memory and large chunks had
to be moved to SWAP.)

### Intersect

The performance of the intersection operation depends on the sparseness of the
two arrays. For example, if the first element of one of the arrays is larger
than all elements in the other array, only the other array has to be searched
(linearly, binarily, or exponentially). Similarly, if the common elements are
far apart in the arrays (sparseness), large chunks of the arrays can be skipped.
The arrays in the benchmark contain random (unique) integers. The sparseness is
defined as the average difference between two consecutive elements in one array.

The first set of tests studies the performance dependence on the size of the
arrays. The second set of tests studies the dependence on the sparseness of the
array for a fixed size of array. Every shows a color-coded comparison of the
performance of intersecting more than two arrays.

<table>
  <tr>
    <th>Test</th>
    <th>Simple Search</th>
    <th>Binary Search</th>
    <th>Galloping Search</th>
  </tr>
  <tr>
    <th>Intersect</th>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_assume_sorted_simple.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_assume_sorted_binary.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_assume_sorted_galloping.png?job=benchmark" /> </td>
  </tr>
  <tr>
    <th>Sparseness</th>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_sparseness_assume_sorted_simple.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_sparseness_assume_sorted_binary.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_sparseness_assume_sorted_galloping.png?job=benchmark" /> </td>
  </tr>
</table>

### Merge
The following chart shows the performance of merging 2 or more arrays as a
function of the array size. It is assumed that the arrays are already sorted.
<img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_merge_assume_sorted.png?job=benchmark" /> 
