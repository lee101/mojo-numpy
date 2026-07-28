# mojo-numpy

`mojo-numpy` is a compute-oriented subset of [NumPy](https://numpy.org/)
implemented in Mojo and callable from Python. It keeps NumPy's familiar
function names and signatures for the covered operations while moving the
numeric loops into one compiled shared library.

The Python package is named `mojonumpy`, so it can be installed alongside real
NumPy:

```python
import mojonumpy as np

x = np.arange(12, dtype=np.float64).reshape(3, 4)
print(np.mean(np.square(x), axis=1))

a = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
print(np.linalg.solve(a, b))
```

This prints:

```text
[  3.5  31.5  91.5]
[2. 3.]
```

## Covered subset

| area | implemented API |
| --- | --- |
| Elementwise ufuncs | `add`, `subtract`, `multiply`, `divide`, `maximum`, `minimum`, `negative`, `absolute`, `square`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `clip` |
| Reductions | `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `argmin`, `argmax`; arbitrary axes, tuples of axes, `keepdims`, `initial`, and `out` where applicable |
| Sorting | `sort`, `argsort`; any axis or flattened, stable ordering, NumPy-compatible NaN placement |
| Linear algebra | `dot` and broadcasting `matmul` for vectors/matrices, `norm`, `solve`, `inv`, `det`, `cholesky` |
| Random | `default_rng`, `Generator.random`, `uniform`, `normal`, `standard_normal`, `integers`, plus the common module-level functions |
| Array plumbing | Common creation, reshape, stacking, comparison, dtype, and constant names are forwarded to NumPy |

The accelerated numeric contract is C-contiguous `float64`, with `int64` used
for indices and random integers. Non-contiguous real inputs are converted once
before entering Mojo. Integer inputs are accepted only when every value is
exactly representable as `float64`; complex values and floating types wider
than `float64` are rejected instead of silently narrowed. Basic NumPy
broadcasting is supported for elementwise operations and matrix batches.

This is not a replacement for all of NumPy. Integer and complex ufunc loops,
masked `where` execution, generalized `einsum`, FFTs, eigen/SVD routines,
structured sorting, polynomial APIs, file I/O, and the rest of NumPy's broad
surface are not implemented. `solve`, `inv`, `det`, and `cholesky` currently
cover one matrix rather than stacked matrix batches. Random streams are
deterministic for a given seed but intentionally do not reproduce NumPy's
PCG64 bitstream.

## Install

The repository carries its own pinned Mojo nightly:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-numpy.so`. Set `PYTHONPATH=python` when
using the package outside a Pixi task.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic. These are best-of-five wall-clock times from the same
process. The benchmark checks numerical correctness before timing.

| case | mojo-numpy | NumPy | result |
| --- | ---: | ---: | ---: |
| add (5M) | 39.53 ms | 39.48 ms | 1.00x slower |
| tanh (2M) | 10.25 ms | 50.53 ms | 4.93x faster |
| sum (5M) | 6.35 ms | 5.44 ms | 1.17x slower |
| sort (1M) | 107.49 ms | 22.98 ms | 4.68x slower |
| matmul (256x256) | 5.75 ms | 0.21 ms | 26.92x slower |
| solve (256x256) | 13.11 ms | 571.51 ms | 43.58x faster |
| standard_normal (2M) | 16.61 ms | 39.34 ms | 2.37x faster |

The result remains mixed. Thresholded parallel execution puts `tanh` and normal
generation ahead, while the multi-accumulator SIMD sum is at parity. Parallel
chunk sorting and the register-accumulating matrix microkernel narrow the two
largest gaps, but NumPy's tuned sort and BLAS matrix multiplication remain well
ahead. The large `solve` win is specific to the NumPy/BLAS build on this
measured machine; it should not be assumed on a system linked to a faster
LAPACK.

Run the locked benchmark to reproduce the table:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-numpy.so`.

The `python/mojonumpy` layer owns all arrays and scratch space. It normalizes
shapes and strides, then makes one `ctypes` call per bulk operation. Buffers
cross the C ABI as integer addresses and are reconstructed in Mojo as
`UnsafePointer[..., AnyOrigin[mut=True]]`; this avoids parametric exported
functions. Arrays remain row-major, and Mojo never retains or frees
Python-owned memory.

Axis reductions and sorting move the selected axis or axes to a contiguous
trailing dimension, then process the data as a batch of rows. Reductions use
architecture-selected SIMD widths and multiple accumulators. Large default
sorts use parallel in-place quicksort chunks followed by parallel merges;
explicitly stable sorting retains stable index ordering. Matrix multiplication
transposes the right operand once and accumulates four SIMD dot products in
registers. Linear solves use LU decomposition with partial pivoting. Normal
random values use a 64-bit LCG with Box-Muller and deterministic jump-ahead
states for independent parallel chunks.

## License

MIT
