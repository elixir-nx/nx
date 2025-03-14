# Changelog

## v0.9.2 (2024-11-16)

### Bug fixes

  * [Nx] Fix deprecation warnings on latest Elixir
  * [Nx.LinAlg] Fix `least_squares` implementation
  * [Nx.Random] Fix `Nx.Random.shuffle` repeating a single value in certain cases on GPU

## v0.9.1 (2024-10-08)

### Deprecations

  * [Nx] Deprecate `Nx.Defn.stream`

## v0.9.0 (2024-09-26)

### Enhancements

  * [Nx] Add 8-bit Floating Point numerical type
  * [Nx] Add quantized int types (s2, s4, u2, u4)

### Bug fixes

  * [Nx.LinAlg] Minor range slicing fixes on QR decomposition
  * [Nx] Nx.Defn.Grad now supports more vectorization cases

### Deprecations and incompatibilities

  * [Nx] Default integer type is now `s32`
  * [Nx] Interface breaking changes for `Nx.to_pointer` and `Nx.from_pointer`

## v0.8.0 (2024-08-19)

### Enhancements

  * [Nx] Add `Nx.to_pointer/2` and `Nx.from_pointer/5`
  * [Nx] Introduce `~VEC` sigil for 1d tensors
  * [Nx] Introduce `~MAT` sigil for 2d tensors
  * [Nx] Implement `stack` as a callback for performance
  * [Nx] Make `take` an optional callback
  * [Nx] Make `take_along_axis` an optional callback
  * [Nx.LinAlg] Support `:keep_axes` in eigh

### Bug fixes

  * [Nx] Fix a bug with `gather` when `indices` had more dimensions than the input `tensor`
  * [Nx] Fix min/max value for 16 bit signed type
  * [Nx] Fix argmax/argmin behaviour with NaNs
  * [Nx.Serving] Fix a bug where streaming responses were never closing

### Deprecations and incompatibilities

  * [Nx] Deprecate `~V` in favor of `~VEC`
  * [Nx] Deprecate `~M` in favor of `~MAT`
  * [Nx] Remove `Nx.map/2`

## v0.7.1 (2024-02-27)

  * [Nx.LinAlg] Minor speed up to `Nx.LinAlg.qr/2` default implementation

## v0.7.0 (2024-02-22)

### Enhancements

  * [Nx] Add `Nx.fft2` and `Nx.ifft2`
  * [Nx] Add `Nx.fill/2`
  * [Nx] Implement QR decomposition as optional callback
  * [Nx] Support `:type` option in argmin/argmax
  * [Nx] Default all sorting operations to unstable sorting (pass `stable: true` to change it)
  * [Nx.BinaryBackend] Improve performance of `Nx.concatenate/2`
  * [Nx.Defn] Support a mapping function in `print_value/2`
  * [Nx.Defn] Add `c:Nx.Defn.Compiler.__to_backend__/1` callback
  * [Nx.LinAlg] Add `Nx.least_squares/2`

### Bug fixes

  * [Nx.Constants] Fix min and max finite values for `:bf16`
  * [Nx.Defn] Do not discard arguments on optional grads

### Incompatible changes

  * [Nx] Default to non-stable sorting
  * [Nx] Remove deprecated random_uniform, random_normal, shuffle
  * [Nx.Defn] `Nx.Defn.rewrite_types/2` has been removed

## v0.6.4 (2023-11-13)

### Enhancements

  * [Nx] Allow non-scalar tensors on access

### Bug fixes

  * [Nx] Improve the `:axes` option in `gather`, `indexed_add`, and `indexed_put`
  * [Nx] Fix grad of `gather`, `indexed_add`, and `indexed_put` with axes
  * [Nx.BinaryBackend] Fix sorting of negative infinity
  * [Nx.BinaryBackend] Always sort NaN last
  * [Nx.Serving] Fix `Nx.Batch` padding with multi-device backends

## v0.6.3 (2023-11-09)

### Enhancements

  * [Nx] Allow non-scalars as updates on `indexed_add` and `indexed_put`
  * [Nx] Allow non-scalars as return of `gather`
  * [Nx] Support the `:axes` option in `gather`, `indexed_add`, and `indexed_put`
  * [Nx] Add `Nx.covariance`
  * [Nx] Support `:type` in argsort
  * [Nx] Support `:stable` option in argsort for future compatibility
  * [Nx.Serving] Add `:weight` option for static load balancing

### Bug fixes

  * [Nx] Cast input types on slicing
  * [Nx.Defn] Support vectorized tensors in grad
  * [Nx.Defn] Fix bugs when diffing tensor expressions
  * [Nx.Serving] Handle serving getting stuck on timer messages

## v0.6.2 (2023-09-21)

### Enhancements

  * [Nx.Serving] Add `Nx.Serving.batch_size/2` and perform batch splitting on run
  * [Nx.Serving] Support input streaming

## v0.6.1 (2023-09-12)

### Enhancements

  * [Nx] Add multivariate normal distribution
  * [Nx.Serving] Automatically split exceeding batch sizes

### Bug fixes

  * [Nx] Fix `Nx.pad/2` with different backends
  * [Nx] Fix `Nx.clip/3` with non-finite values
  * [Nx.Serving] Emit batches as they arrive in `Nx.Serving.streaming/2`
  * [Nx.Serving] Ensure batch key is preserved when a batch is split

## v0.6.0 (2023-08-15)

### Enhancements

  * [Nx] Add constant creation helpers such as `u8`, `f32`, etc
  * [Nx] Implement Bluestein's algorithm for `fft` and `ifft` in the binary backend
  * [Nx] Support range with steps when accessing tensors
  * [Nx] Support vectorization via `Nx.vectorize/2`, `Nx.devectorize/2`, `Nx.revectorize/2`, `Nx.reshape_vectors/2`, and `Nx.broadcast_vectors/2`
  * [Nx] Add `Nx.logsumexp/2`
  * [Nx] Add `Nx.split/3`
  * [Nx] Add `Nx.tri/2`, `Nx.triu/2`, `Nx.tril/2`
  * [Nx] Introduce a new serialization format that is more suitable to memory mapping
  * [Nx.Defn] Consider Inspect.Opts limit when pretty printing Nx.Defn expressions
  * [Nx.Serving] Support multiple batch keys in Nx.Serving
  * [Nx.Serving] Support streaming in Nx.Serving

### Bug fixes

  * [Nx] Fix `from_numpy` with 1-byte width arrays
  * [Nx] Fix cases where pretty printing large Nx.Defn expressions would take a long time
  * [Nx] Fix `reduce_min`/`reduce_max` for non-finite values

### Deprecations

  * [Nx.Serving] The post-processing function must now be a two-arity function that receives the `{output, metadata}` as a pair or the stream

### Breaking changes

  * [Nx.Serving] The `nx.serving.postprocessing` telemetry event no longer receives the serving output or serving metadata as event metadata

## v0.5.3 (2023-04-14)

### Bug fixes

  * [Nx.Defn] Fix compilation error when Elixir compiler has column tracking enabled
  * [Nx.LinAlg] Fix cases where determinant could return NaN
  * [Nx.LinAlg] Fix SVD when working with f16 and bf16

## v0.5.2 (2023-03-21)

### Enhancements

  * [Nx.Random] Add `stop_grad` to `Nx.Random` creation functions
  * [Nx.Serving] Reduce references sent through serving

### Bug fixes

  * [Nx] Fix `Nx.mode` with `:axis` option

## v0.5.1 (2023-02-18)

Require Elixir v1.14.

### Enhancements

  * [Nx] Support any container or lazy container in `stack`/`concatenate`
  * [Nx] Add `Nx.top_k/2`
  * [Nx] Add `Nx.to_list/1`
  * [Nx] Improve shape validation in `Nx.concatenate/2`
  * [Nx.Constants] Add `pi`, `e`, and `euler_gamma`
  * [Nx.Random] Raise if a non-unary rank tensor is given as probabilities to `Nx.Random.choice/4`
  * [Nx.Random] Make `samples` optional in `Nx.Random.choice/3`

## v0.5.0 (2023-02-10)

### Enhancements

  * [Nx] Support serialization of containers
  * [Nx] Rename `Nx.power` to `Nx.pow`
  * [Nx] Add `Nx.reflect` and `Nx.linspace`
  * [Nx.Defn] Raise at compile time for invalid defn if/cond usage
  * [Nx.LinAlg] Support `full_matrices?` in SVD
  * [Nx.LinAlg] Add `Nx.LinAlg.matrix_rank`
  * [Nx.Random] Add `Nx.Random.choice` and `Nx.Random.shuffle`
  * [Nx.Serving] Add distributed² serving by distributing over devices (GPUs/CPUs) as well as nodes
  * [Nx.Serving] Add telemetry to `Nx.Serving` callbacks

### Backwards incompatible changes

  * [Nx] `from_numpy` and `from_numpy_archive` have been replaced by `load_numpy!` and `load_numpy_archive!`
  * [Nx.Defn.Evaluator] Do not force GC on evaluator

## v0.4.2 (2023-01-13)

### Enhancements

  * [Nx] Allow tensors to be given on `Nx.tensor/2`
  * [Nx] Add `Nx.with_default_backend/2`
  * [Nx] Add `:axes` option to `Nx.flatten/2`
  * [Nx] Add `:axes` option to `Nx.weighted_mean/2`
  * [Nx.Defn] Warn if `Nx.tensor/2` first-argument is not constant inside defn
  * [Nx.LinAlg] Add `Nx.LinAlg.pinv/1`
  * [Nx.LinAlg] Optimize and handle more cases in `Nx.LinAlg.svd/1`

### Bug fixes

  * [Nx] Respect fortran order in loading from numpy
  * [Nx.Defn] Render containers in compile error type+shape mismatch
  * [Nx.Defn] Restore pdict state after compilation

## v0.4.1 (2022-12-07)

### Enhancements

  * [Nx] Add `Nx.Batch` and `Nx.Serving`
  * [Nx] Implement `Nx.Container` for numbers, complex, and tensors for completeness
  * [Nx] Support batches in `Nx.eye/2`

### Bug fixes

  * [Nx] Keep input tensor names on associative scan
  * [Nx.BinaryBackend] Differentiate between complex and real output in `as_type`
  * [Nx.BinaryBackend] Fix loss of precision in `Nx.complex/2`
  * [Nx.BinaryBackend] Preserve NaNs in `window` and `reduce` operations
  * [Nx.Random] Do not return infinity on `normal/2` for f16

## v0.4.0 (2022-10-25)

### Enhancements

  * [Nx] Add `Nx.rename/2`, `Nx.median/2`, `Nx.weighted_mean/3`, and `Nx.mode/2`
  * [Nx] Implement cumulative operations using associative scan for improved performance
  * [Nx.Constants] Add `min` and `max`
  * [Nx.Defn] Allow lists and functions anywhere as arguments in `defn`, `jit` and `compile`
  * [Nx.Defn] Add `Nx.LazyContainer` that allows a data-structure to lazily define tensors
  * [Nx.Defn] Allow tensors and ranges as generators inside `while`
  * [Nx.Defn] Add `debug_expr/2` and `debug_expr_apply/3`
  * [Nx.Defn.Evaluator] Calculate cache lifetime to reduce memory usage on large numerical programs
  * [Nx.LinAlg] Handle Hermitian matrices in `eigh`
  * [Nx.LinAlg] Support batched operations in `adjoint`, `cholesky`, `determinant`, `eigh`, `invert`, `lu`, `matrix_power`, `solve`, `svd`, and `triangular_solve`
  * [Nx.Random] Support pseudo random number generators algorithms

### Bug fixes

  * [Nx] Perform `window_reduce`/`reduce` operations from infinity and negative infinity
  * [Nx.Defn] Ensure `defnp` emits warnings when unused
  * [Nx.Defn] Warn on unused variables in `while`

### Deprecations

  * [Nx] Deprecate tensor as shape in `Nx.eye/2` and `Nx.iota/2`
  * [Nx] Deprecate `Nx.random_uniform/2` and `Nx.random_normal/2`

## v0.3.0 (2022-08-13)

### Enhancements

  * [Nx] Improve support for non-finite values in `Nx.broadcast/2`, `Nx.all_close/2`, and more
  * [Nx] Add `Nx.is_nan/1` and `Nx.is_infinite/1`
  * [Nx] Support booleans in `Nx.tensor/2`
  * [Nx] Add `Nx.fft/2` and `Nx.ifft/2`
  * [Nx] Rename `Nx.logistic/1` to `Nx.sigmoid/1`
  * [Nx] Add `Nx.put_diagonal/3` and `Nx.indexed_put/3`
  * [Nx] Add `:reverse` to cummulative functions
  * [Nx] Add `Nx.to_batched/3` which returns a stream
  * [Nx] Support batched tensors in `Nx.LinAlg.qr/1`
  * [Nx.Defn] Add `Nx.Defn.compile/3` for precompiling expressions
  * [Nx.Defn] Add `deftransform/2` and `deftransformp/2` for easier to define transforms
  * [Nx.Defn] Add `div/2`
  * [Nx.Defn] Support `case/2`, `raise/1`, and `raise/2`
  * [Nx.Defn] Support booleans in `if`, `cond`, and boolean operators
  * [Nx.Defn] Perform branch elimitation in `if` and `cond` and execute branches lazily
  * [Nx.Defn.Evaluator] Garbage collect after evaluation (it can be disabled by setting the `:garbage_collect` compiler option to false)

### Deprecations

  * [Nx] `Nx.to_batched_list/3` is deprecated in favor of `Nx.to_batched/3`
  * [Nx.Defn] `transform/2` is deprecated in favor of `deftransform/2` and `deftransformp/2`
  * [Nx.Defn] `assert_shape/2` and `assert_shape_pattern/2` are deprecated in favor of `case/2` + `raise/2`
  * [Nx.Defn] `inspect_expr/1` and `inspect_value/1` are deprecated in favor of `print_expr/1` and `print_value/1` respectively

## v0.2.1 (2022-06-04)

### Enhancements

  * [Nx] Improve support for non-finite values in `Nx.tensor/1`
  * [Nx] Use iovec on serialization to avoid copying binaries
  * [Nx.BinaryBackend] Improve for complex numbers in `Nx.tensor/1`
  * [Nx.Defn] Improve for complex numbers inside `defn`

### Bug fixes

  * [Nx] Properly normalize type in `Nx.from_binary/3`
  * [Nx.Defn] Raise on `Nx.Defn.Expr` as JIT argument
  * [Nx.Defn.Evaluator] Handle concatenate arguments on evaluator

## v0.2.0 (2022-04-28)

This version requires Elixir v1.13+.

### Enhancements

  * [Nx] Support atom notation as the type option throughout the API (for example, `:u8`, `:f64`, etc)
  * [Nx] Add support for complex numbers (c64, c128)
  * [Nx] Add `Nx.cumulative_sum/2`, `Nx.cumulative_product/2`, `Nx.cumulative_min/2`, `Nx.cumulative_max/2`
  * [Nx] Add `Nx.conjugate/1`, `Nx.phase/1`, `Nx.real/1`, and `Nx.imag/1`
  * [Nx] Initial support for NaN and Infinity
  * [Nx] Add `:axis` option to `Nx.shuffle/2`
  * [Nx] Add `Nx.axis_index/2`
  * [Nx] Add `Nx.variance/2` to `Nx.standard_deviation/2`
  * [Nx] Rename `Nx.slice_axis/3` to `Nx.slice_along_axis/4`
  * [Nx.Backend] Add support for optional backends
  * [Nx.Constants] Provide a convenient module to host constants
  * [Nx.Defn] Improve error messages throughout the compiler

## v0.1.0 (2022-01-06)

First release.
