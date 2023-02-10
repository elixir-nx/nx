# Changelog

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
