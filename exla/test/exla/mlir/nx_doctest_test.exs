defmodule EXLA.MLIR.NxDoctestTest do
  use EXLA.Case, async: true

  setup do
    Nx.default_backend(EXLA.Backend)
    Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)
  end

  @function_clause_error_doctests [
    top_k: 2,
    reduce_max: 2,
    reduce_min: 2,
    window_sum: 3,
    window_mean: 3,
    window_product: 3,
    window_min: 3,
    window_max: 3,
    ifft2: 2,
    ifft: 2,
    fft: 2,
    fft2: 2,
    mode: 2
  ]
  # These tests fail due to the fact that the order of NaNs is
  # not being properly calculated in SortOp
  @nan_order_error_doctests [
    sort: 2,
    argsort: 2
  ]

  # Rounding error doctests are tests that are expected to fail due to rounding errors.
  # This is a permanent category.
  @rounding_error_doctests [
    tan: 1,
    atanh: 1,
    cosh: 1,
    sigmoid: 1,
    expm1: 1,
    erf: 1,
    erfc: 1,
    tanh: 1,
    asinh: 1,
    logsumexp: 2
  ]
  # Sign error doctests are tests that are expected to fail due them returning -0.0
  # instead of 0.0. This is a permanent category.
  @sign_error_doctests [ceil: 1, conjugate: 1]

  @excluded_doctests @function_clause_error_doctests ++
                       @rounding_error_doctests ++
                       @sign_error_doctests ++
                       @nan_order_error_doctests ++
                       [:moduledoc]

  doctest Nx, except: @excluded_doctests
end
