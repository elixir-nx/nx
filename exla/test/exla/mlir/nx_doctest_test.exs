defmodule EXLA.MLIR.NxDoctestTest do
  use EXLA.Case, async: true

  setup do
    Nx.default_backend(EXLA.Backend)
    Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)
  end

  @function_clause_error_doctests [
    gather: 3,
    median: 2,
    top_k: 2,
    take: 3,
    take_along_axis: 3,
    reduce_max: 2,
    reduce_min: 2,
    window_sum: 3,
    window_mean: 3,
    window_product: 3,
    window_min: 3,
    window_max: 3,
    argmin: 2,
    argmax: 2,
    quotient: 2,
    eye: 2,
    iota: 2,
    ifft2: 2,
    ifft: 2,
    fft: 2,
    fft2: 2
  ]
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
  @argument_count_error_doctests [
    reflect: 2,
    mode: 2
  ]
  @sign_error_doctests [logical_not: 1, ceil: 1, conjugate: 1]
  @incorrect_results_error_doctests [
    is_nan: 1,
    tril: 2,
    tri: 3,
    sort: 2,
    argsort: 2,
    all_close: 3
  ]
  @excluded_doctests @argument_count_error_doctests ++
                       @function_clause_error_doctests ++
                       @rounding_error_doctests ++
                       @sign_error_doctests ++
                       @incorrect_results_error_doctests ++
                       [:moduledoc]

  doctest Nx, except: @excluded_doctests
end
