defmodule EXLA.MLIR.NxDoctestTest do
  use EXLA.Case, async: true

  setup do
    Nx.default_backend(EXLA.Backend)
    Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)
  end

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

  @excluded_doctests @rounding_error_doctests ++
                       @sign_error_doctests ++
                       [:moduledoc]

  doctest Nx, except: @excluded_doctests
end
