defmodule EXLA.MLIR.NxLinAlgDoctestTest do
  use EXLA.Case, async: true

  setup do
    Nx.default_backend(EXLA.Backend)
    Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)
  end

  @invalid_type_error_doctests [svd: 2, pinv: 2, matrix_rank: 2]
  @function_clause_error_doctests [
    norm: 2,
    lu: 2,
    solve: 2,
    determinant: 1,
    invert: 1,
    matrix_power: 2
  ]
  @rounding_error_doctests [triangular_solve: 3, eigh: 2, cholesky: 1, least_squares: 2]

  @excluded_doctests @function_clause_error_doctests ++
                       @rounding_error_doctests ++
                       @invalid_type_error_doctests ++
                       [:moduledoc]
  doctest Nx.LinAlg, except: @excluded_doctests
end
