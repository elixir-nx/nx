defmodule EXLA.MLIR.NxLinAlgDoctestTest do
  use EXLA.Case, async: true

  @invalid_type_error_doctests [
    svd: 2,
    pinv: 2
  ]

  @function_clause_error_doctests [
    solve: 2
  ]

  @rounding_error_doctests [
    triangular_solve: 3,
    eigh: 2,
    cholesky: 1,
    least_squares: 3,
    determinant: 1,
    matrix_power: 2,
    lu: 2
  ]

  @excluded_doctests @function_clause_error_doctests ++
                       @rounding_error_doctests ++
                       @invalid_type_error_doctests ++
                       [:moduledoc]
  doctest Nx.LinAlg, except: @excluded_doctests
end
