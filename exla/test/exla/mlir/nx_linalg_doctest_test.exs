defmodule EXLA.MLIR.NxLinAlgDoctestTest do
  use EXLA.Case, async: true

  setup do
    Nx.default_backend(EXLA.Backend)
    Nx.Defn.default_options(compiler: EXLA, compiler_mode: :mlir)
  end

  @invalid_type_error_doctests [svd: 2, pinv: 2, matrix_rank: 2]
  @function_clause_error_doctests [
    norm: 2,
    eigh: 2,
    cholesky: 1,
    lu: 2,
    solve: 2,
    determinant: 1,
    qr: 2,
    invert: 1,
    matrix_power: 2
  ]
  @rounding_error_doctests [triangular_solve: 3]
  # @argument_count_error_doctests [
  #   stack: 2,
  #   reflect: 2,
  #   concatenate: 2,
  #   to_batched: 3,
  #   cumulative_sum: 2,
  #   cumulative_product: 2,
  #   cumulative_min: 2,
  #   cumulative_max: 2,
  #   indexed_add: 4,
  #   indexed_put: 4,
  #   mode: 2,
  #   slice: 4
  # ]
  # @sign_error_doctests [logical_not: 1, ceil: 1, conjugate: 1]
  # @incorrect_results_error_doctests [
  #   equal: 2,
  #   greater: 2,
  #   less_equal: 2,
  #   greater_equal: 2,
  #   less: 2,
  #   is_nan: 1,
  #   tril: 2,
  #   tri: 3,
  #   sort: 2,
  #   is_infinity: 1,
  #   argsort: 2
  # ]
  # @excluded_doctests @argument_count_error_doctests ++
  #                      @function_clause_error_doctests ++
  #                      @rounding_error_doctests ++
  #                      @invalid_type_error_doctests ++
  #                      @sign_error_doctests ++
  #                      @incorrect_results_error_doctests ++
  #                      [:moduledoc]

  @excluded_doctests @function_clause_error_doctests ++
                       @rounding_error_doctests ++
                       @invalid_type_error_doctests ++
                       [:moduledoc]
  doctest Nx.LinAlg, except: @excluded_doctests
end
