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

  describe "eigh" do
    test "properties for matrices with different eigenvalues" do
      # Generate real Hermitian matrices with different eigenvalues
      # from random matrices based on the relation A = Q.Λ.Q^*
      # where Λ is the diagonal matrix of eigenvalues and Q is unitary matrix.

      key = Nx.Random.key(System.unique_integer())

      for type <- [f: 32, c: 64], reduce: key do
        key ->
          # Unitary matrix from a random matrix
          {base, key} = Nx.Random.uniform(key, shape: {2, 3, 3}, type: type)
          {q, _} = Nx.LinAlg.qr(base)

          # Different eigenvalues from random values
          evals_test =
            [100, 10, 1]
            |> Enum.map(fn magnitude ->
              sign =
                if :rand.uniform() - 0.5 > 0 do
                  1
                else
                  -1
                end

              rand = :rand.uniform() * magnitude * 0.1 + magnitude
              rand * sign
            end)
            |> Nx.tensor(type: type)

          evals_test_diag =
            evals_test
            |> Nx.make_diagonal()
            |> Nx.reshape({1, 3, 3})
            |> Nx.tile([2, 1, 1])

          # Hermitian matrix with different eigenvalues
          # using A = A^* = Q^*.Λ.Q.
          a =
            q
            |> Nx.LinAlg.adjoint()
            |> Nx.dot([2], [0], evals_test_diag, [1], [0])
            |> Nx.dot([2], [0], q, [1], [0])

          # Eigenvalues and eigenvectors
          assert {evals, evecs} = Nx.LinAlg.eigh(a, eps: 1.0e-8)

          assert_all_close(evals_test, evals[0], atol: 1.0e-8)
          assert_all_close(evals_test, evals[1], atol: 1.0e-8)

          evals =
            evals
            |> Nx.vectorize(:x)
            |> Nx.make_diagonal()
            |> Nx.devectorize(keep_names: false)

          # Eigenvalue equation
          evecs_evals = Nx.dot(evecs, [2], [0], evals, [1], [0])
          a_evecs = Nx.dot(evecs_evals, [2], [0], Nx.LinAlg.adjoint(evecs), [1], [0])

          assert_all_close(a, a_evecs, atol: 1.0e-8)
          key
      end
    end
  end
end
