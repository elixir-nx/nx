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
    # matrix_power: 2,
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

  describe "cholesky" do
    test "property" do
      key = Nx.Random.key(System.unique_integer())

      for _ <- 1..10, type <- [{:f, 32}, {:c, 64}], reduce: key do
        key ->
          # Generate random L matrix so we can construct
          # a factorizable A matrix:
          shape = {3, 4, 4}

          {a_prime, key} = Nx.Random.normal(key, 0, 1, shape: shape, type: type)

          a_prime = Nx.add(a_prime, Nx.eye(shape))
          b = Nx.dot(Nx.LinAlg.adjoint(a_prime), [-1], [0], a_prime, [-2], [0])

          d = Nx.eye(shape) |> Nx.multiply(0.1)

          a = Nx.add(b, d)

          assert l = Nx.LinAlg.cholesky(a)
          assert_all_close(Nx.dot(l, [2], [0], Nx.LinAlg.adjoint(l), [1], [0]), a, atol: 1.0e-2)
          key
      end
    end
  end

  describe "least_squares" do
    test "properties for linear equations" do
      key = Nx.Random.key(System.unique_integer())

      # Calucate linear equations Ax = y by using least-squares solution
      for {m, n} <- [{2, 2}, {3, 2}, {4, 3}], reduce: key do
        key ->
          # Generate x as temporary solution and A as base matrix
          {a_base, key} = Nx.Random.randint(key, 1, 10, shape: {m, n})
          {x_temp, key} = Nx.Random.randint(key, 1, 10, shape: {n})

          # Generate y as base vector by x and A
          # to prepare an equation that can be solved exactly
          y_base = Nx.dot(a_base, x_temp)

          # Generate y as random noise vector and A as random noise matrix
          noise_eps = 1.0e-2
          {a_noise, key} = Nx.Random.uniform(key, 0, noise_eps, shape: {m, n})
          {y_noise, key} = Nx.Random.uniform(key, 0, noise_eps, shape: {m})

          # Add noise to prepare equations that cannot be solved without approximation,
          # such as the least-squares method
          a = Nx.add(a_base, a_noise)
          y = Nx.add(y_base, y_noise)

          # Calculate least-squares solution to a linear matrix equation Ax = y
          x = Nx.LinAlg.least_squares(a, y)

          # Check linear matrix equation

          assert_all_close(Nx.dot(a, x), y, atol: noise_eps * 10)

          key
      end
    end
  end

  describe "determinant" do
    test "supports batched matrices" do
      two_by_two = Nx.tensor([[[2, 3], [4, 5]], [[6, 3], [4, 8]]])
      assert_equal(Nx.LinAlg.determinant(two_by_two), Nx.tensor([-2.0, 36.0]))

      three_by_three =
        Nx.tensor([
          [[1.0, 2.0, 3.0], [1.0, 5.0, 3.0], [7.0, 6.0, 9.0]],
          [[5.0, 2.0, 3.0], [8.0, 5.0, 4.0], [3.0, 1.0, -9.0]]
        ])

      assert_equal(Nx.LinAlg.determinant(three_by_three), Nx.tensor([-36.0, -98.0]))

      four_by_four =
        Nx.tensor([
          [
            [1.0, 2.0, 3.0, 0.0],
            [1.0, 5.0, 3.0, 0.0],
            [7.0, 6.0, 9.0, 0.0],
            [0.0, -11.0, 2.0, 3.0]
          ],
          [
            [5.0, 2.0, 3.0, 0.0],
            [8.0, 5.0, 4.0, 0.0],
            [3.0, 1.0, -9.0, 0.0],
            [8.0, 2.0, -4.0, 5.0]
          ]
        ])

      assert_all_close(Nx.LinAlg.determinant(four_by_four), Nx.tensor([-108.0, -490]))
    end
  end
end
