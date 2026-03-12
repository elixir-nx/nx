defmodule EXLA.NxLinAlgDoctestTest do
  use EXLA.Case, async: true
  import Nx, only: :sigils

  setup do
    Nx.default_backend(EXLA.Backend)
    :ok
  end

  @function_clause_error_doctests [
    solve: 2,
    triangular_solve: 3
  ]

  @rounding_error_doctests [
    svd: 2,
    pinv: 2,
    eigh: 2,
    cholesky: 1,
    least_squares: 3,
    determinant: 1,
    matrix_power: 2,
    lu: 2,
    qr: 2
  ]

  @excluded_doctests @function_clause_error_doctests ++
                       @rounding_error_doctests ++
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

  describe "matrix_power" do
    test "supports complex with positive exponent" do
      a = ~MAT[
        1 1i
        -1i 1
      ]

      n = 5

      assert_all_close(Nx.LinAlg.matrix_power(a, n), Nx.multiply(2 ** (n - 1), a))
    end

    test "supports complex with 0 exponent" do
      a = ~MAT[
        1 1i
        -1i 1
      ]

      assert_all_close(Nx.LinAlg.matrix_power(a, 0), Nx.eye(Nx.shape(a)))
    end

    test "supports complex with negative exponent" do
      a = ~MAT[
        1 -0.5i
        0 0.5
      ]

      result = ~MAT[
        1 15i
        0 16
      ]

      assert_all_close(Nx.LinAlg.matrix_power(a, -4), result)
    end

    test "supports batched matrices" do
      a =
        Nx.tensor([
          [[5, 3], [1, 2]],
          [[9, 0], [4, 7]]
        ])

      result =
        Nx.tensor([
          [[161, 126], [42, 35]],
          [[729, 0], [772, 343]]
        ])

      assert_all_close(Nx.LinAlg.matrix_power(a, 3), result)
    end
  end

  describe "lu" do
    test "property" do
      key = Nx.Random.key(:rand.uniform(1000))

      for _ <- 1..10, type <- [{:f, 32}, {:c, 64}], reduce: key do
        key ->
          # Generate random L and U matrices so we can construct
          # a factorizable A matrix:
          shape = {3, 4, 4}
          lower_selector = Nx.iota(shape, axis: 1) |> Nx.greater_equal(Nx.iota(shape, axis: 2))
          upper_selector = Nx.LinAlg.adjoint(lower_selector)

          {l_prime, key} = Nx.Random.uniform(key, 0, 1, shape: shape, type: type)
          l_prime = Nx.multiply(l_prime, lower_selector)

          {u_prime, key} = Nx.Random.uniform(key, 0, 1, shape: shape, type: type)
          u_prime = Nx.multiply(u_prime, upper_selector)

          a = Nx.dot(l_prime, [2], [0], u_prime, [1], [0])

          assert {p, l, u} = Nx.LinAlg.lu(a)

          actual = p |> Nx.dot([2], [0], l, [1], [0]) |> Nx.dot([2], [0], u, [1], [0])
          assert_all_close(actual, a)
          key
      end
    end
  end

  describe "svd" do
    test "finds the singular values of tall matrices" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

      assert {%{type: output_type} = u, %{type: output_type} = s, %{type: output_type} = v} =
               Nx.LinAlg.svd(t, max_iter: 1000)

      s_matrix = 0 |> Nx.broadcast({4, 3}) |> Nx.put_diagonal(s)

      assert_all_close(t, u |> Nx.dot(s_matrix) |> Nx.dot(v), atol: 1.0e-2, rtol: 1.0e-2)

      assert_all_close(
        u,
        Nx.tensor([
          [0.140, 0.824, 0.521, -0.166],
          [0.343, 0.426, -0.571, 0.611],
          [0.547, 0.0278, -0.422, -0.722],
          [0.750, -0.370, 0.472, 0.277]
        ]),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )

      assert_all_close(Nx.tensor([25.462, 1.291, 0.0]), s, atol: 1.0e-3, rtol: 1.0e-3)

      assert_all_close(
        Nx.tensor([
          [0.504, 0.574, 0.644],
          [-0.760, -0.057, 0.646],
          [0.408, -0.816, 0.408]
        ]),
        v,
        atol: 1.0e-3,
        rtol: 1.0e-3
      )
    end

    test "works with batched matrices" do
      t =
        Nx.tensor([
          [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          [[1.0, 2.0, 3.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]
        ])

      assert {u, s, v} = Nx.LinAlg.svd(t)

      s_matrix =
        Nx.stack([
          Nx.broadcast(0, {3, 3}) |> Nx.put_diagonal(s[0]),
          Nx.broadcast(0, {3, 3}) |> Nx.put_diagonal(s[1])
        ])

      reconstructed_t =
        u
        |> Nx.dot([2], [0], s_matrix, [1], [0])
        |> Nx.dot([2], [0], v, [1], [0])

      assert_all_close(t, reconstructed_t, atol: 1.0e-2, rtol: 1.0e-2)
    end

    test "works with vectorized tensors matrices" do
      t =
        Nx.tensor([
          [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
          [[[1.0, 2.0, 3.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]]
        ])
        |> Nx.vectorize(x: 2, y: 1)

      assert {u, s, v} = Nx.LinAlg.svd(t)

      s_matrix = Nx.put_diagonal(Nx.broadcast(0, {3, 3}), s)

      reconstructed_t =
        u
        |> Nx.dot(s_matrix)
        |> Nx.dot(v)

      assert reconstructed_t.vectorized_axes == [x: 2, y: 1]
      assert reconstructed_t.shape == {3, 3}

      assert_all_close(Nx.devectorize(t), Nx.devectorize(reconstructed_t),
        atol: 1.0e-2,
        rtol: 1.0e-2
      )
    end

    test "works with vectors" do
      t = Nx.tensor([[-2], [1]])

      {u, s, vt} = Nx.LinAlg.svd(t)
      assert_all_close(u |> Nx.dot(Nx.stack([s, Nx.tensor([0])])) |> Nx.dot(vt), t)
    end

    test "works with zero-tensor" do
      for {m, n, k} <- [{3, 3, 3}, {3, 4, 3}, {4, 3, 3}] do
        t = Nx.broadcast(0, {m, n})
        {u, s, vt} = Nx.LinAlg.svd(t)
        assert_all_close(u, Nx.eye({m, m}))
        assert_all_close(s, Nx.broadcast(0, {k}))
        assert_all_close(vt, Nx.eye({n, n}))
      end
    end
  end

  describe "pinv" do
    test "does not raise for 0 singular values" do
      key = Nx.Random.key(System.unique_integer())

      for {m, n} <- [{3, 4}, {3, 3}, {4, 3}], reduce: key do
        key ->
          # generate u and vt as random orthonormal matrices
          {base_u, key} = Nx.Random.uniform(key, 0, 1, shape: {m, m}, type: :f64)
          {u, _} = Nx.LinAlg.qr(base_u)
          {base_vt, key} = Nx.Random.uniform(key, 0, 1, shape: {n, n}, type: :f64)
          {vt, _} = Nx.LinAlg.qr(base_vt)

          # because min(m, n) is always 3, we can use fixed values here
          # the important thing is that there's at least one zero in the
          # diagonal, to ensure that we're guarding against 0 division
          zeros = Nx.broadcast(0, {m, n})
          s = Nx.put_diagonal(zeros, Nx.f64([1, 4, 0]))
          s_inv = Nx.put_diagonal(Nx.transpose(zeros), Nx.f64([1, 0.25, 0]))

          # construct t with the given singular values
          t = u |> Nx.dot(s) |> Nx.dot(vt)
          pinv = Nx.LinAlg.pinv(t)

          # ensure that the returned pinv is close to what we expect
          assert_all_close(pinv, Nx.transpose(vt) |> Nx.dot(s_inv) |> Nx.dot(Nx.transpose(u)),
            atol: 1.0e-2
          )

          key
      end
    end
  end

  describe "triangular_solve" do
    test "works with batched input" do
      a =
        Nx.tensor([
          [
            [-1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
          ],
          [
            [2, 0, 0],
            [4, -2, 0],
            [-5, 1, 3]
          ]
        ])

      b =
        Nx.tensor([
          [1.0, 2.0, 3.0],
          [6, 10, 1]
        ])

      assert_equal(Nx.dot(a, [2], [0], Nx.LinAlg.triangular_solve(a, b), [1], [0]), b)
    end

    test "works with B that has more columns than rows" do
      a =
        Nx.tensor(
          [
            [1, 0],
            [1, 1]
          ],
          type: :f64
        )

      b =
        Nx.tensor(
          [
            [1, 1, 1],
            [2, 2, 2]
          ],
          type: :f64
        )

      x = Nx.LinAlg.triangular_solve(a, b)

      assert_equal(
        x,
        Nx.tensor(
          [
            [1, 1, 1],
            [1, 1, 1]
          ],
          type: :f64
        )
      )
    end

    test "property" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0], [2.0, 0.0, 1.0]])
      assert_equal(Nx.dot(a, Nx.LinAlg.triangular_solve(a, b)), b)

      upper = Nx.transpose(a)
      assert_equal(Nx.dot(upper, Nx.LinAlg.triangular_solve(upper, b, lower: false)), b)

      assert_equal(
        Nx.dot(
          Nx.LinAlg.triangular_solve(upper, b, left_side: false, lower: false),
          upper
        ),
        b
      )

      assert_equal(
        Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose),
        Nx.LinAlg.triangular_solve(upper, b, lower: false)
      )

      assert_equal(
        Nx.dot(
          Nx.transpose(a),
          Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose)
        ),
        b
      )
    end
  end
end
