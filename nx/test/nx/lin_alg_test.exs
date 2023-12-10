defmodule Nx.LinAlgTest do
  use ExUnit.Case, async: true

  import Nx.Helpers
  import Nx, only: :sigils

  doctest Nx.LinAlg

  @types [{:f, 32}, {:c, 64}]

  describe "triangular_solve" do
    test "works with batched input" do
      a = Nx.tensor([[[-1, 0, 0], [1, 1, 0], [1, 1, 1]], [[2, 0, 0], [4, -2, 0], [-5, 1, 3]]])
      b = Nx.tensor([[1.0, 2.0, 3.0], [6, 10, 1]])

      assert Nx.dot(a, [2], [0], Nx.LinAlg.triangular_solve(a, b), [1], [0]) == b
    end

    test "property" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0], [2.0, 0.0, 1.0]])
      assert Nx.dot(a, Nx.LinAlg.triangular_solve(a, b)) == b

      upper = Nx.transpose(a)
      assert Nx.dot(upper, Nx.LinAlg.triangular_solve(upper, b, lower: false)) == b

      assert Nx.dot(
               Nx.LinAlg.triangular_solve(upper, b, left_side: false, lower: false),
               upper
             ) == b

      assert Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose) ==
               Nx.LinAlg.triangular_solve(upper, b, lower: false)

      assert Nx.dot(
               Nx.transpose(a),
               Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose)
             ) == b
    end
  end

  describe "solve/2" do
    test "does not preserve names" do
      a = Nx.tensor([[1, 0, 1], [1, 1, 0], [1, 1, 1]], names: [:x, :y])

      assert Nx.LinAlg.solve(a, Nx.tensor([0, 2, 1], names: [:z])) |> Nx.round() ==
               Nx.tensor([1.0, 1.0, -1.0])
    end

    test "works with batched input" do
      a = Nx.tensor([[[1, 3, -2], [3, 5, 6], [2, 4, 3]], [[1, 1, 1], [6, -4, 5], [5, 2, 2]]])
      b = Nx.tensor([[5, 7, 8], [2, 31, 13]])

      assert_all_close(Nx.dot(a, [2], [0], Nx.LinAlg.solve(a, b), [1], [0]), b)
    end

    test "works with complex tensors" do
      a = ~M[
        1 0 i
       -1i 0 1i
        1 1 1
      ]

      b = ~V[3+i 4 2-2i]

      result = ~V[i 2 -3i]

      assert_all_close(Nx.LinAlg.solve(a, b), result)
    end
  end

  describe "invert" do
    test "works with batched input" do
      a = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      expected_result = Nx.tensor([[[-2, 1], [1.5, -0.5]], [[-4, 3], [3.5, -2.5]]])

      result = Nx.LinAlg.invert(a)

      assert_all_close(result, expected_result)

      assert_all_close(
        Nx.dot(a, [2], [0], result, [1], [0]),
        Nx.broadcast(Nx.eye(2), Nx.shape(a))
      )

      assert_all_close(
        Nx.dot(result, [2], [0], a, [1], [0]),
        Nx.broadcast(Nx.eye(2), Nx.shape(a))
      )
    end

    test "works with complex tensors" do
      a = ~M[
        1 0 i
        0 -1i 0
        0 0 2
      ]

      expected_result = ~M[
        1 0 -0.5i
        0 1i 0
        0 0 0.5
      ]

      result = Nx.LinAlg.invert(a)

      assert_all_close(result, expected_result)

      assert_all_close(Nx.dot(a, result), Nx.eye(Nx.shape(a)))
      assert_all_close(Nx.dot(result, a), Nx.eye(Nx.shape(a)))
    end

    test "regression for numerical stability" do
      q =
        Nx.tensor(
          [
            [1, 0.5, 0.5, 0.5],
            [0.5, 1, 0.5, 0.5],
            [0.5, 0.5, 1, 0.5],
            [0.5, 0.5, 0.5, 1]
          ],
          type: :f64
        )
        |> Nx.multiply(1.0e-10)

      invq = Nx.LinAlg.invert(q)
      assert_all_close(Nx.dot(invq, q), Nx.eye(q.shape), atol: 1.0e-15)
    end
  end

  describe "determinant/1" do
    test "does not preserve names" do
      two_by_two = Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
      assert Nx.LinAlg.determinant(two_by_two) == Nx.tensor(-2.0)

      three_by_three =
        Nx.tensor([[1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [7.0, 8.0, 9.0]], names: [:x, :y])

      assert Nx.LinAlg.determinant(three_by_three) == Nx.tensor(48.0)
    end

    test "supports batched matrices" do
      two_by_two = Nx.tensor([[[2, 3], [4, 5]], [[6, 3], [4, 8]]])
      assert Nx.LinAlg.determinant(two_by_two) == Nx.tensor([-2.0, 36.0])

      three_by_three =
        Nx.tensor([
          [[1.0, 2.0, 3.0], [1.0, 5.0, 3.0], [7.0, 6.0, 9.0]],
          [[5.0, 2.0, 3.0], [8.0, 5.0, 4.0], [3.0, 1.0, -9.0]]
        ])

      assert Nx.LinAlg.determinant(three_by_three) == Nx.tensor([-36.0, -98.0])

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

    test "returns 0 for LD rows" do
      for type <- [bf: 16, f: 16, f: 32, f: 64] do
        assert_all_close(
          Nx.LinAlg.determinant(Nx.tensor([[1.0, 0.0], [3.0, 0.0]], type: type)),
          Nx.tensor(0)
        )

        assert_all_close(
          Nx.LinAlg.determinant(
            Nx.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]], type: type)
          ),
          Nx.tensor(0)
        )

        assert_all_close(
          Nx.LinAlg.determinant(
            Nx.tensor(
              [
                [1.0, 2.0, 3.0, 0],
                [-1.0, -2.0, -3.0, 0],
                [4.0, 5.0, 6.0, 0],
                [4.0, 5.0, 6.0, 0]
              ],
              type: type
            )
          ),
          Nx.tensor(0)
        )

        assert_all_close(
          Nx.LinAlg.determinant(
            Nx.tensor(
              [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0]
              ],
              type: type
            )
          ),
          Nx.tensor(0)
        )
      end
    end
  end

  describe "norm/2" do
    test "raises for rank 3 or greater tensors" do
      t = Nx.iota({2, 2, 2})

      assert_raise(
        ArgumentError,
        "expected 1-D or 2-D tensor, got tensor with shape {2, 2, 2}",
        fn ->
          Nx.LinAlg.norm(t)
        end
      )
    end

    test "raises for unknown :ord value" do
      t = Nx.iota({2, 2})

      assert_raise(ArgumentError, "unknown ord :blep", fn ->
        Nx.LinAlg.norm(t, ord: :blep)
      end)
    end

    test "raises for invalid :ord integer value" do
      t = Nx.iota({2, 2})

      assert_raise(ArgumentError, "invalid :ord for 2-D tensor, got: -3", fn ->
        Nx.LinAlg.norm(t, ord: -3)
      end)
    end
  end

  describe "matrix_power" do
    test "supports complex with positive exponent" do
      a = ~M[
        1 1i
        -1i 1
      ]

      n = 5

      assert_all_close(Nx.LinAlg.matrix_power(a, n), Nx.multiply(2 ** (n - 1), a))
    end

    test "supports complex with 0 exponent" do
      a = ~M[
        1 1i
        -1i 1
      ]

      assert_all_close(Nx.LinAlg.matrix_power(a, 0), Nx.eye(Nx.shape(a)))
    end

    test "supports complex with negative exponent" do
      a = ~M[
        1 -0.5i
        0 0.5
      ]

      result = ~M[
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

  describe "qr" do
    test "factors a square matrix" do
      t = Nx.tensor([[2, -2, 18], [2, 1, 0], [1, 2, 0]])
      assert {q, %{type: output_type} = r} = Nx.LinAlg.qr(t)
      assert t |> Nx.round() |> Nx.as_type(output_type) == q |> Nx.dot(r) |> Nx.round()

      expected_q =
        Nx.tensor([
          [2 / 3, 2 / 3, 1 / 3],
          [2 / 3, -1 / 3, -2 / 3],
          [1 / 3, -2 / 3, 2 / 3]
        ])

      assert_all_close(q, expected_q, atol: 1.0e-7)

      expected_r =
        Nx.tensor([
          [3.0, 0.0, 12.0],
          [0.0, -3.0, 12.0],
          [0.0, 0.0, 6.0]
        ])

      assert_all_close(r, expected_r, atol: 1.0e-6)

      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-6)
    end

    test "factors tall matrix" do
      t = Nx.tensor([[1.0, -1.0, 4.0], [1.0, 4.0, -2.0], [1.0, 4.0, 2.0], [1.0, -1.0, 0.0]])

      # Reduced mode
      {q, r} = Nx.LinAlg.qr(t, mode: :reduced)

      expected_q =
        Nx.tensor([
          [0.5, -0.5, 0.5],
          [0.5, 0.5, -0.5],
          [0.5, 0.5, 0.5],
          [0.5, -0.5, -0.5]
        ])

      assert_all_close(q, expected_q, atol: 1.0e-7)

      expected_r =
        Nx.tensor([
          [2.0, 3.0, 2.0],
          [0.0, 5.0, -2.0],
          [0.0, 0.0, 4.0]
        ])

      assert_all_close(r, expected_r, atol: 1.0e-7)

      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-7)

      # Complete mode
      {q, r} = Nx.LinAlg.qr(t, mode: :complete)

      expected_q =
        Nx.tensor([
          [0.5, -0.5, 0.5, -0.5],
          [0.5, 0.5, -0.5, -0.5],
          [0.5, 0.5, 0.5, 0.5],
          [0.5, -0.5, -0.5, 0.5]
        ])

      assert_all_close(q, expected_q, atol: 1.0e-7)

      expected_r =
        Nx.tensor([
          [2.0, 3.0, 2.0],
          [0.0, 5.0, -2.0],
          [0.0, 0.0, 4.0],
          [0.0, 0.0, 0.0]
        ])

      assert_all_close(r, expected_r, atol: 1.0e-7)

      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-7)
    end

    test "factors wide matrix" do
      t =
        Nx.tensor([
          [1.0, 1.0, 1.0, 1.0],
          [-1.0, 4.0, 4.0, -1.0],
          [4.0, -2.0, 2.0, 0.0]
        ])

      # Reduced mode
      {q, r} = Nx.LinAlg.qr(t, mode: :reduced)

      expected_q =
        Nx.tensor([
          [0.23570226, 0.42637839, -0.87329601],
          [-0.23570226, 0.89686489, 0.37426972],
          [0.94280904, 0.11762162, 0.31189143]
        ])

      assert_all_close(q, expected_q, atol: 1.0e-7)

      expected_r =
        Nx.tensor([
          [4.24264069, -2.59272486, 1.1785113, 0.47140452],
          [0.0, 3.77859468, 4.24908118, -0.4704865],
          [0.0, 0.0, 1.24756572, -1.24756572]
        ])

      assert_all_close(r, expected_r, atol: 1.0e-6)
      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-7)

      # Complete mode
      {q, r} = Nx.LinAlg.qr(t, mode: :complete)

      assert_all_close(q, expected_q, atol: 1.0e-7)
      assert_all_close(r, expected_r, atol: 1.0e-6)
    end

    test "works with complex matrix" do
      t = ~M[
        1 0 1i
        0 2 -1i
        1 1 1
      ]

      {q, r} = Nx.LinAlg.qr(t)

      assert_all_close(q, ~M[
        -0.7071 0.2357  -0.6666
         0      -0.9428   -0.3333
        -0.7071 -0.2357  0.6666
      ])

      assert_all_close(r, ~M[
        -1.4142 -0.7071 -0.7071-0.7071i
        0      -2.1213  -0.2357+1.1785i
        0      0      0.6666-0.3333i
      ])

      assert_all_close(Nx.dot(q, r), t)
    end

    test "works with batches of matrices" do
      t =
        Nx.tensor([
          [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]],
          [[1.0, 2.0, 3.0], [0.0, 10.0, 5.0], [0.0, 0.0, 20.0]]
        ])

      {q, r} = Nx.LinAlg.qr(t)

      expected_q =
        Nx.tensor([
          [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ],
          [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
          ]
        ])

      assert_all_close(q, expected_q, atol: 1.0e-10)

      expected_r =
        Nx.tensor([
          [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0]
          ],
          [
            [1.0, 2.0, 3.0],
            [0.0, 10.0, 5.0],
            [0.0, 0.0, 20.0]
          ]
        ])

      assert_all_close(r, expected_r, atol: 1.0e-10)

      assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), t, atol: 1.0e-10)
    end

    test "property" do
      key = Nx.Random.key(System.unique_integer())

      for _ <- 1..10, type <- [{:f, 32}, {:c, 64}], reduce: key do
        key ->
          {square, key} = Nx.Random.uniform(key, shape: {2, 4, 4}, type: type)
          {tall, key} = Nx.Random.uniform(key, shape: {2, 4, 3}, type: type)
          {wide, key} = Nx.Random.uniform(key, shape: {2, 3, 4}, type: type)

          assert {q, r} = Nx.LinAlg.qr(square)
          assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), square, atol: 1.0e-6)

          assert {q, r} = Nx.LinAlg.qr(tall)
          assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), tall, atol: 1.0e-6)

          assert {q, r} = Nx.LinAlg.qr(wide)
          assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), wide, atol: 1.0e-6)

          key
      end
    end
  end

  describe "eigh" do
    test "computes eigenvalues and eigenvectors" do
      t =
        Nx.tensor([
          [5, -1, 0, 1, 2],
          [-1, 5, 0, 5, 3],
          [0, 0, 4, 7, 2],
          [1, 5, 7, 0, 9],
          [2, 3, 2, 9, 2]
        ])

      assert {eigenvals, eigenvecs} = Nx.LinAlg.eigh(t)

      # Eigenvalues
      assert round(eigenvals, 3) ==
               Nx.tensor([16.394, -9.738, 5.901, 4.334, -0.892])

      # Eigenvectors
      assert round(eigenvecs, 3) ==
               Nx.tensor([
                 [0.112, -0.005, -0.831, -0.436, -0.328],
                 [0.395, 0.163, 0.530, -0.537, -0.497],
                 [0.427, 0.326, -0.133, 0.700, -0.452],
                 [0.603, -0.783, -0.007, 0.079, 0.130],
                 [0.534, 0.504, -0.104, -0.160, 0.651]
               ])
    end

    test "computes eigenvalues and eigenvectors for a Hermitian matrix case" do
      # Hermitian matrix
      t =
        Nx.tensor([
          [1, Complex.new(0, 2), 2],
          [Complex.new(0, -2), -3, Complex.new(0, 2)],
          [2, Complex.new(0, -2), 1]
        ])

      assert {eigenvals, eigenvecs} = Nx.LinAlg.eigh(t, max_iter: 10_000)

      # Eigenvalues
      assert eigenvals ==
               Nx.tensor([Complex.new(-5, 0), Complex.new(3, 0), Complex.new(1, 0)])

      # Eigenvectors
      assert round(eigenvecs, 3) ==
               Nx.tensor([
                 [
                   Complex.new(-0.408, 0.0),
                   Complex.new(0.0, 0.707),
                   Complex.new(0.577, 0.0)
                 ],
                 [
                   Complex.new(0.0, -0.816),
                   Complex.new(0.0, 0.0),
                   Complex.new(0.0, -0.577)
                 ],
                 [
                   Complex.new(0.408, 0.0),
                   Complex.new(0.0, 0.707),
                   Complex.new(-0.577, 0.0)
                 ]
               ])
    end

    test "properties for matrices with different eigenvalues" do
      # Generate real Hermitian matrices with different eigenvalues
      # from random matrices based on the relation A = Q.Λ.Q^*
      # where Λ is the diagonal matrix of eigenvalues and Q is unitary matrix.

      key = Nx.Random.key(System.unique_integer())

      for type <- [f: 32, c: 64], reduce: key do
        key ->
          # Unitary matrix from a random matrix
          {base, key} = Nx.Random.uniform(key, shape: {3, 3, 3}, type: type)
          {q, _} = Nx.LinAlg.qr(base)

          # Different eigenvalues from random values
          evals_test =
            [{100, 30}, {4, 6}, {0.7, 0.9}]
            |> Enum.map(fn {low, up} ->
              if :rand.uniform() - 0.5 > 0 do
                {low, up}
              else
                {-up, -low}
              end
            end)
            |> Enum.map(fn {low, up} ->
              rand = :rand.uniform() * (up - low) + low
              Nx.tensor([rand], type: :f64)
            end)
            |> Nx.concatenate()

          # Hermitian matrix with different eigenvalues
          # using A = A^* = Q^*.Λ.Q.
          a =
            q
            |> Nx.LinAlg.adjoint()
            |> Nx.multiply(evals_test)
            |> Nx.dot([2], [0], q, [1], [0])
            |> round(2)

          # Eigenvalues and eigenvectors
          assert {evals, evecs} = Nx.LinAlg.eigh(a, max_iter: 10_000)
          assert_all_close(evals_test, evals, atol: 1.0e-1)

          # Eigenvalue equation
          evecs_evals = Nx.multiply(evecs, evals)
          a_evecs = Nx.dot(a, [2], [0], evecs, [1], [0])

          assert_all_close(evecs_evals, a_evecs, atol: 1.0e-1)
          key
      end
    end

    test "properties for matrices with close eigenvalues" do
      # Generate real Hermitian matrices with close eigenvalues
      # from random matrices based on the relation A = Q.Λ.Q^*

      key = Nx.Random.key(System.unique_integer())

      for _ <- 1..3, reduce: key do
        key ->
          # Unitary matrix from a random matrix
          {base_q, key} = Nx.Random.uniform(key, 0, 1, shape: {3, 3, 3})
          {q, _} = Nx.LinAlg.qr(base_q)

          # ensure that eval1 is far apart from the other two eigenvals
          eval1 = :rand.uniform() * 10 + 10
          # eval2 is in the range 1 < eval2 < 1.01
          eval2 = :rand.uniform() * 0.01 + 1
          # eval3 also in the same range as eval2
          eval3 = :rand.uniform() * 0.01 + 1

          evals_test = Nx.tensor([eval1, eval2, eval3])

          # Hermitian matrix with different eigenvalues
          # using A = A^* = Q^*ΛQ.
          a =
            q
            |> Nx.LinAlg.adjoint()
            |> Nx.multiply(evals_test)
            |> Nx.dot([2], [0], q, [1], [0])
            |> round(3)

          # Eigenvalues and eigenvectors
          assert {evals, evecs} = Nx.LinAlg.eigh(a)
          assert_all_close(evals_test, evals, atol: 0.1)

          # Eigenvalue equation
          evecs_evals = Nx.multiply(evecs, evals)
          a_evecs = Nx.dot(a, [2], [0], evecs, [1], [0])
          assert_all_close(evecs_evals, a_evecs, atol: 0.1)
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

      assert round(t, 2) == u |> Nx.dot(s_matrix) |> Nx.dot(v) |> round(2)

      assert round(u, 3) ==
               Nx.tensor([
                 [0.141, 0.825, -0.007, 0.019],
                 [0.344, 0.426, 0.01, 0.382],
                 [0.547, 0.028, 0.002, -0.822],
                 [0.75, -0.371, -0.005, 0.421]
               ])
               |> round(3)

      assert Nx.tensor([25.462, 1.291, 0.0]) |> round(3) == round(s, 3)

      assert Nx.tensor([
               [0.505, 0.575, 0.644],
               [-0.761, -0.057, 0.647],
               [-0.408, 0.817, -0.408]
             ])
             |> round(3) == round(v, 3)
    end

    test "finds the singular values of square matrices" do
      t = Nx.iota({5, 5})

      assert {u, s, vt} = Nx.LinAlg.svd(t)

      assert round(Nx.as_type(t, :f32), 2) == u |> Nx.multiply(s) |> Nx.dot(vt) |> round(2)
    end

    test "finds the singular values of wide matrices" do
      t = Nx.iota({3, 5})

      assert {u, s, vt} = Nx.LinAlg.svd(t)

      assert round(Nx.as_type(t, :f32), 1) ==
               u |> Nx.dot(t.shape |> Nx.eye() |> Nx.put_diagonal(s)) |> Nx.dot(vt) |> round(1)
    end

    test "finds the singular values triangular matrices" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]])

      assert {%{type: output_type} = u, %{type: output_type} = s, %{type: output_type} = v} =
               Nx.LinAlg.svd(t)

      # turn s into a {4, 3} tensor
      s_matrix =
        0
        |> Nx.broadcast({3, 3})
        |> Nx.put_diagonal(s)

      assert round(t, 1) == u |> Nx.dot(s_matrix) |> Nx.dot(v) |> round(1)

      assert round(u, 3) ==
               Nx.tensor([
                 [0.336, -0.407, -0.849],
                 [0.037, -0.895, 0.444],
                 [0.941, 0.181, 0.286]
               ])
               |> round(3)

      # The expected value is ~ [9, 5, 1] since the eigenvalues of
      # a triangular matrix are the diagonal elements. Close enough!
      assert Nx.tensor([9.52, 4.433, 0.853]) |> round(3) == round(s, 3)

      assert Nx.tensor([
               [0.035, 0.0869, 0.996],
               [-0.091, -0.992, 0.09],
               [-0.995, 0.094, 0.027]
             ])
             |> round(3) == round(v, 3)
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

    test "works with f16" do
      x = Nx.tensor([[0, 0], [0, 0]], type: :f16)
      assert Nx.LinAlg.svd(x) == {Nx.eye(2, type: :f16), ~V"0.0 0.0"f16, Nx.eye(2, type: :f16)}

      x = Nx.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], type: :f16)
      assert {u, s, vt} = Nx.LinAlg.svd(x)
      assert u.type == {:f, 16}
      assert s.type == {:f, 16}
      assert vt.type == {:f, 16}
    end

    test "works with f64" do
      x = Nx.tensor([[0, 0], [0, 0]], type: :f64)
      assert Nx.LinAlg.svd(x) == {Nx.eye(2, type: :f64), ~V"0.0 0.0"f64, Nx.eye(2, type: :f64)}

      x = Nx.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], type: :f64)
      assert {u, s, vt} = Nx.LinAlg.svd(x)
      assert u.type == {:f, 64}
      assert s.type == {:f, 64}
      assert vt.type == {:f, 64}
    end
  end

  describe "lu" do
    test "property" do
      key = Nx.Random.key(System.unique_integer())

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
          assert_all_close(p |> Nx.dot([2], [0], l, [1], [0]) |> Nx.dot([2], [0], u, [1], [0]), a)
          key
      end
    end
  end

  describe "cholesky" do
    test "property" do
      key = Nx.Random.key(System.unique_integer())

      for _ <- 1..10, type <- @types, reduce: key do
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

  describe "pinv" do
    test "does not raise for 0 singular values" do
      key = Nx.Random.key(System.unique_integer())

      for {m, n} <- [{3, 4}, {3, 3}, {4, 3}], reduce: key do
        key ->
          # generate u and vt as random orthonormal matrices
          {base_u, key} = Nx.Random.uniform(key, 0, 1, shape: {m, m})
          {u, _} = Nx.LinAlg.qr(base_u)
          {base_vt, key} = Nx.Random.uniform(key, 0, 1, shape: {n, n})
          {vt, _} = Nx.LinAlg.qr(base_vt)

          # because min(m, n) is always 3, we can use fixed values here
          # the important thing is that there's at least one zero in the
          # diagonal, to ensure that we're guarding against 0 division
          zeros = Nx.broadcast(0, {m, n})
          s = Nx.put_diagonal(zeros, Nx.tensor([1, 4, 0]))
          s_inv = Nx.put_diagonal(Nx.transpose(zeros), Nx.tensor([1, 0.25, 0]))

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

  defp round(tensor, places) do
    round_real = fn x -> Float.round(Complex.real(x), places) end
    round_imag = fn x -> Float.round(Complex.imag(x), places) end

    Nx.map(tensor, fn x ->
      x = Nx.to_number(x)

      cond do
        is_integer(x) ->
          x

        is_float(x) ->
          Float.round(x, places)

        x in [:nan, :neg_infinity, :infinity] ->
          x

        true ->
          # Complex case
          Complex.new(round_real.(x), round_imag.(x))
      end
    end)
  end
end
