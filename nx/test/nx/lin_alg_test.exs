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

      assert_all_close(Nx.dot(a, result), Nx.eye(a))
      assert_all_close(Nx.dot(result, a), Nx.eye(a))
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

      assert_all_close(Nx.LinAlg.matrix_power(a, 0), Nx.eye(a))
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
    test "correctly factors a square matrix" do
      t = Nx.tensor([[2, -2, 18], [2, 1, 0], [1, 2, 0]])
      assert {q, %{type: output_type} = r} = Nx.LinAlg.qr(t)
      assert t |> Nx.round() |> Nx.as_type(output_type) == q |> Nx.dot(r) |> Nx.round()

      expected_q =
        Nx.tensor([
          [2 / 3, 2 / 3, 1 / 3],
          [2 / 3, -1 / 3, -2 / 3],
          [1 / 3, -2 / 3, 2 / 3]
        ])

      assert_all_close(q, expected_q, atol: 1.0e-10)

      expected_r =
        Nx.tensor([
          [3.0, 0.0, 12.0],
          [0.0, -3.0, 12.0],
          [0.0, 0.0, 6.0]
        ])

      assert_all_close(r, expected_r, atol: 1.0e-10)

      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-10)
    end

    test "factors rectangular matrix" do
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

      assert_all_close(q, expected_q, atol: 1.0e-10)

      expected_r =
        Nx.tensor([
          [2.0, 3.0, 2.0],
          [0.0, 5.0, -2.0],
          [0.0, 0.0, 4.0]
        ])

      assert_all_close(r, expected_r, atol: 1.0e-10)

      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-10)

      # Complete mode
      {q, r} = Nx.LinAlg.qr(t, mode: :complete)

      expected_q =
        Nx.tensor([
          [0.5, -0.5, 0.5, -0.5],
          [0.5, 0.5, -0.5, -0.5],
          [0.5, 0.5, 0.5, 0.5],
          [0.5, -0.5, -0.5, 0.5]
        ])

      assert_all_close(q, expected_q, atol: 1.0e-10)

      expected_r =
        Nx.tensor([
          [2.0, 3.0, 2.0],
          [0.0, 5.0, -2.0],
          [0.0, 0.0, 4.0],
          [0.0, 0.0, 0.0]
        ])

      assert_all_close(r, expected_r, atol: 1.0e-10)

      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-10)
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
      for _ <- 1..10, type <- [{:f, 32}, {:c, 64}] do
        square = Nx.random_uniform({2, 4, 4}, type: type)
        tall = Nx.random_uniform({2, 4, 3}, type: type)

        assert {q, r} = Nx.LinAlg.qr(square)
        assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), square, atol: 1.0e-6)

        assert {q, r} = Nx.LinAlg.qr(tall)
        assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), tall, atol: 1.0e-6)
      end
    end
  end

  describe "eigh" do
    test "correctly a eigenvalues and eigenvectors" do
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
               Nx.tensor([16.394, -9.739, 5.901, 4.334, -0.892])

      # Eigenvectors
      assert round(eigenvecs, 3) ==
               Nx.tensor([
                 [0.112, -0.004, -0.828, 0.440, 0.328],
                 [0.395, 0.163, 0.533, 0.534, 0.497],
                 [0.427, 0.326, -0.137, -0.699, 0.452],
                 [0.603, -0.783, -0.008, -0.079, -0.130],
                 [0.534, 0.504, -0.103, 0.160, -0.651]
               ])
    end

    test "correctly a eigenvalues and eigenvectors for a Hermitian matrix case" do
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

    test "property for matrices with different eigenvalues" do
      # Generate real Hermitian matrices with different eigenvalues
      # from random matrices based on the relation A = Q.Λ.Q^*
      # where Λ is the diagonal matrix of eigenvalues and Q is unitary matrix.

      for _ <- 1..3, type <- [f: 32, c: 64] do
        # Unitary matrix from a random matrix
        {q, _} = Nx.random_uniform({3, 3, 3}, type: type) |> Nx.LinAlg.qr()

        # Different eigenvalues from random values
        evals_test =
          [{1, 3}, {0.4, 0.6}, {0.07, 0.09}]
          |> Enum.map(fn {low, up} ->
            if :rand.uniform() - 0.5 > 0 do
              {low, up}
            else
              {-up, -low}
            end
          end)
          |> Enum.map(fn {low, up} ->
            Nx.random_uniform({1}, low, up, type: {:f, 64})
          end)
          |> Nx.concatenate()

        # Hermitian matrix with different eigenvalues
        # using A = A^* = Q^*.Λ.Q.
        a =
          q
          |> Nx.LinAlg.adjoint()
          |> Nx.multiply(evals_test)
          |> Nx.dot([2], [0], q, [1], [0])
          |> round(3)

        # Eigenvalues and eigenvectors
        assert {evals, evecs} = Nx.LinAlg.eigh(a, max_iter: 10_000)
        assert_all_close(evals_test, evals, atol: 1.0e-2)

        # Eigenvalue equation
        evecs_evals = Nx.multiply(evecs, evals)
        a_evecs = Nx.dot(a, [2], [0], evecs, [1], [0])

        assert_all_close(evecs_evals, a_evecs, atol: 1.0e-2)
      end
    end

    test "properties for matrices with close eigenvalues" do
      # Generate real Hermitian matrices with close eigenvalues
      # from random matrices based on the relation A = Q.Λ.Q^*

      for _ <- 1..3 do
        # Unitary matrix from a random matrix
        {q, _} = Nx.random_uniform({3, 3, 3}) |> Nx.LinAlg.qr()

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
      end
    end
  end

  describe "svd" do
    test "correctly finds the singular values of full matrices" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

      assert {%{type: output_type} = u, %{type: output_type} = s, %{type: output_type} = v} =
               Nx.LinAlg.svd(t, max_iter: 1000)

      zero_row = List.duplicate(0, 3)

      # turn s into a {4, 3} tensor
      s_matrix =
        s
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.map(fn {x, idx} -> List.replace_at(zero_row, idx, x) end)
        |> Enum.concat([zero_row])
        |> Nx.tensor()

      assert round(t, 2) == u |> Nx.dot(s_matrix) |> Nx.dot(v) |> round(2)

      assert round(u, 3) ==
               Nx.tensor([
                 [-0.141, -0.825, -0.547, 0.019],
                 [-0.344, -0.426, 0.744, 0.382],
                 [-0.547, -0.028, 0.153, -0.822],
                 [-0.75, 0.371, -0.35, 0.421]
               ])
               |> round(3)

      assert Nx.tensor([25.462, 1.291, 0.0]) |> round(3) == round(s, 3)

      assert Nx.tensor([
               [-0.505, -0.575, -0.644],
               [0.761, 0.057, -0.646],
               [-0.408, 0.816, -0.408]
             ])
             |> round(3) == round(v, 3)
    end

    test "correctly finds the singular values triangular matrices" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]])

      assert {%{type: output_type} = u, %{type: output_type} = s, %{type: output_type} = v} =
               Nx.LinAlg.svd(t)

      zero_row = List.duplicate(0, 3)

      # turn s into a {4, 3} tensor
      s_matrix =
        s
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.map(fn {x, idx} -> List.replace_at(zero_row, idx, x) end)
        |> Nx.tensor()

      assert round(t, 2) == u |> Nx.dot(s_matrix) |> Nx.dot(v) |> round(2)

      assert round(u, 3) ==
               Nx.tensor([
                 [-0.335, 0.408, -0.849],
                 [-0.036, 0.895, 0.445],
                 [-0.941, -0.18, 0.286]
               ])
               |> round(3)

      # The expected value is ~ [9, 5, 1] since the eigenvalues of
      # a triangular matrix are the diagonal elements. Close enough!
      assert Nx.tensor([9.52, 4.433, 0.853]) |> round(3) == round(s, 3)

      assert Nx.tensor([
               [-0.035, -0.086, -0.996],
               [0.092, 0.992, -0.089],
               [-0.995, 0.095, 0.027]
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

      assert round(t, 2) ==
               u
               |> Nx.dot([2], [0], s_matrix, [1], [0])
               |> Nx.dot([2], [0], v, [1], [0])
               |> round(2)
    end

    test "works with vectors" do
      t = Nx.tensor([[-2], [1]])

      {u, s, vt} = Nx.LinAlg.svd(t)
      assert_all_close(u |> Nx.dot(Nx.stack([s, Nx.tensor([0])])) |> Nx.dot(vt), t)
    end
  end

  describe "lu" do
    test "property" do
      for _ <- 1..10, type <- [{:f, 32}, {:c, 64}] do
        # Generate random L and U matrices so we can construct
        # a factorizable A matrix:
        shape = {3, 4, 4}
        lower_selector = Nx.iota(shape, axis: 1) |> Nx.greater_equal(Nx.iota(shape, axis: 2))
        upper_selector = Nx.LinAlg.adjoint(lower_selector)

        l_prime =
          shape
          |> Nx.random_uniform(type: type)
          |> Nx.multiply(lower_selector)

        u_prime = shape |> Nx.random_uniform(type: type) |> Nx.multiply(upper_selector)

        a = Nx.dot(l_prime, [2], [0], u_prime, [1], [0])

        assert {p, l, u} = Nx.LinAlg.lu(a)
        assert_all_close(p |> Nx.dot([2], [0], l, [1], [0]) |> Nx.dot([2], [0], u, [1], [0]), a)
      end
    end
  end

  describe "cholesky" do
    test "property" do
      for _ <- 1..10, type <- @types do
        # Generate random L matrix so we can construct
        # a factorizable A matrix:
        shape = {3, 4, 4}
        lower_selector = Nx.iota(shape, axis: 1) |> Nx.greater_equal(Nx.iota(shape, axis: 2))

        l_prime =
          shape
          |> Nx.random_uniform(type: type)
          |> Nx.multiply(lower_selector)

        a = Nx.dot(l_prime, [2], [0], Nx.LinAlg.adjoint(l_prime), [1], [0])

        assert l = Nx.LinAlg.cholesky(a)
        assert_all_close(Nx.dot(l, [2], [0], Nx.LinAlg.adjoint(l), [1], [0]), a, atol: 1.0e-2)
      end
    end
  end

  defp round(tensor, places) do
    round_real = fn x -> Float.round(Complex.real(Nx.to_number(x)), places) end
    round_imag = fn x -> Float.round(Complex.imag(Nx.to_number(x)), places) end

    Nx.map(tensor, fn x ->
      if is_float(Nx.to_number(x)) do
        # Float case
        Float.round(Nx.to_number(x), places)
      else
        # Complex case
        Complex.new(round_real.(x), round_imag.(x))
      end
    end)
  end
end
