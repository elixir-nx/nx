defmodule Nx.LinAlgTest do
  use ExUnit.Case, async: true

  import Nx.Testing
  import Nx, only: :sigils

  doctest Nx.LinAlg

  @types [{:f, 32}, {:c, 64}]

  describe "triangular_solve" do
    test "works with batched input" do
      a = Nx.tensor([[[-1, 0, 0], [1, 1, 0], [1, 1, 1]], [[2, 0, 0], [4, -2, 0], [-5, 1, 3]]])
      b = Nx.tensor([[1.0, 2.0, 3.0], [6, 10, 1]])

      assert Nx.dot(a, [2], [0], Nx.LinAlg.triangular_solve(a, b), [1], [0]) == b
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

      assert x ==
               Nx.tensor(
                 [
                   [1, 1, 1],
                   [1, 1, 1]
                 ],
                 type: :f64
               )
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

    test "transform_a: :conjugate with real tensors" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0], [2.0, 0.0, 1.0]])

      # For real tensors, conjugate is identity, so results match :none
      assert Nx.LinAlg.triangular_solve(a, b, transform_a: :conjugate) ==
               Nx.LinAlg.triangular_solve(a, b)

      upper = Nx.transpose(a)

      assert Nx.LinAlg.triangular_solve(upper, b, transform_a: :conjugate, lower: false) ==
               Nx.LinAlg.triangular_solve(upper, b, lower: false)

      assert Nx.LinAlg.triangular_solve(upper, b,
               transform_a: :conjugate,
               left_side: false,
               lower: false
             ) ==
               Nx.LinAlg.triangular_solve(upper, b, left_side: false, lower: false)
    end
  end

  describe "solve/2" do
    test "does not preserve names" do
      a = Nx.tensor([[1, 0, 1], [1, 1, 0], [1, 1, 1]], names: [:x, :y])

      assert Nx.LinAlg.solve(a, Nx.tensor([0, 2, 1], names: [:z])) |> Nx.round() ==
               Nx.tensor([1.0, 1.0, -1.0])
    end

    test "raises for singular matrices from the exhaustively checked domains" do
      # every 2x2 matrix with entries in -5..5 and every 3x3 matrix with
      # entries in -1..1 was checked; the test samples those domains
      for type <- [bf: 16, f: 16, f: 32, f: 64], {n, lo, hi} <- [{2, -5, 5}, {3, -1, 1}] do
        b = Nx.broadcast(1, {n})

        for {rows, 0} <- small_integer_matrices(n, lo, hi, 50) do
          t = Nx.tensor(rows, type: type)

          assert_raise ArgumentError, "can't solve for singular matrix", fn ->
            Nx.LinAlg.solve(t, b)
          end
        end
      end
    end

    test "is accurate for nonsingular small integer matrices" do
      for type <- [f: 32, f: 64],
          {n, lo, hi, count} <- [{2, -5, 5, 50}, {3, -3, 3, 50}, {4, -2, 2, 20}, {5, -1, 1, 10}] do
        epsilon =
          type
          |> Nx.Constants.epsilon()
          |> Nx.to_number()

        tolerance = 32 * n * epsilon

        for {rows, exact} <- small_integer_matrices(n, lo, hi, count), exact != 0 do
          t = Nx.tensor(rows, type: type)
          b = Nx.broadcast(1, {n})
          x = Nx.LinAlg.solve(t, b)
          product = Nx.dot(t, x)

          assert relative_residual(product, b, t, x) <= tolerance
        end
      end
    end

    test "works with batched input" do
      a = Nx.tensor([[[1, 3, -2], [3, 5, 6], [2, 4, 3]], [[1, 1, 1], [6, -4, 5], [5, 2, 2]]])
      b = Nx.tensor([[5, 7, 8], [2, 31, 13]])

      assert_all_close(Nx.dot(a, [2], [0], Nx.LinAlg.solve(a, b), [1], [0]), b)
    end

    test "works with complex tensors" do
      a = ~MAT[
        1 0 i
       -1i 0 1i
        1 1 1
      ]

      b = ~VEC[3+i 4 2-2i]

      result = ~VEC[i 2 -3i]

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
      a = ~MAT[
        1 0 i
        0 -1i 0
        0 0 2
      ]

      expected_result = ~MAT[
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

    test "keeps the input type" do
      for type <- [bf: 16, f: 16, f: 32, f: 64] do
        t =
          Nx.tensor(
            [
              [2, 0],
              [0, 4]
            ],
            type: type
          )

        expected =
          Nx.tensor(
            [
              [0.5, 0],
              [0, 0.25]
            ],
            type: type
          )

        result = Nx.LinAlg.invert(t)
        assert result.type == type
        assert_equal(result, expected)
      end
    end

    test "returns NaN in the input type for singular matrices" do
      for type <- [bf: 16, f: 16, f: 32, f: 64] do
        # rows in arithmetic progression, so the matrix is exactly singular
        t =
          Nx.tensor(
            [
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]
            ],
            type: type
          )

        result = Nx.LinAlg.invert(t)
        assert result.type == type
        assert_equal(result, Nx.broadcast(:nan, {3, 3}))
      end
    end

    test "returns NaN for a singular matrix whose determinant rounds away from zero" do
      # exactly singular, but rounding leaves a tiny pivot instead of zero
      t =
        Nx.tensor([
          [2, -1, -1, 0],
          [2, 0, -1, 1],
          [-1, 0, 1, 1],
          [1, 2, -2, -2]
        ])

      in_f64 = Nx.as_type(t, :f64)
      nan = Nx.broadcast(:nan, {4, 4})

      assert_equal(Nx.LinAlg.invert(t), nan)
      assert_equal(Nx.LinAlg.invert(in_f64), nan)
    end

    test "returns NaN for singular matrices in a batch" do
      a =
        Nx.tensor([
          [
            [2, 0],
            [0, 4]
          ],
          [
            [1, 2],
            [2, 4]
          ]
        ])

      expected =
        Nx.tensor([
          [
            [0.5, 0],
            [0, 0.25]
          ],
          [
            [:nan, :nan],
            [:nan, :nan]
          ]
        ])

      vectorized = Nx.vectorize(a, :x)
      expected_vectorized = Nx.vectorize(expected, :x)

      assert_equal(Nx.LinAlg.invert(a), expected)
      assert_equal(Nx.LinAlg.invert(vectorized), expected_vectorized)
    end

    test "inverts well-conditioned matrices with small entries" do
      a =
        Nx.tensor(
          [
            [0.2, 0],
            [0, 0.2]
          ],
          type: :f16
        )

      expected =
        Nx.tensor(
          [
            [5, 0],
            [0, 5]
          ],
          type: :f16
        )

      assert_equal(Nx.LinAlg.invert(a), expected)

      identity = Nx.eye(10)
      scaled = Nx.multiply(identity, 1.0e-4)
      expected_scaled = Nx.multiply(identity, 1.0e4)

      assert_all_close(Nx.LinAlg.invert(scaled), expected_scaled)
    end

    test "inverts a graded matrix whose smallest pivot is far below its largest" do
      # the tiny pivot is an entry of the matrix, not the result of cancellation
      t =
        Nx.tensor([
          [1.0, 0.0],
          [1.0, 1.0e-6]
        ])

      expected =
        Nx.tensor([
          [1.0, 0.0],
          [-1.0e6, 1.0e6]
        ])

      assert_equal(Nx.LinAlg.invert(t), expected)
    end

    test "inverts bf16 diagonal and block-diagonal matrices at sizes where the tolerance is large" do
      # the tolerance is 0.18 at n = 32 and 0.25 at n = 64 in bf16; a pivot that
      # nothing was subtracted from still counts as nonsingular
      identity = Nx.eye(64, type: :bf16)
      assert_equal(Nx.LinAlg.invert(identity), identity)

      # the 32x32 block-diagonal matrix with sixteen copies of a 2x2 block on
      # its diagonal: the Kronecker product of the identity with the block
      block_diagonal = fn block ->
        eye = Nx.eye(16, type: :bf16)
        eye_axes = Nx.reshape(eye, {16, 1, 16, 1})

        block
        |> Nx.tensor(type: :bf16)
        |> Nx.reshape({1, 2, 1, 2})
        |> Nx.multiply(eye_axes)
        |> Nx.reshape({32, 32})
      end

      block = [
        [1, 1],
        [1, 2]
      ]

      inverse_block = [
        [2, -1],
        [-1, 1]
      ]

      t = block_diagonal.(block)
      expected = block_diagonal.(inverse_block)

      assert_equal(Nx.LinAlg.invert(t), expected)
    end

    test "inverts well-conditioned matrices in every type" do
      # LAPACK's test suite builds random matrices with a prescribed condition
      # number and expects systems with cond * epsilon <= 0.1 to be solvable,
      # after the perturbation bound of Higham, Accuracy and Stability of
      # Numerical Algorithms, 2nd ed., SIAM, 2002, ch. 7; these are its cond 2
      # and cond 0.1 / epsilon types
      for type <- [f: 64, f: 32, f: 16, bf: 16], n <- [4, 8, 16] do
        epsilon =
          type
          |> Nx.Constants.epsilon()
          |> Nx.to_number()

        for cond <- [2.0, 0.1 / epsilon] do
          matrices = conditioned_matrices(n, cond, 10, type)
          inverses = Nx.LinAlg.invert(matrices)

          has_nan =
            inverses
            |> Nx.is_nan()
            |> Nx.any()
            |> Nx.to_number()

          assert has_nan == 0

          for i <- 0..9 do
            t = matrices[i]
            inverse = inverses[i]
            product = Nx.dot(t, inverse)

            assert relative_residual(product, Nx.eye(n), t, inverse) <= 32 * n * epsilon
          end
        end
      end

      # The Hadamard matrix (Higham, ACM TOMS 17(3), 1991,
      # doi:10.1145/114697.116805) has entries of 1 and -1 and is orthogonal
      # up to scaling, so its inverse is its transpose over n. Its elimination
      # is exact integer arithmetic that cancels many products at each pivot,
      # which the criterion must not mistake for rounding noise.
      for type <- [f: 64, f: 32, f: 16, bf: 16], n <- [4, 8, 16] do
        t = hadamard(n, type)

        expected =
          t
          |> Nx.transpose()
          |> Nx.divide(n)

        assert_equal(Nx.LinAlg.invert(t), expected)
      end
    end

    test "returns NaN when the elimination overflows the type" do
      # The Wilkinson matrix is the classic example of pivot growth (Wilkinson,
      # J. ACM 8(3), 1961, doi:10.1145/321075.321076): every elimination step
      # doubles its last column, so the last pivot is 2 ** (n - 1), beyond f16
      # from n = 17.
      t = wilkinson(17, :f16)
      assert_equal(Nx.LinAlg.invert(t), Nx.broadcast(:nan, {17, 17}))
    end

    test "inverts matrices holding the minimum of a signed integer type" do
      # the absolute value of the minimum wraps around in the integer type, so
      # the row magnitudes must be taken after the cast to the floating type
      for width <- [8, 16, 32, 64] do
        minimum = -(2 ** (width - 1))

        t =
          Nx.tensor(
            [
              [minimum, 0],
              [0, 1]
            ],
            type: {:s, width}
          )

        expected =
          Nx.tensor([
            [1 / minimum, 0.0],
            [0.0, 1.0]
          ])

        assert_equal(Nx.LinAlg.invert(t), expected)
      end
    end

    test "inverts a matrix whose rows are below the normal range of f64" do
      # the row scale is clamped to the largest power of two of the type
      m = 2.0 ** -1024
      big = 2.0 ** 1023

      t =
        Nx.tensor(
          [
            [m, m],
            [m, -m]
          ],
          type: :f64
        )

      expected =
        Nx.tensor(
          [
            [big, big],
            [big, -big]
          ],
          type: :f64
        )

      assert_equal(Nx.LinAlg.invert(t), expected)
    end

    test "returns NaN for singular matrices from the exhaustively checked domains, alone and in a batch" do
      # every 2x2 matrix with entries in -5..5 and every 3x3 matrix with
      # entries in -1..1 was checked; the test samples those domains
      for type <- [bf: 16, f: 16, f: 32, f: 64], {n, lo, hi} <- [{2, -5, 5}, {3, -1, 1}] do
        matrices = small_integer_matrices(n, lo, hi, 50)
        nan = Nx.broadcast(:nan, {n, n})

        for {rows, 0} <- matrices do
          t = Nx.tensor(rows, type: type)
          assert_equal(Nx.LinAlg.invert(t), nan)
        end

        # a batch mixing singular and nonsingular matrices
        batch =
          matrices
          |> Enum.map(&elem(&1, 0))
          |> Nx.tensor(type: type)

        inverses = Nx.LinAlg.invert(batch)

        for {{rows, exact}, i} <- Enum.with_index(matrices) do
          if exact == 0 do
            assert_equal(inverses[i], nan)
          else
            t = Nx.tensor(rows, type: type)
            assert_all_close(inverses[i], Nx.LinAlg.invert(t))
          end
        end
      end
    end

    test "is accurate for nonsingular small integer matrices, alone and in a batch" do
      for type <- [f: 32, f: 64],
          {n, lo, hi, count} <- [{2, -5, 5, 50}, {3, -3, 3, 50}, {4, -2, 2, 20}, {5, -1, 1, 10}] do
        epsilon =
          type
          |> Nx.Constants.epsilon()
          |> Nx.to_number()

        tolerance = 32 * n * epsilon

        matrices =
          for {rows, exact} <- small_integer_matrices(n, lo, hi, count), exact != 0, do: rows

        batch = Nx.tensor(matrices, type: type)
        inverses = Nx.LinAlg.invert(batch)

        for {rows, i} <- Enum.with_index(matrices) do
          t = Nx.tensor(rows, type: type)
          inverse = Nx.LinAlg.invert(t)
          product = Nx.dot(t, inverse)

          assert relative_residual(product, Nx.eye(n), t, inverse) <= tolerance
          assert_all_close(inverses[i], inverse)
        end
      end
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

    test "regression" do
      tensor =
        Nx.f64([
          [-1.0, 1.0, -1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      result = Nx.LinAlg.determinant(tensor)

      assert_all_close(result, Nx.tensor(1.0, type: :f64))
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

    test "correctly support axes option" do
      t =
        Nx.tensor([
          [-1.0, -1.0],
          [0.0, 0.0],
          [1.0, 1.0]
        ])

      result = Nx.tensor([1.4142135381698608, 0.0, 1.4142135381698608])
      assert Nx.LinAlg.norm(t, axes: [1]) == result
      assert Nx.LinAlg.norm(t, axes: [1], keep_axes: true) == Nx.reshape(result, {3, 1})
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
      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-6)

      # Complete mode
      {q, r} = Nx.LinAlg.qr(t, mode: :complete)

      assert_all_close(q, expected_q, atol: 1.0e-7)
      assert_all_close(r, expected_r, atol: 1.0e-6)
    end

    test "factors wide and rank-deficient f16 matrices" do
      # A column with no entries below the diagonal has a zero norm in the
      # Householder step. The default eps must stay representable in f16
      # for that guard to work, otherwise the reflector divides by zero.
      for t <- [
            Nx.iota({3, 4}, type: :f16),
            Nx.tensor([[1, 0, 2], [3, 0, 4], [5, 0, 6]], type: :f16)
          ] do
        {q, r} = Nx.LinAlg.qr(t)
        assert q.type == {:f, 16}
        assert_all_close(Nx.dot(q, r), t, atol: 2.0e-2)
      end
    end

    test "works with complex matrix" do
      t = ~MAT[
        1 0 1i
        0 2 -1i
        1 1 1
      ]

      {q, r} = Nx.LinAlg.qr(t)

      assert_all_close(q, ~MAT[
        -0.7071 0.2357  -0.6666
         0      -0.9428   -0.3333
        -0.7071 -0.2357  0.6666
      ])

      assert_all_close(r, ~MAT[
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
          assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), square, atol: 1.0e-5)

          assert {q, r} = Nx.LinAlg.qr(tall)
          assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), tall, atol: 1.0e-5)

          assert {q, r} = Nx.LinAlg.qr(wide)
          assert_all_close(Nx.dot(q, [2], [0], r, [1], [0]), wide, atol: 1.0e-5)

          key
      end
    end
  end

  describe "eigh" do
    test "supports batched 1x1 matrices" do
      t = Nx.tensor([[[2.0]], [[5.0]]], type: :f64)

      {eigenvals, eigenvecs} = Nx.LinAlg.eigh(t)

      assert eigenvals == Nx.tensor([[2.0], [5.0]], type: :f64)
      assert eigenvecs == Nx.tensor([[[1.0]], [[1.0]]], type: :f64)

      # double batch
      t2 = Nx.iota({3, 2, 1, 1}, type: :f64) |> Nx.add(1.0)
      {vals2, vecs2} = Nx.LinAlg.eigh(t2)
      assert Nx.shape(vals2) == {3, 2, 1}
      assert Nx.shape(vecs2) == {3, 2, 1, 1}
      assert_all_close(vals2, Nx.reshape(t2, {3, 2, 1}))
    end

    test "batched size-1 dimensions work through svd and pinv" do
      {_u, s, _vt} = Nx.LinAlg.svd(Nx.tensor([[[3.0, 4.0]], [[6.0, 8.0]]], type: :f64))
      assert_all_close(s, Nx.tensor([[5.0], [10.0]], type: :f64))

      {_u2, s2, _vt2} = Nx.LinAlg.svd(Nx.tensor([[[3.0], [4.0]], [[6.0], [8.0]]], type: :f64))
      assert_all_close(s2, Nx.tensor([[5.0], [10.0]], type: :f64))

      pinv = Nx.LinAlg.pinv(Nx.tensor([[[2.0]], [[4.0]]], type: :f64))
      assert_all_close(pinv, Nx.tensor([[[0.5]], [[0.25]]], type: :f64))
    end

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
      assert_all_close(eigenvals, Nx.tensor([16.394, -9.738, 5.901, 4.334, -0.892]),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )

      # Eigenvectors
      assert_all_close(
        eigenvecs,
        Nx.tensor([
          [0.112, 0.004, 0.828, -0.440, -0.328],
          [0.395, -0.163, -0.533, -0.534, -0.497],
          [0.427, -0.326, 0.137, 0.700, -0.452],
          [0.603, 0.783, 0.008, 0.079, 0.130],
          [0.534, -0.504, 0.103, -0.160, 0.651]
        ]),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )
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
               Nx.tensor([
                 Complex.new(-5, 0),
                 Complex.new(3, 0),
                 Complex.new(0.9999998807907104, 0)
               ])

      # Eigenvectors
      assert_all_close(
        eigenvecs,
        ~MAT[
            0.0000-0.4082i 0.7071-0.0i 00.5773-0.0000i
            0.8164-0.0000i 0.0000+0.0i 00.0000-0.5773i
            0.0000+0.4082i 0.7071-0.0i -0.5773-0.0000i
          ],
        atol: 1.0e-3,
        rtol: 1.0e-3
      )
    end

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

          assert_all_close(evals_test, evals[0], atol: 1.0e-7)
          assert_all_close(evals_test, evals[1], atol: 1.0e-7)

          evals =
            evals
            |> Nx.vectorize(:x)
            |> Nx.make_diagonal()
            |> Nx.devectorize(keep_names: false)

          # Eigenvalue equation
          evecs_evals = Nx.dot(evecs, [2], [0], evals, [1], [0])
          a_evecs = Nx.dot(evecs_evals, [2], [0], Nx.LinAlg.adjoint(evecs), [1], [0])

          assert_all_close(a, a_evecs, atol: 1.0e-7)
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

          # Eigenvalues and eigenvectors
          assert {evals, evecs} = Nx.LinAlg.eigh(a)
          assert_all_close(Nx.broadcast(evals_test, Nx.shape(evals)), evals, atol: 0.1)

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

      assert_all_close(t, u |> Nx.dot(s_matrix) |> Nx.dot(v), atol: 1.0e-2, rtol: 1.0e-2)

      # Only the first `rank` columns of U are uniquely determined
      # (columns corresponding to zero singular values span the null space)
      assert_all_close(
        Nx.slice(u, [0, 0], [4, 2]),
        Nx.tensor([
          [0.141, -0.825],
          [0.344, -0.426],
          [0.547, -0.028],
          [0.75, 0.371]
        ]),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )

      assert_all_close(Nx.tensor([25.462, 1.291, 0.0]), s, atol: 1.0e-3, rtol: 1.0e-3)

      assert_all_close(
        Nx.tensor([
          [0.504, 0.575, 0.644],
          [0.761, 0.057, -0.647],
          [-0.408, 0.816, -0.408]
        ]),
        v,
        atol: 1.0e-3,
        rtol: 1.0e-3
      )
    end

    test "finds the singular values of square matrices" do
      t = Nx.iota({5, 5})

      assert {u, s, vt} = Nx.LinAlg.svd(t)

      assert_all_close(Nx.as_type(t, :f32), u |> Nx.multiply(s) |> Nx.dot(vt) |> Nx.abs(),
        atol: 1.0e-2,
        rtol: 1.0e-2
      )
    end

    test "finds the singular values of wide matrices" do
      t = Nx.iota({3, 5})

      assert {u, s, vt} = Nx.LinAlg.svd(t)

      assert_all_close(
        Nx.as_type(t, :f32),
        u |> Nx.dot(t.shape |> Nx.eye() |> Nx.put_diagonal(s)) |> Nx.dot(vt),
        atol: 1.0e-1,
        rtol: 1.0e-1
      )
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

      assert_all_close(t, u |> Nx.dot(s_matrix) |> Nx.dot(v) |> Nx.abs(),
        atol: 1.0e-1,
        rtol: 1.0e-1
      )

      assert_all_close(
        u,
        Nx.tensor([
          [0.335, 0.408, 0.849],
          [0.036, 0.895, -0.445],
          [0.941, -0.18, -0.286]
        ]),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )

      # The expected value is ~ [9, 5, 1] since the eigenvalues of
      # a triangular matrix are the diagonal elements. Close enough!
      assert_all_close(Nx.tensor([9.52, 4.433, 0.853]), s, atol: 1.0e-3, rtol: 1.0e-3)

      assert_all_close(
        Nx.tensor([
          [0.035, 0.0856, 0.996],
          [0.092, 0.992, -0.089],
          [0.995, -0.094, -0.027]
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

    test "works with f16" do
      x = Nx.tensor([[0, 0], [0, 0]], type: :f16)
      assert Nx.LinAlg.svd(x) == {Nx.eye(2, type: :f16), ~VEC"0.0 0.0"f16, Nx.eye(2, type: :f16)}

      x = Nx.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], type: :f16)
      assert {u, s, vt} = Nx.LinAlg.svd(x)
      assert u.type == {:f, 16}
      assert s.type == {:f, 16}
      assert vt.type == {:f, 16}
    end

    test "works with f64" do
      x = Nx.tensor([[0, 0], [0, 0]], type: :f64)
      assert Nx.LinAlg.svd(x) == {Nx.eye(2, type: :f64), ~VEC"0.0 0.0"f64, Nx.eye(2, type: :f64)}

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

          actual = p |> Nx.dot([2], [0], l, [1], [0]) |> Nx.dot([2], [0], u, [1], [0])
          assert_all_close(actual, a)
          key
      end
    end

    test "regression" do
      matrix =
        Nx.tensor([
          [-1.0, 1.0, -1.0, 0.0],
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      {p, l, u} = Nx.LinAlg.lu(matrix)

      assert_all_close(p |> Nx.dot(l) |> Nx.dot(u), matrix)
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

    test "supports multiple batch dimensions of differing sizes" do
      # The zero-branch shape was built from a reversed dimension list and
      # never re-reversed, so unequal batch dims came out transposed.
      key = Nx.Random.key(7)

      for shape <- [{3, 2, 2, 2}, {5, 4, 2, 3}], reduce: key do
        key ->
          {t, key} = Nx.Random.uniform(key, -1, 1, shape: shape, type: :f64)

          rank = tuple_size(shape)
          {rows, cols} = {elem(shape, rank - 2), elem(shape, rank - 1)}
          batch_axes = Enum.to_list(0..(rank - 3))

          pinv = Nx.LinAlg.pinv(t)

          assert Nx.shape(pinv) ==
                   shape |> put_elem(rank - 2, cols) |> put_elem(rank - 1, rows)

          apa =
            t
            |> Nx.dot([rank - 1], batch_axes, pinv, [rank - 2], batch_axes)
            |> Nx.dot([rank - 1], batch_axes, t, [rank - 2], batch_axes)

          assert_all_close(apa, t, atol: 1.0e-3)

          key
      end
    end

    test "supports multiple batch dimensions for the all-zeros branch" do
      zeros = Nx.broadcast(0.0, {3, 2, 1, 2})
      assert Nx.shape(Nx.LinAlg.pinv(zeros)) == {3, 2, 2, 1}
      assert Nx.to_flat_list(Nx.LinAlg.pinv(zeros)) |> Enum.all?(&(&1 == 0.0))
    end

    test "supports batched input" do
      key = Nx.Random.key(42)

      for shape <- [{2, 3, 3}, {2, 3, 4}, {2, 4, 3}, {2, 2, 3, 3}], reduce: key do
        key ->
          {t, key} = Nx.Random.uniform(key, -1, 1, shape: shape, type: :f64)

          pinv = Nx.LinAlg.pinv(t)

          rank = tuple_size(shape)
          {rows, cols} = {elem(shape, rank - 2), elem(shape, rank - 1)}

          expected_shape =
            shape |> Tuple.delete_at(rank - 1) |> Tuple.delete_at(rank - 2)

          assert Nx.shape(pinv) ==
                   expected_shape
                   |> Tuple.insert_at(rank - 2, cols)
                   |> Tuple.insert_at(rank - 1, rows)

          # batched result equals the stacked per-matrix results
          batch_dims = shape |> Tuple.to_list() |> Enum.take(rank - 2)
          flat = Nx.reshape(t, {Enum.product(batch_dims), rows, cols})

          per_matrix =
            for i <- 0..(Enum.product(batch_dims) - 1) do
              Nx.LinAlg.pinv(flat[i])
            end

          assert_all_close(
            Nx.reshape(pinv, {Enum.product(batch_dims), cols, rows}),
            Nx.stack(per_matrix),
            atol: 1.0e-6
          )

          # Moore-Penrose: A P A = A, per batch
          batch_axes = Enum.to_list(0..(rank - 3))

          apa =
            t
            |> Nx.dot([rank - 1], batch_axes, pinv, [rank - 2], batch_axes)
            |> Nx.dot([rank - 1], batch_axes, t, [rank - 2], batch_axes)

          assert_all_close(apa, t, atol: 1.0e-3)

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
          Nx.dot(a, x)
          |> assert_all_close(y, atol: noise_eps * 10)

          key
      end
    end
  end

  # Random n-by-n integer matrices with entries in lo..hi, paired with their
  # exact determinant, so a test knows which ones are singular.
  defp small_integer_matrices(n, lo, hi, count) do
    key = Nx.Random.key(42)
    {ints, _} = Nx.Random.randint(key, lo, hi + 1, shape: {count, n, n})
    for rows <- Nx.to_list(ints), do: {rows, exact_determinant(rows)}
  end

  defp exact_determinant([[a]]), do: a

  defp exact_determinant([first | rest]) do
    first
    |> Enum.with_index()
    |> Enum.map(fn {x, j} ->
      minor = Enum.map(rest, &List.delete_at(&1, j))
      Integer.pow(-1, j) * x * exact_determinant(minor)
    end)
    |> Enum.sum()
  end

  # Random n-by-n matrices with the given condition number in the given type,
  # built as LAPACK's test generator does: random orthogonal factors around
  # singular values spread geometrically from 1 to 1 / cond (Stewart, SIAM J.
  # Numer. Anal. 17(3), 1980, doi:10.1137/0717034).
  defp conditioned_matrices(n, cond, count, type) do
    key = Nx.Random.key(42)
    {left, key} = Nx.Random.normal(key, shape: {count, n, n}, type: :f64)
    {right, _} = Nx.Random.normal(key, shape: {count, n, n}, type: :f64)
    {u, _} = Nx.LinAlg.qr(left)
    {v, _} = Nx.LinAlg.qr(right)
    ratios = for i <- 0..(n - 1), do: cond ** (-i / max(n - 1, 1))
    singular_values = Nx.tensor(ratios)

    u
    |> Nx.multiply(singular_values)
    |> Nx.dot([2], [0], v, [2], [0])
    |> Nx.as_type(type)
  end

  # The Sylvester Hadamard matrix of order n, a power of two, with entries of
  # 1 and -1: each order doubles the previous one into four blocks, the last
  # negated.
  defp hadamard(n, type) do
    n
    |> hadamard_rows()
    |> Nx.tensor(type: type)
  end

  defp hadamard_rows(1), do: [[1]]

  defp hadamard_rows(n) do
    half = hadamard_rows(div(n, 2))
    top = Enum.map(half, &(&1 ++ &1))
    bottom = Enum.map(half, fn row -> row ++ Enum.map(row, &(-&1)) end)
    top ++ bottom
  end

  # The Wilkinson matrix: ones on the diagonal and in the last column, minus
  # ones below the diagonal, zeros elsewhere.
  defp wilkinson(n, type) do
    entries =
      for i <- 1..n, j <- 1..n do
        cond do
          j == n -> 1
          i == j -> 1
          j < i -> -1
          true -> 0
        end
      end

    entries
    |> Nx.tensor(type: type)
    |> Nx.reshape({n, n})
  end

  # The backward-error style residual |product - expected| / (|a| |x| + |expected|),
  # in the max norm and in f64, so it means the same thing for every type
  defp relative_residual(product, expected, a, x) do
    norm = fn t ->
      t
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()
    end

    product_f64 = Nx.as_type(product, :f64)
    expected_f64 = Nx.as_type(expected, :f64)
    residual = Nx.subtract(product_f64, expected_f64)
    norm.(residual) / (norm.(a) * norm.(x) + norm.(expected))
  end
end
