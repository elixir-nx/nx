defmodule Torchx.NxLinAlgTest do
  use Torchx.Case, async: true

  describe "matrix_power" do
    test "integers" do
      assert_all_close(
        Nx.LinAlg.matrix_power(Nx.tensor([[1, 2], [3, 4]]), 0),
        Nx.tensor([
          [1, 0],
          [0, 1]
        ])
      )

      assert_all_close(
        Nx.LinAlg.matrix_power(Nx.tensor([[1, 2], [3, 4]]), 6),
        Nx.tensor([
          [5743, 8370],
          [12555, 18298]
        ])
      )

      assert_all_close(
        Nx.LinAlg.matrix_power(Nx.eye(3), 65535),
        Nx.tensor([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ])
      )
    end

    test "floats" do
      assert_all_close(
        Nx.LinAlg.matrix_power(Nx.tensor([[1, 2], [3, 4]]), -1),
        Nx.tensor([
          [-2.0, 1.0],
          [1.5, -0.5]
        ])
      )
    end
  end

  describe "triangular_solve" do
    test "base case 1D (s64)" do
      a = Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])

      result = Nx.LinAlg.triangular_solve(a, Nx.tensor([4, 2, 4, 2]))

      assert_all_close(result, Nx.tensor([1.33333337, -0.6666666, 2.6666667, -1.33333]))
    end

    test "base case 1D (f64)" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      %{type: {:f, 64}} = result = Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]))
      assert_all_close(result, Nx.tensor([1.0, 1.0, -1.0]))
    end

    test "base case 2D" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1, 2, 3], [2, 2, 4], [2, 0, 1]])
      result = Nx.LinAlg.triangular_solve(a, b)

      expected =
        Nx.tensor([
          [1.0, 2.0, 3.0],
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.0]
        ])

      assert_all_close(result, expected)
    end

    test "lower: false" do
      a = Nx.tensor([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 0, 3]])
      b = Nx.tensor([2, 4, 2, 4])
      result = Nx.LinAlg.triangular_solve(a, b, lower: false)

      expected =
        Nx.tensor([
          -1.3333,
          2.66666,
          -0.6666,
          1.33333
        ])

      assert_all_close(result, expected)
    end

    test "left_side: false" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      b = Nx.tensor([[0, 2, 1], [1, 1, 0], [3, 3, 1]])

      assert_raise ArgumentError, "left_side: false option not supported in Torchx", fn ->
        Nx.LinAlg.triangular_solve(a, b, left_side: false)
      end
    end

    test "transform_a: :transpose" do
      a = Nx.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], type: {:f, 64})
      b = Nx.tensor([1, 2, 1])
      result = Nx.LinAlg.triangular_solve(a, b, transform_a: :transpose, lower: false)

      assert_all_close(result, Nx.tensor([1.0, 1.0, -1.0]))
    end

    test "explicit transform_a: :none" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      b = Nx.tensor([1, 2, 1])
      result = Nx.LinAlg.triangular_solve(a, b, transform_a: :none)

      assert_all_close(result, Nx.tensor([1.0, 1.0, -1.0]))
    end

    test "explicit left_side: true" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
      b = Nx.tensor([[0, 2], [3, 0], [0, 0]])
      result = Nx.LinAlg.triangular_solve(a, b, left_side: true)

      assert_all_close(
        result,
        Nx.tensor([
          [0.0, 2.0],
          [3.0, -2.0],
          [-6.0, 2.0]
        ])
      )
    end

    test "validates transform_a" do
      assert_raise ArgumentError,
                   "invalid value for :transform_a option, expected :none, :transpose, or :conjugate, got: :other",
                   fn ->
                     a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
                     Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]), transform_a: :other)
                   end
    end
  end

  describe "solve" do
    test "base case 1D (s64)" do
      a = Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])

      result = Nx.LinAlg.solve(a, Nx.tensor([4, 2, 4, 2]))

      assert_all_close(result, Nx.tensor([1.33333337, -0.6666666, 2.6666667, -1.33333]))
    end

    test "base case 1D (f64)" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
      %{type: {:f, 64}} = result = Nx.LinAlg.solve(a, Nx.tensor([1, 2, 1]))
      assert_all_close(result, Nx.tensor([1.0, 1.0, -1.0]))
    end

    test "base case 2D" do
      a = Nx.tensor([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
      b = Nx.tensor([[1, 2, 3], [2, 2, 4], [2, 0, 1]])
      result = Nx.LinAlg.solve(a, b)

      expected =
        Nx.tensor([
          [1.0, 2.0, 3.0],
          [1.0, 0.0, 1.0],
          [1.0, 0.0, 0.0]
        ])

      assert_all_close(result, expected)
    end
  end

  describe "invert" do
    test "works for matrix" do
      a = Nx.tensor([[1, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
      a_inv = Nx.LinAlg.invert(a)

      assert_all_close(Nx.dot(a, a_inv), Nx.eye(Nx.shape(a)))
      assert_all_close(Nx.dot(a_inv, a), Nx.eye(Nx.shape(a)))
    end

    test "fails silently for singular matrix" do
      assert_equal(
        Nx.LinAlg.invert(Nx.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])),
        Nx.broadcast(:nan, {4, 4})
      )
    end
  end

  describe "qr" do
    test "property" do
      for _ <- 1..10, type <- [{:f, 32}, {:c, 64}] do
        square = random_uniform({4, 4}, type: type)
        tall = random_uniform({4, 3}, type: type)
        wide = random_uniform({3, 4}, type: type)

        assert {q, r} = Nx.LinAlg.qr(square)
        assert_all_close(Nx.dot(q, r), square, rtol: 1.0e-2)

        assert {q, r} = Nx.LinAlg.qr(tall)
        assert_all_close(Nx.dot(q, r), tall, rtol: 1.0e-2)

        assert {q, r} = Nx.LinAlg.qr(wide)
        assert_all_close(Nx.dot(q, r), wide, rtol: 1.0e-2)
      end
    end

    test "rectangular matrix" do
      t = Nx.tensor([[1.0, -1.0, 4.0], [1.0, 4.0, -2.0], [1.0, 4.0, 2.0], [1.0, -1.0, 0.0]])
      {q, r} = Nx.LinAlg.qr(t, mode: :reduced)
      assert_all_close(Nx.dot(q, r), t, atol: 1.0e-6)
    end
  end

  describe "lu" do
    test "property" do
      for _ <- 1..20, type <- [{:f, 32}, {:c, 64}] do
        a = random_uniform({3, 3}, type: type)
        {p, l, u} = Nx.LinAlg.lu(a)

        a_reconstructed = p |> Nx.dot(l) |> Nx.dot(u)

        assert_all_close(a, a_reconstructed)
      end
    end
  end

  describe "eigh" do
    test "property" do
      for _ <- 1..20 do
        a = random_uniform({3, 3}) |> then(&Nx.add(&1, Nx.transpose(&1)))

        {eigenval, eigenvec} = Nx.LinAlg.eigh(a)

        a_reconstructed = eigenvec |> Nx.multiply(eigenval) |> Nx.dot(eigenvec |> Nx.transpose())

        assert_all_close(a, a_reconstructed)
      end
    end
  end

  describe "svd" do
    test "factors square matrix" do
      t = Nx.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, -1]])
      {u, s, vt} = Nx.LinAlg.svd(t)

      assert_all_close(u, Nx.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
      assert_all_close(s, Nx.tensor([1, 1, 1]))

      assert_all_close(
        vt,
        Nx.tensor([
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, -1.0]
        ])
      )

      assert_all_close(t, u |> Nx.multiply(s) |> Nx.dot(vt))
    end

    test "factors tall matrix" do
      t = Nx.tensor([[2.0, 0, 0], [0, 3, 0], [0, 0, -1], [0, 0, 0]])
      {u, s, vt} = Nx.LinAlg.svd(t)

      assert_all_close(u, Nx.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
      assert_all_close(s, Nx.tensor([3, 2, 1]))

      assert_all_close(
        vt,
        Nx.tensor([
          [0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.0, -1.0]
        ])
      )

      eye = {4, 4} |> Nx.eye() |> Nx.slice([0, 0], [4, 3])

      s_full = Nx.multiply(eye, s)

      assert_all_close(t, u |> Nx.dot(s_full) |> Nx.dot(vt))
    end
  end

  describe "determinant" do
    test "works for 2x2" do
      assert_all_close(
        Nx.LinAlg.determinant(Nx.tensor([[1, 2], [3, 4]])),
        Nx.tensor(-2)
      )
    end

    test "works for 3x3" do
      assert_all_close(
        Nx.LinAlg.determinant(Nx.tensor([[1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [7.0, 8.0, 9.0]])),
        Nx.tensor(48)
      )
    end

    test "linearly dependent rows/cols return 0" do
      assert_all_close(
        Nx.LinAlg.determinant(Nx.tensor([[1.0, 0.0], [3.0, 0.0]])),
        Nx.tensor(0)
      )

      assert_all_close(
        Nx.LinAlg.determinant(Nx.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]])),
        Nx.tensor(0)
      )
    end

    test "works for order bigger than 3" do
      assert_all_close(
        Nx.LinAlg.determinant(
          Nx.tensor([
            [1, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 1, -2, 3],
            [0, 7, 8, 9.0]
          ])
        ),
        Nx.tensor(48.0)
      )

      assert_all_close(
        Nx.LinAlg.determinant(
          Nx.tensor([
            [0, 0, 0, 0, -1.0],
            [0, 1, 2, 3, 0],
            [0, 1, -2, 3, 0],
            [0, 7, 8, 9, 0],
            [1, 0, 0, 0, 0]
          ])
        ),
        Nx.tensor(48)
      )
    end
  end

  describe "eig" do
    test "computes eigenvalues and eigenvectors for diagonal matrix" do
      t = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
      assert {eigenvals, _eigenvecs} = Nx.LinAlg.eig(t)
      expected = Nx.tensor([3.0, 2.0, 1.0]) |> Nx.as_type({:c, 64})
      assert_all_close(Nx.abs(eigenvals), Nx.abs(expected), atol: 1.0e-2)
    end

    test "computes eigenvalues and eigenvectors for upper triangular matrix" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]])
      assert {eigenvals, eigenvecs} = Nx.LinAlg.eig(t)
      expected = Nx.tensor([6.0, 4.0, 1.0])
      assert_all_close(Nx.abs(eigenvals), Nx.abs(expected), atol: 1.0e-2)

      assert_all_close(Nx.dot(t, eigenvecs), Nx.dot(eigenvecs, Nx.make_diagonal(eigenvals)),
        atol: 1.0e-2
      )
    end

    test "computes eigenvalues and eigenvectors for lower triangular matrix" do
      t = Nx.tensor([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
      assert {eigenvals, eigenvecs} = Nx.LinAlg.eig(t)
      expected = Nx.tensor([6.0, 3.0, 1.0])
      assert_all_close(Nx.abs(eigenvals), Nx.abs(expected), atol: 1.0e-2)

      assert_all_close(Nx.dot(t, eigenvecs), Nx.dot(eigenvecs, Nx.make_diagonal(eigenvals)),
        atol: 1.0e-2
      )
    end

    test "computes complex eigenvalues for rotation matrix" do
      t = Nx.tensor([[0.0, -1.0], [1.0, 0.0]])
      assert {eigenvals, _eigenvecs} = Nx.LinAlg.eig(t)
      assert_all_close(Nx.abs(eigenvals), Nx.tensor([1.0, 1.0]), atol: 1.0e-3)
      assert_all_close(Nx.sum(Nx.imag(eigenvals)), Nx.tensor(0.0), atol: 1.0e-3)
    end

    test "works with batched matrices" do
      t = Nx.tensor([[[1.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 4.0]]])
      assert {eigenvals, _eigenvecs} = Nx.LinAlg.eig(t)
      expected = Nx.tensor([[2.0, 1.0], [4.0, 3.0]])
      assert_all_close(Nx.abs(eigenvals), expected, atol: 1.0e-3)
    end

    test "works with vectorized matrices" do
      t =
        Nx.tensor([
          [[[1.0, 0.0], [0.0, 2.0]]],
          [[[3.0, 0.0], [0.0, 4.0]]]
        ])
        |> Nx.vectorize(x: 2, y: 1)

      assert {eigenvals, eigenvecs} = Nx.LinAlg.eig(t)
      assert eigenvals.vectorized_axes == [x: 2, y: 1]
      assert eigenvecs.vectorized_axes == [x: 2, y: 1]

      eigenvals = Nx.devectorize(eigenvals)
      assert_all_close(Nx.abs(eigenvals[0][0]), Nx.tensor([2.0, 1.0]), atol: 1.0e-3)
      assert_all_close(Nx.abs(eigenvals[1][0]), Nx.tensor([4.0, 3.0]), atol: 1.0e-3)

      eigenvecs_dev = Nx.devectorize(eigenvecs)

      for batch <- 0..1, col <- 0..1 do
        v = eigenvecs_dev[batch][0][[.., col]]
        norm = Nx.LinAlg.norm(v) |> Nx.to_number()
        assert_in_delta(norm, 1.0, 0.1)
      end
    end

    test "property: eigenvalue equation A*v = λ*v" do
      key = Nx.Random.key(System.unique_integer())

      for _ <- 1..3, type <- [{:f, 32}, {:f, 64}], reduce: key do
        key ->
          {base_q, key} = Nx.Random.uniform(key, -2, 2, shape: {2, 3, 3}, type: :f32)

          {q, _} = Nx.LinAlg.qr(base_q)

          evals_test =
            [10, 1, 0.1]
            |> Enum.map(fn magnitude ->
              sign = if :rand.uniform() - 0.5 > 0, do: 1, else: -1
              rand = :rand.uniform() * magnitude * 0.1 + magnitude
              rand * sign
            end)
            |> Nx.tensor(type: type)

          evals_test_diag =
            evals_test
            |> Nx.make_diagonal()
            |> Nx.reshape({1, 3, 3})
            |> Nx.tile([2, 1, 1])

          q_adj = Nx.LinAlg.adjoint(q)

          a =
            q
            |> Nx.dot([2], [0], evals_test_diag, [1], [0])
            |> Nx.dot([2], [0], q_adj, [1], [0])

          assert {eigenvals, eigenvecs} = Nx.LinAlg.eig(a, balance: 0)

          evals =
            eigenvals
            |> Nx.vectorize(x: 2)
            |> Nx.make_diagonal()
            |> Nx.devectorize(keep_names: false)

          assert_all_close(
            Nx.dot(eigenvecs, [-1], [0], evals, [-2], [0]),
            Nx.dot(a, [-1], [0], eigenvecs, [-2], [0]),
            atol: 1.0e-3
          )

          key
      end
    end
  end

  defp random_uniform(shape, opts \\ [type: :f32]) do
    values = Enum.map(1..Tuple.product(shape), fn _ -> :rand.uniform() end)

    values
    |> Nx.tensor(opts)
    |> Nx.reshape(shape)
  end
end
