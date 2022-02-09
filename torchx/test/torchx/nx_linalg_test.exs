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

    test "invalid a shape" do
      assert_raise ArgumentError, "expected a square matrix, got matrix with shape: {2, 4}", fn ->
        Nx.LinAlg.triangular_solve(
          Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0]]),
          Nx.tensor([4, 2, 4, 2])
        )
      end
    end

    test "incompatible dims" do
      assert_raise ArgumentError, "incompatible dimensions for a and b on triangular solve", fn ->
        Nx.LinAlg.triangular_solve(
          Nx.tensor([[3, 0, 0, 0], [2, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]),
          Nx.tensor([4])
        )
      end
    end

    test "singular matrix" do
      assert_raise ArgumentError, "can't solve for singular matrix", fn ->
        Nx.LinAlg.triangular_solve(
          Nx.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]),
          Nx.tensor([4, 2, 4, 2])
        )
      end
    end

    test "complex numbers not supported" do
      assert_raise ArgumentError, "complex numbers not supported yet", fn ->
        a = Nx.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], type: {:f, 64})
        Nx.LinAlg.triangular_solve(a, Nx.tensor([1, 2, 1]), transform_a: :conjugate)
      end
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

  describe "qr" do
    test "property" do
      for _ <- 1..10 do
        square = Nx.random_uniform({4, 4})
        tall = Nx.random_uniform({4, 3})

        assert {q, r} = Nx.LinAlg.qr(square)
        assert_all_close(Nx.dot(q, r), square, atol: 1.0e-6)

        assert {q, r} = Nx.LinAlg.qr(tall)
        assert_all_close(Nx.dot(q, r), tall, atol: 1.0e-6)
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
      for _ <- 1..20 do
        a = Nx.random_uniform({3, 3})
        {p, l, u} = Nx.LinAlg.lu(a)

        a_reconstructed = p |> Nx.dot(l) |> Nx.dot(u)

        assert_all_close(a, a_reconstructed)
      end
    end

    test "invalid a shape" do
      assert_raise ArgumentError,
                   "tensor must have as many rows as columns, got shape: {3, 4}",
                   fn ->
                     Nx.LinAlg.lu(Nx.tensor([[1, 1, 1, 1], [-1, 4, 4, -1], [4, -2, 2, 0]]))
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

    # TO-DO: deal with `Nx.sum` returning s64 on Torchx instead of u64
    @tag :skip
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
        Nx.tensor(-48)
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

    test "eigh" do
      a =
        Nx.tensor([
          [0, 1, 2],
          [1, 0, 2],
          [2, 2, 3]
        ])

      {values, vectors} = Nx.LinAlg.eigh(a)
      assert_all_close(values, Nx.tensor([-1.0, -1.0, 5.0]))

      assert_all_close(
        vectors,
        Nx.tensor([
          [-0.796406, -0.44617220, -0.408248],
          [0.596439, -0.6910815, -0.408248],
          [0.099983, 0.568626, -0.816496]
        ])
      )
    end
  end
end
