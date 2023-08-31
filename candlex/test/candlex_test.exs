defmodule CandlexTest do
  use Candlex.Case, async: true
  doctest Candlex

  describe "creation" do
    test "tensor" do
      check(255, type: :u8)
      check(100_002, type: :u32)
      check(-101, type: :s64)
      check(1.16, type: :f16)
      check(1.32, type: :f32)
      check([1, 2, 3], type: :f32)
      check(-0.002, type: :f64)
      check([1, 2], type: :u32)
      check([[1, 2], [3, 4]], type: :u32)
      check([[1, 2, 3, 4], [5, 6, 7, 8]], type: :u32)
      check([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], type: :u32)
      check([0, 255], type: :u8)
      check([-0.5, 0.88], type: :f32)
      check([-0.5, 0.88], type: :f64)
      check(2.16, type: :bf16)
    end

    # test "gpu" do
    #   t([1, 2, 3], backend: {Candlex.Backend, device: :cuda})
    #   |> assert_equal(t([1, 2, 3]))
    # end

    test "named dimensions" do
      check([[1, 2, 3], [4, 5, 6]], names: [:x, :y])

      t([[1, 2, 3], [4, 5, 6]], names: [:x, :y])
      |> assert_equal(t([[1, 2, 3], [4, 5, 6]]))
    end

    test "tensor tensor" do
      t(t([1, 2, 3]))
      |> assert_equal(t([1, 2, 3]))
    end

    test "tril" do
      t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      |> Nx.tril()
      |> assert_equal(t([[1, 0, 0], [4, 5, 0], [7, 8, 9]]))
    end

    test "triu" do
      t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      |> Nx.triu()
      |> assert_equal(t([[1, 2, 3], [0, 5, 6], [0, 0, 9]]))
    end

    test "addition" do
      t([1, 2, 3])
      |> Nx.add(t([10, 20, 30]))
      |> assert_equal(t([11, 22, 33]))

      Nx.add(1, 2.2)
      |> assert_equal(t(3.2))

      t([1, 2, 3])
      |> Nx.add(1.0)
      |> assert_equal(t([2.0, 3.0, 4.0]))
    end

    test "iota" do
      Nx.iota({})
      |> assert_equal(t(0))

      Nx.iota({}, type: :f32)
      |> assert_equal(t(0.0))

      Nx.iota({5})
      |> assert_equal(t([0, 1, 2, 3, 4]))

      Nx.iota({5}, type: :f32)
      |> assert_equal(t([0.0, 1.0, 2.0, 3.0, 4.0]))

      Nx.iota({2, 3})
      |> assert_equal(t([[0, 1, 2], [3, 4, 5]]))
    end

    test "max" do
      Nx.max(1, 2)
      |> assert_equal(t(2))

      Nx.max(1, t([1.0, 2.0, 3.0], names: [:data]))
      |> assert_equal(t([1.0, 2.0, 3.0]))

      t([[1], [2]], type: :f32, names: [:x, nil])
      |> Nx.max(t([[10, 20]], type: :f32, names: [nil, :y]))
      |> assert_equal(t([[10.0, 20.0], [10.0, 20.0]]))
    end

    test "min" do
      Nx.min(1, 2)
      |> assert_equal(t(1))

      Nx.min(1, t([1.0, 2.0, 3.0], names: [:data]))
      |> assert_equal(t([1.0, 1.0, 1.0]))

      t([[1], [2]], type: :f32, names: [:x, nil])
      |> Nx.min(t([[10, 20]], type: :f32, names: [nil, :y]))
      |> assert_equal(t([[1.0, 1.0], [2.0, 2.0]]))
    end

    test "multiply" do
      t([1, 2])
      |> Nx.multiply(t([3, 4]))
      |> assert_equal(t([3, 8]))

      t([[1], [2]])
      |> Nx.multiply(t([3, 4]))
      |> assert_equal(t([[3, 4], [6, 8]]))

      t([1, 2])
      |> Nx.multiply(t([[3], [4]]))
      |> assert_equal(t([[3, 6], [4, 8]]))
    end

    test "broadcast" do
      Nx.broadcast(1, {1, 2, 3})
      |> assert_equal(t([[[1, 1, 1], [1, 1, 1]]]))

      t([1, 2, 3])
      |> Nx.broadcast({3, 2}, axes: [0])
      |> assert_equal(t([[1, 1], [2, 2], [3, 3]]))
    end

    test "access" do
      tensor = t([[1, 2], [3, 4]])

      assert_equal(tensor[0], t([1, 2]))
      assert_equal(tensor[1], t([3, 4]))
    end

    test "concatenate" do
      [t([1, 2, 3])]
      |> Nx.concatenate()
      |> assert_equal(t([1, 2, 3]))

      [t([1, 2, 3]), t([4, 5, 6])]
      |> Nx.concatenate()
      |> assert_equal(t([1, 2, 3, 4, 5, 6]))

      t1 = Nx.iota({2, 2, 2}, names: [:x, :y, :z], type: :f32)
      t2 = Nx.iota({1, 2, 2}, names: [:x, :y, :z], type: :u8)
      t3 = Nx.iota({1, 2, 2}, names: [:x, :y, :z], type: :s64)

      [t1, t2, t3]
      |> Nx.concatenate(axis: :x)
      |> assert_equal(
        t([
          [
            [0.0, 1.0],
            [2.0, 3.0]
          ],
          [
            [4.0, 5.0],
            [6.0, 7.0]
          ],
          [
            [0.0, 1.0],
            [2.0, 3.0]
          ],
          [
            [0.0, 1.0],
            [2.0, 3.0]
          ]
        ])
      )
    end

    # test "bitwise_and" do
    #   Nx.bitwise_and(1, 0)
    #   |> assert_equal(t(0))

    #   t([0, 1, 2])
    #   |> Nx.bitwise_and(1)
    #   |> assert_equal(t([0, 1, 0]))

    #   t([0, -1, -2])
    #   |> Nx.bitwise_and(-1)
    #   |> assert_equal(t([0, -1, -2]))

    #   t([0, 0, 1, 1])
    #   |> Nx.bitwise_and(t([0, 1, 0, 1]))
    #   |> assert_equal(t([0, 0, 0, 1]))
    # end

    # test "bitwise_not" do
    #   Nx.bitwise_not(1)
    #   |> assert_equal(t(-2))
    # end

    # test "bitwise_or" do
    #   Nx.bitwise_or(1, 0)
    #   |> assert_equal(t(1))

    #   t([0, 1, 2])
    #   |> Nx.bitwise_or(1)
    #   |> assert_equal(t([1, 1, 3]))

    #   t([0, -1, -2])
    #   |> Nx.bitwise_or(-1)
    #   |> assert_equal(t([-1, -1, -1]))

    #   t([0, 0, 1, 1])
    #   |> Nx.bitwise_or(t([0, 1, 0, 1]))
    #   |> assert_equal(t([0, 1, 1, 1]))
    # end

    # test "bitwise_xor" do
    #   Nx.bitwise_xor(1, 0)
    #   |> assert_equal(t(1))

    #   t([1, 2, 3])
    #   |> Nx.bitwise_xor(2)
    #   |> assert_equal(t([3, 0, 1]))

    #   t([-1, -2, -3])
    #   |> Nx.bitwise_xor(2)
    #   |> assert_equal(t([-3, -4, -1]))

    #   t([0, 0, 1, 1])
    #   |> Nx.bitwise_xor(t([0, 1, 0, 1]))
    #   |> assert_equal(t([0, 1, 1, 0]))
    # end

    test "less" do
      Nx.less(1, 2)
      |> assert_equal(t(1))

      Nx.less(1, t([1, 2, 3]))
      |> assert_equal(t([0, 1, 1]))

      t([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]])
      |> Nx.less(t([1, 2, 3]))
      |> assert_equal(t([[0, 0, 0], [0, 0, 1]]))
    end

    test "less_equal" do
      Nx.less_equal(1, 2)
      |> assert_equal(t(1))

      Nx.less_equal(1, t([1, 2, 3]))
      |> assert_equal(t([1, 1, 1]))

      t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      |> Nx.less_equal(t([1, 2, 3]))
      |> assert_equal(t([[1, 1, 1], [0, 0, 0]]))
    end

    # test "left_shift" do
    #   Nx.left_shift(1, 0)
    #   |> assert_equal(t(1))

    #   t([1, 2, 3])
    #   |> Nx.left_shift(2)
    #   |> assert_equal(t([4, 8, 12]))

    #   t([1, 1, -1, -1])
    #   |> Nx.left_shift(t([1, 2, 3, 4]))
    #   |> assert_equal(t([2, 4, -8, -16]))
    # end

    # test "right_shift" do
    #   Nx.right_shift(1, 0)
    #   |> assert_equal(t(1))

    #   t([2, 4, 8])
    #   |> Nx.right_shift(2)
    #   |> assert_equal(t([0, 1, 2]))

    #   t([16, 32, -64, -128])
    #   |> Nx.right_shift(t([1, 2, 3, 4]))
    #   |> assert_equal(t([8, 8, -8, -8]))
    # end

    test "bitcast" do
      t([0, 0, 0], type: :s64)
      |> Nx.bitcast(:f64)
      |> assert_equal(t([0.0, 0.0, 0.0]))
    end

    # test "erf_inv" do
    #   Nx.erf_inv(0.10000000149011612)
    #   |> assert_equal(t(0.08885598927736282))

    #   t([0.10000000149011612, 0.5, 0.8999999761581421])
    #   |> Nx.erf_inv()
    #   |> assert_equal(t([0.08885598927736282, 0.4769362807273865, 1.163087010383606]))
    # end

    test "eye" do
      Nx.eye(2)
      |> assert_equal(t([[1, 0], [0, 1]]))

      Nx.eye(3, type: :f32)
      |> assert_equal(
        t([
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]
        ])
      )

      Nx.eye({1, 2})
      |> assert_equal(t([[1, 0]]))

      Nx.eye({2, 4, 3})
      |> assert_equal(
        t([
          [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
          ],
          [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
          ]
        ])
      )

      # assert_equal doesn't yet work with vectorized axes
      # Nx.eye({3}, vectorized_axes: [x: 1, y: 2])
      # |> assert_equal(t(
      #   [
      #     [
      #       [1, 0, 0],
      #       [1, 0, 0]
      #     ]
      #   ]
      # ))

      # Nx.eye({2, 3}, vectorized_axes: [x: 2])
      # |> assert_equal(t(
      #   [
      #     [
      #       [1, 0, 0],
      #       [0, 1, 0]
      #     ],
      #     [
      #       [1, 0, 0],
      #       [0, 1, 0]
      #     ]
      #   ]
      # ))
    end

    test "dot" do
      Nx.dot(5, 5)
      |> assert_equal(t(25))

      Nx.dot(-2.0, 5.0)
      |> assert_equal(t(-10.0))

      Nx.dot(2, 2.0)
      |> assert_equal(t(4.0))

      # TODO:
      # t([1, 2, 3])
      # |> Nx.dot(t([4, 5, 6]))
      # |> assert_equal(t(32))

      # t([1.0, 2.0, 3.0])
      # |> Nx.dot(t([1, 2, 3]))
      # |> assert_equal(t(14.0))

      # t([[1, 2, 3], [4, 5, 6]])
      # |> Nx.dot(t([[7, 8], [9, 10], [11, 12]]))
      # |> assert_equal(t(
      #   [
      #     [58, 64],
      #     [139, 154]
      #   ]
      # ))
    end

    test "negate" do
      # TODO: candle doesn't support unary functions for integers yet
      # Nx.negate(1)
      # |> assert_equal(t(-1))

      Nx.negate(1.0)
      |> assert_equal(t(-1.0))

      t([1.0, 2.0, -3.0], type: :f32)
      |> Nx.negate()
      |> assert_equal(t([-1.0, -2.0, 3.0]))
    end
  end

  defp t(values, opts \\ []) do
    opts =
      [backend: Candlex.Backend]
      |> Keyword.merge(opts)

    Nx.tensor(values, opts)
  end

  defp check(value, opts \\ []) do
    tensor = t(value, opts)

    tensor
    # |> IO.inspect()
    |> Nx.to_binary()
    # |> IO.inspect()

    opts =
      [backend: Nx.BinaryBackend]
      |> Keyword.merge(opts)

    assert Nx.backend_copy(tensor) == t(value, opts)
    assert Nx.backend_transfer(tensor) == t(value, opts)
  end
end
