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

    test "divide/2" do
      1.0
      |> Nx.divide(2)
      |> assert_equal(t(0.5))

      t([1.0, 2, 3])
      |> Nx.divide(1)
      |> assert_equal(t([1.0, 2.0, 3.0]))

      t([[1.0], [2]])
      |> Nx.divide(t([[10, 20]]))
      |> assert_equal(t(
        [
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ]
      ))

      # TODO: Support integers
      # 1
      # |> Nx.divide(2)
      # |> assert_equal(t(0.5))

      # t([1, 2, 3])
      # |> Nx.divide(1)
      # |> assert_equal(t([1.0, 2.0, 3.0]))

      # t([[1], [2]])
      # |> Nx.divide(t([[10, 20]]))
      # |> assert_equal(t(
      #   [
      #     [0.10000000149011612, 0.05000000074505806],
      #     [0.20000000298023224, 0.10000000149011612]
      #   ]
      # ))
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

    test "bitcast" do
      t([0, 0, 0], type: :s64)
      |> Nx.bitcast(:f64)
      |> assert_equal(t([0.0, 0.0, 0.0]))

      t([0, 0, 0], type: :u32)
      |> Nx.bitcast(:f32)
      |> assert_equal(t([0.0, 0.0, 0.0]))

      t([0, 0, 0], type: :u32)
      |> Nx.bitcast(:u32)
      |> assert_equal(t([0, 0, 0]))
    end

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

    test "dot/2" do
      # Dot product of scalars

      Nx.dot(5, 5)
      |> assert_equal(t(25))

      Nx.dot(-2.0, 5.0)
      |> assert_equal(t(-10.0))

      Nx.dot(2, 2.0)
      |> assert_equal(t(4.0))

      # Dot product of vectors

      # TODO:
      # t([1, 2, 3])
      # |> Nx.dot(t([4, 5, 6]))
      # |> assert_equal(t(32))

      # t([1.0, 2.0, 3.0])
      # |> Nx.dot(t([1, 2, 3]))
      # |> assert_equal(t(14.0))

      # Dot product of matrices (2-D tensors)

      # TODO: Candle matmul doesn't support integers yet
      # t([[1, 2, 3], [4, 5, 6]])
      # |> Nx.dot(t([[7, 8], [9, 10], [11, 12]]))
      # |> assert_equal(t(
      #   [
      #     [58, 64],
      #     [139, 154]
      #   ]
      # ))

      t([[1.0, 2, 3], [4, 5, 6]])
      |> Nx.dot(t([[7.0, 8], [9, 10], [11, 12]]))
      |> assert_equal(t(
        [
          [58.0, 64],
          [139, 154]
        ]
      ))

      # Dot product of vector and n-D tensor

      # t([[[1.0, 2], [3, 4]], [[5, 6], [7, 8]]], names: [:i, :j, :k])
      # |> Nx.dot(t([5.0, 10], names: [:x]))
      # |> assert_equal(t(
      #   [
      #     [25, 55],
      #     [85, 115]
      #   ]
      # ))

      # t([5.0, 10], names: [:x])
      # |> Nx.dot(t([[1.0, 2, 3], [4, 5, 6]], names: [:i, :j]))
      # |> assert_equal(t(
      #   [45, 60, 75]
      # ))

      # t([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], names: [:shard, :batch, :x, :y, :z])
      # |> Nx.dot(t([2.0, 2.0], names: [:data]))
      # |> assert_equal(t(
      #   [
      #     [
      #       [
      #         [6.0, 14.0],
      #         [22.0, 30.0]
      #       ]
      #     ]
      #   ]
      # ))

      # Dot product of n-D and m-D tensors

      # t([[[1.0, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:x, :y, :z])
      # |> Nx.dot(t([[[1.0, 2, 3], [3, 4, 5], [5, 6, 7]]], names: [:i, :j, :k]))
      # |> assert_equal(t(
      #   [
      #     [
      #       [
      #         [22, 28, 34]
      #       ],
      #       [
      #         [49, 64, 79]
      #       ],
      #       [
      #         [76, 100, 124]
      #       ]
      #     ],
      #     [
      #       [
      #         [22, 28, 34]
      #       ],
      #       [
      #         [49, 64, 79]
      #       ],
      #       [
      #         [76, 100, 124]
      #       ]
      #     ]
      #   ]
      # ))
    end

    test "dot/6" do
      # Contracting along axes

      t1 = t([[1.0, 2], [3, 4]], names: [:x, :y])
      t2 = t([[10.0, 20], [30, 40]], names: [:height, :width])

      t1
      |> Nx.dot([0], [], t2, [0], [])
      |> assert_equal(t(
        [
          [100, 140],
          [140, 200]
        ]
      ))

      # TODO:
      t1
      |> Nx.dot([0], [], t2, [1], [])
      |> assert_equal(t(
        [
          [70, 150],
          [100, 220]
        ]
      ))

      t1
      |> Nx.dot([1], [], t2, [0], [])
      |> assert_equal(t(
        [
          [70, 100],
          [150, 220]
        ]
      ))

      # t1
      # |> Nx.dot([1], [], t2, [1], [])
      # |> assert_equal(t(
      #   [
      #     [50, 110],
      #     [110, 250]
      #   ]
      # ))

      # t1
      # |> Nx.dot([0, 1], [], t2, [0, 1], [])
      # |> assert_equal(t(300))
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

    test "sin" do
      Nx.sin(1.0)
      |> assert_equal(t(0.8414709568023682))

      t([1.0, 2.0, 3.0])
      |> Nx.sin()
      |> assert_equal(t([0.8414709568023682, 0.9092974066734314, 0.14112000167369843]))
    end

    test "exp" do
      Nx.exp(1.0)
      |> assert_equal(t(2.7182817459106445))

      t([1.0, 2, 3])
      |> Nx.exp()
      |> assert_equal(t([2.7182817459106445, 7.389056205749512, 20.08553695678711]))
    end

    test "cos" do
      Nx.cos(1.0)
      |> assert_equal(t(0.5403022766113281))

      t([1.0, 2, 3])
      |> Nx.cos()
      |> assert_equal(t([0.5403022766113281, -0.416146844625473, -0.9899924993515015]))
    end

    test "log" do
      Nx.log(1.0)
      |> assert_equal(t(0.0))

      t([1.0, 2, 3])
      |> Nx.log()
      |> assert_equal(t([0.0, 0.6931471824645996, 1.0986123085021973]))
    end

    test "tanh" do
      Nx.tanh(1.0)
      |> assert_equal(t(0.7615941762924194))

      t([1.0, 2, 3])
      |> Nx.tanh()
      |> assert_equal(t([0.7615941762924194, 0.9640275835990906, 0.9950547814369202]))
    end

    test "abs" do
      t([-2.0, -1, 0, 1, 2])
      |> Nx.abs()
      |> assert_equal(t([2, 1, 0, 1, 2]))
    end

    test "sqrt" do
      Nx.sqrt(1.0)
      |> assert_equal(t(1.0))

      t([1.0, 2, 3])
      |> Nx.sqrt()
      |> assert_equal(t([1.0, 1.4142135381698608, 1.7320507764816284]))
    end

    test "argmax" do
      Nx.argmax(4)
      |> assert_equal(t(0))

      # TODO: Support argmax without specific axis
      # t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])
      # |> Nx.argmax()
      # |> assert_equal(t(10))

      # t([2.0, 4.0])
      # |> Nx.argmax()
      # |> assert_equal(t(1))

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])
      |> Nx.argmax(axis: 0)
      |> assert_equal(t(
        [
          [1, 0, 0],
          [1, 1, 0]
        ]
      ))

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      |> Nx.argmax(axis: :z)
      |> assert_equal(t(
        [
          [0, 2],
          [0, 1]
        ]
      ))

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      |> Nx.argmax(axis: :y, keep_axis: true)
      |> assert_equal(t(
        [
          [
            [0, 0, 0]
          ],
          [
            [0, 1, 0]
          ]
        ]
      ))
    end

    test "argmin" do
      Nx.argmin(4)
      |> assert_equal(t(0))

      # TODO: Support argmin without specific axis
      # t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])
      # |> Nx.argmin()
      # |> assert_equal(t(4))

      # t([2.0, 4.0])
      # |> Nx.argmin()
      # |> assert_equal(t(0))

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]])
      |> Nx.argmin(axis: 0)
      |> assert_equal(t(
        [
          [0, 0, 0],
          [0, 0, 0]
        ]
      ))

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      |> Nx.argmin(axis: 1)
      |> assert_equal(t(
        [
          [1, 1, 0],
          [1, 0, 0]
        ]
      ))

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      |> Nx.argmin(axis: :z)
      |> assert_equal(t(
        [
          [1, 1],
          [1, 2]
        ]
      ))
    end

    test "acos" do
      Nx.acos(0.10000000149011612)
      |> assert_equal(t(1.4706288576126099))

      t([0.10000000149011612, 0.5, 0.8999999761581421])
      |> Nx.acos()
      |> assert_equal(t([1.4706288576126099, 1.0471975803375244, 0.4510268568992615]))
    end

    test "asin" do
      Nx.asin(0.10000000149011612)
      |> assert_equal(t(0.1001674234867096))

      t([0.10000000149011612, 0.5, 0.8999999761581421])
      |> Nx.asin()
      |> assert_equal(t([0.1001674234867096, 0.5235987901687622, 1.1197694540023804]))
    end

    test "tan" do
      Nx.tan(1.0)
      |> assert_equal(t(1.5574077367782593))

      t([1.0, 2, 3])
      |> Nx.tan()
      |> assert_equal(t([1.5574077367782593, -2.185039758682251, -0.14254654943943024]))
    end

    test "atan" do
      Nx.atan(0.10000000149011612)
      |> assert_equal(t(0.09966865181922913))

      t([0.10000000149011612, 0.5, 0.8999999761581421])
      |> Nx.atan()
      |> assert_equal(t([0.09966865181922913, 0.46364760398864746, 0.7328150868415833]))
    end

    test "ceil" do
      t([-1, 0, 1])
      |> Nx.ceil()
      |> assert_equal(t([-1, 0, 1]))

      t([-1.5, -0.5, 0.5, 1.5])
      |> Nx.ceil()
      |> assert_equal(t([-1.0, 0.0, 1.0, 2.0]))
    end

    test "floor" do
      t([-1, 0, 1])
      |> Nx.floor()
      |> assert_equal(t([-1, 0, 1]))

      t([-1.5, -0.5, 0.5, 1.5])
      |> Nx.floor()
      |> assert_equal(t([-2.0, -1.0, 0.0, 1.0]))
    end

    test "round" do
      t([-1, 0, 1])
      |> Nx.round()
      |> assert_equal(t([-1, 0, 1]))

      t([-1.5, -0.5, 0.5, 1.5])
      |> Nx.round()
      |> assert_equal(t([-2.0, -1.0, 1.0, 2.0]))
    end

    test "cbrt" do
      Nx.cbrt(1.0)
      |> assert_equal(t(1.0))

      t([1.0, 2, 3])
      |> Nx.cbrt()
      |> assert_equal(t([1.0, 1.2599210739135742, 1.4422495365142822]))
    end

    test "log1p" do
      Nx.log1p(1.0)
      |> assert_equal(t(0.6931471824645996))

      t([1.0, 2, 3])
      |> Nx.log1p()
      |> assert_equal(t([0.6931471824645996, 1.0986123085021973, 1.3862943649291992]))
    end

    test "bitwise_and" do
      Nx.bitwise_and(1, 0)
      |> assert_equal(t(0))

      t([0, 1, 2])
      |> Nx.bitwise_and(1)
      |> assert_equal(t([0, 1, 0]))

      t([0, -1, -2])
      |> Nx.bitwise_and(-1)
      |> assert_equal(t([0, -1, -2]))

      t([0, 0, 1, 1])
      |> Nx.bitwise_and(t([0, 1, 0, 1]))
      |> assert_equal(t([0, 0, 0, 1]))
    end

    test "bitwise_or" do
      Nx.bitwise_or(1, 0)
      |> assert_equal(t(1))

      t([0, 1, 2])
      |> Nx.bitwise_or(1)
      |> assert_equal(t([1, 1, 3]))

      t([0, -1, -2])
      |> Nx.bitwise_or(-1)
      |> assert_equal(t([-1, -1, -1]))

      t([0, 0, 1, 1])
      |> Nx.bitwise_or(t([0, 1, 0, 1]))
      |> assert_equal(t([0, 1, 1, 1]))
    end

    test "bitwise_xor" do
      Nx.bitwise_xor(1, 0)
      |> assert_equal(t(1))

      t([1, 2, 3])
      |> Nx.bitwise_xor(2)
      |> assert_equal(t([3, 0, 1]))

      t([1, 2, 3], type: :u32)
      |> Nx.bitwise_xor(2)
      |> assert_equal(t([3, 0, 1]))

      t([-1, -2, -3])
      |> Nx.bitwise_xor(2)
      |> assert_equal(t([-3, -4, -1]))

      t([0, 0, 1, 1])
      |> Nx.bitwise_xor(t([0, 1, 0, 1]))
      |> assert_equal(t([0, 1, 1, 0]))
    end

    test "bitwise_not" do
      Nx.bitwise_not(1)
      |> assert_equal(t(-2))

      t([-1, 0, 1])
      |> Nx.bitwise_not()
      |> assert_equal(t([0, -1, -2]))

      t([0, 1, 254, 255], type: :u8)
      |> Nx.bitwise_not()
      |> assert_equal(t([255, 254, 1, 0]))
    end

    test "left_shift" do
      Nx.left_shift(1, 0)
      |> assert_equal(t(1))

      t([1, 2, 3])
      |> Nx.left_shift(2)
      |> assert_equal(t([4, 8, 12]))

      t([1, 1, -1, -1])
      |> Nx.left_shift(t([1, 2, 3, 4]))
      |> assert_equal(t([2, 4, -8, -16]))

      t([1, 2, 3], type: :u32)
      |> Nx.left_shift(2)
      |> assert_equal(t([4, 8, 12]))

      t([1, 2, 3], type: :u32)
      |> Nx.left_shift(t(2, type: :u8))
      |> assert_equal(t([4, 8, 12]))

      t([1, 1, 0, 0], type: :u32)
      |> Nx.left_shift(t([1, 2, 3, 4]))
      |> assert_equal(t([2, 4, 0, 0]))

      t([1, 1, 0, 0], type: :u32)
      |> Nx.left_shift(t([1, 2, 3, 4], type: :u8))
      |> assert_equal(t([2, 4, 0, 0]))
    end

    test "right_shift" do
      Nx.right_shift(1, 0)
      |> assert_equal(t(1))

      t([2, 4, 8])
      |> Nx.right_shift(2)
      |> assert_equal(t([0, 1, 2]))

      t([16, 32, -64, -128])
      |> Nx.right_shift(t([1, 2, 3, 4]))
      |> assert_equal(t([8, 8, -8, -8]))

      t([2, 4, 8], type: :u32)
      |> Nx.right_shift(2)
      |> assert_equal(t([0, 1, 2]))

      t([16, 32, -64, -128], type: :u32)
      |> Nx.right_shift(t([1, 2, 3, 4]))
      |> assert_equal(t([8, 8, 536870904, 268435448]))
    end

    test "is_infinity" do
      t([:infinity, :nan, :neg_infinity, 1, 0])
      |> Nx.is_infinity()
      |> assert_equal(t([1, 0, 1, 0, 0]))

      t([:infinity, 1, :neg_infinity])
      |> Nx.is_infinity()
      |> assert_equal(t([1, 0, 1]))

      # TODO: Not supported for :s64
      # t([1, 0])
      # |> Nx.is_infinity()
      # |> assert_equal(t([0, 0]))
    end

    test "logical_or" do
      Nx.logical_or(0, t([-1, 0, 1]))
      |> assert_equal(t([1, 0, 1]))

      t([-1, 0, 1])
      |> Nx.logical_or(t([[-1], [0], [1]]))
      |> assert_equal(t(
        [
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1]
        ]
      ))

      t([-1.0, 0.0, 1.0])
      |> Nx.logical_or(t([[-1], [0], [1]]))
      |> assert_equal(t(
        [
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1]
        ]
      ))
    end

    test "erf_inv" do
      Nx.erf_inv(0.10000000149011612)
      |> assert_close(t(0.08885598927736282))

      t([0.10000000149011612, 0.5, 0.8999999761581421])
      |> Nx.erf_inv()
      |> assert_close(t([0.08885598927736282, 0.4769362807273865, 1.163087010383606]))

      t(0.10000000149011612, type: :f64)
      |> Nx.erf_inv()
      |> assert_close(t(0.08885598927736282, type: :f64))

      t([0.10000000149011612, 0.5, 0.8999999761581421], type: :f64)
      |> Nx.erf_inv()
      |> assert_close(t([0.0888559891358877, 0.47693629334671295, 1.1630870196442271], type: :f64))
    end

    test "sum/2" do
      t(42)
      |> Nx.sum()
      |> assert_equal(t(42))

      t([1, 2, 3])
      |> Nx.sum()
      |> assert_equal(t(6))

      t([[1.0, 2.0], [3.0, 4.0]])
      |> Nx.sum()
      |> assert_equal(t(10.0))

      t = Nx.iota({2, 2, 3}, names: [:x, :y, :z])
      Nx.sum(t, axes: [:x])
      |> assert_equal(t(
        [
          [6, 8, 10],
          [12, 14, 16]
        ]
      ))

      Nx.sum(t, axes: [:y])
      |> assert_equal(t(
        [
          [3, 5, 7],
          [15, 17, 19]
        ]
      ))

      Nx.sum(t, axes: [:z])
      |> assert_equal(t(
        [
          [3, 12],
          [21, 30]
        ]
      ))

      Nx.sum(t, axes: [:x, :z])
      |> assert_equal(t([24, 42]))

      Nx.sum(t, axes: [-3])
      |> assert_equal(t(
        [
          [6, 8, 10],
          [12, 14, 16]
        ]
      ))

      t([[1, 2], [3, 4]], names: [:x, :y])
      |> Nx.sum(axes: [:x], keep_axes: true)
      |> assert_equal(t(
        [
          [4, 6]
        ]
      ))
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
