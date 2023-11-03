defmodule CandlexTest do
  use Nx.Case, async: true
  doctest Candlex

  describe "creation" do
    test "tensor" do
      check(255, type: :u8)
      check(100_002, type: :u32)
      check(100_102, type: :u64)
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

      t([1, 2, 3], type: :u64)
      |> Nx.add(t([10, 20, 30], type: :u64))
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

      Nx.iota({5}, type: :u64)
      |> assert_equal(t([0, 1, 2, 3, 4]))

      Nx.iota({5}, type: :f32)
      |> assert_equal(t([0.0, 1.0, 2.0, 3.0, 4.0]))

      Nx.iota({2, 3})
      |> assert_equal(t([[0, 1, 2], [3, 4, 5]]))

      Nx.iota({3, 3}, axis: 1)
      |> assert_equal(
        t([
          [0, 1, 2],
          [0, 1, 2],
          [0, 1, 2]
        ])
      )

      Nx.iota({3, 3}, axis: -1)
      |> assert_equal(
        t([
          [0, 1, 2],
          [0, 1, 2],
          [0, 1, 2]
        ])
      )

      Nx.iota({3, 4, 3}, axis: 0, type: :f64)
      |> assert_equal(
        t([
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
          ],
          [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
          ],
          [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0]
          ]
        ])
      )

      Nx.iota({1, 3, 2}, axis: 2)
      |> assert_equal(
        t([
          [
            [0, 1],
            [0, 1],
            [0, 1]
          ]
        ])
      )
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

      t([1, 2], type: :u64)
      |> Nx.multiply(t([3, 4], type: :u64))
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
      |> assert_equal(
        t([
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ])
      )

      1
      |> Nx.divide(2)
      |> assert_equal(t(0.5))

      t([1, 2, 3])
      |> Nx.divide(2)
      |> assert_equal(t([0.5, 1.0, 1.5]))

      t([[1], [2]])
      |> Nx.divide(t([[10, 20]]))
      |> assert_equal(
        t([
          [0.10000000149011612, 0.05000000074505806],
          [0.20000000298023224, 0.10000000149011612]
        ])
      )
    end

    test "remainder" do
      Nx.remainder(1, 2)
      |> assert_equal(t(1))

      t([1, 2, 3])
      |> Nx.remainder(2)
      |> assert_equal(t([1, 0, 1]))

      2
      |> Nx.remainder(t([1.0, 2.0, 3.0]))
      |> assert_equal(t([0.0, 0.0, 2.0]))

      t([[10], [20]], names: [:x, :y])
      |> Nx.remainder(t([[3, 4]], names: [nil, :y]))
      |> assert_equal(
        t([
          [1, 2],
          [2, 0]
        ])
      )

      left = t(-11)
      right = t(10, type: :u8)

      Nx.remainder(left, right)
      |> assert_equal(t(-1))

      left
      |> Nx.add(t(20))
      |> Nx.remainder(right)
      |> assert_equal(t(9))

      positive_left = t(9, type: :u8)

      Nx.remainder(positive_left, right)
      |> assert_equal(t(9))

      positive_left
      |> Nx.add(Nx.tensor(20, type: :u8))
      |> Nx.remainder(right)
      |> assert_equal(t(9))
    end

    test "quotient" do
      Nx.quotient(11, 2)
      |> assert_equal(t(5))

      t([2, 4, 5])
      |> Nx.quotient(2)
      |> assert_equal(t([1, 2, 2]))

      10
      |> Nx.quotient(t([1, 2, 3]))
      |> assert_equal(t([10, 5, 3]))

      t([[10, 20]], names: [nil, :y])
      |> Nx.quotient(t([[1], [2]], names: [:x, nil]))
      |> assert_equal(
        t([
          [10, 20],
          [5, 10]
        ])
      )

      t([[10, 20]])
      |> Nx.quotient(t([[1], [2]]))
      |> assert_equal(
        t([
          [10, 20],
          [5, 10]
        ])
      )

      t([[10, 20]], type: :u8)
      |> Nx.quotient(t([[1], [2]], type: :u32))
      |> assert_equal(
        t([
          [10, 20],
          [5, 10]
        ])
      )
    end

    test "sign" do
      t([-2, -1, 0, 1, 2])
      |> Nx.sign()
      |> assert_equal(t([-1, -1, 0, 1, 1]))
    end

    test "atan2" do
      Nx.atan2(1.0, 2.0)
      |> assert_close(t(0.46364760398864746))

      t([1.0, 2, 3])
      |> Nx.atan2(1)
      |> assert_close(t([0.7853981852531433, 1.1071487665176392, 1.249045729637146]))

      1.0
      |> Nx.atan2(t([1.0, 2.0, 3.0]))
      |> assert_close(t([0.7853981852531433, 0.46364760398864746, 0.32175055146217346]))

      t([[-0.0], [0.0]], type: :f64)
      |> Nx.atan2(t([-0.0, 0.0], type: :f64))
      |> assert_close(
        t([
          [-3.141592653589793, -0.0],
          [3.141592653589793, 0.0]
        ])
      )
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

    test "greater" do
      Nx.greater(1, 2)
      |> assert_equal(t(0))

      Nx.greater(1, t([1, 2, 3]))
      |> assert_equal(t([0, 0, 0]))

      t([1, 2, 3])
      |> Nx.greater(t([1, 2, 2]))
      |> assert_equal(t([0, 0, 1]))

      t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      |> Nx.greater(t([1, 2, 3]))
      |> assert_equal(
        t([
          [0, 0, 0],
          [1, 1, 1]
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

      t([1, 2, 3])
      |> Nx.dot(t([4, 5, 6]))
      |> assert_equal(t(32))

      t([1.0, 2, 3])
      |> Nx.dot(t([1, 2, 3]))
      |> assert_equal(t(14.0))

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
      |> Nx.dot(t([[7, 8], [9, 10], [11, 12]]))
      |> assert_equal(
        t([
          [58.0, 64],
          [139, 154]
        ])
      )

      # Dot product of vector and n-D tensor

      t([[0.0]])
      |> Nx.dot(t([55.0]))
      |> assert_equal(t([0.0]))

      t([[[1.0, 2], [3, 4]], [[5, 6], [7, 8]]])
      |> Nx.dot(t([5, 10]))
      |> assert_equal(
        t([
          [25.0, 55],
          [85, 115]
        ])
      )

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
      |> assert_equal(
        t([
          [100, 140],
          [140, 200]
        ])
      )

      # TODO:
      t1
      |> Nx.dot([0], [], t2, [1], [])
      |> assert_equal(
        t([
          [70, 150],
          [100, 220]
        ])
      )

      t1
      |> Nx.dot([1], [], t2, [0], [])
      |> assert_equal(
        t([
          [70, 100],
          [150, 220]
        ])
      )

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
      |> assert_close(t(0.8414709568023682))

      t([1.0, 2.0, 3.0])
      |> Nx.sin()
      |> assert_close(t([0.8414709568023682, 0.9092974066734314, 0.14112000167369843]))
    end

    test "sinh" do
      Nx.sinh(1.0)
      |> assert_close(t(1.175201177597046))

      t([1.0, 2, 3])
      |> Nx.sinh()
      |> assert_close(t([1.175201177597046, 3.6268603801727295, 10.017874717712402]))
    end

    test "exp" do
      Nx.exp(1.0)
      |> assert_equal(t(2.7182817459106445))

      t([1.0, 2, 3])
      |> Nx.exp()
      |> assert_equal(t([2.7182817459106445, 7.389056205749512, 20.08553695678711]))
    end

    test "expm1" do
      Nx.expm1(1.0)
      |> assert_close(t(1.718281865119934))

      t([1.0, 2, 3])
      |> Nx.expm1()
      |> assert_close(t([1.718281865119934, 6.389056205749512, 19.08553695678711]))
    end

    test "cos" do
      Nx.cos(1.0)
      |> assert_close(t(0.5403022766113281))

      t([1.0, 2, 3])
      |> Nx.cos()
      |> assert_close(t([0.5403022766113281, -0.416146844625473, -0.9899924993515015]))
    end

    test "cosh" do
      Nx.cosh(1.0)
      |> assert_close(t(1.5430806875228882))

      t([1.0, 2, 3])
      |> Nx.cosh()
      |> assert_close(t([1.5430806875228882, 3.762195587158203, 10.067662239074707]))
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

      t([-2, -1, 0, 1, 2])
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

    test "rsqrt" do
      Nx.rsqrt(1.0)
      |> assert_equal(t(1.0))

      t([1.0, 2, 3])
      |> Nx.rsqrt()
      |> assert_equal(t([1.0, 0.7071067690849304, 0.5773502588272095]))
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
      |> assert_equal(
        t([
          [1, 0, 0],
          [1, 1, 0]
        ])
      )

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      |> Nx.argmax(axis: :z)
      |> assert_equal(
        t([
          [0, 2],
          [0, 1]
        ])
      )

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      |> Nx.argmax(axis: :y, keep_axis: true)
      |> assert_equal(
        t([
          [
            [0, 0, 0]
          ],
          [
            [0, 1, 0]
          ]
        ])
      )
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
      |> assert_equal(
        t([
          [0, 0, 0],
          [0, 0, 0]
        ])
      )

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      |> Nx.argmin(axis: 1)
      |> assert_equal(
        t([
          [1, 1, 0],
          [1, 0, 0]
        ])
      )

      t([[[4, 2, 3], [1, -5, 3]], [[6, 2, 3], [4, 8, 3]]], names: [:x, :y, :z])
      |> Nx.argmin(axis: :z)
      |> assert_equal(
        t([
          [1, 1],
          [1, 2]
        ])
      )
    end

    test "acos" do
      Nx.acos(0.10000000149011612)
      |> assert_equal(t(1.4706288576126099))

      t([0.10000000149011612, 0.5, 0.8999999761581421])
      |> Nx.acos()
      |> assert_equal(t([1.4706288576126099, 1.0471975803375244, 0.4510268568992615]))
    end

    test "acosh" do
      Nx.acosh(1.0)
      |> assert_equal(t(0.0))

      t([1.0, 2, 3])
      |> Nx.acosh()
      |> assert_close(t([0.0, 1.316957950592041, 1.7627471685409546]))
    end

    test "asin" do
      Nx.asin(0.10000000149011612)
      |> assert_equal(t(0.1001674234867096))

      t([0.10000000149011612, 0.5, 0.8999999761581421])
      |> Nx.asin()
      |> assert_equal(t([0.1001674234867096, 0.5235987901687622, 1.1197694540023804]))
    end

    test "asinh" do
      Nx.asinh(1.0)
      |> assert_close(t(0.8813735842704773))

      t([1.0, 2, 3])
      |> Nx.asinh()
      |> assert_close(t([0.8813735842704773, 1.4436354637145996, 1.8184465169906616]))
    end

    test "tan" do
      Nx.tan(1.0)
      |> assert_close(t(1.5574077367782593))

      t([1.0, 2, 3])
      |> Nx.tan()
      |> assert_close(t([1.5574077367782593, -2.185039758682251, -0.14254654943943024]))
    end

    test "atan" do
      Nx.atan(0.10000000149011612)
      |> assert_close(t(0.09966865181922913))

      t([0.10000000149011612, 0.5, 0.8999999761581421])
      |> Nx.atan()
      |> assert_close(t([0.09966865181922913, 0.46364760398864746, 0.7328150868415833]))
    end

    test "atanh" do
      Nx.atanh(0.10000000149011612)
      |> assert_close(t(0.10033535212278366))

      t([0.10000000149011612, 0.5, 0.8999999761581421])
      |> Nx.atanh()
      |> assert_close(t([0.10033535212278366, 0.5493061542510986, 1.4722193479537964]))
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
      |> assert_equal(t([8, 8, 536_870_904, 268_435_448]))
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

    test "is_nan" do
      t([:nan, 1.0, 0.0])
      |> Nx.is_nan()
      |> assert_equal(t([1, 0, 0]))

      t([:nan, :infinity])
      |> Nx.is_nan()
      |> assert_equal(t([1, 0]))

      # Complex not yet supported
      # t(Complex.new(0, :nan))
      # |> Nx.is_nan()
      # |> assert_equal(t(1))

      t([1.0, 0.0])
      |> Nx.is_nan()
      |> assert_equal(t([0, 0]))
    end

    test "logical_and" do
      Nx.logical_and(1, t([-1, 0, 1]))
      |> assert_equal(t([1, 0, 1]))

      t([-1, 0, 1])
      |> Nx.logical_and(t([[-1], [0], [1]]))
      |> assert_equal(
        t([
          [1, 0, 1],
          [0, 0, 0],
          [1, 0, 1]
        ])
      )

      t([-1.0, 0.0, 1.0])
      |> Nx.logical_and(t([[-1], [0], [1]]))
      |> assert_equal(
        t([
          [1, 0, 1],
          [0, 0, 0],
          [1, 0, 1]
        ])
      )
    end

    test "logical_or" do
      Nx.logical_or(0, t([-1, 0, 1]))
      |> assert_equal(t([1, 0, 1]))

      t([-1, 0, 1])
      |> Nx.logical_or(t([[-1], [0], [1]]))
      |> assert_equal(
        t([
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1]
        ])
      )

      t([-1.0, 0.0, 1.0])
      |> Nx.logical_or(t([[-1], [0], [1]]))
      |> assert_equal(
        t([
          [1, 1, 1],
          [1, 0, 1],
          [1, 1, 1]
        ])
      )
    end

    test "logical_xor" do
      0
      |> Nx.logical_xor(t([-1, 0, 1]))
      |> assert_equal(t([1, 0, 1]))

      t([-1, 0, 1])
      |> Nx.logical_xor(t([[-1], [0], [1]]))
      |> assert_equal(
        t([
          [0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]
        ])
      )

      t([-1.0, 0.0, 1.0])
      |> Nx.logical_xor(t([[-1], [0], [1]]))
      |> assert_equal(
        t([
          [0, 1, 0],
          [1, 0, 1],
          [0, 1, 0]
        ])
      )
    end

    test "erf" do
      Nx.erf(1.0)
      |> assert_close(t(0.8427007794380188))

      Nx.erf(t([1.0, 2, 3]))
      |> assert_close(t([0.8427007794380188, 0.9953222870826721, 0.9999778866767883]))
    end

    test "erfc" do
      Nx.erfc(1.0)
      |> assert_close(t(0.15729920566082))

      Nx.erfc(t([1.0, 2, 3]))
      |> assert_close(t([0.15729920566082, 0.004677734803408384, 2.2090496713644825e-5]))
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
      |> assert_close(
        t([0.0888559891358877, 0.47693629334671295, 1.1630870196442271], type: :f64)
      )
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
      |> assert_equal(
        t([
          [6, 8, 10],
          [12, 14, 16]
        ])
      )

      Nx.sum(t, axes: [:y])
      |> assert_equal(
        t([
          [3, 5, 7],
          [15, 17, 19]
        ])
      )

      Nx.sum(t, axes: [:z])
      |> assert_equal(
        t([
          [3, 12],
          [21, 30]
        ])
      )

      Nx.sum(t, axes: [:x, :z])
      |> assert_equal(t([24, 42]))

      Nx.sum(t, axes: [-3])
      |> assert_equal(
        t([
          [6, 8, 10],
          [12, 14, 16]
        ])
      )

      t([[1, 2], [3, 4]], names: [:x, :y])
      |> Nx.sum(axes: [:x], keep_axes: true)
      |> assert_equal(
        t([
          [4, 6]
        ])
      )
    end

    test "to_batched/2" do
      [first, second] =
        Nx.iota({2, 2, 2})
        |> Nx.to_batched(1)
        |> Enum.to_list()

      first
      |> assert_equal(
        t([
          [
            [0, 1],
            [2, 3]
          ]
        ])
      )

      second
      |> assert_equal(
        t([
          [
            [4, 5],
            [6, 7]
          ]
        ])
      )

      [first, second] =
        Nx.iota({10})
        |> Nx.to_batched(5)
        |> Enum.to_list()

      first
      |> assert_equal(Nx.tensor([0, 1, 2, 3, 4]))

      second
      |> assert_equal(Nx.tensor([5, 6, 7, 8, 9]))

      [first, second, third, fourth] =
        Nx.iota({10})
        |> Nx.to_batched(3)
        |> Enum.to_list()

      first
      |> assert_equal(Nx.tensor([0, 1, 2]))

      second
      |> assert_equal(Nx.tensor([3, 4, 5]))

      third
      |> assert_equal(Nx.tensor([6, 7, 8]))

      fourth
      |> assert_equal(Nx.tensor([9, 0, 1]))

      # TODO: Implement with discard
      # [first, second] =
      #   Nx.iota({10})
      #   |> Nx.to_batched(4, leftover: :discard)
      #   |> Enum.to_list()

      # first
      # |> assert_equal(Nx.tensor([0, 1, 2, 3]))

      # second
      # |> assert_equal(Nx.tensor([4, 5, 6, 7]))
    end

    test "sigmoid/1" do
      Nx.sigmoid(1.0)
      |> assert_close(t(0.7310585975646973))

      t([1.0, 2, 3])
      |> Nx.sigmoid()
      |> assert_close(t([0.7310585975646973, 0.8807970881462097, 0.9525741338729858]))
    end

    test "mean/1" do
      t(42)
      |> Nx.mean()
      |> assert_equal(t(42.0))

      t([1, 2, 3])
      |> Nx.mean()
      |> assert_equal(t(2.0))

      t([0.1, 0.2, 0.3])
      |> Nx.mean()
      |> assert_equal(t(0.2))
    end

    test "pow" do
      # Nx.pow(2, 4)
      # |> assert_equal(t(16))

      # t([1, 2, 3], type: :u32)
      # |> Nx.pow(t(2, type: :u32))
      # |> assert_equal(t([1, 4, 9]))

      t([1.0, 2.0, 3.0])
      |> Nx.pow(2)
      |> assert_equal(t([1.0, 4.0, 9.0]))

      2
      |> Nx.pow(t([1.0, 2.0, 3.0]))
      |> assert_equal(t([2.0, 4.0, 8.0]))

      # t([[2], [3]])
      # |> Nx.pow(t([[4, 5]]))
      # |> assert_equal(t(
      #   [
      #     [16, 32],
      #     [81, 243]
      #   ]
      # ))
    end

    test "conv" do
      Nx.iota({9})
      |> Nx.reshape({1, 1, 3, 3})
      |> Nx.conv(
        Nx.iota({4})
        |> Nx.reshape({4, 1, 1, 1}),
        strides: [1, 1]
      )
      |> assert_equal(
        t([
          [
            [
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0]
            ],
            [
              [0.0, 1.0, 2.0],
              [3.0, 4.0, 5.0],
              [6.0, 7.0, 8.0]
            ],
            [
              [0.0, 2.0, 4.0],
              [6.0, 8.0, 10.0],
              [12.0, 14.0, 16.0]
            ],
            [
              [0.0, 3.0, 6.0],
              [9.0, 12.0, 15.0],
              [18.0, 21.0, 24.0]
            ]
          ]
        ])
      )

      # input/output permutation

      result =
        Nx.iota({1, 3, 3, 6})
        |> Nx.conv(
          1 |> Nx.broadcast({2, 6, 1, 1}),
          input_permutation: [0, 3, 1, 2],
          output_permutation: [0, 3, 1, 2]
        )

      assert result.shape == {1, 3, 3, 2}

      result
      |> assert_close(
        t([
          [
            [15.0, 15.0],
            [51.0, 51.0],
            [87.0, 87.0]
          ],
          [
            [123.0, 123.0],
            [159.0, 159.0],
            [195.0, 195.0]
          ],
          [
            [231.0, 231.0],
            [267.0, 267.0],
            [303.0, 303.0]
          ]
        ])
      )

      # Nx.iota({9})
      # |> Nx.reshape({1, 1, 3, 3})
      # |> Nx.conv(
      #   Nx.iota({8})
      #   |> Nx.reshape({4, 1, 2, 1}),
      #   strides: 2,
      #   padding: :same,
      #   kernel_dilation: [2, 1]
      # )
      # |> assert_equal(t(
      #   [
      #     [
      #       [
      #         [3.0, 5.0],
      #         [0.0, 0.0]
      #       ],
      #       [
      #         [9.0, 15.0],
      #         [6.0, 10.0]
      #       ],
      #       [
      #         [15.0, 25.0],
      #         [12.0, 20.0]
      #       ],
      #       [
      #         [21.0, 35.0],
      #         [18.0, 30.0]
      #       ]
      #     ]
      #   ]
      # ))
    end

    test "reduce_max" do
      t(42)
      |> Nx.reduce_max()
      |> assert_equal(t(42))

      t(42.0)
      |> Nx.reduce_max()
      |> assert_equal(t(42.0))

      t([1, 2, 3])
      |> Nx.reduce_max()
      |> assert_equal(t(3))

      t([[3, 1, 4], [2, 1, 1]], names: [:x, :y])
      |> Nx.reduce_max(axes: [:x])
      |> assert_equal(t([3, 1, 4]))

      t([[3, 1, 4], [2, 1, 1]], names: [:x, :y])
      |> Nx.reduce_max(axes: [:y])
      |> assert_equal(t([4, 2]))

      # t([[[1, 2], [4, 5]], [[2, 4], [3, 8]]], names: [:x, :y, :z])
      # |> Nx.reduce_max(axes: [:x, :z])
      # |> assert_equal(t([4, 8]))

      # t([[[1, 2], [4, 5]], [[2, 4], [3, 8]]], names: [:x, :y, :z])
      # |> Nx.reduce_max(axes: [:x, :z], keep_axes: true)
      # |> assert_equal(t(
      #   [
      #     [
      #       [4],
      #       [8]
      #     ]
      #   ]
      # ))
    end

    test "reduce_min" do
      Nx.reduce_min(t(42))
      |> assert_equal(t(42))

      Nx.reduce_min(t(42.0))
      |> assert_equal(t(42.0))

      Nx.reduce_min(t([1, 2, 3]))
      |> assert_equal(t(1))

      t([[3, 1, 4], [2, 1, 1]], names: [:x, :y])
      |> Nx.reduce_min(axes: [:x])
      |> assert_equal(t([2, 1, 1]))

      t([[3, 1, 4], [2, 1, 1]], names: [:x, :y])
      |> Nx.reduce_min(axes: [:y])
      |> assert_equal(t([1, 1]))

      # t([[[1, 2], [4, 5]], [[2, 4], [3, 8]]], names: [:x, :y, :z])
      # |> Nx.reduce_min(axes: [:x, :z])
      # |> assert_equal(t([1, 3]))

      # t([[[1, 2], [4, 5]], [[2, 4], [3, 8]]], names: [:x, :y, :z])
      # |> Nx.reduce_min(axes: [:x, :z], keep_axes: true)
      # |> assert_equal(t(
      #   [
      #     [
      #       [1],
      #       [3]
      #     ]
      #   ]
      # ))
    end

    test "take_along_axis" do
      t([[1, 2, 3], [4, 5, 6]])
      |> Nx.take_along_axis(
        t([
          [0, 0, 2, 2, 1, 1],
          [2, 2, 1, 1, 0, 0]
        ]),
        axis: 1
      )
      |> assert_equal(
        t([
          [1, 1, 3, 3, 2, 2],
          [6, 6, 5, 5, 4, 4]
        ])
      )

      t([[1, 2, 3], [4, 5, 6]])
      |> Nx.take_along_axis(
        t([
          [0, 1, 1],
          [1, 0, 0],
          [0, 1, 0]
        ]),
        axis: 0
      )
      |> assert_equal(
        t([
          [1, 5, 6],
          [4, 2, 3],
          [1, 5, 3]
        ])
      )
    end

    test "gather" do
      t([1, 2, 3, 4])
      |> Nx.gather(t([[3], [1], [2]]))
      |> assert_equal(t([4, 2, 3]))

      # t([[1, 2], [3, 4]])
      # |> Nx.gather(t([[1, 1], [0, 1], [1, 0]]))
      # |> assert_equal(t([4, 2, 3]))

      # t([[1, 2], [3, 4]])
      # |> Nx.gather(t([[[1, 1], [0, 0]], [[1, 0], [0, 1]]]))
      # |> assert_equal(t(
      #   [
      #     [4, 1],
      #     [3, 2]
      #   ]
      # ))

      # t([[[1, 2], [11, 12]], [[101, 102], [111, 112]]])
      # |> Nx.gather(t([[0, 0, 0], [0, 1, 1], [1, 1, 1]]))
      # |> assert_equal(t([1, 12, 112]))
    end

    test "indexed_add" do
      t([1.0])
      |> Nx.indexed_add(t([[0], [0]]), t([1, 1]))
      |> assert_equal(t([3.0]))

      t([1])
      |> Nx.indexed_add(t([[0], [0]]), t([1.0, 1.0]))
      |> assert_equal(t([3.0]))

      t([1], type: :u8)
      |> Nx.indexed_add(t([[0], [0]]), t([1, 1], type: :s64))
      |> assert_equal(t([3]))

      # Nx.iota({1, 2, 3})
      # |> Nx.indexed_add(
      #   t([[0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 0, 2], [0, 1, 2]]),
      #   t([1, 3, 1, -2, 5])
      # )
      # |> assert_equal(t(
      #   [
      #     [
      #       [2, 1, 0],
      #       [3, 7, 10]
      #     ]
      #   ]
      # ))
    end

    test "transpose" do
      t(1)
      |> Nx.transpose()
      |> assert_equal(t(1))

      Nx.iota({2, 3, 4}, names: [:x, :y, :z])
      |> Nx.transpose()
      |> assert_equal(
        t([
          [
            [0, 12],
            [4, 16],
            [8, 20]
          ],
          [
            [1, 13],
            [5, 17],
            [9, 21]
          ],
          [
            [2, 14],
            [6, 18],
            [10, 22]
          ],
          [
            [3, 15],
            [7, 19],
            [11, 23]
          ]
        ])
      )

      t(1)
      |> Nx.transpose(axes: [])
      |> assert_equal(t(1))

      Nx.iota({2, 3, 4}, names: [:batch, :x, :y])
      |> Nx.transpose(axes: [2, 1, :batch])
      |> assert_equal(
        t([
          [
            [0, 12],
            [4, 16],
            [8, 20]
          ],
          [
            [1, 13],
            [5, 17],
            [9, 21]
          ],
          [
            [2, 14],
            [6, 18],
            [10, 22]
          ],
          [
            [3, 15],
            [7, 19],
            [11, 23]
          ]
        ])
      )

      Nx.iota({2, 3, 4}, names: [:batch, :x, :y])
      |> Nx.transpose(axes: [:y, :batch, :x])
      |> assert_equal(
        t([
          [
            [0, 4, 8],
            [12, 16, 20]
          ],
          [
            [1, 5, 9],
            [13, 17, 21]
          ],
          [
            [2, 6, 10],
            [14, 18, 22]
          ],
          [
            [3, 7, 11],
            [15, 19, 23]
          ]
        ])
      )

      Nx.iota({2, 3, 4}, names: [:batch, :x, :y])
      |> Nx.transpose(axes: [:batch, :y, :x])
      |> assert_equal(
        t([
          [
            [0, 4, 8],
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11]
          ],
          [
            [12, 16, 20],
            [13, 17, 21],
            [14, 18, 22],
            [15, 19, 23]
          ]
        ])
      )
    end

    test "put_slice" do
      t([0, 1, 2, 3, 4])
      |> Nx.put_slice([2], Nx.tensor([5, 6]))
      |> assert_equal(t([0, 1, 5, 6, 4]))

      t([[1, 2, 3], [4, 5, 6]])
      |> Nx.put_slice([0, 0], t([[7, 8, 9], [10, 11, 12]]))
      |> assert_equal(
        t([
          [7, 8, 9],
          [10, 11, 12]
        ])
      )

      t([[1, 2, 3], [4, 5, 6]])
      |> Nx.put_slice([0, 1], t([[7, 8], [9, 10]]))
      |> assert_equal(
        t([
          [1, 7, 8],
          [4, 9, 10]
        ])
      )

      # t([[1, 2, 3], [4, 5, 6]])
      # |> Nx.put_slice([t(0), t(1)], t([[10.0, 11.0]]))
      # |> assert_equal(t(
      #   [
      #     [1.0, 10.0, 11.0],
      #     [4.0, 5.0, 6.0]
      #   ]
      # ))

      # t([[1, 2, 3], [4, 5, 6]])
      # |> Nx.put_slice([1, 1], t([[7, 8], [9, 10]]))
      # |> assert_equal(t(
      #   [
      #     [1, 7, 8],
      #     [4, 9, 10]
      #   ]
      # ))

      t([
        [
          [1, 2],
          [3, 4]
        ],
        [
          [4, 5],
          [6, 7]
        ]
      ])
      |> Nx.put_slice([0, 0, 1], t([[[8], [9]], [[10], [11]]]))
      |> assert_equal(
        t([
          [
            [1, 8],
            [3, 9]
          ],
          [
            [4, 10],
            [6, 11]
          ]
        ])
      )
    end

    test "pad" do
      t(1)
      |> Nx.pad(0, [])
      |> assert_equal(t(1))

      t([1, 2, 3], names: [:data])
      |> Nx.pad(0, [{1, 1, 0}])
      |> assert_equal(t([0, 1, 2, 3, 0]))

      # t([[1, 2, 3], [4, 5, 6]])
      # |> Nx.pad(0, [{0, 0, 1}, {0, 0, 1}])
      # |> assert_equal(t(
      #   [
      #     [1, 0, 2, 0, 3],
      #     [0, 0, 0, 0, 0],
      #     [4, 0, 5, 0, 6]
      #   ]
      # ))

      # Nx.pad(Nx.tensor([[1, 2, 3], [4, 5, 6]]), 0, [{1, 1, 0}, {1, 1, 0}])
      # [
      #   [0, 0, 0, 0, 0],
      #   [0, 1, 2, 3, 0],
      #   [0, 4, 5, 6, 0],
      #   [0, 0, 0, 0, 0]
      # ]
      # >

      # tensor = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      # Nx.pad(tensor, 0, [{0, 2, 0}, {1, 1, 0}, {1, 0, 0}])
      # [
      #   [
      #     [0, 0, 0],
      #     [0, 1, 2],
      #     [0, 3, 4],
      #     [0, 0, 0]
      #   ],
      #   [
      #     [0, 0, 0],
      #     [0, 5, 6],
      #     [0, 7, 8],
      #     [0, 0, 0]
      #   ],
      #   [
      #     [0, 0, 0],
      #     [0, 0, 0],
      #     [0, 0, 0],
      #     [0, 0, 0]
      #   ],
      #   [
      #     [0, 0, 0],
      #     [0, 0, 0],
      #     [0, 0, 0],
      #     [0, 0, 0]
      #   ]
      # ]

      # tensor = Nx.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      # Nx.pad(tensor, 0, [{1, 0, 0}, {1, 1, 0}, {0, 1, 0}])
      # [
      #   [
      #     [0, 0, 0],
      #     [0, 0, 0],
      #     [0, 0, 0],
      #     [0, 0, 0]
      #   ],
      #   [
      #     [0, 0, 0],
      #     [1, 2, 0],
      #     [3, 4, 0],
      #     [0, 0, 0]
      #   ],
      #   [
      #     [0, 0, 0],
      #     [5, 6, 0],
      #     [7, 8, 0],
      #     [0, 0, 0]
      #   ]
      # ]

      # tensor = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
      # Nx.pad(tensor, 0.0, [{1, 2, 0}, {1, 0, 0}, {0, 1, 0}])
      # [
      #   [
      #     [0.0, 0.0, 0.0],
      #     [0.0, 0.0, 0.0],
      #     [0.0, 0.0, 0.0]
      #   ],
      #   [
      #     [0.0, 0.0, 0.0],
      #     [1.0, 2.0, 0.0],
      #     [3.0, 4.0, 0.0]
      #   ],
      #   [
      #     [0.0, 0.0, 0.0],
      #     [5.0, 6.0, 0.0],
      #     [7.0, 8.0, 0.0]
      #   ],
      #   [
      #     [0.0, 0.0, 0.0],
      #     [0.0, 0.0, 0.0],
      #     [0.0, 0.0, 0.0]
      #   ],
      #   [
      #     [0.0, 0.0, 0.0],
      #     [0.0, 0.0, 0.0],
      #     [0.0, 0.0, 0.0]
      #   ]
      # ]

      # Nx.pad(Nx.tensor([0, 1, 2, 3, 0]), 0, [{-1, -1, 0}])
      # [1, 2, 3]

      # tensor = Nx.tensor([
      #   [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      #   [[0, 0, 0], [1, 2, 0], [3, 4, 0], [0, 0, 0]],
      #   [[0, 0, 0], [5, 6, 0], [7, 8, 0], [0, 0, 0]]
      # ])
      # Nx.pad(tensor, 0, [{-1, 0, 0}, {-1, -1, 0}, {0, -1, 0}])
      # [
      #   [
      #     [1, 2],
      #     [3, 4]
      #   ],
      #   [
      #     [5, 6],
      #     [7, 8]
      #   ]
      # ]

      # t([[0, 1, 2, 3], [0, 4, 5, 6]])
      # |> Nx.pad(0, [{0, 0, 0}, {-1, 1, 0}])
      # |> assert_equal(t(
      #   [
      #     [1, 2, 3, 0],
      #     [4, 5, 6, 0]
      #   ]
      # ))

      # t([[0, 1, 2], [3, 4, 5]], type: :f32)
      # |> Nx.pad(0, [{-1, 2, 0}, {1, -1, 0}])
      # |> assert_equal(t(
      #   [
      #     [0.0, 3.0, 4.0],
      #     [0.0, 0.0, 0.0],
      #     [0.0, 0.0, 0.0]
      #   ]
      # )
    end

    test "take" do
      t([[1, 2], [3, 4]])
      |> Nx.take(t([1, 0, 1]))
      |> assert_equal(
        t([
          [3, 4],
          [1, 2],
          [3, 4]
        ])
      )

      t([[1, 2], [3, 4]])
      |> Nx.take(t([1, 0, 1]), axis: 1)
      |> assert_equal(
        t([
          [2, 1, 2],
          [4, 3, 4]
        ])
      )

      t([[[1, 2], [11, 12]], [[101, 102], [111, 112]]])
      |> Nx.take(t([1, 0, 1]), axis: 1)
      |> assert_equal(
        t([
          [
            [11, 12],
            [1, 2],
            [11, 12]
          ],
          [
            [111, 112],
            [101, 102],
            [111, 112]
          ]
        ])
      )

      # t([[1, 2], [11, 12]])
      # |> Nx.take(t([[0, 0], [1, 1], [0, 0]]), axis: 1)
      # |> assert_equal(t(
      #   [
      #     [
      #       [1, 1],
      #       [2, 2],
      #       [1, 1]
      #     ],
      #     [
      #       [11, 11],
      #       [12, 12],
      #       [11, 11]
      #     ]
      #   ]
      # ))

      # t([[[1, 2], [11, 12]], [[101, 102], [111, 112]]])
      # |> Nx.take(t([[0, 0, 0], [1, 1, 1], [0, 0, 0]]), axis: 1)
      # |> assert_equal(t(
      #   [
      #     [
      #       [
      #         [1, 2],
      #         [1, 2],
      #         [1, 2]
      #       ],
      #       [
      #         [11, 12],
      #         [11, 12],
      #         [11, 12]
      #       ],
      #       [
      #         [1, 2],
      #         [1, 2],
      #         [1, 2]
      #       ]
      #     ],
      #     [
      #       [
      #         [101, 102],
      #         [101, 102],
      #         [101, 102]
      #       ],
      #       [
      #         [111, 112],
      #         [111, 112],
      #         [111, 112]
      #       ],
      #       [
      #         [101, 102],
      #         [101, 102],
      #         [101, 102]
      #       ]
      #     ]
      #   ]
      # ))
    end

    test "clip" do
      t([[1, 2, 3], [4, 5, 6]])
      |> Nx.clip(2, 4)
      |> assert_equal(
        t([
          [2, 2, 3],
          [4, 4, 4]
        ])
      )

      t([[1, 2, 3], [4, 5, 6]])
      |> Nx.clip(2.0, 3)
      |> assert_equal(
        t([
          [2.0, 2.0, 3.0],
          [3.0, 3.0, 3.0]
        ])
      )

      t([[1, 2, 3], [4, 5, 6]])
      |> Nx.clip(t(2.0), Nx.max(1.0, 3.0))
      |> assert_equal(
        t([
          [2.0, 2.0, 3.0],
          [3.0, 3.0, 3.0]
        ])
      )

      t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      |> Nx.clip(2, 6.0)
      |> assert_equal(
        t([
          [2.0, 2.0, 3.0],
          [4.0, 5.0, 6.0]
        ])
      )

      t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32)
      |> Nx.clip(1, 4)
      |> assert_equal(
        t([
          [1.0, 2.0, 3.0],
          [4.0, 4.0, 4.0]
        ])
      )
    end

    test "not_equal" do
      Nx.not_equal(1, 2)
      |> assert_equal(t(1))

      t([1, 2, 3])
      |> Nx.not_equal(t(1))
      |> assert_equal(t([0, 1, 1]))

      t([1, 1, 2])
      |> Nx.not_equal(t([1, 2, 3]))
      |> assert_equal(t([0, 1, 1]))

      t([[1, 4, 2], [4, 5, 6]])
      |> Nx.not_equal(t([[1, 3, 2], [4, 2, 1]]))
      |> assert_equal(
        t([
          [0, 1, 0],
          [0, 1, 1]
        ])
      )
    end

    test "all" do
      t(0)
      |> Nx.all()
      |> assert_equal(t(0))

      t(10)
      |> Nx.all()
      |> assert_equal(t(1))

      t([0, 1, 2])
      |> Nx.all()
      |> assert_equal(t(0))

      t([[-1, 0, 1], [2, 3, 4]], names: [:x, :y])
      |> Nx.all(axes: [:x])
      |> assert_equal(t([1, 0, 1]))

      t([[-1, 0, 1], [2, 3, 4]], names: [:x, :y])
      |> Nx.all(axes: [:y])
      |> assert_equal(t([0, 1]))

      t([[-1, 0, 1], [2, 3, 4]], names: [:x, :y])
      |> Nx.all(axes: [:y], keep_axes: true)
      |> assert_equal(
        t([
          [0],
          [1]
        ])
      )

      tensor = Nx.tensor([[[1, 2], [0, 4]], [[5, 6], [7, 8]]], names: [:x, :y, :z])

      tensor
      |> Nx.all(axes: [:x, :y])
      |> assert_equal(t([0, 1]))

      tensor
      |> Nx.all(axes: [:y, :z])
      |> assert_equal(t([0, 1]))

      tensor
      |> Nx.all(axes: [:x, :z])
      |> assert_equal(t([1, 0]))

      tensor
      |> Nx.all(axes: [:x, :y], keep_axes: true)
      |> assert_equal(
        t([
          [
            [0, 1]
          ]
        ])
      )

      tensor
      |> Nx.all(axes: [:y, :z], keep_axes: true)
      |> assert_equal(
        t([
          [
            [0]
          ],
          [
            [1]
          ]
        ])
      )

      tensor
      |> Nx.all(axes: [:x, :z], keep_axes: true)
      |> assert_equal(
        t([
          [
            [1],
            [0]
          ]
        ])
      )
    end

    test "any" do
      t([0, 1, 2])
      |> Nx.any()
      |> assert_equal(t(1))

      t([[0, 1, 0], [0, 1, 2]], names: [:x, :y])
      |> Nx.any(axes: [:x])
      |> assert_equal(t([0, 1, 1]))

      t([[0, 1, 0], [0, 1, 2]], names: [:x, :y])
      |> Nx.any(axes: [:y])
      |> assert_equal(t([1, 1]))

      tensor = t([[0, 1, 0], [0, 1, 2]], names: [:x, :y])

      tensor
      |> Nx.any(axes: [:x], keep_axes: true)
      |> assert_equal(t([[0, 1, 1]]))

      tensor
      |> Nx.any(axes: [:y], keep_axes: true)
      |> assert_equal(t([[1], [1]]))
    end

    if Candlex.Backend.cuda_available?() do
      test "different devices" do
        t([1, 2, 3], backend: {Candlex.Backend, device: :cpu})
        |> Nx.add(t([10, 20, 30], backend: {Candlex.Backend, device: :cuda}))
        |> assert_equal(t([11, 22, 33]))

        t([1, 2, 3], backend: {Candlex.Backend, device: :cuda})
        |> Nx.add(t([10, 20, 30], backend: {Candlex.Backend, device: :cpu}))
        |> assert_equal(t([11, 22, 33]))
      end
    end

    test "backend_transfer" do
      t([1, 2, 3], backend: Nx.BinaryBackend)
      |> Nx.backend_transfer({Candlex.Backend, device: :cpu})
      |> assert_equal(t([1, 2, 3]))

      t([1, 2, 3], backend: {Candlex.Backend, device: :cpu})
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> assert_equal(t([1, 2, 3]))

      t([1, 2, 3], backend: {Candlex.Backend, device: :cpu})
      |> Nx.backend_transfer({Candlex.Backend, device: :cpu})
      |> assert_equal(t([1, 2, 3]))
    end
  end

  defp t(values, opts \\ []) do
    opts =
      [backend: Candlex.Backend]
      |> Keyword.merge(opts)

    Nx.tensor(values, opts)
  end

  defp check(value, opts) do
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
