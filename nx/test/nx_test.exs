defmodule NxTest do
  use ExUnit.Case, async: true

  doctest Nx

  defp commute(a, b, fun) do
    fun.(a, b)
    fun.(b, a)
  end

  describe "binary broadcast" do
    test "{2, 1} + {1, 2}" do
      commute(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<11::64-native, 21::64-native, 12::64-native, 22::64-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end

    test "{2} + {2, 2}" do
      commute(Nx.tensor([1, 2]), Nx.tensor([[1, 2], [3, 4]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<2::64-native, 4::64-native, 4::64-native, 6::64-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end

    test "{2, 1, 2} + {1, 2, 1}" do
      a = Nx.tensor([[[10], [20]]])
      assert Nx.shape(a) == {1, 2, 1}
      b = Nx.tensor([[[1, 2]], [[3, 4]]])
      assert Nx.shape(b) == {2, 1, 2}

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)
        assert Nx.shape(t) == {2, 2, 2}

        assert Nx.to_binary(t) ==
                 <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 13::64-native,
                   14::64-native, 23::64-native, 24::64-native>>
      end)
    end

    test "{4, 1, 3} + {1, 3, 1}" do
      a = Nx.tensor([[[100], [200], [300]]])
      assert Nx.shape(a) == {1, 3, 1}
      b = Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]])
      assert Nx.shape(b) == {4, 1, 3}

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)
        assert Nx.shape(t) == {4, 3, 3}

        assert Nx.to_binary(t) ==
                 <<101::64-native, 102::64-native, 103::64-native, 201::64-native, 202::64-native,
                   203::64-native, 301::64-native, 302::64-native, 303::64-native, 104::64-native,
                   105::64-native, 106::64-native, 204::64-native, 205::64-native, 206::64-native,
                   304::64-native, 305::64-native, 306::64-native, 107::64-native, 108::64-native,
                   109::64-native, 207::64-native, 208::64-native, 209::64-native, 307::64-native,
                   308::64-native, 309::64-native, 110::64-native, 111::64-native, 112::64-native,
                   210::64-native, 211::64-native, 212::64-native, 310::64-native, 311::64-native,
                   312::64-native>>
      end)
    end

    test "{2, 1, 2} + {1, 2, 2, 1}" do
      a = Nx.tensor([[[[10], [20]], [[30], [40]]]])
      assert Nx.shape(a) == {1, 2, 2, 1}
      b = Nx.tensor([[[1, 2]], [[3, 4]]])
      assert Nx.shape(b) == {2, 1, 2}

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)
        assert Nx.shape(t) == {1, 2, 2, 2}

        assert Nx.to_binary(t) ==
                 <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 33::64-native,
                   34::64-native, 43::64-native, 44::64-native>>
      end)
    end

    test "{2, 1, 1, 2} + {1, 2, 2, 1}" do
      a = Nx.tensor([[[[10], [20]], [[30], [40]]]])
      assert Nx.shape(a) == {1, 2, 2, 1}
      b = Nx.tensor([[[[1, 2]]], [[[3, 4]]]])
      assert Nx.shape(b) == {2, 1, 1, 2}

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)
        assert Nx.shape(t) == {2, 2, 2, 2}

        assert Nx.to_binary(t) ==
                 <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 31::64-native,
                   32::64-native, 41::64-native, 42::64-native, 13::64-native, 14::64-native,
                   23::64-native, 24::64-native, 33::64-native, 34::64-native, 43::64-native,
                   44::64-native>>
      end)
    end

    test "{1, 2, 2} + {2, 1, 2}" do
      a = Nx.tensor([[[10], [20]], [[30], [40]]])
      b = Nx.tensor([[[1, 2]], [[3, 4]]])

      commute(a, b, fn a, b ->
        t = Nx.add(a, b)

        assert Nx.to_binary(t) ==
                 <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 33::64-native,
                   34::64-native, 43::64-native, 44::64-native>>

        assert Nx.shape(t) == {2, 2, 2}
      end)
    end

    test "raises when it cannot broadcast" do
      a = Nx.tensor([[1, 2], [3, 4]])
      b = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      assert_raise ArgumentError, ~r"cannot broadcast", fn -> Nx.add(a, b) end
    end
  end

  describe "convolution" do
    test "valid padding, no stride" do
      t1 = Nx.iota({2, 2, 4, 4}, type: {:f, 32})
      k1 = Nx.iota({1, 2, 2, 2}, type: {:f, 32})
      k2 = Nx.iota({8, 2, 4, 4}, type: {:f, 32})
      k3 = Nx.iota({2, 2, 3, 3}, type: {:f, 32})

      assert Nx.conv(t1, k1, strides: [1, 1], padding: :valid) ==
               Nx.tensor(
                 [
                   [
                     [
                       [440.0, 468.0, 496.0],
                       [552.0, 580.0, 608.0],
                       [664.0, 692.0, 720.0]
                     ]
                   ],
                   [
                     [
                       [1336.0, 1364.0, 1392.0],
                       [1448.0, 1476.0, 1504.0],
                       [1560.0, 1588.0, 1616.0]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )

      assert Nx.conv(t1, k2, strides: [1, 1], padding: :valid) ==
               Nx.tensor(
                 [
                   [
                     [[10416.0]],
                     [[26288.0]],
                     [[42160.0]],
                     [[58032.0]],
                     [[73904.0]],
                     [[89776.0]],
                     [[105_648.0]],
                     [[121_520.0]]
                   ],
                   [
                     [[26288.0]],
                     [[74928.0]],
                     [[123_568.0]],
                     [[172_208.0]],
                     [[220_848.0]],
                     [[269_488.0]],
                     [[318_128.0]],
                     [[366_768.0]]
                   ]
                 ],
                 type: {:f, 32}
               )

      assert Nx.conv(t1, k3, strides: [1, 1], padding: :valid) ==
               Nx.tensor(
                 [
                   [
                     [
                       [2793.0, 2946.0],
                       [3405.0, 3558.0]
                     ],
                     [
                       [7005.0, 7482.0],
                       [8913.0, 9390.0]
                     ]
                   ],
                   [
                     [
                       [7689.0, 7842.0],
                       [8301.0, 8454.0]
                     ],
                     [
                       [22269.0, 22746.0],
                       [24177.0, 24654.0]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end

    test "valid padding, stride" do
      t1 = Nx.iota({2, 2, 4, 4}, type: {:f, 32})
      k1 = Nx.iota({1, 2, 2, 2}, type: {:f, 32})

      assert Nx.conv(t1, k1, strides: [2, 1], padding: :valid) ==
               Nx.tensor(
                 [
                   [
                     [
                       [440.0, 468.0, 496.0],
                       [664.0, 692.0, 720.0]
                     ]
                   ],
                   [
                     [
                       [1336.0, 1364.0, 1392.0],
                       [1560.0, 1588.0, 1616.0]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end

    test "same padding, no stride" do
      t1 = Nx.iota({2, 2, 4, 4}, type: {:f, 32})
      k1 = Nx.iota({1, 2, 3, 3}, type: {:f, 32})

      assert Nx.conv(t1, k1, strides: [1, 1], padding: :same) ==
               Nx.tensor(
                 [
                   [
                     [
                       [1196.0, 1796.0, 1916.0, 1264.0],
                       [1881.0, 2793.0, 2946.0, 1923.0],
                       [2313.0, 3405.0, 3558.0, 2307.0],
                       [1424.0, 2072.0, 2156.0, 1380.0]
                     ]
                   ],
                   [
                     [
                       [3884.0, 5636.0, 5756.0, 3696.0],
                       [5337.0, 7689.0, 7842.0, 4995.0],
                       [5769.0, 8301.0, 8454.0, 5379.0],
                       [3344.0, 4760.0, 4844.0, 3044.0]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end

    test "3d conv" do
      t1 = Nx.iota({1, 3, 2, 2, 2}, type: {:f, 32})
      k1 = Nx.iota({3, 3, 2, 2, 2}, type: {:f, 32})

      assert Nx.conv(t1, k1, strides: [1, 1, 1], padding: :same) ==
               Nx.tensor(
                 [
                   [
                     [
                       [
                         [4324.0, 2156.0],
                         [2138.0, 1060.0]
                       ],
                       [
                         [2066.0, 1018.0],
                         [997.0, 488.0]
                       ]
                     ],
                     [
                       [
                         [10948.0, 5612.0],
                         [5738.0, 2932.0]
                       ],
                       [
                         [5954.0, 3034.0],
                         [3085.0, 1568.0]
                       ]
                     ],
                     [
                       [
                         [17572.0, 9068.0],
                         [9338.0, 4804.0]
                       ],
                       [
                         [9842.0, 5050.0],
                         [5173.0, 2648.0]
                       ]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end
  end

  test "interior padding" do
    assert Nx.pad(Nx.tensor([[1, 2, 3], [4, 5, 6]]), 0, [{0, 0, 1}, {0, 0, 1}]) ==
             Nx.tensor([[1, 0, 2, 0, 3], [0, 0, 0, 0, 0], [4, 0, 5, 0, 6]])

    assert Nx.pad(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 2.0, [{-2, 1, 4}, {1, 3, 2}]) ==
             Nx.tensor([
               [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
               [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
               [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
               [2.0, 4.0, 2.0, 2.0, 5.0, 2.0, 2.0, 6.0, 2.0, 2.0, 2.0],
               [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
             ])
  end

  describe "reverse" do
    test "works on all dims" do
      assert Nx.reverse(Nx.iota({10})) == Nx.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
      assert Nx.reverse(Nx.iota({2, 4})) == Nx.tensor([[7, 6, 5, 4], [3, 2, 1, 0]])

      assert Nx.reverse(Nx.iota({3, 3, 3})) ==
               Nx.tensor([
                 [[26, 25, 24], [23, 22, 21], [20, 19, 18]],
                 [[17, 16, 15], [14, 13, 12], [11, 10, 9]],
                 [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
               ])

      assert Nx.reverse(Nx.iota({2, 1, 4, 2})) ==
               Nx.tensor([
                 [[[15, 14], [13, 12], [11, 10], [9, 8]]],
                 [[[7, 6], [5, 4], [3, 2], [1, 0]]]
               ])
    end

    test "works with 1 dim" do
      assert Nx.reverse(Nx.iota({2, 3}), axes: [1]) == Nx.tensor([[2, 1, 0], [5, 4, 3]])
    end

    test "works with 2 dims" do
      assert Nx.reverse(Nx.iota({2, 3, 4}), axes: [0, 2]) ==
               Nx.tensor([
                 [
                   [15, 14, 13, 12],
                   [19, 18, 17, 16],
                   [23, 22, 21, 20]
                 ],
                 [
                   [3, 2, 1, 0],
                   [7, 6, 5, 4],
                   [11, 10, 9, 8]
                 ]
               ])
    end

    test "works with 3 dims" do
      assert Nx.reverse(Nx.iota({2, 2, 1, 3, 4}), axes: [1, 2, 4]) ==
               Nx.tensor([
                 [
                   [
                     [
                       [15, 14, 13, 12],
                       [19, 18, 17, 16],
                       [23, 22, 21, 20]
                     ]
                   ],
                   [
                     [
                       [3, 2, 1, 0],
                       [7, 6, 5, 4],
                       [11, 10, 9, 8]
                     ]
                   ]
                 ],
                 [
                   [
                     [
                       [39, 38, 37, 36],
                       [43, 42, 41, 40],
                       [47, 46, 45, 44]
                     ]
                   ],
                   [
                     [
                       [27, 26, 25, 24],
                       [31, 30, 29, 28],
                       [35, 34, 33, 32]
                     ]
                   ]
                 ]
               ])
    end

    test "works across types" do
      assert Nx.reverse(Nx.iota({2, 2, 3, 1, 2}, type: {:f, 32}), axes: [1, 2, 4]) ==
               Nx.tensor(
                 [
                   [
                     [
                       [[11.0, 10.0]],
                       [[9.0, 8.0]],
                       [[7.0, 6.0]]
                     ],
                     [
                       [[5.0, 4.0]],
                       [[3.0, 2.0]],
                       [[1.0, 0.0]]
                     ]
                   ],
                   [
                     [
                       [[23.0, 22.0]],
                       [[21.0, 20.0]],
                       [[19.0, 18.0]]
                     ],
                     [
                       [[17.0, 16.0]],
                       [[15.0, 14.0]],
                       [[13.0, 12.0]]
                     ]
                   ]
                 ],
                 type: {:f, 32}
               )
    end
  end

  describe "inspect" do
    test "scalar" do
      assert inspect(Nx.tensor(123)) == """
             #Nx.Tensor<
               s64
               123
             >\
             """
    end

    test "n-dimensional" do
      assert inspect(Nx.tensor([[1, 2, 3], [4, 5, 6]])) == """
             #Nx.Tensor<
               s64[2][3]
               [
                 [1, 2, 3],
                 [4, 5, 6]
               ]
             >\
             """

      assert inspect(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) == """
             #Nx.Tensor<
               f64[2][3]
               [
                 [1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]
               ]
             >\
             """
    end

    test "limit" do
      assert inspect(Nx.tensor([1, 2]), limit: :infinity) == """
             #Nx.Tensor<
               s64[2]
               [1, 2]
             >\
             """

      assert inspect(Nx.tensor([[1, 2], [3, 4]]), limit: 3) == """
             #Nx.Tensor<
               s64[2][2]
               [
                 [1, 2],
                 [3, ...]
               ]
             >\
             """

      assert inspect(Nx.tensor([[1, 2], [3, 4], [5, 6]]), limit: 3) == """
             #Nx.Tensor<
               s64[3][2]
               [
                 [1, 2],
                 [3, ...],
                 ...
               ]
             >\
             """
    end

    test "infinity and nan for bf16" do
      bin = <<0xFF80::16-native, 0x7F80::16-native, 0xFFC1::16-native, 0xFF81::16-native>>

      assert inspect(Nx.from_binary(bin, {:bf, 16})) == """
             #Nx.Tensor<
               bf16[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "infinity and nan for f32" do
      bin =
        <<0xFF800000::32-native, 0x7F800000::32-native, 0xFF800001::32-native,
          0xFFC00001::32-native>>

      # Assert that none of them are indeed valid
      assert for(<<x::float-32-native <- bin>>, do: x) == []

      assert inspect(Nx.from_binary(bin, {:f, 32})) == """
             #Nx.Tensor<
               f32[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "infinity and nan for f64" do
      bin =
        <<0xFFF0000000000000::64-native, 0x7FF0000000000000::64-native,
          0x7FF0000000000001::64-native, 0x7FF8000000000001::64-native>>

      # Assert that none of them are indeed valid
      assert for(<<x::float-64-native <- bin>>, do: x) == []

      assert inspect(Nx.from_binary(bin, {:f, 64})) == """
             #Nx.Tensor<
               f64[4]
               [-Inf, Inf, NaN, NaN]
             >\
             """
    end

    test "all dimensions named" do
      assert inspect(Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:batch, :x, :y])) ==
               """
               #Nx.Tensor<
                 s64[batch: 1][x: 3][y: 3]
                 [
                   [
                     [1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]
                   ]
                 ]
               >\
               """
    end

    test "some dimensions named" do
      assert inspect(Nx.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], names: [:batch, nil, nil])) ==
               """
               #Nx.Tensor<
                 s64[batch: 1][3][3]
                 [
                   [
                     [1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]
                   ]
                 ]
               >\
               """
    end
  end

  describe "window aggregates" do
    test "computes a window sum" do
      assert Nx.window_sum(
               Nx.tensor([[[4, 2, 1, 3], [4, 2, 1, 7]], [[1, 2, 5, 7], [1, 8, 9, 2]]]),
               {2, 1, 2},
               strides: [2, 1, 1],
               padding: [{2, 1}, {3, 1}, {1, 0}],
               window_dilations: [1, 2, 2]
             ) ==
               Nx.tensor([
                 [
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]
                 ],
                 [
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [4, 11, 14],
                   [10, 15, 19],
                   [0, 0, 0]
                 ]
               ])
    end
  end

  describe "access" do
    test "supports an empty list" do
      tensor = Nx.tensor([1, 2, 3])
      assert tensor[[]] == tensor
    end

    test "raises on duplicated axis" do
      tensor = Nx.tensor([[1, 2, 3]], names: [:x, :y])

      assert_raise ArgumentError,
                   ~r"unknown or duplicate axis 1 found when slicing shape {1, 3}",
                   fn -> tensor[[y: 1, y: 2]] end
    end

    test "raises on invalid ranges" do
      tensor = Nx.tensor([[1, 2, 3]], names: [:x, :y])

      assert_raise ArgumentError,
                   ~r"slicing a tensor requires an increasing range, got: 2..1",
                   fn -> tensor[[y: 2..1]] end
    end
  end

  describe "quotient/2" do
    test "raises for non-integer values" do
      msg = "quotient expects integer tensors as inputs and outputs an integer tensor, got: {:f, 64}"

      assert_raise ArgumentError, msg, fn ->
        Nx.quotient(10, 1.0)
      end

      assert_raise ArgumentError, msg, fn ->
        Nx.quotient(10.0, 2)
      end
    end
  end

  describe "reshape" do
    test "correctly adds names with same shape" do
      t = Nx.tensor([[1, 2], [3, 4]])
      assert Nx.reshape(t, {2, 2}, names: [:x, :y]) == Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
    end
  end

  describe "broadcast" do
    test "correctly adds names with same shape" do
      t = Nx.tensor([[1, 2], [3, 4]])
      assert Nx.broadcast(t, {2, 2}, names: [:x, :y]) == Nx.tensor([[1, 2], [3, 4]], names: [:x, :y])
    end
  end
end
