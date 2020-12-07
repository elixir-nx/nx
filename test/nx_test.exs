defmodule NxTest do
  use ExUnit.Case, async: true

  doctest Nx
  doctest Nx.Util

  defp commute(a, b, fun) do
    fun.(a, b)
    fun.(b, a)
  end

  describe "unary broadcast" do
    test "{2, 1} x {1, 2}" do
      t = Nx.broadcast(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))

      assert Nx.Util.to_bitstring(t) ==
               <<1::64-native, 1::64-native, 2::64-native, 2::64-native>>

      assert Nx.shape(t) == {2, 2}
    end

    test "{2} + {2, 2}" do
      t = Nx.broadcast(Nx.tensor([1, 2]), Nx.tensor([[1, 2], [3, 4]]))

      assert Nx.Util.to_bitstring(t) ==
               <<1::64-native, 2::64-native, 1::64-native, 2::64-native>>

      assert Nx.shape(t) == {2, 2}
    end

    test "{2, 1, 2} + {1, 2, 1}" do
      a = Nx.tensor([[[1, 2]], [[3, 4]]])
      assert Nx.shape(a) == {2, 1, 2}

      b = Nx.tensor([[[10], [20]]])
      assert Nx.shape(b) == {1, 2, 1}

      t = Nx.broadcast(a, b)
      assert Nx.shape(t) == {2, 2, 2}

      assert Nx.Util.to_bitstring(t) ==
               <<1::64-native, 2::64-native, 1::64-native, 2::64-native, 3::64-native,
                 4::64-native, 3::64-native, 4::64-native>>
    end

    test "{4, 1, 3} + {1, 3, 1}" do
      a = Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]])
      assert Nx.shape(a) == {4, 1, 3}

      b = Nx.tensor([[[100], [200], [300]]])
      assert Nx.shape(b) == {1, 3, 1}

      t = Nx.broadcast(a, b)
      assert Nx.shape(t) == {4, 3, 3}

      assert Nx.Util.to_bitstring(t) ==
               <<01::64-native, 02::64-native, 03::64-native, 01::64-native, 02::64-native,
                 03::64-native, 01::64-native, 02::64-native, 03::64-native, 04::64-native,
                 05::64-native, 06::64-native, 04::64-native, 05::64-native, 06::64-native,
                 04::64-native, 05::64-native, 06::64-native, 07::64-native, 08::64-native,
                 09::64-native, 07::64-native, 08::64-native, 09::64-native, 07::64-native,
                 08::64-native, 09::64-native, 10::64-native, 11::64-native, 12::64-native,
                 10::64-native, 11::64-native, 12::64-native, 10::64-native, 11::64-native,
                 12::64-native>>
    end

    test "{2, 1, 2} + {1, 2, 2, 1}" do
      a = Nx.tensor([[[1, 2]], [[3, 4]]])
      assert Nx.shape(a) == {2, 1, 2}

      b = Nx.tensor([[[[10], [20]], [[30], [40]]]])
      assert Nx.shape(b) == {1, 2, 2, 1}

      t = Nx.broadcast(a, b)
      assert Nx.shape(t) == {1, 2, 2, 2}

      assert Nx.Util.to_bitstring(t) ==
               <<1::64-native, 2::64-native, 1::64-native, 2::64-native, 3::64-native,
                 4::64-native, 3::64-native, 4::64-native>>
    end

    test "{2, 1, 1, 2} + {1, 2, 2, 1}" do
      a = Nx.tensor([[[[1, 2]]], [[[3, 4]]]])
      assert Nx.shape(a) == {2, 1, 1, 2}

      b = Nx.tensor([[[[10], [20]], [[30], [40]]]])
      assert Nx.shape(b) == {1, 2, 2, 1}

      t = Nx.broadcast(a, b)
      assert Nx.shape(t) == {2, 2, 2, 2}

      assert Nx.Util.to_bitstring(t) ==
               <<1::64-native, 2::64-native, 1::64-native, 2::64-native, 1::64-native,
                 2::64-native, 1::64-native, 2::64-native, 3::64-native, 4::64-native,
                 3::64-native, 4::64-native, 3::64-native, 4::64-native, 3::64-native,
                 4::64-native>>
    end

    test "{1, 2, 2} + {2, 1, 2}" do
      a = Nx.tensor([[[1, 2]], [[3, 4]]])
      b = Nx.tensor([[[10], [20]], [[30], [40]]])

      t = Nx.broadcast(a, b)

      assert Nx.Util.to_bitstring(t) ==
               <<1::64-native, 2::64-native, 1::64-native, 2::64-native, 3::64-native,
                 4::64-native, 3::64-native, 4::64-native>>

      assert Nx.shape(t) == {2, 2, 2}
    end

    test "raises when it cannot broadcast" do
      a = Nx.tensor([[1, 2], [3, 4]])
      b = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      assert_raise ArgumentError, ~r"cannot broadcast", fn -> Nx.broadcast(a, b) end
    end
  end

  describe "binary broadcast" do
    test "{2, 1} + {1, 2}" do
      commute(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.Util.to_bitstring(t) ==
                 <<11::64-native, 21::64-native, 12::64-native, 22::64-native>>

        assert Nx.shape(t) == {2, 2}
      end)
    end

    test "{2} + {2, 2}" do
      commute(Nx.tensor([1, 2]), Nx.tensor([[1, 2], [3, 4]]), fn a, b ->
        t = Nx.add(a, b)

        assert Nx.Util.to_bitstring(t) ==
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

        assert Nx.Util.to_bitstring(t) ==
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

        assert Nx.Util.to_bitstring(t) ==
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

        assert Nx.Util.to_bitstring(t) ==
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

        assert Nx.Util.to_bitstring(t) ==
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

        assert Nx.Util.to_bitstring(t) ==
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

  test "all broadcast" do
    tensors = [
      {Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([0, 0])},
      {Nx.tensor([[[1, 2], [3, 4]], [[4, 5], [6, 7]]]), Nx.tensor([0, 0])},
      {Nx.tensor([[1], [2]]), Nx.tensor([[0, 0]])},
      {Nx.tensor([[[1, 2]], [[3, 4]]]), Nx.tensor([[[0], [0]]])},
      {Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]), Nx.tensor([[[0], [0], [0]]])},
      {Nx.tensor([[1, 2], [3, 4]]), Nx.tensor([[[[0]]]])},
      {Nx.tensor([1, 2]), Nx.tensor([[[[0]]]])},
      {Nx.tensor([[1, 2]]), Nx.tensor([[[0], [0]], [[0], [0]]])},
      {Nx.tensor([[[1, 2]], [[3, 4]]]), Nx.tensor([[[[0], [0]], [[0], [0]]]])},
      {Nx.tensor([[[[1, 2]]], [[[3, 4]]]]), Nx.tensor([[[[0], [0]], [[0], [0]]]])},
      {Nx.tensor([[[1, 2]], [[3, 4]]]), Nx.tensor([[[0], [0]], [[0], [0]]])},
    ]

    for {left, right} <- tensors do
      assert Nx.add(left, right) == Nx.broadcast(left, right)
    end
  end
end
