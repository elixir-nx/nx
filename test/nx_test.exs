defmodule NxTest do
  use ExUnit.Case, async: true

  doctest Nx

  describe "broadcast" do
    test "{2, 1} + {1, 2}" do
      t = Nx.add(Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]]))
      assert Nx.to_bitstring(t) == <<11::64-native, 21::64-native, 12::64-native, 22::64-native>>
      assert Nx.shape(t) == {2, 2}

      t = Nx.add(Nx.tensor([[10, 20]]), Nx.tensor([[1], [2]]))
      assert Nx.to_bitstring(t) == <<11::64-native, 21::64-native, 12::64-native, 22::64-native>>
      assert Nx.shape(t) == {2, 2}
    end

    test "{2, 1, 2} + {1, 2, 1}" do
      a = Nx.tensor([[[10], [20]]])
      assert Nx.shape(a) == {1, 2, 1}
      b = Nx.tensor([[[1, 2]], [[3, 4]]])
      assert Nx.shape(b) == {2, 1, 2}

      t = Nx.add(a, b)
      assert Nx.shape(t) == {2, 2, 2}

      assert Nx.to_bitstring(t) ==
               <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 13::64-native,
                 14::64-native, 23::64-native, 24::64-native>>

      t = Nx.add(b, a)
      assert Nx.shape(t) == {2, 2, 2}

      assert Nx.to_bitstring(t) ==
               <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 13::64-native,
                 14::64-native, 23::64-native, 24::64-native>>
    end

    test "{2, 1, 2} + {1, 2, 2, 1}" do
      a = Nx.tensor([[[[10], [20]], [[30], [40]]]])
      assert Nx.shape(a) == {1, 2, 2, 1}
      b = Nx.tensor([[[1, 2]], [[3, 4]]])
      assert Nx.shape(b) == {2, 1, 2}

      t = Nx.add(a, b)
      assert Nx.shape(t) == {1, 2, 2, 2}

      assert Nx.to_bitstring(t) ==
               <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 33::64-native,
                 34::64-native, 43::64-native, 44::64-native>>

      t = Nx.add(b, a)
      assert Nx.shape(t) == {1, 2, 2, 2}

      assert Nx.to_bitstring(t) ==
               <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 33::64-native,
                 34::64-native, 43::64-native, 44::64-native>>
    end

    test "{2, 1, 1, 2} + {1, 2, 2, 1}" do
      a = Nx.tensor([[[[10], [20]], [[30], [40]]]])
      assert Nx.shape(a) == {1, 2, 2, 1}
      b = Nx.tensor([[[[1, 2]]], [[[3, 4]]]])
      assert Nx.shape(b) == {2, 1, 1, 2}

      t = Nx.add(a, b)
      assert Nx.shape(t) == {2, 2, 2, 2}

      assert Nx.to_bitstring(t) ==
               <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 31::64-native,
                 32::64-native, 41::64-native, 42::64-native, 13::64-native, 14::64-native,
                 23::64-native, 24::64-native, 33::64-native, 34::64-native, 43::64-native,
                 44::64-native>>

      t = Nx.add(b, a)
      assert Nx.shape(t) == {2, 2, 2, 2}

      assert Nx.to_bitstring(t) ==
               <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 31::64-native,
                 32::64-native, 41::64-native, 42::64-native, 13::64-native, 14::64-native,
                 23::64-native, 24::64-native, 33::64-native, 34::64-native, 43::64-native,
                 44::64-native>>
    end

    test "{1, 2, 2} + {2, 1, 2}" do
      a = Nx.tensor([[[10], [20]], [[30], [40]]])
      b = Nx.tensor([[[1, 2]], [[3, 4]]])
      t = Nx.add(a, b)

      assert Nx.to_bitstring(t) ==
               <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 33::64-native,
                 34::64-native, 43::64-native, 44::64-native>>

      assert Nx.shape(t) == {2, 2, 2}

      t = Nx.add(b, a)

      assert Nx.to_bitstring(t) ==
               <<11::64-native, 12::64-native, 21::64-native, 22::64-native, 33::64-native,
                 34::64-native, 43::64-native, 44::64-native>>

      assert Nx.shape(t) == {2, 2, 2}
    end
  end
end
