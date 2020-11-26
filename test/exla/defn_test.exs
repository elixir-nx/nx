defmodule Exla.DefnTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  @default_defn_compiler Exla

  describe "+/2" do
    defn add_two(a, b), do: a + b

    test "same shape and type" do
      tensor = add_two(1.0, 2.0)
      assert Nx.to_bitstring(tensor) == <<3.0::float-64-native>>
      assert Nx.type(tensor) == {:f, 64}
      assert Nx.shape(tensor) == {}

      tensor = add_two(1, 2)
      assert Nx.to_bitstring(tensor) == <<3::64-native>>
      assert Nx.type(tensor) == {:s, 64}
      assert Nx.shape(tensor) == {}

      tensor = add_two(Nx.tensor([1, 2]), Nx.tensor([3, 4]))
      assert Nx.to_bitstring(tensor) == <<4::64-native, 6::64-native>>
      assert Nx.type(tensor) == {:s, 64}
      assert Nx.shape(tensor) == {2}

      tensor = add_two(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0]))
      assert Nx.to_bitstring(tensor) == <<4.0::float-64-native, 6.0::float-64-native>>
      assert Nx.type(tensor) == {:f, 64}
      assert Nx.shape(tensor) == {2}
    end

    test "broadcast" do
      tensors = [
        {Nx.tensor([[1], [2]]), Nx.tensor([[10, 20]])},
        {Nx.tensor([[10, 20]]), Nx.tensor([[1], [2]])},
        {Nx.tensor([[[10], [20]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[100], [200], [300]]]),
         Nx.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([[1, 2], [3, 4]])},
        {Nx.tensor([[[[1]]]]), Nx.tensor([1, 2])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[1, 2]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[1, 2]], [[3, 4]]])},
        {Nx.tensor([[[[10], [20]], [[30], [40]]]]), Nx.tensor([[[[1, 2]]], [[[3, 4]]]])},
        {Nx.tensor([[[10], [20]], [[30], [40]]]), Nx.tensor([[[1, 2]], [[3, 4]]])}
      ]

      for {left, right} <- tensors do
        exla = add_two(left, right)
        nx = Nx.add(left, right)
        assert Nx.type(exla) == Nx.type(nx)
        assert Nx.shape(exla) == Nx.shape(nx)
        assert Nx.to_bitstring(exla) == Nx.to_bitstring(nx)

        exla = add_two(right, left)
        nx = Nx.add(right, left)
        assert Nx.type(exla) == Nx.type(nx)
        assert Nx.shape(exla) == Nx.shape(nx)
        assert Nx.to_bitstring(exla) == Nx.to_bitstring(nx)
      end
    end

    test "broadcast error" do
      assert_raise RuntimeError, ~r"Binary op add with incompatible shapes", fn ->
        add_two(Nx.tensor([1, 2]), Nx.tensor([1, 2, 3]))
      end
    end
  end

  describe "sum" do
    defn sum(t), do: Nx.sum(t)

    test "compures the sum across types" do
      assert Nx.tensor([1, 2, 3]) |> sum() |> Nx.to_bitstring() ==
               <<6::64-native>>

      assert Nx.tensor([1, 2, 3], type: {:s, 8}) |> sum() |> Nx.to_bitstring() ==
               <<6::8-native>>

      assert Nx.tensor([1, 2, 3], type: {:u, 8}) |> sum() |> Nx.to_bitstring() ==
               <<6::8-native>>

      assert Nx.tensor([1.0, 2.0, 3.0]) |> sum() |> Nx.to_bitstring() ==
               <<6::64-float-native>>

      assert Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}) |> sum() |> Nx.to_bitstring() ==
               <<6::32-float-native>>
    end
  end

  describe "softmax" do
    defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

    test "computes softmax" do
      tensor = softmax(Nx.tensor([1.0, 2.0, 3.0, 4.0]))

      assert Nx.to_bitstring(tensor) ==
               <<0.03205860328008499::float-64-native, 0.08714431874203257::float-64-native,
                 0.23688281808991013::float-64-native, 0.6439142598879722::float-64-native>>

      assert Nx.type(tensor) == {:f, 64}
      assert Nx.shape(tensor) == {4}
    end
  end
end
