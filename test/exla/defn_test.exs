defmodule Exla.DefnTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  @default_defn_compiler Exla

  defn add_two(a, b), do: a + b

  test "adds two tensors" do
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
