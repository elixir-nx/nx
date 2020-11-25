defmodule Exla.DefnTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  @default_defn_compiler Exla

  defn add_two(a, b), do: a + b

  test "adds two tensors" do
    buffer = add_two(1.0, 2.0)
    assert buffer.data == <<3.0::float-64-native>>

    buffer = add_two(1, 2)
    assert buffer.data == <<3::64-native>>

    # buffer = add_two(Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0]))
    # assert buffer.data == <<4.0::float-64-native, 6.0::float-64-native>>

    # buffer = add_two(Nx.tensor([1, 2]), Nx.tensor([3, 4]))
    # assert buffer.data == <<4::64-native, 6::64-native>>
  end

  defn softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t))

  test "computes softmax" do
    buffer = softmax(Nx.tensor([1.0, 2.0, 3.0, 4.0]))

    assert buffer.data ==
             <<0.03205860328008499::float-64-native, 0.08714431874203257::float-64-native,
               0.23688281808991013::float-64-native, 0.6439142598879722::float-64-native>>
  end
end
