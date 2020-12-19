defmodule DefnNewTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  @default_defn_compiler Nx.Defn.New

  describe "exp" do
    defn exp(t), do: Nx.exp(t)

    test "works on tensors" do
      assert Nx.tensor([1, 2, 3]) |> exp() ==
         Nx.tensor([2.718281828459045, 7.38905609893065, 20.085536923187668])
    end
  end
end