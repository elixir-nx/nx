defmodule Nx.GradTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  describe "tanh" do
    defn grad_tanh(t), do: grad(t, Nx.tanh(t))

    test "works" do
      assert grad_tanh(Nx.tensor(1.0)) == Nx.tensor(0.41997434161402614)
    end
  end
end