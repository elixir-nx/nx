defmodule DefnNewTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  alias Nx.Defn.Expr

  @default_defn_compiler Nx.Defn.New

  describe "exp" do
    defn exp(t), do: Nx.exp(t)

    test "to expr" do
      assert %Expr{op: :exp, args: [_], shape: {3}} = exp(Nx.tensor([1, 2, 3]))
    end
  end
end