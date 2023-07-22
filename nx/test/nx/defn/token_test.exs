defmodule Nx.Defn.TokenTest do
  use ExUnit.Case, async: true

  test "inspect" do
    token = Nx.Defn.Token.new()
    assert inspect(token) == "#Nx.Defn.Token<[]>"

    token = Nx.Defn.Token.add_hook(token, Nx.tensor(1), :example, nil)
    assert inspect(token) == "#Nx.Defn.Token<[:example]>"

    token = Nx.Defn.Token.add_hook(token, Nx.tensor(1), :later, nil)
    assert inspect(token) == "#Nx.Defn.Token<[:later, :example]>"
  end
end
