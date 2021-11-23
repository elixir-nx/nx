defmodule Nx.Defn.TokenTest do
  use ExUnit.Case, async: true

  test "Inspect" do
    token = Nx.Defn.Token.new()
    assert inspect(token) == "#Nx.Defn.Token<[]>"

    token = Nx.Defn.Token.add_hook(token, Nx.tensor(1), :example, nil)
    assert inspect(token) == "#Nx.Defn.Token<[:example]>"

    token = Nx.Defn.Token.add_hook(token, Nx.tensor(1), :later, nil)
    assert inspect(token) == "#Nx.Defn.Token<[:later, :example]>"
  end

  import Nx.Defn

  defn token(a, b) do
    token = create_token()
    {token, _expr} = hook_token(token, a + b, :example, &Function.identity/1)
    token
  end

  test "Nx.Container" do
    assert token(1, 2) == %Nx.Defn.Token{
      hooks: [%{name: :example, callback: &Function.identity/1, expr: Nx.tensor(3)}]
    }
  end
end
