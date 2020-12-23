defmodule Nx.Defn.ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr

  test "inspect" do
    a = Expr.parameter({2, 2}, "x")
    b = Expr.parameter({2, 2}, "y")
    c = Expr.parameter({2, 2}, "z")
    d = Nx.tensor([[1, 2], [1, 2]])

    assert Expr.sum(Expr.add(Expr.add(Expr.dot(a, a), Expr.tanh(b)), 2)) |> inspect() == """
           #Nx.Defn.Expr<
             parameter a
             parameter b
             c = tanh [ b ] (2x2)
             d = dot [ a, a ] (2x2)
             e = add [ d, c ] (2x2)
             f = add [ e, 2 ] (2x2)
             g = sum [ f, [] ] ()
           >\
           """

    assert Expr.argmin(Expr.add(Expr.tanh(Expr.dot(Expr.iota({2, 2}), d)), c), tie_break: :high)
           |> inspect() == """
           #Nx.Defn.Expr<
             parameter d
             tensor a
             b = iota [ {2, 2}, [] ] (2x2)
             c = dot [ b, a ] (2x2)
             e = tanh [ c ] (2x2)
             f = add [ e, d ] (2x2)
             g = argmin [ f, tie_break: :high ] ()
           >\
           """
  end
end
