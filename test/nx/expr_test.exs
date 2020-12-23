defmodule Nx.Defn.ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr

  test "inspect" do
    a = Expr.parameter({2, 2}, "x")
    b = Expr.parameter({2, 2}, "y")
    c = Expr.parameter({2, 2}, "z")
    d = Nx.tensor([[1, 2], [1, 2]])

    assert Expr.sum(Expr.add(Expr.add(Expr.dot(a, a), Expr.tanh(b)), 2)) |> inspect() == """
           param[2][2] x
           param[2][2] y
           a = tanh [ y ]
           b = dot [ x, x ]
           c = add [ b, a ]
           d = add [ c, 2 ]
           e = sum [ d, [] ]\
           """

    assert Expr.argmin(Expr.add(Expr.tanh(Expr.dot(Expr.iota({2, 2}), d)), c), tie_break: :high)
           |> inspect() == """
           param[2][2] z
           tensor[2][2] a
           b = iota [ {2, 2}, [] ]
           c = dot [ b, a ]
           d = tanh [ c ]
           e = add [ d, z ]
           f = argmin [ e, tie_break: :high ]\
           """
  end
end
