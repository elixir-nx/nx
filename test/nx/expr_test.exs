defmodule ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr

  test "inspect" do
    a = Expr.parameter({2, 2}, "x")
    b = Expr.parameter({2, 2}, "y")
    c = Expr.parameter({2, 2}, "z")
    d = Nx.tensor([[1, 2], [1, 2]])

    assert inspect(Expr.add(Expr.add(Expr.dot(a, a), Expr.tanh(b)), 2)) ==
             "param[2][2] x\nparam[2][2] y\na = tanh [ y ]\nb = dot [ x, x ]\nc = add [ b, a ]\nd = add [ c, 2 ]"

    assert inspect(Expr.sum(Expr.add(Expr.tanh(Expr.dot(Expr.iota({2, 2}), d)), c))) ==
             "param[2][2] z\ntensor[2][2] a\nb = iota [ {2, 2}, axis: 1 ]\nc = dot [ b, a ]\nd = tanh [ c ]\ne = add [ d, z ]\nf = sum [ e, [] ]"
  end
end
