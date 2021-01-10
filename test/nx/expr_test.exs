defmodule Nx.Defn.ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr

  test "inspect" do
    a = Expr.parameter({2, 2}, {:s, 64}, "x")
    b = Expr.parameter({2, 2}, {:s, 64}, "y")
    c = Expr.parameter({2, 2}, {:s, 64}, "z")
    d = Nx.tensor([[1, 2], [1, 2]])

    assert Nx.sum(Nx.add(Nx.add(Nx.dot(a, a), Nx.tanh(b)), 2))
           |> inspect() == """
           #Nx.Tensor<
             Nx.Defn.Expr
             parameter a                 s64[2][2]
             parameter c                 s64[2][2]
             b = dot [ a, [1], a, [0] ]  s64[2][2]
             d = tanh [ c ]              f64[2][2]
             e = add [ b, d ]            f64[2][2]
             f = add [ e, 2 ]            f64[2][2]
             g = sum [ f, axes: nil ]    f64
           >\
           """

    assert Nx.argmin(Nx.add(Nx.tanh(Nx.dot(Expr.iota({2, 2}, []), d)), c), tie_break: :high)
           |> inspect() == """
           #Nx.Tensor<
             Nx.Defn.Expr
             tensor b                                       s64[2][2]
             parameter e                                    s64[2][2]
             a = iota [ nil ]                               s64[2][2]
             c = dot [ a, [1], b, [0] ]                     s64[2][2]
             d = tanh [ c ]                                 f64[2][2]
             f = add [ d, e ]                               f64[2][2]
             g = argmin [ f, tie_break: :high, axis: nil ]  s64
           >\
           """
  end
end
