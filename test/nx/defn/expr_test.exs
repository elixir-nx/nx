defmodule Nx.Defn.ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr

  test "with parameters" do
    a = Expr.parameter(nil, {:s, 64}, {2, 2}, 0)
    b = Expr.parameter(nil, {:s, 64}, {2, 2}, 1)

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
  end

  test "with tensors" do
    a = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)
    b = Nx.tensor([[1, 2], [1, 2]])

    assert Nx.argmin(Nx.add(Nx.tanh(Nx.dot(Expr.iota({2, 2}, []), b)), a), tie_break: :high)
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

  test "with fun" do
    a = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)

    assert Nx.reduce(a, 0, [], &Nx.add/2) |> inspect() == """
           #Nx.Tensor<
             Nx.Defn.Expr
             parameter a                                s64[2][2]
             b = reduce [ a, 0, axes: nil, &Nx.add/2 ]  s64
           >\
           """
  end
end
