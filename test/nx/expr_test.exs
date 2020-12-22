defmodule ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr
  import Inspect.Algebra

  test "inspect" do
    a = Expr.parameter({2, 2}, "x")
    b = Expr.parameter({2, 2}, "y")
    assert inspect(Expr.add(Expr.dot(a, a), Expr.tanh(b))) == "param[2][2] x\nparam[2][2] y\na = tanh [ y ]\nb = dot [ x, x ]\nc = add [ b, a ]"
  end
end