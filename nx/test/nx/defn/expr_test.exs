defmodule Nx.Defn.ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr

  describe "tensor" do
    test "uses the binary backend" do
      Nx.default_backend(Unknown)
      assert %Nx.Tensor{data: %Expr{op: :tensor, args: [tensor]}} = Expr.tensor(0)
      assert Nx.to_scalar(tensor) == 0
    end
  end

  describe "inspect" do
    test "with scalar" do
      assert inspect(Expr.tensor(Nx.tensor(0)), safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
             >\
             """
    end

    test "with scalar from invalid backend" do
      a = Expr.parameter(nil, {:s, 64}, {2, 2}, 0)
      b = %Nx.Tensor{data: %{__struct__: Unknown}, shape: {}, type: {:s, 64}, names: []}

      assert inspect(Nx.add(a, Expr.tensor(b)), safe: false) == """
             #Nx.Tensor<
               s64[2][2]
             \s\s
               Nx.Defn.Expr
               parameter a            s64[2][2]
               b = add [ a, SCALAR ]  s64[2][2]
             >\
             """
    end


    test "with parameters" do
      a = Expr.parameter(nil, {:s, 64}, {2, 2}, 0)
      b = Expr.parameter(nil, {:s, 64}, {2, 2}, 1)

      assert Nx.sum(Nx.add(Nx.add(Nx.dot(a, a), Nx.tanh(b)), 2))
             |> inspect(safe: false) == """
             #Nx.Tensor<
               f32
             \s\s
               Nx.Defn.Expr
               parameter a                                 s64[2][2]
               parameter c                                 s64[2][2]
               b = dot [ a, [1], a, [0] ]                  s64[2][2]
               d = tanh [ c ]                              f32[2][2]
               e = add [ b, d ]                            f32[2][2]
               f = add [ e, 2 ]                            f32[2][2]
               g = sum [ f, axes: nil, keep_axes: false ]  f32
             >\
             """
    end

    test "with tensors" do
      a = Expr.parameter(nil, {:s, 64}, {2, 2}, 2)
      b = Nx.tensor([[1, 2], [1, 2]])
      c = Nx.iota({2, 2}, backend: Expr)

      assert Nx.argmin(Nx.add(Nx.tanh(Nx.dot(c, b)), a), tie_break: :high)
             |> inspect(safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               tensor b                                       s64[2][2]
               parameter e                                    s64[2][2]
               a = iota [ nil ]                               s64[2][2]
               c = dot [ a, [1], b, [0] ]                     s64[2][2]
               d = tanh [ c ]                                 f32[2][2]
               f = add [ d, e ]                               f32[2][2]
               g = argmin [ f, tie_break: :high, axis: nil ]  s64
             >\
             """
    end

    test "with fun" do
      a = Expr.parameter(nil, {:s, 64}, {2, 2}, 0)

      assert Nx.reduce(a, 0, [], &Nx.add/2) |> inspect(safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a                                                  s64[2][2]
               b = reduce [ a, 0, axes: nil, keep_axes: false, &Nx.add/2 ]  s64
             >\
             """
    end

    test "with metadata" do
      a = Expr.parameter(nil, {:s, 64}, {}, 0)

      assert Expr.metadata(a, %{foo: true}) |> inspect(safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a                 s64
               b = metadata [ a, [:foo] ]  s64
             >\
             """
    end

    test "with tuple and cond" do
      a = Expr.parameter(nil, {:s, 64}, {}, 0)
      b = Expr.parameter(nil, {:s, 64}, {}, 1)
      {left, right} = Expr.cond([{Nx.any?(a), {a, b}}], {b, a})

      assert inspect(left, safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a                                     s64
               parameter c                                     s64
               b = any? [ a, axes: nil, keep_axes: false ]     u8
               d = cond [ b -> {a, c}, :otherwise -> {c, a} ]  tuple2
               e = elem [ d, 0, 2 ]                            s64
             >\
             """

      assert inspect(right, safe: false) == """
             #Nx.Tensor<
               s64
             \s\s
               Nx.Defn.Expr
               parameter a                                     s64
               parameter c                                     s64
               b = any? [ a, axes: nil, keep_axes: false ]     u8
               d = cond [ b -> {a, c}, :otherwise -> {c, a} ]  tuple2
               e = elem [ d, 1, 2 ]                            s64
             >\
             """
    end
  end
end
