defmodule Nx.Defn.ExprTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  describe "rewrite_types" do
    @u64_param Expr.parameter(nil, {:u, 64}, {}, 0)
    @s64_param Expr.parameter(nil, {:s, 64}, {}, 1)
    @f64_param Expr.parameter(nil, {:f, 64}, {}, 2)

    test "wraps parameters" do
      assert %T{data: %Expr{op: :as_type, args: [@u64_param]}, type: {:u, 32}} =
               Expr.rewrite_types(@u64_param, max_unsigned_type: {:u, 32})

      assert %T{data: %Expr{op: :as_type, args: [@s64_param]}, type: {:s, 32}} =
               Expr.rewrite_types(@s64_param, max_signed_type: {:s, 32})

      assert %T{data: %Expr{op: :as_type, args: [@f64_param]}, type: {:f, 32}} =
               Expr.rewrite_types(@f64_param, max_float_type: {:f, 32})

      assert %T{data: %Expr{op: :as_type, args: [@f64_param]}, type: {:bf, 16}} =
               Expr.rewrite_types(@f64_param, max_float_type: {:bf, 16})

      assert @s64_param = Expr.rewrite_types(@s64_param, max_float_type: {:f, 32})
      assert @f64_param = Expr.rewrite_types(@f64_param, max_signed_type: {:s, 32})
      assert @f64_param = Expr.rewrite_types(@f64_param, max_unsigned_type: {:u, 32})
    end

    test "converts tensors" do
      expr = Expr.tensor(Nx.tensor([1, 2, 3], type: {:s, 64}))

      assert Expr.rewrite_types(expr, max_signed_type: {:s, 32}).data.args ==
               [Nx.tensor([1, 2, 3], type: {:s, 32})]

      expr = Expr.tensor(Nx.tensor([1, 2, 3], type: {:u, 64}))

      assert Expr.rewrite_types(expr, max_unsigned_type: {:u, 32}).data.args ==
               [Nx.tensor([1, 2, 3], type: {:u, 32})]

      expr = Expr.tensor(Nx.tensor([1, 2, 3], type: {:f, 64}))

      assert Expr.rewrite_types(expr, max_float_type: {:f, 32}).data.args ==
               [Nx.tensor([1, 2, 3], type: {:f, 32})]

      assert Expr.rewrite_types(expr, max_float_type: {:bf, 16}).data.args ==
               [Nx.tensor([1, 2, 3], type: {:bf, 16})]
    end

    test "converts expressions" do
      assert %T{data: %Expr{op: :exp, args: [_]}, type: {:f, 32}} =
               Expr.rewrite_types(Nx.exp(@s64_param), max_float_type: {:f, 32})

      assert %T{
               data: %Expr{
                 op: :exp,
                 args: [%T{data: %Expr{op: :as_type, args: [@f64_param]}, type: {:f, 32}}]
               },
               type: {:f, 32}
             } = Expr.rewrite_types(Nx.exp(@f64_param), max_float_type: {:f, 32})
    end

    test "converts functions" do
      assert %T{data: %Expr{op: :reduce, args: [_, _, _, fun]}, type: {:f, 32}} =
               Expr.rewrite_types(Nx.reduce(@f64_param, 1, &Nx.divide/2), max_float_type: {:f, 32})

      assert %T{data: %Expr{op: :fun, args: [[arg1, arg2], div, _]}} = fun
      assert %T{data: %Expr{op: :parameter}, type: {:f, 32}} = arg1
      assert %T{data: %Expr{op: :parameter}, type: {:f, 32}} = arg2
      assert %T{data: %Expr{op: :divide}, type: {:f, 32}} = div
    end

    test "converts tuples" do
      assert {%T{data: %Expr{op: :as_type, args: [@s64_param]}, type: {:s, 32}},
              %T{data: %Expr{op: :as_type, args: [@f64_param]}, type: {:f, 32}}} =
               Expr.rewrite_types({@s64_param, @f64_param},
                 max_signed_type: {:s, 32},
                 max_float_type: {:f, 32}
               )
    end

    test "keeps a cache" do
      assert %T{data: %Expr{op: :add, args: [arg, arg]}, type: {:f, 32}} =
               Expr.rewrite_types(Nx.add(@f64_param, @f64_param), max_float_type: {:f, 32})
    end

    test "is no-op with max types" do
      expr = Nx.exp(@f64_param)
      assert Expr.rewrite_types(expr, []) == expr
      assert Expr.rewrite_types(expr, max_float_type: {:f, 64}) == expr
    end
  end

  describe "traverse_args" do
    test "handles regular operations" do
      expr = Expr.add(Nx.tensor(3), Nx.tensor(1), Nx.tensor(2))
      {[arg1, arg2], acc} = Expr.traverse_args(expr, [], &{&1, [&1.data.id | &2]})
      assert acc == [arg2.data.id, arg1.data.id]
    end

    test "handles concatenate" do
      expr = Expr.concatenate(Nx.tensor(1), [Nx.tensor(2), Nx.tensor(3)], 0)
      {[[arg1, arg2], 0], acc} = Expr.traverse_args(expr, [], &{&1, [&1.data.id | &2]})
      assert acc == [arg2.data.id, arg1.data.id]
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

    test "with parameters" do
      a = Expr.parameter(nil, {:s, 64}, {2, 2}, 0)
      b = Expr.parameter(nil, {:s, 64}, {2, 2}, 1)

      assert Nx.sum(Nx.add(Nx.add(Nx.dot(a, a), Nx.tanh(b)), 2))
             |> inspect(safe: false) == """
             #Nx.Tensor<
               f64
             \s\s
               Nx.Defn.Expr
               parameter a                                 s64[2][2]
               parameter c                                 s64[2][2]
               b = dot [ a, [1], a, [0] ]                  s64[2][2]
               d = tanh [ c ]                              f64[2][2]
               e = add [ b, d ]                            f64[2][2]
               f = add [ e, 2 ]                            f64[2][2]
               g = sum [ f, axes: nil, keep_axes: false ]  f64
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
               d = tanh [ c ]                                 f64[2][2]
               f = add [ d, e ]                               f64[2][2]
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
