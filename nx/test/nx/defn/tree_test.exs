defmodule Nx.Defn.TreeTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.{Expr, Tree}
  alias Nx.Tensor, as: T

  describe "rewrite_types" do
    @u64_param Expr.parameter(nil, {:u, 64}, {}, 0)
    @s64_param Expr.parameter(nil, {:s, 64}, {}, 1)
    @f64_param Expr.parameter(nil, {:f, 64}, {}, 2)

    test "wraps parameters" do
      assert %T{data: %Expr{op: :as_type, args: [@u64_param]}, type: {:u, 32}} =
               Tree.rewrite_types(@u64_param, max_unsigned_type: {:u, 32})

      assert %T{data: %Expr{op: :as_type, args: [@s64_param]}, type: {:s, 32}} =
               Tree.rewrite_types(@s64_param, max_signed_type: {:s, 32})

      assert %T{data: %Expr{op: :as_type, args: [@f64_param]}, type: {:f, 32}} =
               Tree.rewrite_types(@f64_param, max_float_type: {:f, 32})

      assert %T{data: %Expr{op: :as_type, args: [@f64_param]}, type: {:bf, 16}} =
               Tree.rewrite_types(@f64_param, max_float_type: {:bf, 16})

      assert @s64_param = Tree.rewrite_types(@s64_param, max_float_type: {:f, 32})
      assert @f64_param = Tree.rewrite_types(@f64_param, max_signed_type: {:s, 32})
      assert @f64_param = Tree.rewrite_types(@f64_param, max_unsigned_type: {:u, 32})
    end

    test "converts tensors" do
      expr = Expr.tensor(Nx.tensor([1, 2, 3], type: {:s, 64}))

      assert Tree.rewrite_types(expr, max_signed_type: {:s, 32}).data.args ==
               [Nx.tensor([1, 2, 3], type: {:s, 32})]

      expr = Expr.tensor(Nx.tensor([1, 2, 3], type: {:u, 64}))

      assert Tree.rewrite_types(expr, max_unsigned_type: {:u, 32}).data.args ==
               [Nx.tensor([1, 2, 3], type: {:u, 32})]

      expr = Expr.tensor(Nx.tensor([1, 2, 3], type: {:f, 64}))

      assert Tree.rewrite_types(expr, max_float_type: {:f, 32}).data.args ==
               [Nx.tensor([1, 2, 3], type: {:f, 32})]

      assert Tree.rewrite_types(expr, max_float_type: {:bf, 16}).data.args ==
               [Nx.tensor([1, 2, 3], type: {:bf, 16})]
    end

    test "converts expressions" do
      assert %T{data: %Expr{op: :exp, args: [_]}, type: {:f, 32}} =
               Tree.rewrite_types(Nx.exp(@s64_param), max_float_type: {:f, 32})

      assert %T{
               data: %Expr{
                 op: :exp,
                 args: [%T{data: %Expr{op: :as_type, args: [@f64_param]}, type: {:f, 32}}]
               },
               type: {:f, 32}
             } = Tree.rewrite_types(Nx.exp(@f64_param), max_float_type: {:f, 32})
    end

    test "converts functions" do
      assert %T{data: %Expr{op: :reduce, args: [_, _, _, fun]}, type: {:f, 32}} =
               Tree.rewrite_types(Nx.reduce(@f64_param, 1, &Nx.divide/2), max_float_type: {:f, 32})

      assert %T{data: %Expr{op: :fun, args: [[arg1, arg2], div, _]}} = fun
      assert %T{data: %Expr{op: :parameter}, type: {:f, 32}} = arg1
      assert %T{data: %Expr{op: :parameter}, type: {:f, 32}} = arg2
      assert %T{data: %Expr{op: :divide}, type: {:f, 32}} = div
    end

    test "converts tuples" do
      assert {%T{data: %Expr{op: :as_type, args: [@s64_param]}, type: {:s, 32}},
              %T{data: %Expr{op: :as_type, args: [@f64_param]}, type: {:f, 32}}} =
               Tree.rewrite_types({@s64_param, @f64_param},
                 max_signed_type: {:s, 32},
                 max_float_type: {:f, 32}
               )
    end

    test "keeps a cache" do
      assert %T{data: %Expr{op: :add, args: [arg, arg]}, type: {:f, 32}} =
               Tree.rewrite_types(Nx.add(@f64_param, @f64_param), max_float_type: {:f, 32})
    end

    test "is no-op with max types" do
      expr = Nx.exp(@f64_param)
      assert Tree.rewrite_types(expr, []) == expr
      assert Tree.rewrite_types(expr, max_float_type: {:f, 64}) == expr
    end
  end

  describe "traverse_args" do
    test "handles regular operations" do
      expr = Expr.add(Nx.tensor(3), Nx.tensor(1), Nx.tensor(2))
      {[arg1, arg2], acc} = Tree.traverse_args(expr, [], &{&1, [&1.data.id | &2]})
      assert acc == [arg2.data.id, arg1.data.id]
    end

    test "handles concatenate" do
      expr = Expr.concatenate(Nx.tensor(1), [Nx.tensor(2), Nx.tensor(3)], 0)
      {[[arg1, arg2], 0], acc} = Tree.traverse_args(expr, [], &{&1, [&1.data.id | &2]})
      assert acc == [arg2.data.id, arg1.data.id]
    end
  end
end
