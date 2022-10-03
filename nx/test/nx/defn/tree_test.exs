defmodule Nx.Defn.TreeTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.{Expr, Tree}
  alias Nx.Tensor, as: T
  doctest Nx.Defn.Tree

  import Nx.Defn

  setup do
    Nx.Defn.default_options(compiler: Nx.Defn.Debug)
    :ok
  end

  defn factorial(x) do
    {factorial, _} =
      while {factorial = 1.0, x}, Nx.greater(x, 1) do
        {factorial * x, x - 1}
      end

    factorial
  end

  defn with_hook(a, b), do: hook(a + b, :example)

  defn hooked_factorial(a, b) do
    {hook(factorial(with_hook(a, b)), :another), b}
  end

  describe "has_hooks?" do
    test "returns true if there are hooks" do
      refute Tree.has_hooks?(factorial(10), %{})
      refute Tree.has_hooks?(hooked_factorial(1, 2), %{})
      assert Tree.has_hooks?(hooked_factorial(1, 2), %{example: & &1})
      assert Tree.has_hooks?(hooked_factorial(1, 2), %{another: & &1})
    end
  end

  describe "scope_ids" do
    defn plus_constant(a), do: a + 10

    test "ignores constants" do
      a = Expr.parameter(:root, {:u, 64}, {}, 0)
      assert [{_, :parameter}, {_, :add}] = plus_constant(a) |> Tree.scope_ids() |> Enum.sort()
    end

    defn inside_cond(bool, a, b) do
      if bool do
        a + b
      else
        0
      end
    end

    test "ignores expressions inside cond" do
      bool = Expr.parameter(:root, {:u, 64}, {}, 0)
      a = Expr.parameter(:root, {:u, 64}, {}, 1)
      b = Expr.parameter(:root, {:u, 64}, {}, 2)

      assert [{_, :cond}] = inside_cond(bool, a, b) |> Tree.scope_ids() |> Enum.sort()
    end

    defn inside_both_cond(bool, a, b) do
      add = a + b

      left =
        if bool do
          add
        else
          1
        end

      right =
        if bool do
          1
        else
          add
        end

      left * right
    end

    test "keeps expressions shared across conds" do
      bool = Expr.parameter(:root, {:u, 64}, {}, 0)
      a = Expr.parameter(:root, {:u, 64}, {}, 1)
      b = Expr.parameter(:root, {:u, 64}, {}, 2)

      assert [
               {_, :parameter},
               {_, :parameter},
               {_, :parameter},
               {_, :add},
               {_, :cond},
               {_, :cond},
               {_, :multiply}
             ] = inside_both_cond(bool, a, b) |> Tree.scope_ids() |> Enum.sort()
    end
  end

  describe "rewrite_types" do
    test "wraps root parameters" do
      u64_param = Expr.parameter(:root, {:u, 64}, {}, 0)
      s64_param = Expr.parameter(:root, {:s, 64}, {}, 1)
      f64_param = Expr.parameter(:root, {:f, 64}, {}, 2)

      assert %T{data: %Expr{op: :as_type, args: [^u64_param]}, type: {:u, 32}} =
               Tree.rewrite_types(u64_param, max_unsigned_type: {:u, 32})

      assert %T{data: %Expr{op: :as_type, args: [^s64_param]}, type: {:s, 32}} =
               Tree.rewrite_types(s64_param, max_signed_type: {:s, 32})

      assert %T{data: %Expr{op: :as_type, args: [^f64_param]}, type: {:f, 32}} =
               Tree.rewrite_types(f64_param, max_float_type: {:f, 32})

      assert %T{data: %Expr{op: :as_type, args: [^f64_param]}, type: {:bf, 16}} =
               Tree.rewrite_types(f64_param, max_float_type: {:bf, 16})

      assert Tree.rewrite_types(s64_param, max_float_type: {:f, 32}) == s64_param
      assert Tree.rewrite_types(f64_param, max_signed_type: {:s, 32}) == f64_param
      assert Tree.rewrite_types(f64_param, max_unsigned_type: {:u, 32}) == f64_param
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

    test "converts scalars" do
      expr = Expr.tensor(Nx.tensor(3, type: {:s, 64}))

      assert %T{data: %Expr{op: :constant}, type: {:s, 32}} =
               Tree.rewrite_types(expr, max_signed_type: {:s, 32})

      expr = Expr.tensor(Nx.tensor(3.0, type: {:f, 64}))

      assert %T{data: %Expr{op: :constant}, type: {:f, 32}} =
               Tree.rewrite_types(expr, max_float_type: {:f, 32})
    end

    test "converts expressions" do
      s64_param = Expr.parameter(:root, {:s, 64}, {}, 1)
      f64_param = Expr.parameter(:root, {:f, 64}, {}, 2)

      assert %T{data: %Expr{op: :exp, args: [_]}, type: {:f, 32}} =
               Tree.rewrite_types(Nx.exp(s64_param), max_float_type: {:f, 32})

      assert %T{
               data: %Expr{
                 op: :exp,
                 args: [%T{data: %Expr{op: :as_type, args: [^f64_param]}, type: {:f, 32}}]
               },
               type: {:f, 32}
             } = Tree.rewrite_types(Nx.exp(f64_param), max_float_type: {:f, 32})
    end

    test "converts functions" do
      f64_param = Expr.parameter(:root, {:f, 64}, {}, 2)

      assert %T{data: %Expr{op: :reduce, args: [_, _, _, fun]}, type: {:f, 32}} =
               Tree.rewrite_types(Nx.reduce(f64_param, 1, &Nx.divide/2), max_float_type: {:f, 32})

      assert %T{data: %Expr{op: :fun, args: [[arg1, arg2], div, _]}} = fun
      assert %T{data: %Expr{op: :parameter}, type: {:f, 32}} = arg1
      assert %T{data: %Expr{op: :parameter}, type: {:f, 32}} = arg2
      assert %T{data: %Expr{op: :divide}, type: {:f, 32}} = div
    end

    test "converts tuples" do
      s64_param = Expr.parameter(:root, {:s, 64}, {}, 1)
      f64_param = Expr.parameter(:root, {:f, 64}, {}, 2)

      assert {%T{data: %Expr{op: :as_type, args: [^s64_param]}, type: {:s, 32}},
              %T{data: %Expr{op: :as_type, args: [^f64_param]}, type: {:f, 32}}} =
               Tree.rewrite_types({s64_param, f64_param},
                 max_signed_type: {:s, 32},
                 max_float_type: {:f, 32}
               )
    end

    test "converts maps" do
      s64_param = Expr.parameter(:root, {:s, 64}, {}, 1)
      f64_param = Expr.parameter(:root, {:f, 64}, {}, 2)

      assert %{
               a: %T{data: %Expr{op: :as_type, args: [^s64_param]}, type: {:s, 32}},
               b: %T{data: %Expr{op: :as_type, args: [^f64_param]}, type: {:f, 32}}
             } =
               Tree.rewrite_types(%{a: s64_param, b: f64_param},
                 max_signed_type: {:s, 32},
                 max_float_type: {:f, 32}
               )
    end

    test "keeps a cache" do
      f64_param = Expr.parameter(:root, {:f, 64}, {}, 2)

      assert %T{data: %Expr{op: :add, args: [arg, arg]}, type: {:f, 32}} =
               Tree.rewrite_types(Nx.add(f64_param, f64_param), max_float_type: {:f, 32})
    end

    test "is no-op with max types" do
      f64_param = Expr.parameter(:root, {:f, 64}, {}, 2)

      expr = Nx.exp(f64_param)
      assert Tree.rewrite_types(expr, []) == expr
      assert Tree.rewrite_types(expr, max_float_type: {:f, 64}) == expr
    end

    defmacro param(type) do
      quote do
        %T{data: %Expr{op: :parameter}, type: unquote(type)}
      end
    end

    defn if3(a, b, c), do: if(a, do: b, else: c)

    test "with cond" do
      int = Nx.template({}, {:s, 64})
      float = Nx.template({}, {:f, 64})

      assert %T{data: %Expr{op: :cond, args: [clauses, last]}} =
               Tree.rewrite_types(if3(int, int, int), max_signed_type: {:s, 32})

      assert [
               {%T{data: %Expr{op: :as_type, args: [param({:s, 64})]}, type: {:s, 32}},
                %T{data: %Expr{op: :as_type, args: [param({:s, 64})]}, type: {:s, 32}}}
             ] = clauses

      assert %T{data: %Expr{op: :as_type, args: [param({:s, 64})]}, type: {:s, 32}} = last

      assert %T{data: %Expr{op: :cond, args: [clauses, last]}} =
               Tree.rewrite_types(if3(int, float, int), max_float_type: {:f, 32})

      assert [
               {param({:s, 64}),
                %T{data: %Expr{op: :as_type, args: [param({:f, 64})]}, type: {:f, 32}}}
             ] = clauses

      assert %T{data: %Expr{op: :as_type, args: [param({:s, 64})]}, type: {:f, 32}} = last
    end

    test "with while" do
      expr = factorial(Nx.template({}, {:s, 64}))

      assert %T{data: %Expr{op: :elem, args: [while, 0]}, type: {:f, 32}} =
               Tree.rewrite_types(expr, max_signed_type: {:s, 32}, max_float_type: {:f, 32})

      assert %T{data: %Expr{op: :while, args: [initial, arg, condition, body]}, type: {:tuple, 2}} =
               while

      assert {%T{type: {:f, 32}},
              %T{data: %Expr{op: :as_type, args: [param({:s, 64})]}, type: {:s, 32}}} = initial

      assert {param({:f, 32}), param({:s, 32})} = arg

      assert %T{data: %Expr{op: :greater}, type: {:u, 8}} = condition

      assert {%T{data: %Expr{op: :multiply}, type: {:f, 32}},
              %T{data: %Expr{op: :subtract}, type: {:s, 32}}} = body
    end

    test "with hook" do
      expr = with_hook(Nx.template({}, {:s, 64}), Nx.template({}, {:s, 64}))

      assert %T{data: %Expr{op: :attach_token, args: [token, expr]}, type: {:s, 32}} =
               Tree.rewrite_types(expr, max_signed_type: {:s, 32})

      assert %T{data: %Expr{op: :add, id: id}, type: {:s, 32}} = expr

      %T{data: %Expr{op: :token, args: [token]}} = token
      [%{name: :example, callback: nil, expr: hook}] = token.hooks
      assert %T{data: %Expr{op: :add, id: ^id}, type: {:s, 32}} = hook
    end
  end

  describe "apply_args" do
    test "handles regular operations" do
      expr = Expr.add(Nx.tensor([0, 1]), Nx.tensor([1, 2]), Nx.tensor([2, 3]))
      {[arg1, arg2], acc} = Tree.apply_args(expr, [], &{&1, [&1.data.id | &2]})
      assert acc == [arg2.data.id, arg1.data.id]
    end

    test "handles concatenate" do
      expr = Expr.concatenate(Nx.tensor(1), [Nx.tensor(2), Nx.tensor(3)], 0)
      {[[arg1, arg2], 0], acc} = Tree.apply_args(expr, [], &{&1, [&1.data.id | &2]})
      assert acc == [arg2.data.id, arg1.data.id]
    end
  end
end
