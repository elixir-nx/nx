defmodule Nx.Defn.TreeTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.{Expr, Tree}
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
      {bool, cond} = Nx.Defn.jit(&{&1, inside_cond(&1, &2, &3)}).(0, 1, 2)

      assert cond |> Tree.scope_ids() |> Enum.sort() ==
               [{bool.data.id, :parameter}, {cond.data.id, :cond}]
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
