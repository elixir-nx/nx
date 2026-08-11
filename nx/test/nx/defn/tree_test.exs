defmodule Nx.Defn.TreeTest do
  use ExUnit.Case, async: true

  alias Nx.Defn.{Expr, Tree}
  doctest Nx.Defn.Tree

  import Nx.Defn

  setup do
    Nx.Defn.default_options(compiler: Nx.Defn.Debug)
    :ok
  end

  defn with_hook(a, b), do: io_call(a + b, :example)

  defn duplicate_hook_names(a, b) do
    ha = io_call(a, :same)
    hb = io_call(b, :same)
    ha + hb
  end

  describe "scope_ids" do
    defn plus_constant(a), do: a + 10

    test "ignores constants" do
      a = Expr.parameter(:root, {:u, 64}, {}, 0)

      assert [{_, :add}, {_, :parameter}] =
               plus_constant(a) |> Tree.scope_ids() |> Enum.sort_by(&elem(&1, 1))
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

      assert cond |> Tree.scope_ids() |> Enum.sort_by(&elem(&1, 1)) ==
               [{cond.data.id, :cond}, {bool.data.id, :parameter}]
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
               {_, :add},
               {_, :cond},
               {_, :cond},
               {_, :multiply},
               {_, :parameter},
               {_, :parameter},
               {_, :parameter}
             ] = inside_both_cond(bool, a, b) |> Tree.scope_ids() |> Enum.sort_by(&elem(&1, 1))
    end

    test "treats io_calls with the same name as distinct nodes" do
      a = Expr.parameter(:root, {:u, 64}, {}, 0)
      b = Expr.parameter(:root, {:u, 64}, {}, 1)

      assert [
               {_, :add},
               {_, :io_call},
               {_, :io_call},
               {_, :parameter},
               {_, :parameter}
             ] =
               tuples =
               duplicate_hook_names(a, b) |> Tree.scope_ids() |> Enum.sort_by(&elem(&1, 1))

      ids = Enum.map(tuples, &elem(&1, 0))
      assert length(ids) == length(Enum.uniq(ids))
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
