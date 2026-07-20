defmodule Nx.Defn.DonationTest do
  use ExUnit.Case, async: true

  describe "donate/1" do
    test "is idempotent" do
      t = Nx.tensor([1, 2, 3])
      donated = Nx.Defn.donate(t)
      assert %Nx.Defn.Donated{value: ^t} = donated
      assert Nx.Defn.donate(donated) == donated
    end

    test "unwraps through jit with the evaluator" do
      fun = Nx.Defn.jit(&Nx.add(&1, 1))
      assert Nx.to_flat_list(fun.(Nx.Defn.donate(Nx.tensor([1, 2, 3])))) == [2, 3, 4]
    end

    test "can donate part of a container" do
      fun = Nx.Defn.jit(fn %{a: a, b: b} -> %{a: Nx.add(a, 1), b: Nx.multiply(b, 2)} end)

      result =
        fun.(%{
          a: Nx.Defn.donate(Nx.tensor([1, 2])),
          b: Nx.tensor([3, 4])
        })

      assert Nx.to_flat_list(result.a) == [2, 3]
      assert Nx.to_flat_list(result.b) == [6, 8]
    end
  end

  describe "donate_argnums" do
    test "accepts valid indices with the evaluator" do
      fun = Nx.Defn.jit(&Nx.add(&1, &2), donate_argnums: [0])
      assert Nx.to_flat_list(fun.(Nx.tensor([1, 2]), Nx.tensor([3, 4]))) == [4, 6]
    end

    test "raises on malformed option" do
      assert_raise ArgumentError, ~r":donate_argnums must be a list", fn ->
        Nx.Defn.jit_apply(&Nx.add(&1, 1), [Nx.tensor([1])], donate_argnums: :foo)
      end

      assert_raise ArgumentError, ~r":donate_argnums must be a list", fn ->
        Nx.Defn.jit_apply(&Nx.add(&1, 1), [Nx.tensor([1])], donate_argnums: [-1])
      end
    end

    test "raises when an index is out of range" do
      assert_raise ArgumentError, ~r":donate_argnums entries must be in the range", fn ->
        Nx.Defn.jit_apply(&Nx.add(&1, 1), [Nx.tensor([1])], donate_argnums: [5])
      end
    end

    test "records donated root parameter indices" do
      opts = [donate_argnums: [0]]
      args = [%{w: Nx.tensor([1.0, 2.0]), b: Nx.tensor([0.0])}, Nx.tensor([3.0, 4.0])]

      {_fun, _params, _templates, _flatten, donated} =
        Nx.Defn.Compiler.to_lazy_params(fn p, _batch -> p end, args, opts)

      # Both leaves of the first argument are donated; batch is not.
      assert donated == [0, 1]
    end

    test "donate/1 records only wrapped leaves" do
      # Map keys are traversed in sorted order: :b then :w
      args = [%{w: Nx.Defn.donate(Nx.tensor([1.0])), b: Nx.tensor([0.0])}]

      {_fun, _params, _templates, _flatten, donated} =
        Nx.Defn.Compiler.to_lazy_params(fn p -> p end, args, [])

      assert donated == [1]
    end

    test "unions donate_argnums with donate/1 marks" do
      # arg0 leaves: :b -> 0, :w -> 1; arg1 leaf -> 2
      args = [
        %{w: Nx.tensor([1.0]), b: Nx.Defn.donate(Nx.tensor([0.0]))},
        Nx.tensor([3.0])
      ]

      {_fun, _params, _templates, _flatten, donated} =
        Nx.Defn.Compiler.to_lazy_params(fn p, b -> {p, b} end, args, donate_argnums: [1])

      assert donated == [0, 2]
    end
  end
end
