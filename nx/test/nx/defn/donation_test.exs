defmodule Nx.Defn.DonationTest do
  use ExUnit.Case, async: true

  describe "Nx.donate/1" do
    test "marks a tensor as donatable and is idempotent" do
      t = Nx.tensor([1, 2, 3])
      donated = Nx.donate(t)
      assert donated.donatable
      assert Nx.donatable?(donated)
      assert Nx.donate(donated).donatable
    end

    test "preserves donatable through to_template/1" do
      template = Nx.donate(Nx.tensor([1, 2, 3])) |> Nx.to_template()
      assert template.donatable
      assert %Nx.TemplateBackend{} = template.data
    end

    test "marks all leaves of a container" do
      %{a: a, b: b} = Nx.donate(%{a: Nx.tensor(1), b: Nx.tensor(2)})
      assert a.donatable
      assert b.donatable
    end

    test "unwraps through jit with the evaluator" do
      fun = Nx.Defn.jit(&Nx.add(&1, 1))
      assert Nx.to_flat_list(fun.(Nx.donate(Nx.tensor([1, 2, 3])))) == [2, 3, 4]
    end

    test "can donate part of a container" do
      fun = Nx.Defn.jit(fn %{a: a, b: b} -> %{a: Nx.add(a, 1), b: Nx.multiply(b, 2)} end)

      result =
        fun.(%{
          a: Nx.donate(Nx.tensor([1, 2])),
          b: Nx.tensor([3, 4])
        })

      assert Nx.to_flat_list(result.a) == [2, 3]
      assert Nx.to_flat_list(result.b) == [6, 8]
    end
  end

  describe "to_lazy_params" do
    test "preserves donatable on templates and expression parameters" do
      # Map keys are traversed in sorted order: :b then :w
      args = [%{w: Nx.donate(Nx.tensor([1.0])), b: Nx.tensor([0.0])}]

      {_fun, params, templates, _flatten} =
        Nx.Defn.Compiler.to_lazy_params(fn p -> p end, args, [])

      assert Enum.map(templates, & &1.donatable) == [false, true]

      assert Enum.map(params, fn param ->
               Nx.Defn.Composite.reduce(param, [], fn t, acc -> [t.donatable | acc] end)
               |> Enum.reverse()
             end) == [[false, true]]
    end
  end
end
