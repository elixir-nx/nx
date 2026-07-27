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
    test "records donated root parameter indices from tensor metadata" do
      # Map keys are traversed in sorted order: :b then :w
      args = [%{w: Nx.donate(Nx.tensor([1.0])), b: Nx.tensor([0.0])}]

      {_fun, params, templates, _flatten, donated} =
        Nx.Defn.Compiler.to_lazy_params(fn p -> p end, args, [])

      assert donated == [1]
      # Templates stored for compile matching are stripped.
      refute Enum.any?(templates, & &1.donatable)
      # Expression parameters do not carry the call-boundary mark.
      refute Enum.any?(params, fn param ->
               Nx.Defn.Composite.reduce(param, false, fn
                 %Nx.Tensor{donatable: true}, _ -> true
                 _, acc -> acc
               end)
             end)
    end

    test "donates every leaf marked on the container" do
      args = [%{w: Nx.donate(Nx.tensor([1.0, 2.0])), b: Nx.donate(Nx.tensor([0.0]))}]

      {_fun, _params, _templates, _flatten, donated} =
        Nx.Defn.Compiler.to_lazy_params(fn p -> p end, args, [])

      assert donated == [0, 1]
    end

    test "compile derives donated_params from donatable templates" do
      template = Nx.donate(Nx.template({3}, {:s, 32}))

      {_fun, _params, templates, _flatten, donated} =
        Nx.Defn.Compiler.to_lazy_params(&Nx.add(&1, 1), [template], [])

      assert donated == [0]
      refute hd(templates).donatable
    end
  end
end
