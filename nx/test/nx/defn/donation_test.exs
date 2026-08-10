defmodule Nx.Defn.DonationTest do
  use ExUnit.Case, async: true

  describe "Nx.donatable/1" do
    test "marks a tensor as donatable and is idempotent" do
      t = Nx.tensor([1, 2, 3])
      donated = Nx.donatable(t)
      assert donated.donatable?
      assert Nx.donatable?(donated)
      assert Nx.donatable(donated).donatable?
    end

    test "preserves donatable through to_template/1" do
      template = Nx.donatable(Nx.tensor([1, 2, 3])) |> Nx.to_template()
      assert template.donatable?
      assert %Nx.TemplateBackend{} = template.data
    end

    test "marks all leaves of a container" do
      %{a: a, b: b} = Nx.donatable(%{a: Nx.tensor(1), b: Nx.tensor(2)})
      assert a.donatable?
      assert b.donatable?
    end

    test "returns false for numbers" do
      refute Nx.donatable?(1)
      refute Nx.donatable?(Complex.new(1, 2))
    end

    test "unwraps through jit with the evaluator" do
      fun = Nx.Defn.jit(&Nx.add(&1, 1))
      assert Nx.to_flat_list(fun.(Nx.donatable(Nx.tensor([1, 2, 3])))) == [2, 3, 4]
    end

    test "can mark part of a container" do
      fun = Nx.Defn.jit(fn %{a: a, b: b} -> %{a: Nx.add(a, 1), b: Nx.multiply(b, 2)} end)

      result =
        fun.(%{
          a: Nx.donatable(Nx.tensor([1, 2])),
          b: Nx.tensor([3, 4])
        })

      assert Nx.to_flat_list(result.a) == [2, 3]
      assert Nx.to_flat_list(result.b) == [6, 8]
    end
  end

  describe "to_lazy_params" do
    test "preserves donatable on templates and expression parameters" do
      # Map keys are traversed in sorted order: :b then :w
      args = [%{w: Nx.donatable(Nx.tensor([1.0])), b: Nx.tensor([0.0])}]

      {_fun, params, templates, _flatten} =
        Nx.Defn.Compiler.to_lazy_params(fn p -> p end, args)

      assert Enum.map(templates, & &1.donatable?) == [false, true]

      assert Enum.map(params, fn param ->
               Nx.Defn.Composite.reduce(param, [], fn t, acc -> [t.donatable? | acc] end)
               |> Enum.reverse()
             end) == [[false, true]]
    end
  end

  describe "Nx.Defn.compile/3" do
    test "raises when donatable? mismatches between templates and args" do
      template = Nx.donatable(Nx.template({3}, {:s, 32}))
      fun = Nx.Defn.compile(&Nx.add(&1, 1), [template])

      assert_raise ArgumentError, ~r"compiled with donatable\?: true but got false", fn ->
        fun.(Nx.tensor([1, 2, 3]))
      end

      fun = Nx.Defn.compile(&Nx.add(&1, 1), [Nx.template({3}, {:s, 32})])

      assert_raise ArgumentError, ~r"compiled with donatable\?: false but got true", fn ->
        fun.(Nx.donatable(Nx.tensor([1, 2, 3])))
      end
    end
  end
end
