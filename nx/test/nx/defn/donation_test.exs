defmodule Nx.Defn.DonationTest do
  use ExUnit.Case, async: true

  import Nx.Defn
  import Nx.Testing

  defn once(x), do: Nx.exp(Nx.add(x, 1))

  defn twice(x) do
    y = Nx.add(x, 1)
    Nx.multiply(y, y)
  end

  defn two_branches(x), do: Nx.add(Nx.exp(x), Nx.negate(x))

  defn shared(x) do
    y = Nx.add(x, 1)
    Nx.add(Nx.exp(y), Nx.negate(y))
  end

  defn held_argument(x), do: Nx.add(x, Nx.negate(x))

  defn cond_alias(pred, x) do
    y = Nx.add(x, 1)

    z =
      if pred do
        y
      else
        Nx.exp(y)
      end

    Nx.add(z, Nx.negate(y))
  end

  defn loop(x) do
    {_, acc} =
      while {i = 0, acc = x}, i < 2 do
        {i + 1, Nx.exp(acc)}
      end

    acc
  end

  defn hooked(x) do
    y = print_value(Nx.add(x, 1))
    Nx.exp(y)
  end

  defp record(fun, args) do
    RecordingBackend.reset()

    result =
      Nx.with_default_backend(RecordingBackend, fn ->
        args = Enum.map(args, &Nx.backend_transfer(&1, RecordingBackend))
        apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), args)
      end)

    {result, RecordingBackend.recorded()}
  end

  defp flags_of(recorded, op), do: for({^op, flags} <- recorded, do: flags)

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

  describe "donatable propagation" do
    test "expression parameters keep the mark, expressions built from them do not" do
      template = Nx.donatable(Nx.template({3}, {:s, 32}))

      {_fun, [param], _templates, _flatten} =
        Nx.Defn.Compiler.to_lazy_params(fn t -> t end, [template])

      assert param.donatable?
      refute Nx.add(param, 1).donatable?
      refute Nx.sum(param).donatable?
      refute Nx.reshape(param, {3, 1}).donatable?
    end

    test "results of a jitted function are not donatable" do
      fun = Nx.Defn.jit(&Nx.add(&1, 1))
      refute Nx.donatable?(fun.(Nx.donatable(Nx.tensor([1, 2, 3]))))
    end

    test "results are not donatable when only part of a container is" do
      fun = Nx.Defn.jit(fn %{a: a, b: b} -> %{a: Nx.add(a, b), b: Nx.multiply(b, 2)} end)

      result = fun.(%{a: Nx.donatable(Nx.tensor([1, 2])), b: Nx.tensor([3, 4])})

      refute Nx.donatable?(result.a)
      refute Nx.donatable?(result.b)
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

  describe "evaluator last-use marks" do
    test "an intermediate used once is donatable" do
      {result, recorded} = record(&once/1, [Nx.tensor([1.0, 2.0])])

      assert flags_of(recorded, :add) == [[false, false]]
      assert flags_of(recorded, :exp) == [[true]]
      refute Nx.donatable?(result)
    end

    test "an intermediate used twice is donatable only at its last use" do
      {result, recorded} = record(&shared/1, [Nx.tensor([1.0, 2.0])])

      # Exactly one of the two readers (whichever runs last) may consume.
      flags = flags_of(recorded, :exp) ++ flags_of(recorded, :negate)
      assert Enum.sort(flags) == [[false], [true]]
      assert_all_close(result, Nx.tensor([:math.exp(2.0) - 2.0, :math.exp(3.0) - 3.0]))
    end

    test "the same value twice in one op is never donatable" do
      {_result, recorded} = record(&twice/1, [Nx.tensor([1.0, 2.0])])
      assert flags_of(recorded, :multiply) == [[false, false]]
    end

    test "a donated argument is only donatable at its last use" do
      x = Nx.donatable(Nx.tensor([1.0, 2.0]))

      {_, recorded} = record(&two_branches/1, [x])
      assert flags_of(recorded, :exp) == [[false]]
      assert flags_of(recorded, :negate) == [[true]]

      {_, recorded} = record(&once/1, [x])
      assert flags_of(recorded, :add) == [[false, true]]
    end

    test "a value held by the op being built is not donatable by a later argument" do
      x = Nx.donatable(Nx.tensor([1.0, 2.0]))
      {result, recorded} = record(&held_argument/1, [x])

      assert flags_of(recorded, :negate) == [[false]]
      assert flags_of(recorded, :add) == [[false, true]]
      assert Nx.to_flat_list(result) == [0.0, 0.0]
    end

    test "a value returned by a cond branch is not donatable" do
      {result, recorded} = record(&cond_alias/2, [Nx.tensor(1), Nx.tensor([1.0, 2.0])])

      assert flags_of(recorded, :negate) == [[false]]
      assert Nx.to_flat_list(result) == [0.0, 0.0]
    end

    test "while loop state is never donatable" do
      {result, recorded} = record(&loop/1, [Nx.donatable(Nx.tensor([0.0]))])

      assert flags_of(recorded, :exp) == [[false], [false]]
      assert_all_close(result, Nx.tensor([:math.exp(:math.exp(0.0))]))
    end

    test "values seen by hooks are not donatable" do
      {_result, recorded} =
        ExUnit.CaptureIO.with_io(fn ->
          record(&hooked/1, [Nx.tensor([1.0])])
        end)
        |> elem(0)

      assert flags_of(recorded, :exp) == [[false]]
    end

    test "results are not donatable" do
      {result, _} = record(&once/1, [Nx.donatable(Nx.tensor([1.0]))])
      refute Nx.donatable?(result)

      {result, _} = record(fn x -> x end, [Nx.donatable(Nx.tensor([1.0]))])
      refute Nx.donatable?(result)
    end
  end
end
