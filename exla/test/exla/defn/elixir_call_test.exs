defmodule EXLA.Defn.ElixirCallEvaluatorTest do
  use ExUnit.Case, async: true
  import Nx.Defn
  import Nx.Testing

  setup do
    Nx.Defn.default_options(compiler: Nx.Defn.Evaluator)
    Nx.default_backend(EXLA.Backend)
    :ok
  end

  defn add_offset(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.elixir_call(out, [x, [offset: 10.0]], fn t, opts ->
      Nx.add(Nx.as_type(t, :f32), opts[:offset])
    end)
  end

  test "elixir_call with single output" do
    x = Nx.iota({5})
    y = add_offset(x)

    expected = Nx.add(Nx.as_type(x, :f32), 10.0)
    assert_equal(y, expected)
  end

  defn split_and_sum(x) do
    fx = Nx.as_type(x, :f32)

    out0 = fx
    out1 = fx
    out_template = {out0, out1}

    {a, b} =
      Nx.elixir_call(out_template, [fx], fn t ->
        {Nx.multiply(t, 2.0), Nx.add(t, 1.0)}
      end)

    Nx.add(a, b)
  end

  test "elixir_call with tuple output" do
    x = Nx.tensor([1, 2, 3])
    y = split_and_sum(x)

    fx = Nx.as_type(x, :f32)
    expected = Nx.add(Nx.multiply(fx, 2.0), Nx.add(fx, 1.0))
    assert_equal(y, expected)
  end

  test "fails when using EXLA compiler" do
    x = Nx.tensor([1, 2, 3])

    assert_raise RuntimeError,
                 "Nx.elixir_call/3 is not supported yet. Use Nx.Defn.Evaluator as your compiler.",
                 fn ->
                   EXLA.jit_apply(&split_and_sum/1, [x])
                 end
  end
end
