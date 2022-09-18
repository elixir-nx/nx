defmodule Nx.LazyLazyOnlyTest do
  use ExUnit.Case, async: true

  test "to_template" do
    assert Nx.to_template(%LazyOnly{a: 1, b: 2, c: 3}) ==
             %LazyWrapped{
               a: Nx.template({}, {:s, 64}),
               b: Nx.template({}, {:s, 64}),
               c: Nx.template({}, {:s, 64})
             }

    assert Nx.to_template(%LazyOnly{a: 1, b: {2, 3.0}, c: 4}) ==
             %LazyWrapped{
               a: Nx.template({}, {:s, 64}),
               b: {Nx.template({}, {:s, 64}), Nx.template({}, {:f, 32})},
               c: Nx.template({}, {:s, 64})
             }
  end

  test "compatible?" do
    assert Nx.compatible?(%LazyOnly{a: 1, b: 2, c: 3}, %LazyOnly{a: 4, b: 5, c: 6})
    refute Nx.compatible?(%LazyOnly{a: 1, b: 2, c: 3}, %LazyOnly{a: 4, b: 5, c: 6.0})
    refute Nx.compatible?(%LazyOnly{a: 1, b: 2, c: 3}, %URI{})
  end

  test "backend_copy" do
    assert_raise Protocol.UndefinedError, fn ->
      Nx.backend_transfer(%LazyOnly{a: 1, b: 2, c: 3})
    end
  end

  import Nx.Defn

  defn match_signature(%LazyWrapped{a: a, c: c}) do
    a + c
  end

  test "matches defn signature and does not invoke :c" do
    assert match_signature(%LazyOnly{a: 1, b: 2, c: 3}) == Nx.tensor(4)
  end

  test "matches jit signature and does not invoke :c" do
    fun = Nx.Defn.jit(fn %LazyWrapped{a: a, c: c} -> Nx.add(a, c) end)
    assert fun.(%LazyOnly{a: 1, b: 2, c: 3}) == Nx.tensor(4)
  end
end
