defmodule Nx.LazyLazyOnlyTest do
  use ExUnit.Case, async: true

  test "to_tensor" do
    assert Nx.to_tensor(%MagicNumber{value: 13}) == Nx.tensor(13)

    assert_raise ArgumentError,
                 ~r"it represents a collection of tensors, use Nx.stack/2 or Nx.concatenate/2 instead",
                 fn -> Nx.to_tensor(%{value: 13}) end
  end

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

  test "concatenate" do
    assert Nx.concatenate(%LazyOnly{a: Nx.tensor([1]), c: {Nx.tensor([2]), Nx.tensor([3])}}) ==
             Nx.tensor([1, 2, 3])

    assert Nx.concatenate([
             %LazyOnly{a: Nx.tensor([1]), c: {Nx.tensor([2]), Nx.tensor([3])}},
             Nx.tensor([4])
           ]) ==
             Nx.tensor([1, 2, 3, 4])

    assert ExUnit.CaptureIO.capture_io(:stderr, fn ->
             Nx.concatenate(%{a: Nx.tensor([1]), b: Nx.tensor([2])})
           end) =~ "a map has been given to stack/concatenate"
  end

  test "stack" do
    assert Nx.stack(%LazyOnly{a: Nx.tensor(1), c: {Nx.tensor(2), Nx.tensor(3)}}) ==
             Nx.tensor([1, 2, 3])

    assert Nx.stack([
             %LazyOnly{a: Nx.tensor(1), c: {Nx.tensor(2), Nx.tensor(3)}},
             Nx.tensor(4)
           ]) ==
             Nx.tensor([1, 2, 3, 4])

    assert ExUnit.CaptureIO.capture_io(:stderr, fn ->
             Nx.stack(%{a: Nx.tensor(1), b: Nx.tensor(2)})
           end) =~ "a map has been given to stack/concatenate"
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

  test "matches defn signature and does not invoke :b" do
    assert match_signature(%LazyOnly{a: 1, b: 2, c: 3}) == Nx.tensor(4)
  end

  test "matches jit signature and does not invoke :b" do
    fun = Nx.Defn.jit(fn %LazyWrapped{a: a, c: c} -> Nx.add(a, c) end)
    assert fun.(%LazyOnly{a: 1, b: 2, c: 3}) == Nx.tensor(4)
  end
end
