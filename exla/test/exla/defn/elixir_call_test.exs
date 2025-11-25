defmodule EXLA.Defn.ElixirCallTest do
  use ExUnit.Case, async: true
  import Nx.Defn
  import Nx.Testing

  setup do
    Nx.default_backend(EXLA.Backend)
    :ok
  end

  defp add_offset_callback(t, opts) do
    t
    |> Nx.as_type(:f32)
    # TODO: if we run on the same device there will be a problem due to the device locking.
    |> Nx.backend_transfer({EXLA.Backend, client: :host, device_id: 1})
    |> Nx.add(opts[:offset]) |> dbg(structs: false)
  end

  defn add_offset(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.elixir_call(out, [x, [offset: 10.0]], &add_offset_callback/2)
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

  defn bad_callback(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.elixir_call(out, [x], fn _t ->
      # Wrong shape on purpose
      Nx.tensor([1.0, 2.0, 3.0])
    end)
  end

  test "elixir_call errors when result shape does not match template" do
    x = Nx.iota({2})

    assert_raise RuntimeError,
                 ~r/expected the elixir_call function to match the given output template/,
                 fn ->
                   bad_callback(x)
                 end
  end

  test "works when using EXLA compiler directly" do
    x = Nx.tensor([1, 2, 3])
    y = EXLA.jit_apply(&split_and_sum/1, [x])

    fx = Nx.as_type(x, :f32)
    expected = Nx.add(Nx.multiply(fx, 2.0), Nx.add(fx, 1.0))
    assert_equal(y, expected)
  end
end
