defmodule EXLA.Defn.RuntimeCallTest do
  use ExUnit.Case, async: true
  import Nx.Defn
  import Nx.Testing

  setup do
    Nx.default_backend(EXLA.Backend)
    Nx.Defn.default_options(compiler: EXLA)
    :ok
  end

  deftransform add_offset_callback(t, opts) do
    t
    |> Nx.as_type(:f32)
    |> Nx.add(opts[:offset])
  end

  defn add_offset(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn t -> add_offset_callback(t, offset: 10.0) end)
  end

  test "runtime_call with single output" do
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
      Nx.runtime_call(out_template, fx, fn t ->
        {Nx.multiply(t, 2.0), Nx.add(t, 1.0)}
      end)

    Nx.add(a, b)
  end

  test "runtime_call with tuple output" do
    x = Nx.tensor([1, 2, 3])
    y = split_and_sum(x)

    fx = Nx.as_type(x, :f32)
    expected = Nx.add(Nx.multiply(fx, 2.0), Nx.add(fx, 1.0))
    assert_equal(y, expected)
  end

  defn bad_callback(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn _t ->
      # Wrong shape on purpose
      Nx.tensor([1.0, 2.0, 3.0])
    end)
  end

  test "runtime_call errors when result shape does not match template" do
    x = Nx.iota({2})

    assert_raise RuntimeError,
                 ~r/expected the runtime_call function to match the given output template/,
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

  def add_and_subtract_callback({x, y}) do
    {Nx.add(x, y), Nx.subtract(x, y)}
  end

  defn add_and_subtract(x, y) do
    Nx.runtime_call({x, x}, {x, y}, &add_and_subtract_callback/1)
  end

  test "runtime_call with tuple input" do
    x = Nx.tensor([1, 2, 3])
    y = Nx.tensor([4, 5, 6])
    assert {add, sub} = add_and_subtract(x, y)

    assert_equal(add, Nx.add(x, y))
    assert_equal(sub, Nx.subtract(x, y))
  end

  deftransform add_and_subtract_with_opts_callback({x, y}, {ref, pid}) do
    send(pid, {:add_and_subtract_with_opts, ref})
    {Nx.add(x, y), Nx.subtract(x, y)}
  end

  defn add_and_subtract_with_opts(x, y, opts) do
    Nx.runtime_call(
      {x, x},
      {x, y},
      &add_and_subtract_with_opts_callback(&1, {opts[:ref], opts[:pid]})
    )
  end

  test "runtime_call with non-list second argument" do
    x = Nx.tensor([1, 2, 3])
    y = Nx.tensor([4, 5, 6])
    ref = make_ref()

    assert {add, sub} = add_and_subtract_with_opts(x, y, ref: ref, pid: self())

    assert_equal(add, Nx.add(x, y))
    assert_equal(sub, Nx.subtract(x, y))

    assert_receive {:add_and_subtract_with_opts, ^ref}
  end

  defn return_as_container(x, y, template_fun, container_fun) do
    Nx.runtime_call(template_fun.(x, y), {x, y}, container_fun)
  end

  test "runtime_call with container output" do
    x = Nx.tensor([1, 2, 3])
    y = Nx.tensor([4, 5, 6])

    ref = make_ref()
    pid = self()

    container_fun = fn {x, y} ->
      send(pid, {:container_fun, ref})
      {x, y}
    end

    template_fun = fn x, y -> {x, y} end

    assert {x_res, y_res} = return_as_container(x, y, template_fun, container_fun)
    assert_equal(x_res, x)
    assert_equal(y_res, y)
    assert_receive {:container_fun, ^ref}

    ref = make_ref()

    container_fun = fn {x, y} ->
      send(pid, {:container_fun, ref})
      %{x: x, y: y}
    end

    template_fun = fn x, y -> %{x: x, y: y} end

    assert result = return_as_container(x, y, template_fun, container_fun)
    assert %{x: _, y: _} = result
    assert_equal(result.x, x)
    assert_equal(result.y, y)
    assert_receive {:container_fun, ^ref}
  end
end
