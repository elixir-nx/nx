defmodule EXLA.Defn.RuntimeCallTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog
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

  @tag :cuda_required
  test "runtime_call with CUDA client (device↔host copies)" do
    x = Nx.iota({5}, backend: {EXLA.Backend, client: :cuda})
    y = EXLA.jit_apply(&add_offset/1, [x], client: :cuda)

    expected = Nx.add(Nx.as_type(x, :f32), 10.0)
    assert_equal(y, expected)
  end

  test "runtime_call with CUDA client fails when CUDA not available" do
    if Map.has_key?(EXLA.Client.get_supported_platforms(), :cuda) do
      # CUDA is available: this test is a no-op, cuda_required covers this case.
      :ok
    else
      x = Nx.iota({5})

      # The BEAM must not crash or segfault: the failure must be a clean exit.
      capture_log(fn ->
        assert {{%RuntimeError{message: message}, _stacktrace}, _call_info} =
                 catch_exit(EXLA.jit_apply(&add_offset/1, [x], client: :cuda))

        assert message =~ "cuda"
      end)
    end
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

  defp plain_add_offset(t, opts) do
    t
    |> Nx.as_type(:f32)
    |> Nx.add(opts[:offset])
  end

  defn add_offset_via_defp(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn t ->
      plain_add_offset(t, offset: 10.0)
    end)
  end

  test "runtime_call callback can call regular defp functions" do
    x = Nx.iota({5})
    y = add_offset_via_defp(x)

    expected = Nx.add(Nx.as_type(x, :f32), 10.0)
    assert_equal(y, expected)
  end

  defn callback_with_enum(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn t ->
      [t] |> Enum.map(&Nx.as_type(&1, :f32)) |> hd()
    end)
  end

  defp double(t), do: Nx.multiply(t, 2)
  defp double_and_add(t, val), do: Nx.add(double(t), val)

  defn nested_defp_callback(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn t ->
      double_and_add(Nx.as_type(t, :f32), 1.0)
    end)
  end

  test "runtime_call callback with nested defp calls" do
    x = Nx.tensor([1, 2, 3])
    result = nested_defp_callback(x)

    expected = Nx.add(Nx.multiply(Nx.as_type(x, :f32), 2), 1.0)
    assert_equal(result, expected)
  end

  defp scale(t, factor), do: Nx.multiply(t, factor)

  defn pipe_chain_callback(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, fn t ->
      t |> Nx.as_type(:f32) |> scale(3.0) |> Nx.add(1.0) |> double()
    end)
  end

  test "runtime_call callback with pipe chain mixing Nx and defp" do
    x = Nx.tensor([1, 2, 3])
    result = pipe_chain_callback(x)

    fx = Nx.as_type(x, :f32)
    expected = Nx.multiply(Nx.add(Nx.multiply(fx, 3.0), 1.0), 2)
    assert_equal(result, expected)
  end

  defp negate_f32(t), do: Nx.negate(Nx.as_type(t, :f32))

  defn two_defp_callbacks(x) do
    fx = Nx.as_type(x, :f32)

    a = Nx.runtime_call(%{fx | type: {:f, 32}}, fx, fn t -> double(t) end)
    b = Nx.runtime_call(%{fx | type: {:f, 32}}, fx, fn t -> negate_f32(t) end)

    Nx.add(a, b)
  end

  test "multiple runtime_calls with different defp callbacks" do
    x = Nx.tensor([1.0, 2.0, 3.0])
    result = two_defp_callbacks(x)

    assert_equal(result, x)
  end

  test "runtime_call callback can call non-Nx module functions" do
    x = Nx.tensor([1, 2, 3])
    result = callback_with_enum(x)

    expected = Nx.as_type(x, :f32)
    assert_equal(result, expected)
  end
end
