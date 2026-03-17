defmodule EXLA.Defn.RuntimeCallTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog
  import Nx.Defn
  import Nx.Testing

  setup do
    Nx.default_backend({EXLA.Backend, client: :host})
    Nx.Defn.default_options(compiler: EXLA, client: :host)
    :ok
  end

  deftransform add_offset_callback(t, opts) do
    t
    |> Nx.as_type(:f32)
    |> Nx.add(opts[:offset])
  end

  defn add_offset(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, [offset: 10.0], &add_offset_callback/2)
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

  def split_and_sum_callback(t, _opts) do
    {Nx.multiply(t, 2.0), Nx.add(t, 1.0)}
  end

  defn split_and_sum(x) do
    fx = Nx.as_type(x, :f32)

    out0 = fx
    out1 = fx
    out_template = {out0, out1}

    {a, b} = Nx.runtime_call(out_template, fx, &split_and_sum_callback/2)

    Nx.add(a, b)
  end

  test "runtime_call with tuple output" do
    x = Nx.tensor([1, 2, 3])
    y = split_and_sum(x)

    fx = Nx.as_type(x, :f32)
    expected = Nx.add(Nx.multiply(fx, 2.0), Nx.add(fx, 1.0))
    assert_equal(y, expected)
  end

  defp bad_callback_fn(_t, _opts) do
    # Wrong shape on purpose
    Nx.tensor([1.0, 2.0, 3.0])
  end

  defn bad_callback(x) do
    out = %{x | type: Nx.Type.to_floating(x.type)}

    Nx.runtime_call(out, x, [], &bad_callback_fn/2)
  end

  @tag :capture_log
  test "runtime_call errors when result shape does not match template" do
    x = Nx.iota({2})
    test_pid = self()

    {pid, ref} =
      spawn_monitor(fn ->
        bad_callback(x)
        send(test_pid, :unexpected_success)
      end)

    assert_receive {:DOWN, ^ref, :process, ^pid, reason}
    message = runtime_error_message(reason)

    assert message =~ "expected the runtime_call function to match the given output template"
    refute_received :unexpected_success
  end

  defp runtime_error_message({{%RuntimeError{message: message}, _stacktrace}, _call_info}),
    do: message

  defp runtime_error_message({%RuntimeError{message: message}, _stacktrace}),
    do: message

  defp runtime_error_message({:error, {:runtime_error, message}}),
    do: message

  defp runtime_error_message(other),
    do: inspect(other)

  test "works when using EXLA compiler directly" do
    x = Nx.tensor([1, 2, 3])
    y = EXLA.jit_apply(&split_and_sum/1, [x])

    fx = Nx.as_type(x, :f32)
    expected = Nx.add(Nx.multiply(fx, 2.0), Nx.add(fx, 1.0))
    assert_equal(y, expected)
  end

  def add_and_subtract_callback({x, y}, _opts) do
    {Nx.add(x, y), Nx.subtract(x, y)}
  end

  defn add_and_subtract(x, y) do
    Nx.runtime_call({x, x}, {x, y}, [], &add_and_subtract_callback/2)
  end

  test "runtime_call with tuple input" do
    x = Nx.tensor([1, 2, 3])
    y = Nx.tensor([4, 5, 6])
    assert {add, sub} = add_and_subtract(x, y)

    assert_equal(add, Nx.add(x, y))
    assert_equal(sub, Nx.subtract(x, y))
  end

  deftransform add_and_subtract_with_opts_callback({x, y}, opts) do
    send(opts[:pid], {:add_and_subtract_with_opts, opts[:ref]})
    {Nx.add(x, y), Nx.subtract(x, y)}
  end

  defn add_and_subtract_with_opts(x, y, opts) do
    Nx.runtime_call({x, x}, {x, y}, opts, &add_and_subtract_with_opts_callback/2)
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

  defp return_as_container_tuple_callback({x, y}, opts) do
    send(opts[:pid], {:container_fun, opts[:ref]})
    {x, y}
  end

  defn return_as_container_tuple(x, y, opts) do
    Nx.runtime_call({x, y}, {x, y}, opts, &return_as_container_tuple_callback/2)
  end

  defp return_as_container_map_callback({x, y}, opts) do
    send(opts[:pid], {:container_fun, opts[:ref]})
    %{x: x, y: y}
  end

  defn return_as_container_map(x, y, opts) do
    Nx.runtime_call(%{x: x, y: y}, {x, y}, opts, &return_as_container_map_callback/2)
  end

  test "runtime_call with container output" do
    x = Nx.tensor([1, 2, 3])
    y = Nx.tensor([4, 5, 6])

    ref = make_ref()
    pid = self()

    assert {x_res, y_res} = return_as_container_tuple(x, y, ref: ref, pid: pid)
    assert_equal(x_res, x)
    assert_equal(y_res, y)
    assert_receive {:container_fun, ^ref}

    ref = make_ref()

    assert result = return_as_container_map(x, y, ref: ref, pid: pid)
    assert %{x: _, y: _} = result
    assert_equal(result.x, x)
    assert_equal(result.y, y)
    assert_receive {:container_fun, ^ref}
  end
end
