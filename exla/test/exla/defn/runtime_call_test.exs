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

  def add_one_callback(t, _opts), do: Nx.add(t, 1)
  def double_callback(t, _opts), do: Nx.multiply(t, 2)
  def negate_callback(t, _opts), do: Nx.negate(t)

  defn runtime_call_in_while(x) do
    while x, Nx.less(x, 10) do
      Nx.runtime_call(x, x, &add_one_callback/2)
    end
  end

  test "runtime_call inside while loop" do
    result = runtime_call_in_while(Nx.tensor(0))
    assert_equal(result, Nx.tensor(10))
  end

  defn runtime_call_in_while_with_tuple(x) do
    {result, _count} =
      while {x, count = Nx.tensor(0)}, Nx.less(count, 3) do
        doubled = Nx.runtime_call(x, x, &double_callback/2)
        {doubled, count + 1}
      end

    result
  end

  test "runtime_call inside while loop with tuple state" do
    result = runtime_call_in_while_with_tuple(Nx.tensor(1.0))
    # 1.0 * 2 * 2 * 2 = 8.0
    assert_equal(result, Nx.tensor(8.0))
  end

  defn runtime_call_in_cond(x) do
    if Nx.greater(x, 0) do
      Nx.runtime_call(x, x, &double_callback/2)
    else
      Nx.runtime_call(x, x, &negate_callback/2)
    end
  end

  test "runtime_call inside cond branches" do
    assert_equal(runtime_call_in_cond(Nx.tensor(5.0)), Nx.tensor(10.0))
    assert_equal(runtime_call_in_cond(Nx.tensor(-3.0)), Nx.tensor(3.0))
  end

  defn multiple_runtime_calls_in_while(x) do
    while x, Nx.less(x, 100) do
      step1 = Nx.runtime_call(x, x, &add_one_callback/2)
      Nx.runtime_call(step1, step1, &double_callback/2)
    end
  end

  test "multiple runtime_calls in one while body" do
    # (0+1)*2=2, (2+1)*2=6, (6+1)*2=14, (14+1)*2=30, (30+1)*2=62, (62+1)*2=126
    result = multiple_runtime_calls_in_while(Nx.tensor(0.0))
    assert_equal(result, Nx.tensor(126.0))
  end

  def cast_to_float_callback(t, _opts), do: Nx.as_type(t, :f32)

  defn runtime_call_type_change(x) do
    out = %{x | type: {:f, 32}}
    Nx.runtime_call(out, x, &cast_to_float_callback/2)
  end

  test "runtime_call where callback changes type" do
    result = runtime_call_type_change(Nx.tensor([1, 2, 3], type: :s32))
    assert Nx.type(result) == {:f, 32}
    assert_equal(result, Nx.tensor([1.0, 2.0, 3.0]))
  end

  def add_ten_callback(t, _opts), do: Nx.add(t, 10)

  defn nested_while_with_runtime_call(x) do
    {result, _} =
      while {x, outer = Nx.tensor(0)}, Nx.less(outer, 2) do
        {inner_result, _} =
          while {x, inner = Nx.tensor(0)}, Nx.less(inner, 3) do
            {Nx.runtime_call(x, x, &add_one_callback/2), inner + 1}
          end

        {inner_result, outer + 1}
      end

    result
  end

  test "runtime_call inside nested while loops" do
    result = nested_while_with_runtime_call(Nx.tensor(0.0))
    # Inner while runs 3 times each outer iteration, outer runs 2 times
    # 0 → +3 = 3 → +3 = 6
    assert_equal(result, Nx.tensor(6.0))
  end

  defn runtime_call_in_while_accumulating(x) do
    {_x, acc} =
      while {x, acc = Nx.tensor(0.0)}, Nx.less(acc, 100) do
        val = Nx.runtime_call(x, x, &double_callback/2)
        {val, acc + val}
      end

    acc
  end

  test "runtime_call in while with separate accumulator" do
    # x=5, iter 1: val=10, acc=10; iter 2: val=20, acc=30; iter 3: val=40, acc=70; iter 4: val=80, acc=150
    result = runtime_call_in_while_accumulating(Nx.tensor(5.0))
    assert_equal(result, Nx.tensor(150.0))
  end
end
