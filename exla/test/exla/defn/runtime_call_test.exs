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
    y = EXLA.jit_apply(&split_and_sum/1, [x], client: :host)

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

  describe "process leak regression" do
    # A simple defn with no runtime_call nodes — previously each JIT call
    # would leak a CallbackServer process.
    defn simple_add(a, b), do: a + b

    test "repeated JIT calls without runtime_call do not leak processes" do
      process_count_before = length(Process.list())

      for _ <- 1..1000 do
        EXLA.jit_apply(&simple_add/2, [Nx.tensor(1), Nx.tensor(2)], client: :host)
      end

      :erlang.garbage_collect()
      Process.sleep(100)

      process_count_after = length(Process.list())

      # Allow some slack for other system processes, but 1000 leaked
      # CallbackServer processes would be obvious.
      assert process_count_after - process_count_before < 50,
             "Process count grew by #{process_count_after - process_count_before} " <>
               "after 1000 JIT calls — suspected CallbackServer leak"
    end

    test "repeated JIT calls with runtime_call do not leak processes" do
      process_count_before = length(Process.list())

      for _ <- 1..100 do
        x = Nx.iota({5})
        add_offset(x)
      end

      :erlang.garbage_collect()
      Process.sleep(100)

      process_count_after = length(Process.list())

      assert process_count_after - process_count_before < 50,
             "Process count grew by #{process_count_after - process_count_before} " <>
               "after 100 runtime_call JIT calls — suspected process leak"
    end
  end

  describe "concurrent runtime_call" do
    # Different defns so they get different callback_ids, exercising
    # the ETS dispatcher routing multiple callbacks simultaneously.
    defn add_ten(x) do
      out = %{x | type: Nx.Type.to_floating(x.type)}
      Nx.runtime_call(out, x, fn t -> Nx.add(Nx.as_type(t, :f32), 10.0) end)
    end

    defn mul_two(x) do
      out = %{x | type: Nx.Type.to_floating(x.type)}
      Nx.runtime_call(out, x, fn t -> Nx.multiply(Nx.as_type(t, :f32), 2.0) end)
    end

    test "concurrent computations with different runtime_calls" do
      tasks =
        for fun <- [&add_ten/1, &mul_two/1], _ <- 1..10 do
          Task.async(fn ->
            x = Nx.iota({5})
            {fun, fun.(x)}
          end)
        end

      results = Task.await_many(tasks, 30_000)

      for {fun, result} <- results do
        x = Nx.iota({5}, type: :f32)

        expected =
          if fun == (&add_ten/1),
            do: Nx.add(x, 10.0),
            else: Nx.multiply(x, 2.0)

        assert_equal(result, expected)
      end
    end

    test "concurrent computations with same cached function" do
      # Multiple tasks calling the same defn concurrently. The cached
      # executable has the same callback_ids, so each task's outfeed
      # task registers the same keys in ETS. The dispatcher must route
      # each message to the correct task.
      tasks =
        for _ <- 1..10 do
          Task.async(fn ->
            x = Nx.iota({5})
            add_ten(x)
          end)
        end

      results = Task.await_many(tasks, 30_000)
      expected = Nx.add(Nx.as_type(Nx.iota({5}), :f32), 10.0)

      for result <- results do
        assert_equal(result, expected)
      end
    end

    test "rapid sequential calls with same cached callback_id do not deadlock" do
      # This specifically tests the select_delete fix: if unregister
      # deleted by callback_id alone (without checking the pid), rapid
      # sequential calls would race and deadlock.
      for _ <- 1..50 do
        x = Nx.iota({5})
        result = add_ten(x)
        expected = Nx.add(Nx.as_type(x, :f32), 10.0)
        assert_equal(result, expected)
      end
    end

    test "callback error in one task does not affect others" do
      # Start several good tasks and one bad task concurrently.
      # The bad task should fail without poisoning the good ones.
      good_tasks =
        for _ <- 1..5 do
          Task.async(fn ->
            x = Nx.iota({5})
            {:ok, add_ten(x)}
          end)
        end

      bad_task =
        Task.async(fn ->
          try do
            bad_callback(Nx.iota({2}))
            :should_not_reach
          rescue
            e in RuntimeError -> {:error, e.message}
          end
        end)

      good_results = Task.await_many(good_tasks, 30_000)
      bad_result = Task.await(bad_task, 30_000)

      expected = Nx.add(Nx.as_type(Nx.iota({5}), :f32), 10.0)

      for {:ok, result} <- good_results do
        assert_equal(result, expected)
      end

      assert {:error, msg} = bad_result
      assert msg =~ "expected the runtime_call function to match the given output template"
    end

    test "ETS table is cleaned up after execution" do
      # Run a runtime_call and verify the dispatcher ETS table
      # doesn't accumulate stale entries.
      ets_count_before = :ets.info(EXLA.Defn.CallbackDispatcher, :size)

      x = Nx.iota({5})
      add_ten(x)

      # Give the outfeed task time to clean up
      Process.sleep(50)

      ets_count_after = :ets.info(EXLA.Defn.CallbackDispatcher, :size)

      assert ets_count_after == ets_count_before,
             "ETS table grew by #{ets_count_after - ets_count_before} " <>
               "entries after runtime_call — stale callback registrations"
    end
  end

  describe "hooks and runtime_call together" do
    defp send_to_self(tag) do
      parent = self()
      fn value -> send(parent, {tag, value}) end
    end

    defn hook_and_callback(x) do
      # Hook observes the intermediate value, runtime_call transforms it.
      hooked = hook(x + 1, :intermediate, send_to_self(:intermediate))
      out = %{hooked | type: Nx.Type.to_floating(hooked.type)}
      Nx.runtime_call(out, hooked, fn t -> Nx.multiply(Nx.as_type(t, :f32), 3.0) end)
    end

    test "hook and runtime_call in same computation" do
      x = Nx.tensor([1, 2, 3])
      result = hook_and_callback(x)

      # Hook should have fired with x + 1
      assert_receive {:intermediate, hooked_value}
      assert_equal(hooked_value, Nx.tensor([2, 3, 4]))

      # Result should be (x + 1) * 3.0
      expected = Nx.multiply(Nx.as_type(Nx.tensor([2, 3, 4]), :f32), 3.0)
      assert_equal(result, expected)
    end

    defn two_hooks_and_callback(x) do
      a = hook(x, :first, send_to_self(:first))
      b = hook(a + 1, :second, send_to_self(:second))
      out = %{b | type: Nx.Type.to_floating(b.type)}
      Nx.runtime_call(out, b, fn t -> Nx.add(Nx.as_type(t, :f32), 100.0) end)
    end

    test "multiple hooks and runtime_call in same computation" do
      x = Nx.tensor([10, 20, 30])
      result = two_hooks_and_callback(x)

      assert_receive {:first, first_value}
      assert_equal(first_value, Nx.tensor([10, 20, 30]))

      assert_receive {:second, second_value}
      assert_equal(second_value, Nx.tensor([11, 21, 31]))

      # Result: (x + 1) + 100.0
      expected = Nx.add(Nx.as_type(Nx.tensor([11, 21, 31]), :f32), 100.0)
      assert_equal(result, expected)
    end

    defn hook_and_callback_cleanup_test(x) do
      hooked = hook(x, :observe, send_to_self(:observe))
      out = %{hooked | type: Nx.Type.to_floating(hooked.type)}
      Nx.runtime_call(out, hooked, fn t -> Nx.add(Nx.as_type(t, :f32), 1.0) end)
    end

    test "callback helper process is cleaned up after hooks+callback execution" do
      process_count_before = length(Process.list())

      for _ <- 1..20 do
        x = Nx.tensor([1, 2, 3])
        hook_and_callback_cleanup_test(x)
        assert_receive {:observe, _}
      end

      :erlang.garbage_collect()
      Process.sleep(100)

      process_count_after = length(Process.list())

      assert process_count_after - process_count_before < 10,
             "Process count grew by #{process_count_after - process_count_before} " <>
               "after 20 hooks+callback calls — suspected helper process leak"
    end
  end

  describe "chained runtime_calls" do
    defn two_chained_calls(x) do
      fx = Nx.as_type(x, :f32)
      out1 = %{fx | type: {:f, 32}}

      # First callback: add 10
      step1 = Nx.runtime_call(out1, fx, fn t -> Nx.add(t, 10.0) end)

      out2 = %{step1 | type: {:f, 32}}

      # Second callback: multiply by 2
      Nx.runtime_call(out2, step1, fn t -> Nx.multiply(t, 2.0) end)
    end

    test "output of one runtime_call feeds into the next" do
      x = Nx.tensor([1, 2, 3])
      result = two_chained_calls(x)

      # (x + 10) * 2
      expected = Nx.multiply(Nx.add(Nx.as_type(x, :f32), 10.0), 2.0)
      assert_equal(result, expected)
    end

    defn three_chained_calls(x) do
      fx = Nx.as_type(x, :f32)

      step1 = Nx.runtime_call(%{fx | type: {:f, 32}}, fx, fn t -> Nx.add(t, 1.0) end)
      step2 = Nx.runtime_call(%{step1 | type: {:f, 32}}, step1, fn t -> Nx.multiply(t, 2.0) end)
      Nx.runtime_call(%{step2 | type: {:f, 32}}, step2, fn t -> Nx.subtract(t, 5.0) end)
    end

    test "three chained runtime_calls" do
      x = Nx.tensor([10, 20, 30])
      result = three_chained_calls(x)

      # ((x + 1) * 2) - 5
      fx = Nx.as_type(x, :f32)
      expected = Nx.subtract(Nx.multiply(Nx.add(fx, 1.0), 2.0), 5.0)
      assert_equal(result, expected)
    end

    defn parallel_calls_then_combine(x) do
      fx = Nx.as_type(x, :f32)

      a = Nx.runtime_call(%{fx | type: {:f, 32}}, fx, fn t -> Nx.add(t, 10.0) end)
      b = Nx.runtime_call(%{fx | type: {:f, 32}}, fx, fn t -> Nx.multiply(t, 3.0) end)

      Nx.add(a, b)
    end

    test "two independent runtime_calls whose results are combined" do
      x = Nx.tensor([1, 2, 3])
      result = parallel_calls_then_combine(x)

      fx = Nx.as_type(x, :f32)
      expected = Nx.add(Nx.add(fx, 10.0), Nx.multiply(fx, 3.0))
      assert_equal(result, expected)
    end
  end

  describe "large tensors" do
    defn large_tensor_callback(x) do
      out = %{x | type: {:f, 32}}
      Nx.runtime_call(out, x, fn t -> Nx.add(Nx.as_type(t, :f32), 1.0) end)
    end

    test "runtime_call with large 1D tensor" do
      x = Nx.iota({10_000})
      result = large_tensor_callback(x)

      expected = Nx.add(Nx.as_type(x, :f32), 1.0)
      assert_equal(result, expected)
    end

    test "runtime_call with large 2D tensor" do
      x = Nx.iota({100, 100})
      result = large_tensor_callback(x)

      expected = Nx.add(Nx.as_type(x, :f32), 1.0)
      assert_equal(result, expected)
    end

    test "runtime_call with large 3D tensor" do
      x = Nx.iota({10, 20, 30})
      result = large_tensor_callback(x)

      expected = Nx.add(Nx.as_type(x, :f32), 1.0)
      assert_equal(result, expected)
    end

    defn large_tuple_roundtrip(x) do
      fx = Nx.as_type(x, :f32)
      out = {fx, fx, fx}

      {a, b, c} =
        Nx.runtime_call(out, fx, fn t ->
          {Nx.add(t, 1.0), Nx.multiply(t, 2.0), Nx.negate(t)}
        end)

      Nx.add(Nx.add(a, b), c)
    end

    test "runtime_call with large tensor and tuple output" do
      x = Nx.iota({1000})
      result = large_tuple_roundtrip(x)

      fx = Nx.as_type(x, :f32)
      # (x + 1) + (x * 2) + (-x) = x + 1 + 2x - x = 2x + 1
      expected = Nx.add(Nx.multiply(fx, 2.0), 1.0)
      assert_equal(result, expected)
    end
  end

  describe "data types" do
    defn f64_callback(x) do
      Nx.runtime_call(%{x | type: {:f, 64}}, x, fn t -> Nx.add(t, 1.0) end)
    end

    defn s32_to_f32_callback(x) do
      out = %{x | type: {:f, 32}}
      Nx.runtime_call(out, x, fn t -> Nx.as_type(Nx.add(t, 1), :f32) end)
    end

    test "runtime_call preserves f64 precision" do
      x = Nx.tensor([1.0, 2.0, 3.0], type: :f64)
      result = f64_callback(x)

      expected = Nx.add(x, 1.0)
      assert_equal(result, expected)
      assert Nx.type(result) == {:f, 64}
    end

    test "runtime_call with integer input and float output" do
      x = Nx.tensor([10, 20, 30], type: :s32)
      result = s32_to_f32_callback(x)

      expected = Nx.tensor([11.0, 21.0, 31.0], type: :f32)
      assert_equal(result, expected)
      assert Nx.type(result) == {:f, 32}
    end

    defn scalar_callback(x) do
      Nx.runtime_call(%{x | type: {:f, 32}}, x, fn t -> Nx.add(Nx.as_type(t, :f32), 42.0) end)
    end

    test "runtime_call with scalar (0-dim tensor)" do
      x = Nx.tensor(5)
      result = scalar_callback(x)

      assert_equal(result, Nx.tensor(47.0))
      assert Nx.shape(result) == {}
    end

    test "runtime_call with very small tensor" do
      # A {1}-shaped tensor — minimal non-empty tensor through the bridge.
      x = Nx.tensor([7])
      result = scalar_callback(x)

      assert_equal(result, Nx.tensor(49.0))
      assert Nx.shape(result) == {1}
      assert Nx.type(result) == {:f, 32}
    end
  end

  describe "runtime_call inside control flow" do
    defn callback_in_while(x) do
      # Repeatedly apply a runtime_call inside a while loop.
      # The same callback_id gets invoked multiple times per execution.
      {result, _i} =
        while {acc = Nx.as_type(x, :f32), i = 0}, i < 3 do
          stepped =
            Nx.runtime_call(%{acc | type: {:f, 32}}, acc, fn t -> Nx.add(t, 1.0) end)

          {stepped, i + 1}
        end

      result
    end

    test "runtime_call invoked multiple times inside while loop" do
      x = Nx.tensor([0.0, 10.0, 20.0])
      result = callback_in_while(x)

      # Each iteration adds 1.0, 3 iterations total
      expected = Nx.add(x, 3.0)
      assert_equal(result, expected)
    end

    defn callback_in_cond(x, flag) do
      fx = Nx.as_type(x, :f32)

      if flag do
        Nx.runtime_call(%{fx | type: {:f, 32}}, fx, fn t -> Nx.add(t, 100.0) end)
      else
        Nx.runtime_call(%{fx | type: {:f, 32}}, fx, fn t -> Nx.multiply(t, -1.0) end)
      end
    end

    test "runtime_call in true branch of cond" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      result = callback_in_cond(x, Nx.tensor(1, type: :u8))

      expected = Nx.add(x, 100.0)
      assert_equal(result, expected)
    end

    test "runtime_call in false branch of cond" do
      x = Nx.tensor([1.0, 2.0, 3.0])
      result = callback_in_cond(x, Nx.tensor(0, type: :u8))

      expected = Nx.negate(x)
      assert_equal(result, expected)
    end
  end

  describe "error edge cases" do
    defn callback_raises_argument_error(x) do
      out = %{x | type: {:f, 32}}

      Nx.runtime_call(out, x, fn _t ->
        raise ArgumentError, "intentional argument error"
      end)
    end

    defn callback_raises_arithmetic_error(x) do
      out = %{x | type: {:f, 32}}

      Nx.runtime_call(out, x, fn _t ->
        raise ArithmeticError, "intentional arithmetic error"
      end)
    end

    test "callback that raises ArgumentError" do
      x = Nx.tensor([1.0, 2.0])

      assert_raise RuntimeError, ~r/intentional argument error/, fn ->
        callback_raises_argument_error(x)
      end
    end

    test "callback that raises ArithmeticError" do
      x = Nx.tensor([1.0, 2.0])

      assert_raise RuntimeError, ~r/intentional arithmetic error/, fn ->
        callback_raises_arithmetic_error(x)
      end
    end
  end

  describe "hooks + multiple runtime_calls" do
    defp send_hook(tag) do
      parent = self()
      fn value -> send(parent, {tag, value}) end
    end

    defn hooks_and_two_callbacks(x) do
      fx = Nx.as_type(x, :f32)
      observed = hook(fx, :input, send_hook(:input))

      step1 =
        Nx.runtime_call(%{observed | type: {:f, 32}}, observed, fn t -> Nx.add(t, 10.0) end)

      mid = hook(step1, :middle, send_hook(:middle))

      Nx.runtime_call(%{mid | type: {:f, 32}}, mid, fn t -> Nx.multiply(t, 2.0) end)
    end

    test "two hooks and two runtime_calls interleaved" do
      x = Nx.tensor([1, 2, 3])
      result = hooks_and_two_callbacks(x)

      assert_receive {:input, input_val}
      assert_equal(input_val, Nx.as_type(x, :f32))

      assert_receive {:middle, middle_val}
      expected_mid = Nx.add(Nx.as_type(x, :f32), 10.0)
      assert_equal(middle_val, expected_mid)

      # (x + 10) * 2
      expected = Nx.multiply(expected_mid, 2.0)
      assert_equal(result, expected)
    end
  end
end
