defmodule EXLA.Defn.APITest do
  use EXLA.Case, async: true

  import Nx.Defn
  import ExUnit.CaptureLog

  defn add_two(a, b), do: a + b

  describe "multi-client" do
    test "converts from host to separate client" do
      a = Nx.tensor(1, backend: {EXLA.Backend, client: :host})
      b = Nx.tensor(2, backend: {EXLA.Backend, client: :host})

      assert_equal(
        EXLA.jit(&add_two/2, client: :other_host).(a, b),
        Nx.tensor(3)
      )
    end

    test "converts from host to separate client through lazy transfers" do
      a = Nx.tensor(1, backend: {EXLA.Backend, client: :host})
      b = Nx.tensor(2, backend: {EXLA.Backend, client: :host})

      assert_equal(
        EXLA.jit(&add_two/2, client: :other_host, lazy_transfers: :always).(a, b),
        Nx.tensor(3)
      )
    end
  end

  describe "options" do
    test "logs when debugging" do
      logs =
        capture_log(fn ->
          EXLA.jit(&add_two/2, debug: true).(2, 3)
        end)

      assert logs =~ ~r"EXLA defn evaluation #Function<[^>]+> cache (hit|miss) in \d+\.\dms"
      assert logs =~ ~r"EXLA compilation #Function<[^>]+> cache (hit|miss) in \d+\.\dms"
      assert logs =~ ~r"EXLA device \d lock in \d+\.\dms"
      assert logs =~ ~r"EXLA execution on device \d in \d+\.\dms"

      logs =
        capture_log(fn ->
          EXLA.jit(&add_two/2, debug: true).(2, 3)
        end)

      assert logs =~ ~r"EXLA defn evaluation #Function<[^>]+> cache hit in \d+\.\dms"
      assert logs =~ ~r"EXLA compilation #Function<[^>]+> cache hit in \d+\.\dms"
      assert logs =~ ~r"EXLA device \d lock in \d+\.\d+ms"
      assert logs =~ ~r"EXLA execution on device \d in \d+\.\dms"
    end
  end

  describe "containers" do
    defn container_as_input(%Container{a: a, b: b}) do
      a * b
    end

    defn update_container(var, x) do
      %{var | b: x}
    end

    defn dot_container(container) do
      container.a * container.b
    end

    defn container_with_map_tuple(%Container{a: {x, y}, b: %{} = b}) do
      %{a: a, b: b} = b
      x * y * a * b
    end

    test "matched as input" do
      inp = %Container{a: Nx.tensor(5), b: Nx.tensor(3)}

      assert_equal(container_as_input(inp), Nx.tensor(15))
    end

    test "updated" do
      inp = %Container{a: Nx.tensor(1), b: 2, c: :reset, d: :keep}

      assert %Container{a: a, b: b, c: c, d: d} = update_container(inp, Nx.tensor(8))
      assert_equal(a, Nx.tensor(1))
      assert_equal(b, Nx.tensor(8))
      assert c == nil
      assert d == :keep
    end

    test "can be used with dot syntax" do
      inp = %Container{a: Nx.tensor(1), b: 2}
      assert_equal(dot_container(inp), Nx.tensor(2))
    end

    test "can be used with nested collections" do
      inp = %Container{a: {Nx.tensor(1), Nx.tensor(2)}, b: %{a: Nx.tensor(3), b: Nx.tensor(4)}}
      assert_equal(container_with_map_tuple(inp), Nx.tensor(24))
    end
  end

  describe "batch" do
    test "when padded" do
      input = Nx.tensor([[1, 2, 3]], backend: EXLA.Backend)
      batch = [input] |> Nx.Batch.concatenate() |> Nx.Batch.pad(1)
      predict = Nx.Defn.jit(fn input -> input end, compiler: EXLA)
      assert_equal(predict.(batch), Nx.tensor([[1, 2, 3], [0, 0, 0]]))
    end
  end

  describe "cache" do
    defn merge(init_map) do
      params = init()
      merge_transform(params, init_map)
    end

    deftransformp(merge_transform(params, init_map), do: Map.merge(params, init_map))

    defn init() do
      %{"x" => rand(), "y" => rand()}
    end

    deftransformp rand, do: :rand.uniform()

    test "considers map keys in cache keys" do
      assert_equal(merge(%{"x" => 10})["x"], Nx.tensor(10))
      assert_equal(merge(%{"y" => 10})["y"], Nx.tensor(10))
    end
  end

  describe "disk cache" do
    @tag :tmp_dir
    test "reads and writes", %{tmp_dir: dir} do
      cache = Path.join(dir, "exla.cache")

      assert capture_log(fn ->
               assert_equal(
                 EXLA.jit(&Nx.multiply/2, cache: cache, debug: true).(2, 3),
                 Nx.tensor(6)
               )
             end) =~ "EXLA disk cache not found"

      assert capture_log(fn ->
               assert_equal(
                 EXLA.jit(&Nx.multiply/2, cache: cache, debug: true).(3, 4),
                 Nx.tensor(12)
               )
             end) =~ "EXLA disk cache found"

      assert capture_log(fn ->
               assert_equal(
                 EXLA.jit(&Nx.multiply/2, cache: cache, debug: true).(
                   Nx.tensor([3]),
                   Nx.tensor([4])
                 ),
                 Nx.tensor([12])
               )
             end) =~ "EXLA disk cache does not match configuration"

      assert capture_log(fn ->
               assert_equal(
                 EXLA.jit(&Nx.multiply/2, cache: cache, debug: true).(
                   Nx.tensor([3]),
                   Nx.tensor([4])
                 ),
                 Nx.tensor([12])
               )
             end) =~ "EXLA disk cache found"
    end
  end

  describe "hooks" do
    defp send_to_self(tag) do
      parent = self()
      fn value -> send(parent, {tag, value}) end
    end

    defn hook_default(a, b) do
      hook(a + b, :default, send_to_self(:default))
    end

    test "executes hook with default" do
      assert hook_default(2, 3)
      assert_receive {:default, tensor}
      assert_equal(tensor, Nx.tensor(5))
    end

    test "executes hook with callback" do
      assert_equal(
        EXLA.jit(&hook_default/2, hooks: %{default: send_to_self(:tag)}).(2, 3),
        Nx.tensor(5)
      )

      assert_receive {:tag, tensor}
      assert_equal(tensor, Nx.tensor(5))

      # Executing again with another tag works
      assert_equal(
        EXLA.jit(&hook_default/2, hooks: %{default: send_to_self(:another_tag)}).(2, 3),
        Nx.tensor(5)
      )

      assert_receive {:another_tag, tensor}
      assert_equal(tensor, Nx.tensor(5))
    end

    defn hook_optional(a, b) do
      hook(a + b, :optional)
    end

    test "executes optional hook" do
      assert_equal(hook_optional(2, 3), Nx.tensor(5))

      assert_equal(
        EXLA.jit(&hook_optional/2, hooks: %{optional: send_to_self(:tag)}).(2, 3),
        Nx.tensor(5)
      )

      assert_receive {:tag, tensor}
      assert_equal(tensor, Nx.tensor(5))
    end

    defn hook_factorial(x) do
      {factorial, _} =
        while {factorial = 1.0, x}, Nx.greater(x, 1) do
          hook({factorial * x, x - 1}, :factorial)
        end

      factorial
    end

    test "executes hook within while" do
      assert_equal(
        EXLA.jit(&hook_factorial/1, hooks: %{factorial: send_to_self(:tag)}).(5),
        Nx.tensor(120.0)
      )

      assert_received {:tag, tuple}
      assert tuple == {Nx.tensor(5.0), Nx.tensor(4)}
      assert_received {:tag, tuple}
      assert tuple == {Nx.tensor(20.0), Nx.tensor(3)}
      assert_received {:tag, tuple}
      assert tuple == {Nx.tensor(60.0), Nx.tensor(2)}
      assert_received {:tag, tuple}
      assert tuple == {Nx.tensor(120.0), Nx.tensor(1)}
    end

    defn hook_cond(a, b) do
      cond do
        a == -1 -> hook(b * 2, :cond)
        a == 1 -> hook(b / 2, :cond)
        true -> hook(Nx.pow(b, 2), :cond)
      end
    end

    test "executes hook within cond" do
      assert_equal(
        EXLA.jit(&hook_cond/2, hooks: %{cond: send_to_self(:tag)}).(1, 4),
        Nx.tensor(2.0)
      )

      assert_received {:tag, tensor}
      assert_equal(tensor, Nx.tensor(2.0))

      assert_equal(
        EXLA.jit(&hook_cond/2, hooks: %{cond: send_to_self(:tag)}).(-1, 4),
        Nx.tensor(8.0)
      )

      assert_received {:tag, tensor}
      assert_equal(tensor, Nx.tensor(8))

      assert_equal(
        EXLA.jit(&hook_cond/2, hooks: %{cond: send_to_self(:tag)}).(0, 4),
        Nx.tensor(16.0)
      )

      assert_received {:tag, tensor}
      assert_equal(tensor, Nx.tensor(16))
    end

    defn hook_container(container) do
      hook(container, :container)
    end

    test "executes hook with container" do
      container = %Container{a: 1, b: 2, c: :reset, d: :elem}
      EXLA.jit(&hook_container/1, hooks: %{container: send_to_self(:tag)}).(container)

      assert_receive {:tag, %Container{a: a, b: b, c: nil, d: :elem}}
      assert_equal(a, Nx.tensor(1))
      assert_equal(b, Nx.tensor(2))
    end
  end

  describe "telemetry" do
    defn telemetry_add_two(a, b), do: a + b

    def telemetry_handler(_event_name, measurements, metadata, _config) do
      send(self(), {measurements, metadata})
    end

    test "executes event when function is compiled" do
      :ok =
        :telemetry.attach(__MODULE__, [:exla, :compilation], &__MODULE__.telemetry_handler/4, nil)

      on_exit(fn -> :telemetry.detach(__MODULE__) end)

      fun = &telemetry_add_two/2
      EXLA.jit(fun).(2, 3)
      assert_received {measurements, metadata}

      assert %{compile_time: compile_time, eval_time: eval_time, total_time: total_time} =
               measurements

      assert metadata == %{key: fun}

      assert is_integer(compile_time)
      assert eval_time > 0
      assert is_integer(eval_time)
      assert compile_time > 0
      assert total_time == compile_time + eval_time

      EXLA.jit(fun).(4, 5)
      refute_received _

      a = Nx.tensor([1, 2])
      b = Nx.tensor([3, 4])
      EXLA.jit(fun).(a, b)

      assert_received {measurements, metadata}

      assert %{compile_time: compile_time, eval_time: eval_time, total_time: total_time} =
               measurements

      assert metadata == %{key: fun}

      assert is_integer(compile_time)
      assert eval_time > 0
      assert is_integer(eval_time)
      assert compile_time > 0
      assert total_time == compile_time + eval_time
    end
  end
end
