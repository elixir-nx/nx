# CPU benchmark of an f32 matmul. Cold, warm, and shape-cache phases are timed
# apart so a first compilation is not mixed into the steady state.
#
#   mix run bench/matmul.exs
#
# Inputs are copied to the host client before the timer starts. The run itself
# waits until the result buffer is ready, so the sample does not include a
# host copy of the output.
#
# [:exla, :compilation] fires once per new shape and should stay quiet on
# warm runs and on shape-cache hits.

sizes = [64, 256, 1024]

defmodule Matmul do
  import Nx.Defn

  defn dot(a, b), do: Nx.dot(a, b)
end

{:ok, _} =
  Agent.start_link(fn -> [] end, name: :exla_matmul_bench_compilations)

:telemetry.attach(
  "exla-matmul-bench",
  [:exla, :compilation],
  fn _event, measurements, _meta, _config ->
    Agent.update(:exla_matmul_bench_compilations, &[measurements | &1])
  end,
  nil
)

EXLA.Client.fetch!(EXLA.Client.default_name())

inputs =
  Map.new(sizes, fn n ->
    left = Nx.iota({n, n}, type: :f32)
    right = Nx.multiply(left, 0.001)

    {n,
     {
       Nx.backend_transfer(left, EXLA.Backend),
       Nx.backend_transfer(right, EXLA.Backend)
     }}
  end)

jit = EXLA.jit(&Matmul.dot/2)

drain = fn ->
  Agent.get_and_update(:exla_matmul_bench_compilations, fn events ->
    {Enum.reverse(events), []}
  end)
end

ms = fn microseconds -> Float.round(microseconds / 1000, 1) end

report = fn label, microseconds, events ->
  detail =
    case events do
      [] ->
        "compilation events 0"

      [%{eval_time: eval_time, compile_time: compile_time}] ->
        "compilation events 1, eval #{ms.(eval_time)} ms, compile #{ms.(compile_time)} ms"

      several ->
        "compilation events #{length(several)}"
    end

  IO.puts("#{label}  wall #{ms.(microseconds)} ms, #{detail}")
end

run = fn n ->
  {left, right} = Map.fetch!(inputs, n)
  jit.(left, right)
end

# Drop anything recorded while the inputs were built.
drain.()

{cold_us, cold_out} = :timer.tc(fn -> run.(64) end)
Nx.backend_deallocate(cold_out)
report.("cold 64x64", cold_us, drain.())

IO.puts("\nwarm 64x64")

Benchee.run(
  %{"64x64" => {fn -> run.(64) end, after_each: &Nx.backend_deallocate/1}},
  time: 5,
  warmup: 1,
  memory_time: 0
)

warm_events = drain.()

if warm_events != [] do
  IO.puts("warm 64x64 compiled #{length(warm_events)} more time(s)")
end

Enum.reduce([256, 1024], 64, fn n, previous ->
  {miss_us, miss_out} = :timer.tc(fn -> run.(n) end)
  Nx.backend_deallocate(miss_out)
  report.("shape cache miss #{n}x#{n}", miss_us, drain.())

  {hit_us, hit_out} = :timer.tc(fn -> run.(previous) end)
  Nx.backend_deallocate(hit_out)
  report.("shape cache hit #{previous}x#{previous}", hit_us, drain.())

  IO.puts("\nwarm #{n}x#{n}")

  Benchee.run(
    %{"#{n}x#{n}" => {fn -> run.(n) end, after_each: &Nx.backend_deallocate/1}},
    time: 5,
    warmup: 1,
    memory_time: 0
  )

  events = drain.()

  if events != [] do
    IO.puts("warm #{n}x#{n} compiled #{length(events)} more time(s)")
  end

  n
end)
