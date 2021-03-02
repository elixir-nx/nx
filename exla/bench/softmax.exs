size = 1_000_000
rand = for(_ <- 1..size, do: :rand.uniform())
t64 = Nx.tensor(rand, type: {:f, 64})
t32 = Nx.tensor(rand, type: {:f, 32})

defmodule Softmax do
  import Nx.Defn

  # This runs on Elixir
  defn softmax(n), do: Nx.exp(n) / Nx.sum(Nx.exp(n))

  # This is JIT+host compiled.
  @defn_compiler EXLA
  defn host(n), do: softmax(n)

  # This is JIT+cuda compiled
  @defn_compiler {EXLA, client: :cuda}
  defn cuda(n), do: softmax(n)

  # This is JIT+cuda+keep_on_device compiled
  @defn_compiler {EXLA, client: :cuda, run_options: [keep_on_device: true]}
  defn cuda_keep(n), do: softmax(n)
end

# Generate a module with an AOT version of softmax.
Nx.Defn.aot(
  AOTSoftmax,
  [
    {:softmax_64, &Softmax.softmax/1, [t64]},
    {:softmax_32, &Softmax.softmax/1, [t32]},
  ],
  EXLA
)

IO.inspect(AOTSoftmax.softmax_32(t32))
IO.inspect(Softmax.softmax(t32))
IO.inspect(Softmax.host(t32))

benches = %{
  "elixir f32" => fn -> Softmax.softmax(t32) end,
  "elixir f64" => fn -> Softmax.softmax(t64) end,
  "xla jit-cpu f32" => fn -> Softmax.host(t32) end,
  "xla jit-cpu f64" => fn -> Softmax.host(t64) end,
  "xla aot-cpu f32" => fn -> AOTSoftmax.softmax_32(t32) end,
  "xla aot-cpu f64" => fn -> AOTSoftmax.softmax_64(t64) end
}

benches =
  if System.get_env("EXLA_TARGET") == "cuda" do
    dt32 = Nx.backend_transfer(t32, EXLA.DeviceBackend, client: :cuda)
    dt64 = Nx.backend_transfer(t64, EXLA.DeviceBackend, client: :cuda)

    Map.merge(benches, %{
      "xla jit-gpu f32" => fn -> Softmax.cuda(t32) end,
      "xla jit-gpu f64" => fn -> Softmax.cuda(t64) end,
      "xla jit-gpu f32 keep" =>
        {fn -> Softmax.cuda_keep(dt32) end, after_each: &Nx.backend_deallocate/1},
      "xla jit-gpu f64 keep" =>
        {fn -> Softmax.cuda_keep(dt64) end, after_each: &Nx.backend_deallocate/1}
    })
  else
    benches
  end

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
