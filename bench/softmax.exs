size = 1_000_000
t64 = Nx.tensor(for _ <- 1..size, do: :rand.uniform())
t32 = Nx.tensor((for _ <- 1..size, do: :rand.uniform()), type: {:f, 32})

defmodule Softmax do
  import Nx.Defn

  # This runs on Elixir
  defn softmax(n), do: Nx.exp(n) / Nx.sum(Nx.exp(n))

  # This is JIT+host compiled
  @defn_compiler EXLA
  defn host(n), do: softmax(n)

  # This is JIT+cuda compiled
  @defn_compiler {EXLA, client: :cuda}
  defn cuda(n), do: softmax(n)

  # This is JIT+cuda+keep_on_device compiled
  @defn_compiler {EXLA, client: :cuda, keep_on_device: true}
  defn cuda_keep(n), do: softmax(n)
end

IO.inspect(Softmax.softmax(t32))
IO.inspect(Softmax.host(t32))

benches = %{
  "elixir f32" => fn -> Softmax.softmax(t32) end,
  "elixir f64" => fn -> Softmax.softmax(t64) end,
  "xla cpu f32" => fn -> Softmax.host(t32) end,
  "xla cpu f64" => fn -> Softmax.host(t64) end
}

benches =
  if System.get_env("EXLA_TARGET") == "cuda" do
    dt32 = Nx.device_transfer(t32, EXLA.NxDevice, client: :cuda)
    dt64 = Nx.device_transfer(t64, EXLA.NxDevice, client: :cuda)

    Map.merge(benches, %{
      "xla gpu f32" => fn -> Softmax.cuda(t32) end,
      "xla gpu f64" => fn -> Softmax.cuda(t64) end,
      "xla gpu f32 keep" => {fn -> Softmax.cuda_keep(dt32) end, after_each: &Nx.device_deallocate/1},
      "xla gpu f64 keep" => {fn -> Softmax.cuda_keep(dt64) end, after_each: &Nx.device_deallocate/1}
    })
  else
    benches
  end

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
