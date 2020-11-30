size = 1_000_000
t = Nx.tensor(for _ <- 1..size, do: :rand.uniform())

defmodule Softmax do
  import Nx.Defn

  # This runs on Elixir
  defn softmax(n), do: Nx.exp(n) / Nx.sum(Nx.exp(n))

  # This is JIT+host compiled
  @defn_compiler Exla
  defn host(n), do: softmax(n)

  # This is JIT+cuda compiled
  @defn_compiler {Exla, client: :cuda}
  defn cuda(n), do: softmax(n)

  # This is JIT+cuda+keep_on_device compiled
  @defn_compiler {Exla, client: :cuda, keep_on_device: true}
  defn cuda_keep(n), do: softmax(n)
end

IO.inspect(Softmax.softmax(t))
IO.inspect(Softmax.host(t))

benches =  %{
  "elixir" => fn -> Softmax.softmax(t) end,
  "xla cpu" => fn -> Softmax.host(t) end
}

benches =
  if System.get_env("EXLA_TARGET") == "cuda" do
    IO.inspect(Softmax.cuda(t))
    Map.put(benches, "xla gpu", fn -> Softmax.cuda(t) end)

    dt = Nx.device_transfer(t, Exla.NxDevice, client: :cuda)
    IO.inspect(Softmax.cuda_keep(dt))
    Map.put(benches, "xla gpu keep", fn -> Softmax.cuda_keep(dt) end)
  else
    benches
  end

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
