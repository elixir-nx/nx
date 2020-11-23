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
  @defn_compiler {Exla, platform: :cuda}
  defn cuda(n), do: softmax(n)
end

IO.inspect(Softmax.softmax(t))
IO.inspect(Softmax.host(t))

Benchee.run(
  %{
    "elixir softmax" => fn -> Softmax.softmax(t) end,
    "xla cpu softmax" => fn -> Softmax.host(t) end
  },
  time: 10,
  memory_time: 2
)
