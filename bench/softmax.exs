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

benches =  %{
  "elixir" => fn -> Softmax.softmax(t) end,
  "xla cpu" => fn -> Softmax.host(t) end
}

benches =
  if System.get_env("EXLA_TARGET") == "cuda" do
    IO.inspect(Softmax.cuda(t))
    Map.put(benches, "xla gpu", Softmax.cuda(t))
  else
    benches
  end

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
