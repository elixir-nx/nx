size = 1_000_000
rand = for(_ <- 1..size, do: :rand.uniform())
t64 = Nx.tensor(rand, type: {:f, 64})
t32 = Nx.tensor(rand, type: {:f, 32})

defmodule Softmax do
  import Nx.Defn

  defn softmax(n), do: Nx.exp(n) / Nx.sum(Nx.exp(n))
end

host_jit = EXLA.jit(&Softmax.softmax/1)

# We call |> Nx.add(1) to force the computation results to be loaded
benches = %{
  "elixir f32" => fn -> Softmax.softmax(t32) |> Nx.add(1) end,
  "elixir f64" => fn -> Softmax.softmax(t64) |> Nx.add(1) end,
  "xla jit-cpu f32" => fn -> host_jit.(t32) |> Nx.add(1) end,
  "xla jit-cpu f64" => fn -> host_jit.(t64) |> Nx.add(1) end
}

benches =
  if System.get_env("EXLA_TARGET") == "cuda" do
    dt32 = Nx.backend_transfer(t32, {EXLA.Backend, client: :cuda})
    dt64 = Nx.backend_transfer(t64, {EXLA.Backend, client: :cuda})

    cuda_jit = EXLA.jit(&Softmax.softmax/1, client: :cuda)

    Map.merge(benches, %{
      "xla jit-gpu f32" =>
        {fn -> cuda_jit.(dt32) |> Nx.add(1) end, after_each: &Nx.backend_deallocate/1},
      "xla jit-gpu f64" =>
        {fn -> cuda_jit.(dt64) |> Nx.add(1) end, after_each: &Nx.backend_deallocate/1}
    })
  else
    benches
  end

Benchee.run(
  benches,
  time: 10,
  memory_time: 2
)
