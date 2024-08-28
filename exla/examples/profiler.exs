session = EXLA.Profiler.start()

defmodule Example do
  import Nx.Defn

  defn function(x, y) do
    Nx.add(x, y)
    |> Nx.cos()
    |> Nx.exp()
    |> Nx.sin()
    |> Nx.tanh()
    |> Nx.sum()
  end
end

Nx.default_backend(EXLA.Backend)
Nx.Defn.default_options(compiler: EXLA)

x = Nx.tensor([1, 2, 3])
y = Nx.tensor([4, 5, 6])

Example.function(x, y) |> IO.inspect

EXLA.Profiler.stop_and_export(session, "profiler")