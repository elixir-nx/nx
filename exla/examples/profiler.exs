defmodule Example do
  import Nx.Defn

  defn add(x, y) do
    Nx.add(x, y)
  end
end

Nx.default_backend(EXLA.Backend)
Nx.Defn.default_options(compiler: EXLA)

x = Nx.tensor([1, 2, 3])
y = Nx.tensor([4, 5, 6])

session = EXLA.Profiler.start()

Example.add(x, y)

EXLA.Profiler.stop_and_export(session, "profiler")