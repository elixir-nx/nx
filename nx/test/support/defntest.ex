defmodule MyDefnTest do
  import Nx.Defn

  defn f(x, opts \\ []) do
    my_transform(opts)

    x + x
  end

  deftransformp my_transform(opts) do
    send(self(), {:mytf, opts})
  end
end
