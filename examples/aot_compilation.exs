defmodule AotTest do
  import Nx.Defn

  @defn_compiler {Exla.Aot.Defn, shapes: [{1_000_000}], types: [{:s, 64}]}
  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t))
  end
end

AotTest.softmax(Nx.tensor(1))