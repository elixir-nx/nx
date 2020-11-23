defmodule Exla do
  @behaviour Nx.Defn.Compiler
  @impl true
  defdelegate __compile__(kind, meta, name, args, ast, opts), to: Exla.Defn
end
