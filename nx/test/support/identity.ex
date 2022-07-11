defmodule Nx.Defn.Identity do
  @behaviour Nx.Defn.Compiler

  def __stream__(_, _, _, _, _, _, _), do: raise("not implemented")

  def __compile__(_, _, _, _), do: raise("not implemented")

  def __jit__(key, vars, fun, args, _opts) do
    Process.put(__MODULE__, key)
    expr = fun.(vars)
    Enum.map(args, fn _ -> expr end)
  end
end
