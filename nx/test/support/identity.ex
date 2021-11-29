defmodule Nx.Defn.Identity do
  @behaviour Nx.Defn.Compiler

  def __stream__(_, _, _, _, _, _), do: raise("not implemented")

  def __jit__(key, vars, fun, _opts) do
    Process.put(__MODULE__, key)
    fun.(vars)
  end
end
