defmodule Nx.Defn.Debug do
  @moduledoc false

  @behaviour Nx.Defn.Compiler

  @impl true
  def __partitions_options__(_), do: raise("not implemented")

  @impl true
  def __to_backend__(_), do: raise("not implemented")

  @impl true
  def __stream__(_, _, _, _, _, _, _), do: raise("not implemented")

  @impl true
  def __compile__(_, _, _, _), do: raise("not implemented")

  @impl true
  def __jit__(key, vars, fun, args, _opts) do
    Process.put(__MODULE__, key)
    expr = fun.(vars)
    Enum.map(args, fn _ -> expr end)
  end
end
