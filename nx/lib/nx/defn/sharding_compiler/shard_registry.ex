defmodule Nx.Defn.ShardingCompiler.ShardRegistry do
  def child_spec(opts) do
    %{
      id: __MODULE__,
      start: {__MODULE__, :start_link, [opts]}
    }
  end

  def start_link(_) do
    Registry.start_link(name: __MODULE__, keys: :unique)
  end

  def lookup(key) do
    results = :erpc.multicall([Node.self() | Node.list()], Registry, :lookup, [__MODULE__, key])

    results
    |> Enum.find_value(fn
      {:ok, [{pid, _}]} -> pid
      _ -> nil
    end)
    |> case do
      nil -> {:error, :pending}
      pid -> {:ok, pid}
    end
  end
end
