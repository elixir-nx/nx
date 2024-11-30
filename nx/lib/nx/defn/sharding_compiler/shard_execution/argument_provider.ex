defmodule Nx.Defn.ShardingCompiler.ShardExecution.ArgumentProvider do
  use GenServer

  alias Nx.Defn.ShardingCompiler.ShardRegistry

  def init(%Nx.Tensor{} = data) do
    {:ok, data}
  end

  def start_link([data, idx, section_id]) do
    GenServer.start_link(__MODULE__, data, name: via_tuple(idx, section_id))
  end

  defp via_tuple(idx, section_id) do
    {:via, Registry, {ShardRegistry, {nil, idx, section_id}}}
  end

  def handle_call(:get, _from, data) do
    {:reply, {:ok, data}, data}
  end
end
