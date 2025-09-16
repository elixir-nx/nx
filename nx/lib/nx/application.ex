defmodule Nx.Application do
  @moduledoc false
  use Application

  def start(_type, _args) do
    children = [
      %{id: Nx.Serving.PG, start: {:pg, :start_link, [Nx.Serving.PG]}},
      {Nx.HiddenServing, Nx.Serving.PG}
    ]

    Supervisor.start_link(children, strategy: :one_for_all, name: Nx.Supervisor)
  end
end

defmodule Nx.HiddenServing do
  # Module to connect hidden nodes with serving.
  # It relies on sending private pg messages to Nx serving.
  @moduledoc false
  use GenServer

  @doc false
  def start_link(scope) do
    GenServer.start_link(__MODULE__, scope)
  end

  @impl true
  def init(scope) do
    pid = Process.whereis(scope)
    :ok = :net_kernel.monitor_nodes(true, node_type: :hidden)

    for node <- Node.list(:hidden) do
      send({scope, node}, {:discover, pid})
    end

    {:ok, scope}
  end

  @impl true
  def handle_info({:nodeup, node, _}, scope) when node != node() do
    send({scope, node}, {:discover, Process.whereis(scope)})
    {:noreply, scope}
  end

  def handle_info(_msg, scope) do
    {:noreply, scope}
  end
end
