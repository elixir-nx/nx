defmodule Nx.Sharding.PartitionedExecutor do
  @moduledoc false
  use GenServer

  # Receives a acyclic directed graph of functions which depend on each other's results
  # and runs them in an as parallel as possible manner.

  # The graph is just represent as a list of Function structs, which themselves
  # contain the function dependencies.

  def start_link(graph) do
    GenServer.start_link(__MODULE__, graph)
  end

  def init(graph) do
    execution_group_id = make_ref()

    graph =
      Enum.map(graph, fn function ->
        id = {execution_group_id, function.id}
        args = Enum.map(function.args, fn {id, index} -> {{execution_group_id, id}, index} end)
        %{function | id: id, args: args}
      end)

    Process.send_after(self(), :start_workflow, 0)

    {:ok, %{graph: graph}}
  end

  def handle_info(:start_workflow, state) do
    # Group functions by node
    functions_by_node =
      Enum.group_by(state.graph, fn function ->
        if function.node == Node.self() do
          nil
        else
          function.node
        end
      end)

    # Start supervisors and functions on each node
    Enum.each(functions_by_node, fn
      {nil, functions} ->
        # Start supervisor on current node
        {:ok, _} =
          Nx.Sharding.PartitionedExecutor.Supervisor.start_link(functions)

      {node, functions} ->
        if !Node.connect(node) do
          raise "Failed to connect to node #{node}"
        end

        # TO-DO: we should be monitoring the supervisor and restarting it if it dies
        {:ok, pid} =
          :rpc.block_call(node, Nx.Sharding.PartitionedExecutor.Supervisor, :start_link, [
            functions
          ])

        Process.link(pid)
    end)

    :global.sync()

    {:noreply, state}
  end

  def handle_info(_, state) do
    {:noreply, state}
  end
end
