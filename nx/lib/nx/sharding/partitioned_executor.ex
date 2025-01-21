defmodule Nx.Sharding.PartitionedExecutor do
  @moduledoc false

  # Receives a acyclic directed graph of functions which depend on each other's results
  # and runs them in an as parallel as possible manner.

  # The graph is just represent as a list of Function structs, which themselves
  # contain the function dependencies.

  alias Nx.Sharding.PartitionedExecutor.Function, as: F
  alias Nx.Sharding.PartitionedExecutor.Supervisor, as: S

  def start_link(graph) do
    GenServer.start_link(__MODULE__, graph)
  end

  def init(graph) do
    Process.send_after(self(), :start_workflow, 0)

    {:ok, %{graph: graph, supervisor_pid: nil, producers: nil}}
  end

  def check_status(executor_pid, producer_id) do
    GenServer.call(executor_pid, {:check_status, producer_id})
  end

  def handle_info(:start_workflow, state) do
    {:ok, supervisor_pid} =
      S.start_link({self(), state.graph})

    producers =
      supervisor_pid
      |> Supervisor.which_children()
      |> Map.new(fn {{F, id}, pid, _, _} ->
        # TODO: deal with the possibility of
        # a producer dying
        {id, pid}
      end)

    {:noreply, %{state | supervisor_pid: supervisor_pid, producers: producers}}
  end

  def handle_call({:check_status, producer_id}, _from, state) do
    pid = Map.fetch!(state.producers, producer_id)

    result =
      case F.check_status(pid) do
        :ok -> {:ok, pid}
        {:error, :pending} -> {:error, :pending}
      end

    {:reply, result, state}
  end
end
