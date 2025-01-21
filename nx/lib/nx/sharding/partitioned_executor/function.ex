defmodule Nx.Sharding.PartitionedExecutor.Function do
  @moduledoc false

  use GenServer

  alias Nx.Sharding.PartitionedExecutor

  # args is a list of argument ids
  # results is a list of result ids
  # id is a unique identifier for the function
  # function is a function of arity length(args) which returns a list of length length(results)

  # a result id is a tuple of {producer_id, index}
  # an argument id is always a result id from some other function
  # functions should always receive and outputs lists for uniformity

  defstruct [:id, :args, :results, :function]

  def start_link([executor_pid, function]) do
    GenServer.start_link(__MODULE__, {executor_pid, function})
  end

  def init({executor_pid, function}) do
    Process.send_after(self(), :run, 0)
    {:ok, %{executor_pid: executor_pid, function: function}}
  end

  def get(function_pid, arg_index) do
    GenServer.call(function_pid, {:get, arg_index})
  end

  def check_status(function_pid) do
    GenServer.call(function_pid, :check_status)
  end

  def handle_info(:run, %{function: %__MODULE__{} = function, executor_pid: executor_pid} = state) do
    %__MODULE__{args: args, function: code} = function

    {args_pids, all_args_available} =
      Enum.map_reduce(args, true, fn {producer_id, index}, acc ->
        case PartitionedExecutor.check_status(executor_pid, producer_id) do
          {:ok, pid} ->
            {{pid, index}, acc}

          {:error, :pending} ->
            {nil, false}
        end
      end)

    new_state =
      if all_args_available do
        # check if all args are available. If they are, fetch them and run the function
        args =
          Enum.map(args_pids, fn {pid, index} ->
            {:ok, result} = get(pid, index)
            result
          end)

        results = code.(args)
        put_in(state.function.results, results)
      else
        # if not, wait a few ms
        Process.send_after(self(), :run, 20)
        state
      end

    {:noreply, new_state}
  end

  def handle_call(:check_status, _from, state) do
    result =
      if state.function.results do
        :ok
      else
        {:error, :pending}
      end

    {:reply, result, state}
  end

  def handle_call({:get, arg_index}, _from, state) do
    result =
      if results = state.function.results do
        {:ok, Enum.fetch!(results, arg_index)}
      else
        {:error, :pending}
      end

    {:reply, result, state}
  end
end
