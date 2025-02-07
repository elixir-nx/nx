defmodule Nx.Sharding.PartitionedExecutor.Function do
  @moduledoc false

  use GenServer

  require Logger

  # args is a list of {argument id, indices}
  # results is a tuple with the results
  # id is a unique identifier for the function
  # code is a function of arity 1 + length(extra_args) which returns a tuple of size length(results)

  # a result id is a tuple of {producer_id, index}
  # an argument id is always a result id from some other function
  # functions should always receive and outputs lists for uniformity

  defstruct [:id, :args, :results, :code, :node, extra_args: []]

  defp get_name(function_id) do
    {:global, {__MODULE__, function_id}}
  end

  def start_link(function) do
    name = get_name(function.id)
    GenServer.start_link(__MODULE__, function, name: name)
  end

  def init(function) do
    Process.send_after(self(), :run, 0)
    {:ok, %{function: function}}
  end

  def get(function_id, arg_index) do
    try do
      :global.sync()
      GenServer.call(get_name(function_id), {:get, arg_index})
    catch
      :exit, _ -> {:error, :not_started}
    end
  end

  def check_status(function_id) do
    try do
      :global.sync()

      GenServer.call(get_name(function_id), :check_status)
    catch
      :exit, _ -> {:error, :not_started}
    end
  end

  def handle_info(:run, %{function: %__MODULE__{} = function} = state) do
    %__MODULE__{args: args, code: code} = function

    all_args_available =
      function.results ||
        Enum.all?(function.args, fn {producer_id, _} -> check_status(producer_id) == :ok end)

    new_state =
      cond do
        function.results ->
          state

        all_args_available ->
          # check if all args are available. If they are, fetch them and run the function
          args =
            args
            |> Enum.map(fn {producer_id, index} ->
              {:ok, result} = get(producer_id, index)
              result
            end)
            |> List.to_tuple()

          results = apply(code, [args | function.extra_args])
          put_in(state.function.results, results)

        true ->
          # if not, wait a few ms
          Process.send_after(self(), :run, 200)
          state
      end

    {:noreply, new_state}
  end

  def handle_info(_, state) do
    {:noreply, state}
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
        {:ok, elem(results, arg_index)}
      else
        {:error, :pending}
      end

    {:reply, result, state}
  end
end
