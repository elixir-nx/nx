defmodule EXLA.Defn.CallbackDispatcher do
  @moduledoc false
  use GenServer
  require Logger

  @table __MODULE__

  @doc """
  Registers callback IDs to be routed to the given pid.

  Note: cached executables reuse the same callback_ids across calls,
  so a new task will overwrite a previous task's entry for the same
  callback_id. This is expected — only the latest task should receive
  messages for a given callback_id.
  """
  def register(callback_ids, pid) when is_list(callback_ids) and is_pid(pid) do
    entries = Enum.map(callback_ids, &{&1, pid})
    :ets.insert(@table, entries)
  end

  @doc """
  Unregisters callback IDs, but only if the given pid is still the
  registered handler.

  This prevents a race condition with cached executables: when the same
  compiled function runs repeatedly, each invocation gets the same
  callback_ids. Without the pid check, a finishing task could delete
  the ETS entry that a newer task just registered, causing the newer
  task's runtime_call messages to be lost (deadlock).
  """
  def unregister(callback_ids, pid) when is_list(callback_ids) and is_pid(pid) do
    Enum.each(callback_ids, fn id ->
      :ets.select_delete(@table, [{{id, pid}, [], [true]}])
    end)
  end

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  @impl true
  def init([]) do
    :ets.new(@table, [:named_table, :public, :set])
    EXLA.NIF.start_runtime_callback_bridge(self())
    {:ok, %{}}
  end

  @impl true
  def handle_info({:exla_runtime_call, callback_id, args_spec, reply_tag}, state) do
    case :ets.lookup(@table, callback_id) do
      [{_, pid}] ->
        send(pid, {:exla_runtime_call, callback_id, args_spec, reply_tag})

      [] ->
        Logger.error(
          "EXLA CallbackDispatcher received callback_id #{inspect(callback_id)} " <>
            "with no registered handler"
        )
    end

    {:noreply, state}
  end

  def handle_info(_msg, state) do
    {:noreply, state}
  end
end
