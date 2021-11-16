defmodule EXLA.Defn.Lock do
  @moduledoc false

  use GenServer
  require Logger

  @name __MODULE__
  @timeout :infinity

  @doc """
  Locks the given `key`.

  It will wait until the key becomes available.
  """
  def lock(key) do
    GenServer.call(@name, {:lock, key}, @timeout)
  end

  @doc """
  Relocks the given `ref`.

  The `to_lock` is executed and then the new `to_unlock` is registered.
  """
  def relock(ref, to_lock, to_unlock)
      when is_reference(ref) and
             is_function(to_lock, 0) and is_function(to_unlock, 0) do
    GenServer.call(@name, {:relock, ref, to_lock, to_unlock}, @timeout)
  end

  @doc """
  Unlocks the given `ref`.

  It will execute the registered `to_unlock` callback, if any.
  """
  def unlock(ref) when is_reference(ref) do
    GenServer.call(@name, {:unlock, ref}, @timeout)
  end

  ## Callbacks

  @doc false
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: @name)
  end

  @impl true
  def init(:ok) do
    Process.flag(:trap_exit, true)
    {:ok, {%{}, %{}}}
  end

  @impl true
  def handle_call({:lock, key}, from, {refs, devices}) do
    {refs, devices} =
      get_and_update_in(devices[key], fn device ->
        device
        |> enqueue(from)
        |> dequeue_if_possible(key, refs)
      end)

    {:noreply, {refs, devices}}
  end

  def handle_call({:unlock, ref}, _from, {refs, devices}) do
    _ = Process.demonitor(ref, [:flush])
    {:reply, :ok, unlock(ref, refs, devices)}
  end

  def handle_call({:relock, ref, to_lock, to_unlock}, _from, {refs, devices}) do
    key = Map.fetch!(refs, ref)
    res = to_lock.()
    devices = update_in(devices[key], fn {_to_unlock, queue} -> {to_unlock, queue} end)
    {:reply, res, {refs, devices}}
  end

  @impl true
  def handle_info({:DOWN, ref, _, _, _}, {refs, devices}) do
    {:noreply, unlock(ref, refs, devices)}
  end

  defp enqueue(nil, entry), do: enqueue({:unlocked, :queue.new()}, entry)
  defp enqueue({state, queue}, entry), do: {state, :queue.in(entry, queue)}

  defp unlock(ref, refs, devices) do
    case Map.pop(refs, ref, nil) do
      {nil, refs} ->
        {refs, devices}

      {key, refs} ->
        get_and_update_in(devices[key], fn {to_unlock, queue} ->
          case run_to_unlock(key, to_unlock) do
            {:lock, to_watch, to_unlock} ->
              ref = Process.monitor(to_watch)
              {Map.put(refs, ref, key), {to_unlock, queue}}

            :unlock ->
              dequeue_if_possible({:unlocked, queue}, key, refs)
          end
        end)
    end
  end

  defp run_to_unlock(key, to_unlock) do
    to_unlock.()
  rescue
    exception ->
      Logger.error(
        "Unlocking #{inspect(key)} with #{inspect(to_unlock)} failed. " <>
          Exception.format(:error, exception, __STACKTRACE__)
      )

      :unlock
  end

  defp dequeue_if_possible({:unlocked, queue}, key, refs) do
    case :queue.out(queue) do
      {{:value, {pid, _} = from}, queue} ->
        ref = Process.monitor(pid)
        GenServer.reply(from, ref)
        {Map.put(refs, ref, key), {fn -> :unlock end, queue}}

      {:empty, queue} ->
        {refs, {:unlocked, queue}}
    end
  end

  defp dequeue_if_possible({to_unlock, queue}, _key, refs) do
    {refs, {to_unlock, queue}}
  end
end
