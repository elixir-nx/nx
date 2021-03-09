defmodule EXLA.LockedCache do
  @moduledoc false

  # EXLA has many singleton resources like clients
  # and expensive resources like executable that we
  # want to compute just once, this module provides
  # a cache functionality so that those are done only
  # once even in face of concurrency.
  use GenServer

  @name __MODULE__
  @timeout :infinity

  @doc """
  Reads cache key or executes the given function if not
  cached yet.
  """
  def run(key, fun) do
    # TODO: Move this to ETS once we add a process.
    case :persistent_term.get(key, __MODULE__) do
      __MODULE__ ->
        case GenServer.call(@name, {:lock, key}, @timeout) do
          {:uncached, ref} ->
            try do
              fun.()
            catch
              kind, reason ->
                GenServer.cast(@name, {:uncached, ref})
                :erlang.raise(kind, reason, __STACKTRACE__)
            else
              {return, result} ->
                :persistent_term.put(key, result)
                GenServer.cast(@name, {:cached, ref})
                {return, result}
            end

          :cached ->
            {nil, :persistent_term.get(key)}
        end

      value ->
        {nil, value}
    end
  end

  @doc false
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: @name)
  end

  @impl true
  def init(:ok) do
    {:ok, %{keys: %{}, ref_to_key: %{}}}
  end

  @impl true
  def handle_call({:lock, key}, from, state) do
    case state.keys do
      %{^key => {ref, waiting}} ->
        {:noreply, put_in(state.keys[key], {ref, [from | waiting]})}

      %{} ->
        {:noreply, lock(key, from, [], state)}
    end
  end

  @impl true
  def handle_cast({:cached, ref}, state) do
    Process.demonitor(ref, [:flush])
    {key, state} = pop_in(state.ref_to_key[ref])
    {{^ref, waiting}, state} = pop_in(state.keys[key])
    for from <- waiting, do: GenServer.reply(from, :cached)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:uncached, ref}, state) do
    Process.demonitor(ref, [:flush])
    {:noreply, unlock(ref, state)}
  end

  @impl true
  def handle_info({:DOWN, ref, _, _, _}, state) do
    {:noreply, unlock(ref, state)}
  end

  defp lock(key, {pid, _} = from, waiting, state) do
    ref = Process.monitor(pid)
    state = put_in(state.keys[key], {ref, waiting})
    state = put_in(state.ref_to_key[ref], key)
    GenServer.reply(from, {:uncached, ref})
    state
  end

  defp unlock(ref, state) do
    {key, state} = pop_in(state.ref_to_key[ref])
    {{^ref, waiting}, state} = pop_in(state.keys[key])

    case waiting do
      [] -> state
      [from | waiting] -> lock(key, from, waiting, state)
    end
  end
end
