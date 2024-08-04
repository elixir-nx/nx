defmodule EXLA.Plugin do
  @moduledoc """
  Plugin system for registering custom calls.
  """
  use GenServer

  # TODO: Register and lookup per client

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  def register(key, library_path) do
    GenServer.cast(__MODULE__, {:register, key, library_path})
  end

  def lookup(key) do
    GenServer.call(__MODULE__, {:lookup, key})
  end

  def register_symbol(key, symbol, dimensions) do
    if ref = lookup(key) do
      EXLA.NIF.register_custom_call_symbol(ref, symbol, dimensions)
    end
  end

  @impl true
  def init(_opts) do
    {:ok, %{}}
  end

  @impl true
  def handle_cast({:register, key, library_path}, state) do
    case state do
      %{^key => _ref} ->
        {:noreply, state}

      %{} ->
        ref =
          library_path
          |> EXLA.NIF.load_custom_call_plugin_library()
          |> unwrap!()

        {:noreply, Map.put(state, key, ref)}
    end
  end

  @impl true
  def handle_call({:lookup, key}, _from, state) do
    value = Map.get(state, key)
    {:reply, value, state}
  end

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise("#{reason}")
end
