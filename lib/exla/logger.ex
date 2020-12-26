
defmodule Exla.Logger do
  @moduledoc false
  use GenServer
  require Logger

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @impl true
  def init(:ok) do
    :ok = Exla.NIF.start_log_sink(self())
    {:ok, :unused_state}
  end

  @impl true
  def handle_info({:info, msg}, _state) do
    Logger.info(msg)
    {:noreply, :ok}
  end

  @impl true
  def handle_info({:warning, msg}, _state) do
    Logger.warning(msg)
    {:noreply, :ok}
  end

  @impl true
  def handle_info({:error, msg}, _state) do
    Logger.error(msg)
    {:noreply, :ok}
  end

  @impl true
  def handle_info({:fatal, msg}, _state) do
    Logger.error(msg)
    {:noreply, :ok}
  end
end
