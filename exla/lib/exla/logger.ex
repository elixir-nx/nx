defmodule EXLA.Logger do
  @moduledoc false
  use GenServer
  require Logger

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @impl true
  def init(:ok) do
    :ok = EXLA.NIF.start_log_sink(self())
    {:ok, :unused_state}
  end

  @impl true
  def handle_info({:info, msg, file, line}, state) do
    Logger.info(msg, domain: [:xla], file: file, line: line)
    {:noreply, state}
  end

  @impl true
  def handle_info({:warning, msg, file, line}, state) do
    Logger.warning(msg, domain: [:xla], file: file, line: line)
    {:noreply, state}
  end

  @impl true
  def handle_info({:error, msg, file, line}, state) do
    Logger.error(msg, domain: [:xla], file: file, line: line)
    {:noreply, state}
  end
end
