
defmodule Exla.Logger do
  use GenServer
  require Logger

  @impl true
  def init(_) do
    Logger.info("Exla Logger started.")
    {Exla.NIF.start_log_sink(self()), :ok}
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
