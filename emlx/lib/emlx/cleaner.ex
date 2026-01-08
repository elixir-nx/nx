defmodule EMLX.Cleaner do
  use GenServer

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    {:ok, %{}}
  end

  @impl true
  def handle_info({:cleanup, key}, state) do
    :persistent_term.erase(key)
    {:noreply, state}
  end
end
