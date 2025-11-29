defmodule EXLA.CallbackServer.Supervisor do
  @moduledoc false

  use DynamicSupervisor

  @impl true
  def start_link(init_arg) do
    DynamicSupervisor.start_link(__MODULE__, init_arg, name: __MODULE__)
  end

  @impl true
  def init(_init_arg) do
    DynamicSupervisor.init(strategy: :one_for_one)
  end

  def start_callback_server do
    DynamicSupervisor.start_child(__MODULE__, {EXLA.CallbackServer, []})
  end

  def terminate_callback_server(pid) do
    DynamicSupervisor.terminate_child(__MODULE__, pid)
  end
end
