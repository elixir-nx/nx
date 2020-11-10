defmodule Exla.Client do
  alias __MODULE__, as: Client

  @enforce_keys [:ref]
  defstruct [:ref]

  use GenServer

  # TODO: This should match LocalClientOptions when those are handled
  # There's a few scenarios where this could fail, is this the best way to handle failure on startup?
  def start_link(options) do
    case GenServer.start_link(__MODULE__, options) do
      {:ok, pid} -> {:ok, pid}
      {:error, :normal} -> {:error, :init_error}
    end
  end

  def get_device_count(pid) do
    GenServer.call(pid, :get_device_count)
  end

  # TODO: Store some of the options for reference as well
  @impl true
  def init(options) do
    case Exla.NIF.get_or_create_local_client(options) do
      {:ok, ref} -> {:ok, %Client{ref: ref}}
      {:error, _msg} -> {:error, :normal}
    end
  end

  @impl true
  def handle_call(:get_device_count, _from, client = %Client{}) do
    device_count = Exla.NIF.get_device_count(client.ref)
    {:reply, device_count, client}
  end
end
