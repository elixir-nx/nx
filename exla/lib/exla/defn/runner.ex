defmodule EXLA.Defn.Runner do
  # Executes a given function and then exits once the value is read once.
  # This is similar to a task but we don't use a task to avoid polluting
  # inboxes
  @moduledoc false
  use GenServer

  def start_link(ref, fun) do
    GenServer.start_link(__MODULE__, {ref, fun})
  end

  def read(pid) do
    GenServer.call(pid, :read, :infinity)
  end

  @impl true
  def init({ref, fun}) do
    {:ok, nil, {:continue, {ref, fun}}}
  end

  @impl true
  def handle_continue({ref, fun}, nil) do
    receive do
      ^ref -> {:noreply, fun.()}
    end
  end

  @impl true
  def handle_call(:read, _from, value), do: {:stop, :normal, value, value}
end
