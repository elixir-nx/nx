defmodule Nx.Defn.Async do
  # Default implementation for Nx.Async
  @moduledoc false

  @derive {Inspect, only: [:ref]}
  @enforce_keys [:ref, :pid]
  defstruct [:ref, :pid]

  @doc false
  def async(fun) do
    {backend, backend_options} = Nx.default_backend()
    ref = make_ref()

    {:ok, pid} =
      Task.start_link(fn ->
        Nx.default_backend(backend, backend_options)
        result = fun.()

        receive do
          {:await!, pid, ^ref} ->
            send(pid, {ref, result})
            :ok
        end
      end)

    %Nx.Defn.Async{ref: ref, pid: pid}
  end

  defimpl Nx.Async do
    def await!(%{ref: ref, pid: pid} = async) do
      monitor = Process.monitor(pid)
      send(pid, {:await!, self(), ref})

      receive do
        {^ref, result} ->
          Process.demonitor(monitor, [:flush])
          result

        {:DOWN, ^monitor, _, _, reason} ->
          exit({reason, {Nx.Async, :await!, [async]}})
      end
    end
  end
end
