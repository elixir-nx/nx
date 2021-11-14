defmodule EXLA.Defn.Outfeed do
  @moduledoc false

  @doc """
  Receives a client, device_id, and mappings of u16 to
  `{shapes, {pid, ref}}` pairs to deliver the outputs
  to. The computation must emit a 0 flag on exit.
  """
  def start_child(client, device_id, mappings) do
    Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
      init(client, device_id, mappings)
    end)
  end

  defp init(client, device_id, mappings) do
    Process.flag(:trap_exit, true)
    ref = make_ref()
    shape = EXLA.Shape.make_shape({:u, 16}, {})
    loop(client, device_id, ref, shape, mappings)
  end

  defp loop(client, device_id, ref, shape, mappings) do
    :ok = EXLA.Client.from_outfeed(client, device_id, [shape], self(), ref)

    receive do
      {^ref, <<0::native-unsigned-16>>} ->
        :ok

      {^ref, <<flag::native-unsigned-16>>} ->
        {shapes, {recv_pid, recv_ref}} = Map.fetch!(mappings, flag)
        :ok = EXLA.Client.from_outfeed(client, device_id, shapes, recv_pid, recv_ref)
        loop(client, device_id, ref, shape, mappings)
    end
  end
end
