defmodule EXLA.Defn.Outfeed do
  @moduledoc false

  @doc """
  Receives a client, device_id, and mappings of u16 to
  `{shapes, {pid, ref} | {fun, template}}` pairs to
  deliver/execute the outputs. The computation must emit
  a 0 flag on exit.
  """
  def start_child(%EXLA.Executable{client: client, device_id: device_id}, hooks, group_leader) do
    Task.Supervisor.start_child(EXLA.Defn.TaskSupervisor, fn ->
      init(client, device_id, hooks, group_leader)
    end)
  end

  defp init(client, device_id, hooks, group_leader) do
    Process.flag(:trap_exit, true)
    # Copy the group leader so we report to the proper device
    Process.group_leader(self(), group_leader)
    ref = make_ref()
    shape = EXLA.Shape.make_shape({:u, 16}, {})
    loop(client, device_id, ref, shape, hooks)
  end

  defp loop(client, device_id, ref, shape, hooks) do
    :ok = EXLA.Client.from_outfeed(client, device_id, [shape], self(), ref)

    receive do
      {^ref, <<0::native-unsigned-16>>} ->
        :ok

      {^ref, <<flag::native-unsigned-16>>} ->
        case Map.fetch!(hooks, flag) do
          {shapes, {recv_pid, recv_ref}} when is_pid(recv_pid) ->
            :ok = EXLA.Client.from_outfeed(client, device_id, shapes, recv_pid, recv_ref)
            loop(client, device_id, ref, shape, hooks)

          {shapes, {fun, template}} when is_function(fun, 1) ->
            length = length(shapes)
            feed_ref = make_ref()

            {hook_pid, hook_ref} =
              spawn_monitor(fn -> apply_hook(feed_ref, length, fun, template) end)

            :ok = EXLA.Client.from_outfeed(client, device_id, shapes, hook_pid, feed_ref)

            receive do
              {:DOWN, ^hook_ref, _, _, _} -> loop(client, device_id, ref, shape, hooks)
            end
        end
    end
  end

  defp apply_hook(ref, length, fun, template) do
    buffers =
      for _ <- 1..length//1 do
        receive do
          {^ref, binary} -> binary
        end
      end

    fun.(EXLA.Defn.Buffers.to_nx!(buffers, template))
  end
end
