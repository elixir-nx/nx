defmodule EXLA.MLIR.ExecutableFeedTest do
  # infeed/outfeed are global resources, so they either
  # need to be locked or we cannot run them concurrently.
  use ExUnit.Case, async: false

  alias EXLA.BinaryBuffer
  alias EXLA.DeviceBuffer
  alias EXLA.Client
  alias EXLA.Shape
  alias EXLA.MLIR.Value
  import EXLAHelpers

  @moduletag :mlir
  describe "infeed/outfeed" do
    test "successfully sends to/from device asynchronously" do
      t = BinaryBuffer.from_binary(<<1::32-native>>, Shape.make_shape({:s, 32}, {}))

      task =
        Task.async(fn ->
          :timer.tc(fn ->
            run_one([], [compiler_mode: :mlir], Nx.template({}, {:s, 32}), fn b ->
              token = Value.create_token(b)
              val_and_token = Value.infeed(token, t.shape)
              val = Value.get_tuple_element(val_and_token, 0)
              new_token = Value.get_tuple_element(val_and_token, 1)
              outfeed_val = Value.add(b, val, val)

              _outfeed_token = Value.outfeed([outfeed_val], new_token)
              Value.tuple(b, [Value.add(b, outfeed_val, val)])
            end)
          end)
        end)

      refute_received _
      # sleep here to ensure that we have a known measureable delay in the task's execution
      # that is correlated to how much time we wait for the infeed to have something in the queue
      Process.sleep(1_000)
      assert :ok = Client.to_infeed(client(), 0, [{t.data, t.shape}])
      assert from_outfeed(client(), 0, Shape.make_shape({:s, 32}, {})) == <<2::32-native>>
      assert {time, [a = %DeviceBuffer{}]} = Task.await(task)
      assert time >= 1_000
      assert DeviceBuffer.read(a) == <<3::32-native>>
    end
  end

  defp from_outfeed(client, device_id, shape) do
    ref = make_ref()
    Client.from_outfeed(client, device_id, [shape], self(), ref)

    receive do
      {^ref, msg} -> msg
    end
  end
end
