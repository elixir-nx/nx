defmodule EXLA.DeviceMemorySharingTest do
  use EXLA.Case, async: false

  @moduletag :cuda_required

  test "buffer sharing works as expected" do
    client = EXLA.Client.fetch!(:cuda)
    t1 = Nx.tensor([1, 2, 3], backend: {EXLA.Backend, client: :cuda})
    buffer = t1.data.buffer

    assert {:ok, {pointer, 24}} =
             EXLA.NIF.get_buffer_device_pointer(client.ref, buffer.ref, :local)

    assert {:ok, new_buffer_ref} =
             EXLA.NIF.create_buffer_from_device_pointer(
               client.ref,
               pointer,
               :local,
               buffer.shape.ref,
               buffer.device_id
             )

    t2 = put_in(t1.data.buffer.ref, new_buffer_ref)

    assert {:ok, {pointer, 24}} ==
             EXLA.NIF.get_buffer_device_pointer(client.ref, new_buffer_ref, :local)

    assert Nx.to_binary(t1) == Nx.to_binary(t2)
  end
end
