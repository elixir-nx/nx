defmodule EXLA.DeviceMemorySharingTest do
  use EXLA.Case, async: false

  @moduletag :cuda_required

  test "buffer sharing works as expected" do
    t1 = Nx.tensor([1, 2, 3], backend: {EXLA.Backend, client: :cuda})

    assert inspect(t1) =~ "1, 2, 3"

    assert {:ok, pointer} = Nx.to_pointer(t1, mode: :local)

    assert {:ok, t2} =
             Nx.from_pointer(EXLA.Backend, pointer, t1.type, t1.shape,
               backend_opts: [client_name: :cuda]
             )

    assert t1.data.buffer.ref != t2.data.buffer.ref
    assert Nx.to_binary(t1) == Nx.to_binary(t2)
  end
end
