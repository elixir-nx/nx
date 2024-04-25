defmodule EXLA.DeviceMemorySharingTest do
  use EXLA.Case, async: false

  for client_name <- [:host, :cuda] do
    if client_name == :cuda do
      @tag :cuda_required
    end

    test "buffer sharing on #{inspect(client_name)} works as expected" do
      t1 = Nx.tensor([1, 2, 3], backend: {EXLA.Backend, client: unquote(client_name)})

      assert inspect(t1) =~ "1, 2, 3"

      assert {:ok, pointer} = Nx.to_pointer(t1, mode: :local)

      assert {:ok, t2} =
               Nx.from_pointer(
                 {EXLA.Backend, client_name: unquote(client_name)},
                 pointer,
                 t1.type,
                 t1.shape
               )

      assert t1.data.buffer.ref != t2.data.buffer.ref
      assert Nx.to_binary(t1) == Nx.to_binary(t2)
    end
  end
end
