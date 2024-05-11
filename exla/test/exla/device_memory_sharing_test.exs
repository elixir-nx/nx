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

  @tag :cuda_required
  test "ipc handles don't crash the runtime when :local mode is selected" do
    assert {:error, ~c"Invalid pointer size for selected mode."} ==
             Nx.from_pointer(
               {EXLA.Backend, client_name: :cuda},
               Enum.to_list(0..63),
               {:f, 32},
               {1},
               mode: :local
             )
  end
end
