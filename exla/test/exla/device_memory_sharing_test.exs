defmodule EXLA.DeviceMemorySharingTest do
  use EXLA.Case, async: false

  for client_name <- [:host, :cuda] do
    if client_name == :cuda do
      @tag :cuda_required
    end

    test "buffer sharing on #{inspect(client_name)} works as expected" do
      t1 = Nx.tensor([1, 2, 3], backend: {EXLA.Backend, client: unquote(client_name)})

      assert inspect(t1) =~ "1, 2, 3"

      assert pointer = Nx.to_pointer(t1, mode: :local)

      assert t2 =
               Nx.from_pointer(
                 {EXLA.Backend, client: unquote(client_name)},
                 pointer,
                 t1.type,
                 t1.shape
               )

      assert t1.data.buffer.ref != t2.data.buffer.ref
      assert Nx.to_binary(t1) == Nx.to_binary(t2)
    end
  end

  @tag :cuda_required
  test "invalid ipc handles don't crash the runtime" do
    assert_raise RuntimeError, "unable to get pointer for IPC handle", fn ->
      Nx.from_pointer(
        {EXLA.Backend, client: :cuda},
        %Nx.Pointer{handle: "#{System.unique_integer()}", kind: :ipc, data_size: 4},
        {:f, 32},
        {1}
      )
    end
  end

  describe ":shm_permissions option" do
    setup do
      t = Nx.tensor([1, 2, 3], backend: {EXLA.Backend, client: :host})
      {:ok, tensor: t}
    end

    test "defaults to 0o400 (owner-read-only, functional-by-default)", %{tensor: t} do
      if File.dir?("/dev/shm") do
        %Nx.Pointer{handle: handle} = Nx.to_pointer(t, mode: :ipc)
        shm_path = Path.join("/dev/shm", handle)
        on_exit(fn -> File.rm(shm_path) end)

        assert File.exists?(shm_path)
        %File.Stat{mode: mode} = File.stat!(shm_path)
        # Mask off the file-type bits (S_IFREG etc.); we care about the
        # lower 12 permission bits.
        assert Bitwise.band(mode, 0o7777) == 0o400
      end
    end

    test "accepts an explicit permission value", %{tensor: t} do
      if File.dir?("/dev/shm") do
        %Nx.Pointer{handle: handle} = Nx.to_pointer(t, mode: :ipc, shm_permissions: 0o600)
        shm_path = Path.join("/dev/shm", handle)
        on_exit(fn -> File.rm(shm_path) end)

        assert File.exists?(shm_path)
        %File.Stat{mode: mode} = File.stat!(shm_path)
        assert Bitwise.band(mode, 0o7777) == 0o600
      end
    end

    test "rejects non-integer shm_permissions", %{tensor: t} do
      assert_raise ArgumentError, ~r/:shm_permissions must be an integer/, fn ->
        Nx.to_pointer(t, mode: :ipc, shm_permissions: :not_an_int)
      end
    end

    test "rejects negative shm_permissions", %{tensor: t} do
      assert_raise ArgumentError, ~r/:shm_permissions must be an integer/, fn ->
        Nx.to_pointer(t, mode: :ipc, shm_permissions: -1)
      end
    end

    test "rejects shm_permissions above 0o7777", %{tensor: t} do
      assert_raise ArgumentError, ~r/:shm_permissions must be an integer/, fn ->
        Nx.to_pointer(t, mode: :ipc, shm_permissions: 0o10000)
      end
    end
  end
end
