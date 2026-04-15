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

  @tag :distributed
  test "host IPC sharing works across peer node processes" do
    [peer | _] = EXLAHelpers.test_peer_nodes()

    {pointer, type, shape, expected_binary} =
      :erpc.call(peer, EXLAHelpers, :export_host_ipc_pointer, [[1, 2, 3, 4]])

    if File.dir?("/dev/shm") do
      shm_path = Path.join("/dev/shm", pointer.handle)
      on_exit(fn -> File.rm(shm_path) end)
    end

    tensor = Nx.from_pointer({EXLA.Backend, client: :host}, pointer, type, shape)

    assert pointer.kind == :ipc
    assert Nx.to_binary(tensor) == expected_binary
    assert Nx.to_flat_list(tensor) == [1, 2, 3, 4]
  end

  @tag :distributed
  test "0o400 shm is importable via the O_RDONLY fallback in open_existing_ipc_handle" do
    # Pins the EACCES fallback path in `open_existing_ipc_handle`: with the
    # default 0o400 mode, `shm_open(name, O_RDWR)` fails with EACCES even for
    # the owning user, so the importer has to retry with `O_RDONLY`. If that
    # retry is removed, `Nx.from_pointer/4` raises "unable to get fd for IPC
    # handle" here before any assertion runs.

    [peer | _] = EXLAHelpers.test_peer_nodes()

    {pointer, type, shape, expected_binary} =
      :erpc.call(peer, EXLAHelpers, :export_host_ipc_pointer, [[1, 2, 3, 4]])

    if File.dir?("/dev/shm") do
      shm_path = Path.join("/dev/shm", pointer.handle)
      on_exit(fn -> File.rm(shm_path) end)

      # Sanity-check the mode bits on disk before we exercise the fallback.
      {:ok, stat} = File.stat(shm_path)
      assert Bitwise.band(stat.mode, 0o777) == 0o400
    end

    tensor = Nx.from_pointer({EXLA.Backend, client: :host}, pointer, type, shape)
    assert Nx.to_binary(tensor) == expected_binary
    assert Nx.to_flat_list(tensor) == [1, 2, 3, 4]
  end

  @tag :distributed
  test "writable permissions (0o600) allow zero-copy mutation visible on both nodes", %{} do
    [peer | _] = EXLAHelpers.test_peer_nodes()

    # Secondary creates tensor and exports a writable IPC segment (MFA — test
    # module is not loaded on peer nodes, so lambdas cannot be used).
    {pointer, type, shape} =
      :erpc.call(peer, EXLAHelpers, :export_writable_ipc_pointer, [[1, 2, 3]])

    if File.dir?("/dev/shm") do
      on_exit(fn -> File.rm(Path.join("/dev/shm", pointer.handle)) end)
    end

    # Secondary also imports from the same shm and holds the tensor alive in
    # a named Agent so the MAP_SHARED mapping survives across erpc calls.
    :erpc.call(peer, EXLAHelpers, :hold_ipc_pointer, [pointer, type, shape])
    on_exit(fn -> :erpc.call(peer, EXLAHelpers, :stop_held_ipc, []) end)

    # Local imports — also a writable MAP_SHARED view of the same shm.
    t_local = Nx.from_pointer({EXLA.Backend, client: :host}, pointer, type, shape)

    peer_binary = fn -> :erpc.call(peer, EXLAHelpers, :get_held_ipc_binary, []) end

    # Both nodes read [1, 2, 3] before the write.
    assert Nx.to_flat_list(t_local) == [1, 2, 3]
    assert peer_binary.() == Nx.to_binary(t_local)

    # Overwrite element at index 1 via the test NIF — mutates the shm in-place.
    %Nx.Pointer{address: addr} = Nx.to_pointer(t_local, mode: :local)
    {_, bits} = type
    EXLA.NIF.write_to_pointer(addr, <<99::32-native>>, 1 * div(bits, 8))

    # Nx.to_binary always re-reads from the EXLA buffer, so both sides
    # see the mutation through their respective MAP_SHARED mappings.
    assert Nx.to_flat_list(t_local) == [1, 99, 3]
    assert peer_binary.() == Nx.to_binary(t_local)
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
