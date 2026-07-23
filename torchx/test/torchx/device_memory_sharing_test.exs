defmodule Torchx.DeviceMemorySharingTest do
  use ExUnit.Case, async: false

  alias Torchx.Backend, as: TB

  @devices [:cpu] ++
             if(Torchx.device_available?(:cuda), do: [{:cuda, 0}], else: []) ++
             if(Torchx.device_available?(:mps), do: [:mps], else: [])

  for device <- @devices do
    test "local buffer sharing on #{inspect(device)} works as expected" do
      device = unquote(Macro.escape(device))
      t1 = Nx.tensor([1, 2, 3], type: :s32, backend: {TB, device: device})

      assert inspect(t1) =~ "1, 2, 3"

      assert pointer = Nx.to_pointer(t1, mode: :local)

      assert t2 =
               Nx.from_pointer(
                 {TB, device: device},
                 pointer,
                 t1.type,
                 t1.shape
               )

      assert t1.data.ref != t2.data.ref
      assert Nx.to_binary(t1) == Nx.to_binary(t2)
    end
  end

  unless match?({:win32, _}, :os.type()) do
    test "host IPC sharing on cpu works as expected" do
      t1 = Nx.tensor([1, 2, 3, 4], type: :s32, backend: {TB, device: :cpu})

      assert %Nx.Pointer{kind: :ipc, handle: handle, data_size: data_size} =
               pointer = Nx.to_pointer(t1, mode: :ipc)

      assert is_binary(handle)
      assert data_size == 16

      if File.dir?("/dev/shm") do
        shm_path = Path.join("/dev/shm", handle)
        on_exit(fn -> File.rm(shm_path) end)
        assert File.exists?(shm_path)
      end

      t2 = Nx.from_pointer({TB, device: :cpu}, pointer, t1.type, t1.shape)

      assert t1.data.ref != t2.data.ref
      assert Nx.to_binary(t1) == Nx.to_binary(t2)
      assert Nx.to_flat_list(t2) == [1, 2, 3, 4]
      # Keep t1 alive through the assertions: host IPC unlinks when the
      # exporter tensor is GC'd.
      assert Nx.to_flat_list(t1) == [1, 2, 3, 4]
    end

    test "host IPC :permissions defaults to 0o400" do
      t = Nx.tensor([1, 2, 3], type: :s32, backend: {TB, device: :cpu})

      if File.dir?("/dev/shm") do
        %Nx.Pointer{handle: handle} = Nx.to_pointer(t, mode: :ipc)
        shm_path = Path.join("/dev/shm", handle)
        on_exit(fn -> File.rm(shm_path) end)

        assert File.exists?(shm_path)
        %File.Stat{mode: mode} = File.stat!(shm_path)
        assert Bitwise.band(mode, 0o7777) == 0o400
        assert Nx.to_flat_list(t) == [1, 2, 3]
      end
    end

    test "host IPC accepts an explicit :permissions value" do
      t = Nx.tensor([1, 2, 3], type: :s32, backend: {TB, device: :cpu})

      if File.dir?("/dev/shm") do
        %Nx.Pointer{handle: handle} = Nx.to_pointer(t, mode: :ipc, permissions: 0o600)
        shm_path = Path.join("/dev/shm", handle)
        on_exit(fn -> File.rm(shm_path) end)

        assert File.exists?(shm_path)
        %File.Stat{mode: mode} = File.stat!(shm_path)
        assert Bitwise.band(mode, 0o7777) == 0o600
        assert Nx.to_flat_list(t) == [1, 2, 3]
      end
    end
  end

  if Torchx.device_available?(:cuda) do
    test "cuda IPC sharing works as expected" do
      t1 = Nx.tensor([1, 2, 3], type: :s32, backend: {TB, device: {:cuda, 0}})

      assert %Nx.Pointer{kind: :ipc, handle: handle, address: 0, data_size: data_size} =
               pointer = Nx.to_pointer(t1, mode: :ipc)

      assert is_binary(handle)
      assert data_size == 12

      t2 = Nx.from_pointer({TB, device: {:cuda, 0}}, pointer, t1.type, t1.shape)

      assert t1.data.ref != t2.data.ref
      assert Nx.to_binary(t1) == Nx.to_binary(t2)
      assert Nx.to_flat_list(t1) == [1, 2, 3]
    end

    test "invalid cuda ipc handles don't crash the runtime" do
      assert_raise RuntimeError, "unable to get pointer for IPC handle", fn ->
        Nx.from_pointer(
          {TB, device: {:cuda, 0}},
          %Nx.Pointer{handle: "#{System.unique_integer()}", kind: :ipc, data_size: 4},
          {:f, 32},
          {1}
        )
      end
    end
  end

  if Torchx.device_available?(:mps) do
    test "rejects :ipc on mps as unsupported for the device" do
      t = Nx.tensor([1, 2, 3], backend: {TB, device: :mps})

      assert_raise ArgumentError, ~r/not supported for the mps device yet/, fn ->
        Nx.to_pointer(t, mode: :ipc)
      end

      assert_raise ArgumentError, ~r/not supported for the mps device yet/, fn ->
        Nx.from_pointer(
          {TB, device: :mps},
          %Nx.Pointer{kind: :ipc, handle: "unused", data_size: 12},
          {:s, 32},
          {3}
        )
      end
    end
  end

  test "rejects invalid :permissions" do
    t = Nx.tensor([1, 2, 3], backend: {TB, device: :cpu})

    assert_raise ArgumentError, ~r/:permissions must be an integer/, fn ->
      Nx.to_pointer(t, mode: :local, permissions: :not_an_int)
    end

    assert_raise ArgumentError, ~r/:permissions must be an integer/, fn ->
      Nx.to_pointer(t, mode: :local, permissions: -1)
    end

    assert_raise ArgumentError, ~r/:permissions must be an integer/, fn ->
      Nx.to_pointer(t, mode: :local, permissions: 0o10000)
    end
  end

  test "rejects invalid pointer data_size" do
    t = Nx.tensor([1, 2, 3], type: :s32, backend: {TB, device: :cpu})
    pointer = Nx.to_pointer(t, mode: :local)

    assert_raise ArgumentError, ~r/invalid pointer data_size/, fn ->
      Nx.from_pointer({TB, device: :cpu}, %{pointer | data_size: 1}, t.type, t.shape)
    end
  end

  test "non-contiguous tensors are made contiguous before exposing the pointer" do
    t =
      Nx.iota({2, 3}, type: :s32, backend: {TB, device: :cpu})
      |> Nx.transpose()

    assert %Nx.Pointer{kind: :local, address: address} =
             pointer = Nx.to_pointer(t, mode: :local)

    assert address > 0

    t2 = Nx.from_pointer({TB, device: :cpu}, pointer, t.type, t.shape)

    # Evaluate into bound vars so both tensors stay reachable across the
    # comparison (from_blob views do not keep the exporter storage alive).
    exported = Nx.to_flat_list(t)
    imported = Nx.to_flat_list(t2)

    assert exported == [0, 3, 1, 4, 2, 5]
    assert imported == exported
  end
end
