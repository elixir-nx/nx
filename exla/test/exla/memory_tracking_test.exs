defmodule EXLA.MemoryTrackingTest do
  use EXLA.Case, async: false

  alias EXLA.{Client, DeviceBuffer, Typespec}

  @client :memory_tracking_host

  test "tracks memory allocation and deallocation" do
    client = Client.fetch!(@client)
    other_host_client = Client.fetch!(:other_host)
    Client.reset_peak_memory(client)
    Client.reset_peak_memory(other_host_client)

    baseline_stats_other_host = Client.get_memory_statistics(other_host_client)

    # Get baseline state
    baseline_stats = Client.get_memory_statistics(client)
    baseline_allocated = baseline_stats.allocated
    baseline_peak = baseline_stats.peak

    # Allocate a buffer
    data = <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>
    buffer = place(data, client)

    # Check memory increased
    stats = Client.get_memory_statistics(client)
    # 4 * 4 bytes
    assert stats.allocated == baseline_allocated + byte_size(data)
    assert stats.peak == baseline_peak + byte_size(data)

    # Allocate another buffer
    data2 = <<5::32-native, 6::32-native, 7::32-native, 8::32-native>>
    place(data2, client)

    # Check memory increased
    stats = Client.get_memory_statistics(client)
    # 8 * 4 bytes
    assert stats.allocated == baseline_allocated + byte_size(data) + byte_size(data2)
    assert stats.peak == baseline_peak + byte_size(data) + byte_size(data2)

    # Deallocate first buffer
    DeviceBuffer.deallocate(buffer)

    # Check memory decreased
    stats = Client.get_memory_statistics(client)
    # 4 * 4 bytes
    assert stats.allocated == baseline_allocated + byte_size(data2)
    assert stats.peak == baseline_peak + byte_size(data) + byte_size(data2)

    assert baseline_stats_other_host == Client.get_memory_statistics(other_host_client)
  end

  test "tracks per-device memory" do
    client = Client.fetch!(@client)

    # Get baseline
    baseline_stats = Client.get_memory_statistics(client)
    baseline_device_memory = Map.get(baseline_stats.per_device, 0, 0)

    # Allocate on device 0
    data = <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>
    place(data, client)

    stats = Client.get_memory_statistics(client)
    assert Map.get(stats.per_device, 0, 0) == baseline_device_memory + 16
  end

  test "reset peak memory" do
    client = Client.fetch!(@client)

    # Get baseline
    baseline_stats = Client.get_memory_statistics(client)
    baseline_allocated = baseline_stats.allocated

    # Allocate and track peak
    data = <<1::32-native, 2::32-native, 3::32-native, 4::32-native>>
    buffer = place(data, client)

    stats = Client.get_memory_statistics(client)
    peak_after_alloc = stats.peak
    assert peak_after_alloc >= baseline_allocated + 16

    DeviceBuffer.deallocate(buffer)

    # Reset peak memory
    Client.reset_peak_memory(client)

    stats = Client.get_memory_statistics(client)
    # After reset, peak should equal current allocated
    assert stats.peak == stats.allocated
    assert stats.peak == baseline_allocated
  end

  test "tracks memory deallocation when buffer is garbage collected" do
    client = Client.fetch!(@client)
    :erlang.garbage_collect()
    Client.reset_peak_memory(client)

    assert %{allocated: baseline_allocated} = Client.get_memory_statistics(client)

    t = free_on_exit(Nx.iota({1000}, type: :u8, backend: {EXLA.Backend, client: @client}))
    f = fn t, f -> f.(t, f) end

    test_pid = self()
    ref = make_ref()

    task =
      Task.async(fn ->
        t2 = Nx.iota({10000}, type: :u8, backend: {EXLA.Backend, client: @client})
        send(test_pid, {ref, :allocated})
        f.(t2, f)
      end)

    Process.unlink(task.pid)

    assert_receive {^ref, :allocated}

    stats = Client.get_memory_statistics(client)
    assert stats.allocated == 11000 + baseline_allocated
    assert stats.peak == 11000 + baseline_allocated
    assert Map.get(stats.per_device, 0) == 11000 + baseline_allocated

    Process.exit(task.pid, :stop)

    task_ref = task.ref
    assert_receive {:DOWN, ^task_ref, :process, _, :stop}

    :erlang.garbage_collect()

    stats = Client.get_memory_statistics(client)

    assert stats.allocated == 1000 + baseline_allocated
    assert stats.peak == 11000 + baseline_allocated
    assert Map.get(stats.per_device, 0) == 1000 + baseline_allocated

    assert :ok == Nx.backend_deallocate(t)

    stats = Client.get_memory_statistics(client)
    assert stats.allocated == baseline_allocated
    assert stats.peak == 11000 + baseline_allocated
    assert Map.get(stats.per_device, 0) == baseline_allocated
  end

  test "untracks donated buffers after execution" do
    client = Client.fetch!(@client)
    Nx.default_backend({EXLA.Backend, client: @client})
    :erlang.garbage_collect()

    assert %{allocated: baseline} = Client.get_memory_statistics(client)

    t1 = Nx.iota({200_000}) |> free_on_exit() |> Nx.donatable()
    t2 = free_on_exit(Nx.iota({100_000}))

    assert %{allocated: before} = Client.get_memory_statistics(client)
    assert before == baseline + Nx.byte_size(t1) + Nx.byte_size(t2)

    res =
      free_on_exit(
        EXLA.jit_apply(fn x, y -> Nx.put_slice(x, [0], y) end, [t1, t2], client: @client)
      )

    assert %{allocated: allocated} = Client.get_memory_statistics(client)
    # t1 was donated; only the result and live t2 should remain counted.
    assert allocated == baseline + Nx.byte_size(res) + Nx.byte_size(t2)
    assert :already_deallocated = Nx.backend_deallocate(t1)
    assert %{allocated: ^allocated} = Client.get_memory_statistics(client)
  end

  defp place(data, client) do
    data
    |> DeviceBuffer.place_on_device(Typespec.tensor({:s, 32}, {4}), client, 0)
    |> free_on_exit()
  end

  defp free_on_exit(%DeviceBuffer{} = buffer) do
    on_exit(fn -> DeviceBuffer.deallocate(buffer) end)
    buffer
  end

  defp free_on_exit(%Nx.Tensor{} = tensor) do
    on_exit(fn -> Nx.backend_deallocate(tensor) end)
    tensor
  end
end
