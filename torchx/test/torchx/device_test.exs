defmodule Torchx.DeviceTest do
  use ExUnit.Case, async: true

  alias Torchx.Backend, as: TB

  if Torchx.device_available?(:cuda) do
    @device {:cuda, 0}
  else
    @device :cpu
  end

  describe "creation" do
    test "from_binary" do
      t = Nx.tensor([1, 2, 3], backend: {TB, device: @device})
      assert {@device, _} = t.data.ref
    end

    test "tensor" do
      t = Nx.tensor(7.77, backend: {TB, device: @device})
      assert {@device, _} = t.data.ref
    end
  end

  describe "deletion" do
    test "delete multiple times" do
      t = Nx.tensor([1, 2, 3], backend: {TB, device: @device})
      assert Nx.backend_deallocate(t) == :ok
      assert Nx.backend_deallocate(t) == :already_deallocated
      assert Nx.backend_deallocate(t) == :already_deallocated
    end
  end

  describe "backend_transfer" do
    test "to" do
      t = Nx.tensor([1, 2, 3], backend: Nx.BinaryBackend)
      td = Nx.backend_transfer(t, {TB, device: @device})
      assert {@device, _} = td.data.ref
    end

    test "from" do
      t = Nx.tensor([1, 2, 3], backend: {TB, device: @device})
      assert Nx.backend_transfer(t) == Nx.tensor([1, 2, 3], backend: Nx.BinaryBackend)

      # TODO: we need to raise once the data has been transferred once
      # assert_raise ArgumentError, fn -> Nx.backend_transfer(t) end
    end
  end

  describe "indices_to_flatten" do
    test "works" do
      t = Nx.tensor([[1, 2], [3, 4]], backend: {TB, device: @device})
      t2 = Nx.tensor([[2, 6], [3, 1]], backend: {TB, device: @device})
      assert Nx.window_scatter_max(t, t2, 0, {2, 3}, backend: {TB, device: @device}) == Nx.tensor([[0, 0, 0, 0, 6, 0], [0, 0, 2, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 1]], backend: {TB, device: @device})
    end
  end
end
