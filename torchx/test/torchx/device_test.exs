defmodule Torchx.DeviceTest do
  use ExUnit.Case, async: true
  import Nx.Testing

  alias Torchx.Backend, as: TB

  cond do
    Torchx.device_available?(:cuda) ->
      @device {:cuda, 0}

    Torchx.device_available?(:mps) ->
      @device :mps

    true ->
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
      t = Nx.tensor([[1, 0, 2], [3, 0, 4], [5, 0, 6]], backend: {TB, device: @device})
      source = Nx.tensor([[2, 6, 3]], backend: {TB, device: @device})

      expected =
        Nx.tensor([[0, 6, 0], [0, 0, 0], [2, 0, 3]], backend: {TB, device: @device})

      assert_equal(
        Nx.window_scatter_max(t, source, 0, {3, 1}, strides: [1, 1]),
        expected
      )
    end
  end
end
