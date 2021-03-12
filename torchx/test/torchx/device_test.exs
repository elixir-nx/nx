defmodule Torchx.DeviceTest do
  use ExUnit.Case, async: true

  alias Torchx.Backend, as: TB

  setup do
    Nx.default_backend(Torchx.Backend)
    :ok
  end

  describe "creation" do
    test "from_binary" do
      t = Nx.tensor([1, 2, 3], backend_options: [device: :cpu])
      Nx.backend_transfer(t, TB, backend_options: [device: :cpu])
    end

    test "scalar" do
      t = Nx.tensor(7.77, backend_options: [device: :cpu])
    end
  end
end
