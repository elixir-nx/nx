defmodule Nx.PytorchTest do
  use ExUnit.Case, async: true

  alias Nx.PytorchBackend

  describe "tensor" do
    test "transfers new tensor" do
      Nx.tensor([1, 2, 3], backend: PytorchBackend, backend_options: [key: :example])
      assert Process.get(:example) == <<1::64-native, 2::64-native, 3::64-native>>
    end
  end
end
