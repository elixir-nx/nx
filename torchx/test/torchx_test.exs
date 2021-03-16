defmodule TorchxTest do
  use ExUnit.Case, async: true

  describe "creation" do
    test "arange" do
      {device, ref} = Torchx.arange(0, 26, 2, type: {:s, 16})

      assert Torchx.device_of(ref) == device
      assert Torchx.type_of(ref) == {:s, 16}
      assert Torchx.shape_of(ref) == {13}
    end
  end
end
