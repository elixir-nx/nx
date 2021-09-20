defmodule TorchxTest do
  use ExUnit.Case, async: true

  describe "creation" do
    test "arange" do
      {:cpu, ref} = tensor = Torchx.arange(0, 26, 2, :short, :cpu)

      assert is_reference(ref)
      assert Torchx.type_of(tensor) == :short
      assert Torchx.shape_of(tensor) == {13}
    end
  end

  describe "operations" do
    test "dot" do
      a = Torchx.arange(0, 3, 1, :float, :cpu)
      b = Torchx.arange(4, 7, 1, :float, :cpu)

      {:cpu, ref} = Torchx.tensordot(a, b, [0], [0])
      assert is_reference(ref)
    end
  end
end
