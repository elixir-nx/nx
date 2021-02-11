defmodule Nx.TypeTest do
  use ExUnit.Case, async: true

  alias Nx.Type
  doctest Type

  describe "requires_int/1" do
    test "raises for non-integer types" do
      assert_raise ArgumentError, "an integer type is required, but got {:bf, 16}", fn ->
        Type.requires_int({:bf, 16})
      end

      assert_raise ArgumentError, "an integer type is required, but got {:f, 32}", fn ->
        Type.requires_int({:f, 32})
      end

      assert_raise ArgumentError, "an integer type is required, but got {:f, 64}", fn ->
        Type.requires_int({:f, 64})
      end
    end

    test "does not raise for integer types" do
      assert {:u, 8} = Type.requires_int({:u, 8})
      assert {:u, 16} = Type.requires_int({:u, 16})
      assert {:u, 32} = Type.requires_int({:u, 32})
      assert {:u, 64} = Type.requires_int({:u, 64})
      assert {:u, 8} = Type.requires_int({:u, 8})
      assert {:u, 16} = Type.requires_int({:u, 16})
      assert {:u, 32} = Type.requires_int({:u, 32})
      assert {:u, 64} = Type.requires_int({:u, 64})
    end
  end
end
