defmodule EXLA.PluginTest do
  use ExUnit.Case

  describe "register/1" do
    test "raises if file does not exist" do
      assert_raise ArgumentError, ~r/does not exist/, fn ->
        EXLA.Plugin.register("test/support/c/doesnotexist.so")
      end
    end

    test "does not crash on invalid files" do
      assert_raise RuntimeError, ~r/Unable to open/, fn ->
        EXLA.Plugin.register(__ENV__.file)
      end
    end

    test "registers a plugin" do
      assert :ok = EXLA.Plugin.register("test/support/c/libcustom_plugin.so")
    end
  end
end