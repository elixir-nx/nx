defmodule EXLA.PluginTest do
  use ExUnit.Case

  describe "register/1" do
    test "registers a plugin" do
      assert :ok = EXLA.Plugin.register(:custom_plugin, "test/support/c/libcustom_plugin.so")
    end
  end
end
