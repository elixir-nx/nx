defmodule EXLA.LoadDylibTest do
  use ExUnit.Case, async: false

  test "loads a library and is safe to call again for the same one" do
    # EXLA's own NIF: already loaded, so opening it again only bumps its
    # reference count.
    path = Application.app_dir(:exla, "priv/libexla.so")

    assert EXLA.load_dylib(path) == :ok
    assert EXLA.load_dylib(path) == :ok
  end

  test "raises with the dynamic loader message when the library cannot be opened" do
    path = Path.join(System.tmp_dir!(), "exla_does_not_exist.so")

    assert_raise ArgumentError, ~r/exla_does_not_exist\.so/, fn ->
      EXLA.load_dylib(path)
    end
  end
end
