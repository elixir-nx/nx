defmodule EXLATest do
  use ExUnit.Case, async: true
  doctest EXLA
end

defmodule EXLA.GlobalTest do
  use ExUnit.Case

  test "set_as_nx_default" do
    Nx.Defn.global_default_options([])
    assert Nx.default_backend() == {Nx.BinaryBackend, []}
    assert Nx.Defn.default_options() == []

    assert EXLA.set_as_nx_default() == :ok
    assert Nx.default_backend() == {EXLA.Backend, []}
  after
    assert Nx.default_backend(Nx.BinaryBackend)
  end
end
