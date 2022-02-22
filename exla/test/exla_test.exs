defmodule EXLATest do
  use ExUnit.Case, async: true
  doctest EXLA
end

defmodule EXLA.GlobalTest do
  use ExUnit.Case

  @default Nx.Defn.global_default_options([])

  test "set_as_nx_default" do
    assert @default[:compiler] == EXLA

    Nx.Defn.global_default_options([])
    assert Nx.Defn.default_options() == []

    assert EXLA.set_as_nx_default([:unknown]) == nil
    assert Nx.Defn.default_options() == []

    assert EXLA.set_as_nx_default([:host]) == :host
    assert Nx.Defn.default_options() == [compiler: EXLA, client: :host]
  after
    assert Nx.Defn.global_default_options(@default)
  end
end
