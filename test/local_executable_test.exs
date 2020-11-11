defmodule LocalExecutableTest do
  use ExUnit.Case
  alias Exla.Builder
  alias Exla.Client
  alias Exla.LocalExecutable
  alias Exla.Op

  test "run/3 succeeds with no inputs and default options" do
    # TODO: This should be in setup
    # TODO: Not sure if this is the most efficient way to test all of this
    client = Client.create_client()
    builder = Builder.new("test")
    op = Op.constant(builder, 1)
    comp = Builder.build(op)
    exec = Client.compile(client, comp, {})
    assert :ok = LocalExecutable.run(exec, {})
  end
end
