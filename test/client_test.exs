defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client
  alias Exla.Op
  alias Exla.LocalExecutable
  alias Exla.Builder

  test "create_client/1 succeeds on host device with default args" do
    assert %Client{} = Client.create_client()
  end

  test "compile/4 succeeds on host device with constant computation and no args" do
    # TODO: Setup stuff for easier testing
    client = Client.create_client()
    builder = Builder.new("test")
    op = Op.constant(builder, 1)
    comp = Builder.build(op)
    assert %LocalExecutable{} = Client.compile(client, comp, {})
  end
end
