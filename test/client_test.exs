defmodule ClientTest do
  use ExUnit.Case
  alias Exla.Client
  alias Exla.Op
  alias Exla.LocalExecutable
  alias Exla.Computation
  alias Exla.Builder

  test "create_client/1 succeeds on host device with default args" do
    assert {:ok, %Client{}} = Client.create_client()
  end

  test "compile/4 succeeds on host device with constant computation and no args" do
    # TODO: Setup stuff for easier testing
    {:ok, client = %Client{}} = Client.create_client()
    {:ok, builder = %Builder{}} = Builder.new("test")
    {:ok, op = %Op{}} = Op.constant(builder, 1)
    {:ok, comp = %Computation{}} = Builder.build(op)
    assert {:ok, %LocalExecutable{}} = Client.compile(client, comp, {})
  end
end
