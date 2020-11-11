defmodule LocalExecutableTest do
  use ExUnit.Case
  alias Exla.Builder
  alias Exla.Client
  alias Exla.Computation
  alias Exla.LocalExecutable
  alias Exla.Op

  test "run/3 succeeds with no inputs and default options" do
    # TODO: This should be in setup
    # TODO: Not sure if this is the most efficient way to test all of this
    {:ok, client = %Client{}} = Client.create_client()
    {:ok, builder = %Builder{}} = Builder.new("test")
    {:ok, op = %Op{}} = Op.constant(builder, 1)
    {:ok, comp = %Computation{}} = Builder.build(op)
    {:ok, exec = %LocalExecutable{}} = Client.compile(client, comp, {})
    assert :ok = LocalExecutable.run(exec, {})
  end
end
