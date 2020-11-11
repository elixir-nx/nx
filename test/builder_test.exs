defmodule BuilderTest do
  use ExUnit.Case
  alias Exla.Computation
  alias Exla.Op
  alias Exla.Builder

  test "new/1 succeeds in creating a new builder" do
    assert %Builder{} = Builder.new("builder")
  end

  test "build/1 succeeds on constant" do
    # TODO: Add this to setup
    builder = Builder.new("test")
    op = Op.constant(builder, 1)
    assert %Computation{} = Builder.build(op)
  end
end
