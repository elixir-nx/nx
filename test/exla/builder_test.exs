defmodule BuilderTest do
  use ExUnit.Case
  alias Exla.Computation
  alias Exla.Op
  alias Exla.Builder

  test "new/1 succeeds in creating a new builder" do
    assert b = %Builder{} = Builder.new("builder")
    assert b.name == "builder"
    assert is_reference(b.ref)
    assert is_nil(b.parent)
  end

  test "new/2 succeeds in creating a new subbuilder" do
    parent = Builder.new("builder")
    assert b = Builder.new(parent, "subbuilder")
    assert b.name == "subbuilder"
    assert is_reference(b.ref)
    assert b.parent == parent
  end

  test "build/1 succeeds on constant" do
    # TODO: Add this to setup
    builder = Builder.new("test")
    op = Op.constant_r0(builder, 1, {:s, 32})
    assert %Computation{} = Builder.build(op)
  end
end
