defmodule EXLA.BuilderTest do
  use ExUnit.Case, async: true

  alias EXLA.{Builder, Computation, Op}

  @moduletag skip: :mlir
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

  test "build/1 returns a computation" do
    builder = Builder.new("test")
    op = Op.constant_r0(builder, 1, {:s, 32})
    assert %Computation{} = Builder.build(op)
  end
end
