defmodule BuilderTest do
  use ExUnit.Case
  alias Exla.Computation
  alias Exla.Op
  alias Exla.Builder

  test "build/1 succeeds on constant" do
    {:ok, op = %Op{}} = Op.constant(1)
    assert {:ok, %Computation{}} = Builder.build(op)
  end
end
