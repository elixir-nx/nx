defmodule Exla.ClientTest do
  use ExUnit.Case, async: true

  alias Exla.Op
  alias Exla.Executable
  alias Exla.Shape

  import ExlaHelpers

  test "compile/4 succeeds with constant computation and no args" do
    assert %Executable{} = compile([], fn builder -> Op.constant_r0(builder, 1, {:s, 32}) end)
  end

  test "compile/4 succeeds with basic computation and args" do
    shape = Shape.make_shape({:s, 32}, {})
    assert %Executable{} = compile([shape], fn _builder, x -> Op.add(x, x) end)
  end
end
