defmodule Exla.Op do
  alias __MODULE__, as: Op
  alias Exla.Shape
  alias Exla.Builder

  @enforce_keys [:builder, :ref]
  defstruct [:builder, :ref]

  # The XLA API is explicit about the rank of the constant being created e.g. ConstantR0, ConstantR1
  # We can be just as explicit, or we can use pattern matching on the inputs, I lean pattern matching
  # as I think it makes the API feel more flexible
  def constant(%Builder{ref: builder}, value) when is_number(value) do
    {:ok, ref} = Exla.NIF.constant_r0(builder, value)
    %Op{builder: builder, ref: ref}
  end

  def constant(%Builder{ref: builder}, value, length) when is_number(value) and is_integer(length) and length >= 0 do
    {:ok, ref} =  Exla.NIF.constant_r1(builder, value, length)
    %Op{builder: builder, ref: ref}
  end

  def dot(%Op{builder: builder, ref: left}, %Op{builder: builder, ref: right}) do
    {:ok, ref} = Exla.NIF.dot(left, right)
    %Op{builder: builder, ref: ref}
  end

  def parameter(%Builder{ref: builder}, i, %Shape{ref: shape}, name)
      when is_integer(i) and i >= 0 and is_binary(name) do
    {:ok, ref} = Exla.NIF.parameter(builder, i, shape, name)
    %Op{builder: builder, ref: ref}
  end

  # TODO: builder is redundant here because it's contained within each op
  def add(
        %Op{builder: builder, ref: left},
        %Op{builder: builder, ref: right},
        broadcast_dims \\ {}
      ) do
    {:ok, ref} = Exla.NIF.add(left, right, broadcast_dims)
    %Op{builder: builder, ref: ref}
  end
end
