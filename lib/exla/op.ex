defmodule Exla.Op do
  alias __MODULE__, as: Op
  alias Exla.Shape
  alias Exla.Builder

  @enforce_keys [:builder, :ref]
  defstruct [:builder, :ref]

  # The XLA API is explicit about the rank of the constant being created e.g. ConstantR0, ConstantR1
  # We can be just as explicit, or we can use pattern matching on the inputs, I lean pattern matching
  # as I think it makes the API feel more flexible
  def constant(builder = %Builder{}, value) when is_number(value) do
    case Exla.NIF.constant_r0(builder.ref, value) do
      {:ok, ref} -> {:ok, %Op{builder: builder.ref, ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end

  def parameter(builder = %Builder{}, i, shape = %Shape{}, name)
      when is_integer(i) and i >= 0 and is_binary(name) do
    case Exla.NIF.parameter(builder.ref, i, shape.ref, name) do
      {:ok, ref} -> {:ok, %Op{builder: builder.ref, ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end

  # TODO: builder is redundant here because it's contained within each op
  def add(
        builder = %Builder{ref: builder},
        lhs = %Op{builder: builder, ref: left},
        rhs = %Op{builder: builder, ref: right},
        broadcast_dims \\ {}
      ) do
    case Exla.NIF.add(builder, left, right, broadcast_dims) do
      {:ok, ref} -> {:ok, %Op{builder.ref}}
      {:error, msg} -> {:error, msg}
    end
  end
end
