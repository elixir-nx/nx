defmodule Exla.Op do
  alias __MODULE__, as: Op
  alias Exla.Shape

  @enforce_keys [:ref]
  defstruct [:ref]

  # The XLA API is explicit about the rank of the constant being created e.g. ConstantR0, ConstantR1
  # We can be just as explicit, or we can use pattern matching on the inputs, I lean pattern matching
  # as I think it makes the API feel more flexible
  def constant(value) when is_number(value) do
    case Exla.NIF.constant_r0(value) do
      {:ok, ref} -> {:ok, %Op{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end

  def parameter(i, shape = %Shape{}, name) when is_integer(i) and i >= 0 and is_binary(name) do
    case Exla.NIF.parameter(i, shape.ref, name) do
      {:ok, ref} -> {:ok, %Op{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end

  def add(lhs = %Op{}, rhs = %Op{}, broadcast_dims \\ {}) do
    case Exla.NIF.add(lhs.ref, rhs.ref, broadcast_dims) do
      {:ok, ref} -> {:ok, %Op{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end
end
