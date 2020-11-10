defmodule Exla.Op do
  alias __MODULE__, as: Op
  alias Exla.Shape

  @enforce_keys [:ref]
  defstruct [:ref]

  def parameter(i, shape = %Shape{}, name) when is_integer(i) and i >= 0 and is_binary(name) do
    case Exla.NIF.parameter(i, shape.ref, name) do
      {:ok, ref} -> {:ok, %Op{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end

  def parameter(_i, _shape, _name),
    do: {:error, "Invalid arguments passed to `#{__MODULE__}.parameter/3.`"}

  def add(lhs = %Op{}, rhs = %Op{}, broadcast_dims \\ {}) do
    case Exla.NIF.add(lhs.ref, rhs.ref, broadcast_dims) do
      {:ok, ref} -> {:ok, %Op{ref: ref}}
      {:error, msg} -> {:error, msg}
    end
  end
end
