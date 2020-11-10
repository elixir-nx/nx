defmodule Exla.Op do
  alias __MODULE__, Op
  alias Exla.Shape

  @enforce_keys [:ref]
  defstruct [:ref]

  def parameter(i, shape, name) when is_integer(i) and i >= 0 and is_binary(name) do
    case shape do
      # Alternatively, we can construct the shape for them but they need the reference later
      # and it's weird if we return it to them here
      %Shape{ref: nil} ->
        {:error, "Unconstructed shape. Pass your shape to `Exla.Shape.make_shape/1`"}

      %Shape{ref: ref} ->
        case Exla.NIF.parameter(i, ref, name) do
          {:ok, ref} -> {:ok, %Op{ref: ref}}
          {:error, msg} -> {:error, msg}
        end

      _ ->
        {:error, "Invalid shape passed to `#{__MODULE__}.parameter/3.`"}
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
