defmodule Exla.Shape do
  alias __MODULE__, as: Shape

  @enforce_keys [:dims, :dtype]
  defstruct [:dims, :dtype, :ref]

  def make_shape(shape = %Shape{}) do
    case shape do
      %Shape{ref: nil} ->
        case Exla.NIF.make_shape(shape.dtype, shape.dims) do
          {:ok, ref} -> {:ok, %Shape{ref: ref | shape}}
          {:error, msg} -> {:error, msg}
        end

      %Shape{ref: ref} ->
        {:error, "Attempting to construct an already constructed shape."}

      _ ->
        {:error, "Invalid shape passed to `#{__MODULE__}.make_shape/2`"}
    end
  end
end
