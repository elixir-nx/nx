defmodule Exla.Shape do
  alias __MODULE__, as: Shape

  @enforce_keys [:dims, :dtype, :ref]
  defstruct [:dims, :dtype, :ref]

  # TODO: convert dtype to integer representation to use in NIF
  def make_shape(dtype, dims) when is_atom(dtype) and is_tuple(dims) do
    case Exla.NIF.make_shape(dtype, dims) do
      {:ok, ref} -> {:ok, %Shape{ref: ref, dims: dims, dtype: dtype}}
      {:error, msg} -> {:error, msg}
    end
  end
end
