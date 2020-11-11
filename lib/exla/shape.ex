defmodule Exla.Shape do
  alias __MODULE__, as: Shape

  @enforce_keys [:dims, :dtype, :ref]
  defstruct [:dims, :dtype, :ref]

  # TODO: convert dtype to integer representation to use in NIF
  def make_shape(dtype, dims) when is_atom(dtype) and is_tuple(dims) do
    {:ok, ref} = Exla.NIF.make_shape(dtype, dims)
    %Shape{ref: ref, dtype: dtype, dims: dims}
  end
end
