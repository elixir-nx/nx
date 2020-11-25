defmodule Exla.Shape do
  alias __MODULE__, as: Shape

  @enforce_keys [:ref]
  defstruct [:ref, dims: nil, dtype: nil]

  def make_shape({type, size}, dims) do
    {:ok, ref} = Exla.NIF.make_shape(dtype_to_str({type, size}), dims)
    %Shape{ref: ref, dtype: {type, size}, dims: dims}
  end

  def str_to_type('bf16'), do: {:bf, 16}
  def str_to_type([letter | integer]), do: {List.to_atom([letter]), List.to_integer(integer)}

  # TODO: Check valid dtype first
  def dtype_to_str({:i, size}), do: dtype_to_str({:s, size})
  def dtype_to_str({type, size}), do: Enum.join([Atom.to_string(type), Integer.to_string(size)])
end
