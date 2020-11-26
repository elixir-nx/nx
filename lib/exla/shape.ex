defmodule Exla.Shape do
  alias __MODULE__, as: Shape

  @enforce_keys [:ref]
  defstruct [:ref, dims: nil, dtype: nil]

  @doc """
  Creates a shape with the given type-size tuple and dimensions.
  """
  def make_shape({type, size}, dims) do
    _ = Nx.Type.validate!({type, size})
    ref = Exla.NIF.make_shape(dtype_to_str({type, size}), dims) |> unwrap!()
    %Shape{ref: ref, dtype: {type, size}, dims: dims}
  end

  @doc """
  Gets the shape of an operator.
  """
  def get_shape(%Exla.Op{builder: builder, ref: operand}) do
    {dims, type_str, shape_ref} = Exla.NIF.get_shape(builder, operand) |> unwrap!()
    %Shape{ref: shape_ref, dims: dims, dtype: str_to_dtype(type_str)}
  end

  defp str_to_dtype('bf16'), do: {:bf, 16}
  defp str_to_dtype([letter | integer]), do: {List.to_atom([letter]), List.to_integer(integer)}

  # TODO: Make this private once we remove zero.
  def dtype_to_str({type, size}), do: Enum.join([Atom.to_string(type), Integer.to_string(size)])

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
