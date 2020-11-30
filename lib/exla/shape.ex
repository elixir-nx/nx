defmodule Exla.Shape do
  alias __MODULE__

  @enforce_keys [:ref, :dims, :dtype]
  defstruct [:ref, :dims, :dtype]

  @doc """
  Gets shape information from given shape reference.
  """
  def get_shape_info(ref) when is_reference(ref) do
    case Exla.NIF.get_shape_info(ref) |> unwrap!() do
      {dims_term, type_str} ->
        %Shape{dims: dims_term, dtype: charlist_to_dtype(type_str), ref: ref}

      children when is_list(children) ->
        children = Enum.map(children, &get_shape_info/1)
        %Shape{dims: {length(children)}, dtype: {:t, children}, ref: ref}
    end
  end

  @doc """
  Creates a shape with the given type-size tuple and dimensions.
  """
  def make_shape({type, size}, dims) when is_tuple(dims) do
    _ = Nx.Type.validate!({type, size})
    ref = Exla.NIF.make_shape(dtype_to_charlist({type, size}), dims) |> unwrap!()
    %Shape{ref: ref, dtype: {type, size}, dims: dims}
  end

  @doc """
  Converts a charlist type into Nx' tuple format.
  """
  def charlist_to_dtype('bf16'), do: {:bf, 16}

  def charlist_to_dtype([letter | integer]),
    do: {List.to_atom([letter]), List.to_integer(integer)}

  @doc """
  Converts Nx's tuple format into charlist.
  """
  def dtype_to_charlist({type, size}), do: Atom.to_charlist(type) ++ Integer.to_charlist(size)

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
