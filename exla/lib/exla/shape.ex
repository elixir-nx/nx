defmodule EXLA.Shape do
  @moduledoc """
  Wrapper around XLA's shape.
  """

  alias __MODULE__
  import Kernel, except: [byte_size: 1]

  @enforce_keys [:ref, :dims, :dtype]
  defstruct [:ref, :dims, :dtype]

  @doc false
  def get_shape_info(ref) when is_reference(ref) do
    case EXLA.NIF.get_shape_info(ref) |> unwrap!() do
      {dims_term, type_str} ->
        %Shape{dims: dims_term, dtype: charlist_to_dtype(type_str), ref: ref}

      children when is_list(children) ->
        children = Enum.map(children, &get_shape_info/1)
        %Shape{dims: {length(children)}, dtype: {:tuple, children}, ref: ref}
    end
  end

  @doc """
  Creates a shape with the given type-size tuple and dimensions.
  """
  def make_shape({type, size}, dims) when is_tuple(dims) do
    validate_dims!(dims, tuple_size(dims))
    ref = EXLA.NIF.make_shape(dtype_to_charlist({type, size}), dims) |> unwrap!()
    %Shape{ref: ref, dtype: {type, size}, dims: dims}
  end

  @doc """
  Creates a token shape.
  """
  def make_token_shape() do
    ref = EXLA.NIF.make_token_shape() |> unwrap!()
    %Shape{dims: {}, dtype: :token, ref: ref}
  end

  @doc """
  Creates a tuple shape with the given shapes.
  """
  def make_tuple_shape(shapes) when is_list(shapes) do
    refs =
      shapes
      |> Enum.map(& &1.ref)

    ref = EXLA.NIF.make_tuple_shape(refs) |> unwrap!()
    %Shape{dims: {length(shapes)}, dtype: {:tuple, shapes}, ref: ref}
  end

  defp validate_dims!(_dims, 0), do: :ok

  defp validate_dims!(dims, i)
       when is_integer(:erlang.element(i, dims)),
       do: validate_dims!(dims, i - 1)

  defp validate_dims!(dims, _i) do
    raise ArgumentError, "dimensions must be a tuple of integers, got: #{inspect(dims)}"
  end

  @doc """
  Returns the shape size in bytes.
  """
  def byte_size(%EXLA.Shape{dtype: {:tuple, shapes}}) do
    Enum.reduce(shapes, 0, &(byte_size(&1) + &2))
  end

  def byte_size(%EXLA.Shape{dtype: {_, bit_size}, dims: dims}) do
    Tuple.product(dims) * div(bit_size, 8)
  end

  @doc """
  Converts a charlist type into Nx' tuple format.
  """
  def charlist_to_dtype('token'), do: :token
  def charlist_to_dtype('bf16'), do: {:bf, 16}
  def charlist_to_dtype('pred'), do: {:pred, 8}
  def charlist_to_dtype([letter | int]), do: {List.to_atom([letter]), List.to_integer(int)}

  @doc """
  Converts Nx's tuple format into charlist.
  """
  def dtype_to_charlist({:pred, _}), do: 'pred'
  def dtype_to_charlist({type, size}), do: Atom.to_charlist(type) ++ Integer.to_charlist(size)

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, error}), do: raise(List.to_string(error))
end
