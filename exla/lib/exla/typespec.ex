defmodule EXLA.Typespec do
  @moduledoc """
  Combined type and shape information about tensors.

  In addition to the Nx types, also supports `{:pred, 8}` and `:token`,
  which are used internally within the compiler.

  This struct corresponds to the `xla::Shape` class in the XLA compiler,
  but is also meant as a lightweight data structure for passing the
  information around.

  Note: the name "typespec" has been chosen intentionally to distinguish
  it from both "type" and "shape".
  """

  @enforce_keys [:type, :shape]
  defstruct [:type, :shape]

  @doc """
  Builds a tensor typespec.
  """
  def tensor(type, shape) do
    %__MODULE__{type: type, shape: shape}
  end

  @doc """
  Builds a token typespec.
  """
  def token() do
    %__MODULE__{type: :token, shape: {}}
  end

  @doc """
  Returns an updated typespec with the given type.
  """
  def to_type(typespec, type), do: %{typespec | type: type}

  @doc """
  Returns an updated typespec with the given shape.
  """
  def to_shape(typespec, shape), do: %{typespec | shape: shape}

  @doc false
  def nif_encode(typespec) do
    {type_to_charlist(typespec.type), typespec.shape}
  end

  @doc false
  def nif_decode({type_charlist, shape}) do
    %__MODULE__{shape: shape, type: charlist_to_type(type_charlist)}
  end

  defp charlist_to_type(~c"token"), do: :token
  defp charlist_to_type(~c"pred"), do: {:pred, 8}
  defp charlist_to_type(~c"bf16"), do: {:bf, 16}
  defp charlist_to_type([letter | int]), do: {List.to_atom([letter]), List.to_integer(int)}

  defp type_to_charlist(:token), do: ~c"token"
  defp type_to_charlist({:pred, 8}), do: ~c"pred"
  defp type_to_charlist({type, size}), do: Atom.to_charlist(type) ++ Integer.to_charlist(size)
end
