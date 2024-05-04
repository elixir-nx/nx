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

  type_to_charlist = %{
    :token => ~c"token",
    {:pred, 8} => ~c"pred",
    {:s, 8} => ~c"s8",
    {:s, 16} => ~c"s16",
    {:s, 32} => ~c"s32",
    {:s, 64} => ~c"s64",
    {:u, 8} => ~c"u8",
    {:u, 16} => ~c"u16",
    {:u, 32} => ~c"u32",
    {:u, 64} => ~c"u64",
    {:f, 16} => ~c"f16",
    {:f, 32} => ~c"f32",
    {:f, 64} => ~c"f64",
    {:bf, 16} => ~c"bf16",
    {:c, 64} => ~c"c64",
    {:c, 128} => ~c"c128"
  }

  for {type, charlist} <- type_to_charlist do
    defp charlist_to_type(unquote(charlist)), do: unquote(type)
    defp type_to_charlist(unquote(type)), do: unquote(charlist)
  end
end
