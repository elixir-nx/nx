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
end
