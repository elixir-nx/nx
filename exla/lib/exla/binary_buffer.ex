defmodule EXLA.BinaryBuffer do
  @moduledoc """
  A buffer where data is kept in a binary.
  """

  @enforce_keys [:data, :shape]
  defstruct [:data, :shape]

  def from_binary(data, shape) do
    %EXLA.BinaryBuffer{data: data, shape: shape}
  end
end
