defmodule EXLA.BinaryBuffer do
  @moduledoc """
  A buffer where data is kept in a binary.
  """

  @enforce_keys [:data, :typespec]
  defstruct [:data, :typespec]

  def from_binary(data, typespec) do
    %EXLA.BinaryBuffer{data: data, typespec: typespec}
  end
end
