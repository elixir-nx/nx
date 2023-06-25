defmodule EXLA.MLIR.Type do
  @moduledoc """
  MLIR Type Helpers.
  """
  defstruct [:dims, :type]

  alias EXLA.Shape

  def new(dims, {_type, _size} = nx_type) when is_list(dims) do
    %__MODULE__{dims: dims, type: nx_type_to_mlir_type_int(nx_type)}
  end

  def new(%Shape{dims: dims, dtype: nx_type}) do
    %__MODULE__{dims: Tuple.to_list(dims), type: nx_type_to_mlir_type_int(nx_type)}
  end

  def nx_type_to_mlir_type_int({:s, 8}), do: 0
  def nx_type_to_mlir_type_int({:s, 16}), do: 1
  def nx_type_to_mlir_type_int({:s, 32}), do: 2
  def nx_type_to_mlir_type_int({:s, 64}), do: 3
  def nx_type_to_mlir_type_int({:u, 8}), do: 4
  def nx_type_to_mlir_type_int({:u, 16}), do: 5
  def nx_type_to_mlir_type_int({:u, 32}), do: 6
  def nx_type_to_mlir_type_int({:u, 64}), do: 7
  def nx_type_to_mlir_type_int({:f, 16}), do: 8
  def nx_type_to_mlir_type_int({:f, 32}), do: 9
  def nx_type_to_mlir_type_int({:f, 64}), do: 10
  def nx_type_to_mlir_type_int({:bf, 16}), do: 11
end
