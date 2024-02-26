defmodule EXLA.MLIR.Function do
  @moduledoc """
  Representation of an MLIR Function or `func.func` type.
  """
  defstruct [:module, :ref, :name, :return_shape]

  alias __MODULE__, as: Function
  alias EXLA.MLIR.Value
  alias EXLA.MLIR.Region

  @doc """
  Returns a list of references to the function's positional arguments
  which can be used in MLIR operations.
  """
  def get_arguments(%Function{ref: ref} = function) do
    arg_refs = EXLA.NIF.get_mlir_function_arguments(ref) |> unwrap!()
    Enum.map(arg_refs, fn arg -> %Value{ref: arg, function: function} end)
  end

  def push_region(%Function{ref: ref}, %Region{ref: region}) do
    EXLA.NIF.mlir_push_region(ref, region)
  end

  def pop_region(%Function{ref: ref}) do
    EXLA.NIF.mlir_pop_region(ref)
  end

  defp unwrap!({:ok, value}), do: value
  defp unwrap!(_other), do: raise("unable to get value")
end
