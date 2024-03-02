defmodule EXLA.MLIR.Function do
  @moduledoc false
  # Representation of an MLIR Function or `func.func` type.

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

  def push_region(%Function{ref: ref} = function, %Region{ref: region}) do
    ref
    |> EXLA.NIF.mlir_push_region(region)
    |> unwrap!()
    |> Enum.map(&%Value{function: function, ref: &1})
  end

  def pop_region(%Function{ref: ref}) do
    EXLA.NIF.mlir_pop_region(ref) |> unwrap!()
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, value}), do: value
  defp unwrap!(other), do: raise("error: #{inspect(other)}")
end
