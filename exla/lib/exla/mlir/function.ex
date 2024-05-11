defmodule EXLA.MLIR.Function do
  @moduledoc false
  # Representation of an MLIR Function or `func.func` type.

  defstruct [:module, :ref, :name, :return_typespecs]

  alias __MODULE__, as: Function
  alias EXLA.MLIR.Value
  alias EXLA.MLIR.Region

  @doc """
  Returns a list of references to the function's positional arguments
  which can be used in MLIR operations.
  """
  def get_arguments(%Function{ref: ref} = function) do
    arg_refs = EXLA.NIF.mlir_get_function_arguments(ref) |> unwrap!()
    Enum.map(arg_refs, fn arg -> %Value{ref: arg, function: function} end)
  end

  @doc """
  Creates a new region within the current function and sets it as the
  insertion point for subsequent operations.

  Returns `{region, args}`, where args is a list of values referencing
  the block arguments.
  """
  def push_region(%Function{ref: ref} = function, arg_typespecs) do
    arg_mlir_types = Value.typespecs_to_mlir_types(arg_typespecs)
    {region, args} = EXLA.NIF.mlir_push_region(ref, arg_mlir_types) |> unwrap!()
    {%Region{ref: region}, Enum.map(args, &%Value{function: function, ref: &1})}
  end

  @doc """
  Pops region created with `push_region/2`.
  """
  def pop_region(%Function{ref: ref}) do
    EXLA.NIF.mlir_pop_region(ref) |> unwrap!()
  end

  defp unwrap!(:ok), do: :ok
  defp unwrap!({:ok, value}), do: value
  defp unwrap!(other), do: raise("error: #{inspect(other)}")
end
