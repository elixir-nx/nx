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
    arg_refs = EXLA.NIF.mlir_get_function_arguments(ref)
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
    {region, args} = EXLA.NIF.mlir_push_region(ref, arg_mlir_types)
    {%Region{ref: region}, Enum.map(args, &%Value{function: function, ref: &1})}
  end

  @doc """
  Pops region created with `push_region/2`.
  """
  def pop_region(%Function{ref: ref}) do
    EXLA.NIF.mlir_pop_region(ref)
  end

  @doc """
  Sets sharding annotation for a function argument.

  Accepts either:
  - A tuple `{mesh, dim_shardings}` where mesh is a %Nx.Defn.Mesh{} and dim_shardings
    is a list of lists of axis indices (e.g., `[[0], [1]]`)
  - The old %EXLA.Sharding.TensorSharding{} struct (for backwards compatibility)
  """
  def set_arg_sharding(
        %Function{ref: ref},
        arg_index,
        {%Nx.Defn.Mesh{name: mesh_name}, dim_shardings}
      ) do
    # Convert axis indices to axis names
    # E.g., [[0], [1]] -> [["axis_0"], ["axis_1"]]
    dims =
      Enum.map(dim_shardings, fn axis_indices ->
        Enum.map(axis_indices, &"axis_#{&1}")
      end)

    EXLA.NIF.mlir_set_function_argument_attribute(ref, arg_index, "sdy.sharding", mesh_name, dims)
  end
end
