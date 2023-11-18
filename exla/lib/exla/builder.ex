defmodule EXLA.Builder do
  @moduledoc """
  Wrapper around XLA's builder.
  """

  alias EXLA.Computation
  alias EXLA.Op
  alias EXLA.MLIR.Module, as: M

  alias EXLA.Shape

  @enforce_keys [:ref]
  defstruct [:ref, :parent, :name]

  def new(name, inputs, outputs, type, sub? \\ false)

  def new(name, _inputs, _outputs, :xla, _sub?) do
    new(name)
  end

  def new(_name, inputs, outputs, :mlir, sub?) do
    # TO-DO (mlir): check if using the function name makes sense
    arg_shapes = Enum.map(inputs, fn {_, %Shape{} = s} -> s end)

    return_shape =
      if sub? do
        exla_shape(outputs)
      else
        [outputs] |> Nx.Defn.Composite.flatten_list() |> List.to_tuple() |> exla_shape()
      end

    module = M.new()
    M.create_function(module, "main", arg_shapes, return_shape)
  end

  defp exla_shape(tensors) when is_tuple(tensors) do
    tensors
    |> Tuple.to_list()
    |> Enum.map(&exla_shape/1)
    |> EXLA.Shape.make_tuple_shape()
  end

  defp exla_shape(%{shape: shape, type: type}) do
    EXLA.Shape.make_shape(type, shape)
  end

  defp new(name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.new_builder(name)
    %__MODULE__{ref: ref, parent: nil, name: name}
  end

  def new(builder = %__MODULE__{ref: ref}, name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.create_sub_builder(ref, name)
    %__MODULE__{ref: ref, parent: builder, name: name}
  end

  def build(root, use_mhlo_return? \\ false)

  def build(%Op{} = root, _) do
    shape = EXLA.Op.get_shape(root)
    {:ok, ref} = EXLA.NIF.build(root.builder, root.ref)
    %Computation{ref: ref, output_shape: shape}
  end

  def build(%EXLA.MLIR.Value{function: function, ref: root_ref}, use_mhlo_return?) do
    %EXLA.MLIR.Function{ref: function_ref} = function
    return_int = if use_mhlo_return?, do: 1, else: 0

    :ok = EXLA.NIF.mlir_build(function_ref, root_ref, return_int)
    function
  end
end
