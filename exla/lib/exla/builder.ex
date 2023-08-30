defmodule EXLA.Builder do
  @moduledoc """
  Wrapper around XLA's builder.
  """

  alias EXLA.Computation
  alias EXLA.Op
  alias EXLA.MLIR.Module, as: M
  alias EXLA.MLIR.Type

  @enforce_keys [:ref]
  defstruct [:ref, :parent, :name]

  def new(name, _inputs, _outputs, :xla) do
    new(name)
  end

  def new(name, inputs, outputs, :mlir) do
    mlir_arg_types = mlir_type(Enum.map(inputs, &elem(&1, 1)))
    mlir_ret_type = mlir_type(outputs)

    xla_ret_shape = exla_shape(outputs)

    module = M.new()
    M.create_function(module, "main", mlir_arg_types, mlir_ret_type, xla_ret_shape)
  end

  defp mlir_type(input) when is_tuple(input) do
    input
    |> Tuple.to_list()
    |> Enum.map(&mlir_type/1)
  end

  defp mlir_type(inputs) when is_list(inputs) do
    Enum.map(inputs, &mlir_type/1)
  end

  defp mlir_type(%EXLA.Shape{} = shape) do
    Type.new(shape)
  end

  defp mlir_type(%Nx.Tensor{} = t) do
    t.type
    |> EXLA.Shape.make_shape(t.shape)
    |> Type.new()
  end

  defp exla_shape(%Nx.Tensor{} = t) do
    EXLA.Shape.make_shape(t.type, t.shape)
  end

  def new(name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.new_builder(name)
    %__MODULE__{ref: ref, parent: nil, name: name}
  end

  def new(builder = %__MODULE__{ref: ref}, name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.create_sub_builder(ref, name)
    %__MODULE__{ref: ref, parent: builder, name: name}
  end

  def build(%Op{} = root) do
    shape = EXLA.Op.get_shape(root)
    {:ok, ref} = EXLA.NIF.build(root.builder, root.ref)
    %Computation{ref: ref, output_shape: shape}
  end

  def build(%EXLA.MLIR.Value{} = val) do
    %EXLA.MLIR.Value{function: function, ref: root_ref} =
      EXLA.MLIR.Value.get_tuple_element(val, 0)

    %EXLA.MLIR.Function{ref: function_ref, module: %EXLA.MLIR.Module{ref: module_ref}} = function
    :ok = EXLA.NIF.mlir_build(function_ref, root_ref)
    # EXLA.NIF.dump_mlir_module(module_ref)
    function
  end
end
