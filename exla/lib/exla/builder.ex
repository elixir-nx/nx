defmodule EXLA.Builder do
  @moduledoc """
  Wrapper around XLA's builder.
  """

  alias EXLA.Computation
  alias EXLA.Op
  alias EXLA.MLIR.Module, as: M

  @enforce_keys [:ref]
  defstruct [:ref, :parent, :name]

  def new_mlir(module_and_name, arg_shapes, return_shape) do
    {module, name, is_public} =
      case module_and_name do
        {%M{} = module, name} -> {module, name, false}
        _name -> {M.new(), "main", true}
      end

    M.create_function(
      module,
      name,
      arg_shapes,
      return_shape,
      is_public
    )
  end

  def new(name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.new_builder(name)
    %__MODULE__{ref: ref, parent: nil, name: name}
  end

  def new(builder = %__MODULE__{ref: ref}, name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.create_sub_builder(ref, name)
    %__MODULE__{ref: ref, parent: builder, name: name}
  end

  def exla_shape(tensors, flatten_tuple) when is_list(tensors) do
    result = Enum.map(tensors, &exla_shape(&1, flatten_tuple))

    if flatten_tuple do
      List.flatten(result)
    else
      result
    end
  end

  def exla_shape(tensors, flatten_tuple) when is_tuple(tensors) do
    tuple =
      tensors
      |> Tuple.to_list()
      |> Enum.map(&exla_shape(&1, flatten_tuple))

    if flatten_tuple do
      List.flatten(tuple)
    else
      EXLA.Shape.make_tuple_shape(tuple)
    end
  end

  def exla_shape(%{type: :token}, _flatten_tuple) do
    EXLA.Shape.make_token_shape()
  end

  def exla_shape(%Nx.Tensor{type: {:tuple, _size}, data: %{args: args}}, flatten_tuple) do
    tuple = Enum.map(args, &exla_shape(&1, flatten_tuple))

    if flatten_tuple do
      List.flatten(tuple)
    else
      EXLA.Shape.make_tuple_shape(tuple)
    end
  end

  def exla_shape(%{shape: shape, type: type}, _flatten_tuple) do
    EXLA.Shape.make_shape(type, shape)
  end

  def exla_shape(%EXLA.Shape{} = shape, _flatten_tuple) do
    shape
  end

  def build(root)

  def build(%Op{} = root) do
    shape = EXLA.Op.get_shape(root)
    {:ok, ref} = EXLA.NIF.build(root.builder, root.ref)
    %Computation{ref: ref, output_shape: shape}
  end

  def build(%EXLA.MLIR.Value{function: function} = value) do
    EXLA.MLIR.Value.variadic_return([value])
    function
  end
end
