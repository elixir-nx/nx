defmodule EXLA.Builder do
  @moduledoc """
  Wrapper around XLA's builder.
  """

  alias EXLA.Computation
  alias EXLA.Op
  alias EXLA.MLIR.Module, as: M

  @enforce_keys [:ref]
  defstruct [:ref, :parent, :name]

  def new(name, inputs, outputs, type)

  def new(name, _inputs, _outputs, :xla) do
    new(name)
  end

  def new(module_and_name, arg_shapes, outputs, :mlir) do
    {module, name, is_public} =
      case module_and_name do
        {%M{} = module, name} -> {module, name, false}
        _name -> {M.new(), "main", true}
      end

    return_shape = exla_shape(outputs)

    arg_shapes = exla_shape(arg_shapes)

    if Enum.any?(arg_shapes, &match?({:tuple, _}, &1.dtype)) do
      raise "Tuple shapes are not allowed"
    end

    if Enum.any?(return_shape, &match?({:tuple, _}, &1.dtype)) do
      raise "Tuple shapes are not allowed"
    end

    M.create_function(
      module,
      name,
      arg_shapes,
      return_shape,
      is_public
    )
  end

  def exla_shape(tensors) when is_list(tensors) do
    Enum.flat_map(tensors, &exla_shape/1)
  end

  def exla_shape(tensors) when is_tuple(tensors) do
    tensors
    |> Tuple.to_list()
    |> Enum.flat_map(&exla_shape/1)
  end

  def exla_shape(%{type: :token}) do
    [EXLA.Shape.make_token_shape()]
  end

  def exla_shape(%Nx.Tensor{type: {:tuple, _size}, data: %{args: args}}) do
    Enum.flat_map(args, &exla_shape/1)
  end

  def exla_shape(%{shape: shape, type: type}) do
    [EXLA.Shape.make_shape(type, shape)]
  end

  def exla_shape(%EXLA.Shape{} = shape) do
    [shape]
  end

  def new(name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.new_builder(name)
    %__MODULE__{ref: ref, parent: nil, name: name}
  end

  def new(builder = %__MODULE__{ref: ref}, name) when is_binary(name) do
    {:ok, ref} = EXLA.NIF.create_sub_builder(ref, name)
    %__MODULE__{ref: ref, parent: builder, name: name}
  end

  def build(root)

  def build(%Op{} = root) do
    shape = EXLA.Op.get_shape(root)
    {:ok, ref} = EXLA.NIF.build(root.builder, root.ref)
    %Computation{ref: ref, output_shape: shape}
  end

  def build([%EXLA.MLIR.Value{function: function} | _] = values) do
    EXLA.MLIR.Value.variadic_return(function, values, true)

    function
  end

  def build(%EXLA.MLIR.Value{function: function} = value) do
    EXLA.MLIR.Value.variadic_return(function, value, true)
    function
  end
end
