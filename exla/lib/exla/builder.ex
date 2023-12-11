defmodule EXLA.Builder do
  @moduledoc """
  Wrapper around XLA's builder.
  """

  alias EXLA.Computation
  alias EXLA.Op
  alias EXLA.MLIR.Module, as: M

  @enforce_keys [:ref]
  defstruct [:ref, :parent, :name]

  def new(name, inputs, outputs, type, sub? \\ false, variadic_return? \\ false)

  def new(name, _inputs, _outputs, :xla, _sub?, _variadic_return?) do
    new(name)
  end

  def new(module_and_name, inputs, outputs, :mlir, sub?, variadic_return?) do
    {_arg_names, arg_shapes} = Enum.unzip(inputs)

    {module, name, is_public} =
      case module_and_name do
        {%M{} = module, name} -> {module, name, false}
        _name -> {M.new(), "main", true}
      end

    return_shape =
      if sub? do
        exla_shape(outputs, false)
      else
        out_types = [outputs] |> Nx.Defn.Composite.flatten_list()

        if variadic_return? do
          exla_shape(out_types, true)
        else
          out_types |> List.to_tuple() |> exla_shape(false)
        end
      end

    M.create_function(
      module,
      name,
      exla_shape(arg_shapes, false),
      List.wrap(return_shape),
      is_public
    )
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

  defp new(name) when is_binary(name) do
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

  def build(%EXLA.MLIR.Value{function: function, ref: root_ref}) do
    %EXLA.MLIR.Function{ref: function_ref} = function

    :ok = EXLA.NIF.mlir_build(function_ref, root_ref)
    function
  end
end
