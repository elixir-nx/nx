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
        exla_shape(outputs)
      else
        out_types = [outputs] |> Nx.Defn.Composite.flatten_list()

        if variadic_return? do
          exla_shape(out_types)
        else
          out_types |> List.to_tuple() |> exla_shape()
        end
      end

    M.create_function(module, name, arg_shapes, List.wrap(return_shape), is_public)
  end

  defp exla_shape(tensors) when is_list(tensors) do
    Enum.map(tensors, &exla_shape/1)
  end

  defp exla_shape(tensors) when is_tuple(tensors) do
    tensors
    |> Tuple.to_list()
    |> Enum.map(&exla_shape/1)
    |> EXLA.Shape.make_tuple_shape()
  end

  defp exla_shape(%{type: :token}) do
    EXLA.Shape.make_token_shape()
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
