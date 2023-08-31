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

  def new(name, _inputs, _outputs, :xla) do
    new(name)
  end

  def new(_name, inputs, outputs, :mlir) do
    # TO-DO (mlir): check if using name makes sense
    arg_shapes = Enum.map(inputs, fn {_, %Shape{} = s} -> s end)
    return_shape = exla_shape(outputs)

    module = M.new()
    M.create_function(module, "main", arg_shapes, return_shape)
  end

  defp exla_shape(%Nx.Tensor{} = t) do
    EXLA.Shape.make_shape(t.type, t.shape)
  end

  defp new(name) when is_binary(name) do
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
    # TO-DO: do not hardcode fetching just the first item as the output
    %EXLA.MLIR.Value{function: function, ref: root_ref} =
      EXLA.MLIR.Value.get_tuple_element(val, 0)

    %EXLA.MLIR.Function{ref: function_ref} = function
    :ok = EXLA.NIF.mlir_build(function_ref, root_ref)
    function
  end
end
