defmodule EXLA.Lib do
  @moduledoc false
  # High-level operations built on top of `EXLA.MLIR.Value`.

  alias EXLA.Shape

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value

  @doc """
  Element-wise tangent function.
  """
  def tan(%Value{} = op) do
    Value.tan(op)
  end

  @doc """
  Builds iota along axis.
  """
  def iota(%EXLA.MLIR.Function{} = function, shape, nil) do
    total_elems = Nx.size(shape.dims)

    Value.reshape(
      Value.iota(function, EXLA.Shape.make_shape(shape.dtype, {total_elems}), 0),
      shape.dims
    )
  end

  def iota(%EXLA.MLIR.Function{} = function, shape, axis) do
    Value.iota(function, shape, axis)
  end

  @doc """
  Computes the argmax of the given operation.

  ## Options

    * `:axis` - the axis to reduce on
    * `:keep_axis` - whether or not to keep reduced axis
    * `:tie_break` - how to break ties
  """
  def argmax(builder, op, type, opts \\ [])

  def argmax(%Function{} = builder, %Value{} = op, type, opts) do
    argmin_or_max(builder, op, false, type, opts)
  end

  @doc """
  Computes the argmin of the given operation.

  ## Options

    * `:axis` - the axis to reduce on
    * `:keep_axis` - whether or not to keep reduced axis
    * `:tie_break` - how to break ties
  """
  def argmin(builder, op, type, opts \\ [])

  def argmin(%Function{} = builder, %Value{} = op, type, opts) do
    argmin_or_max(builder, op, true, type, opts)
  end

  defp argmin_or_max(builder, %Value{} = op, is_min?, type, opts) do
    tie_break = opts[:tie_break] || :low
    keep_axis = opts[:keep_axis] || false
    op_shape = Value.get_shape(op)

    init_value =
      if is_min?,
        do: max_number(builder, op_shape.dtype),
        else: min_number(builder, op_shape.dtype)

    axis = opts[:axis]
    index_init_value = Value.constant_r0(builder, 0, type)
    iota = iota(builder, Shape.make_shape(type, op_shape.dims), axis)
    reduction = create_min_max_computation(builder, op_shape.dtype, type, is_min?, tie_break)

    dims =
      if axis do
        {axis}
      else
        List.to_tuple(Nx.axes(op_shape.dims))
      end

    [_, result] =
      Value.reduce(reduction, [init_value, index_init_value], [op, iota], dims)

    if keep_axis do
      Value.reshape(result, put_elem(op_shape.dims, axis, 1))
    else
      result
    end
  end

  defp create_min_max_computation(%Function{} = builder, type, index_type, is_min?, tie_break) do
    %{module: module, name: name} = subbuilder(builder, "min-max")

    function =
      EXLA.MLIR.Module.add_function(
        module,
        name,
        [
          EXLA.Shape.make_shape(type, {}),
          EXLA.Shape.make_shape(index_type, {}),
          EXLA.Shape.make_shape(type, {}),
          EXLA.Shape.make_shape(index_type, {})
        ],
        [EXLA.Shape.make_shape(type, {}), EXLA.Shape.make_shape(index_type, {})]
      )

    [lhs_value, lhs_index, rhs_value, rhs_index] = EXLA.MLIR.Function.get_arguments(function)

    cmp =
      if is_min?,
        do: Value.less_equal(function, lhs_value, rhs_value),
        else: Value.greater_equal(function, lhs_value, rhs_value)

    max = Value.select(cmp, lhs_value, rhs_value)
    arg_max = Value.select(cmp, lhs_index, rhs_index)

    arg_max =
      case tie_break do
        :low ->
          eq? = Value.equal(function, lhs_value, rhs_value)
          id = Value.min(function, lhs_index, rhs_index)
          Value.select(eq?, id, arg_max)

        :high ->
          eq? = Value.equal(function, lhs_value, rhs_value)
          id = Value.max(function, lhs_index, rhs_index)
          Value.select(eq?, id, arg_max)
      end

    Value.variadic_return(function, [max, arg_max])
    function
  end

  @doc """
  Returns a minimum value scalar operator for the given type.

  It will be negative infinity for floating point types.
  """
  def min_number(%Function{} = builder, type) do
    Value.constant_from_binary(builder, min_binary(type), Shape.make_shape(type, {}))
  end

  @doc """
  Returns a maximum value scalar operator for the given type.

  Maximum values are defined in `Nx.Type.max_finite_binary/1`.
  """
  def max_number(builder, type) do
    Value.constant_from_binary(builder, max_binary(type), Shape.make_shape(type, {}))
  end

  defp subbuilder(%EXLA.MLIR.Function{name: name} = function, description) do
    suffix = System.unique_integer([:positive])
    %{function | name: name <> "-" <> description <> "-" <> Integer.to_string(suffix)}
  end

  defp min_binary({:pred, 8}), do: <<0>>
  defp min_binary(type), do: Nx.Type.min_binary(type)
  defp max_binary({:pred, 8}), do: <<1>>
  defp max_binary(type), do: Nx.Type.max_binary(type)

  @doc """
  Sorts a tensor and returns the corresponding indices in the new positions.
  """
  def argsort(builder, %Value{} = operand, dimension, stable, comparator, iota_type) do
    shape = Value.get_shape(operand)
    iota = iota(builder, Shape.make_shape(iota_type, shape.dims), dimension)

    [_, result] = Value.sort([operand, iota], comparator, dimension, stable)

    result
  end
end
