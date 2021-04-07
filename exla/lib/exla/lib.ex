defmodule EXLA.Lib do
  @moduledoc """
  High-level operations built on top of `EXLA.Op`.
  """

  alias EXLA.{Builder, Op, Shape}

  @doc """
  Element-wise tangent function.
  """
  def tan(%Op{} = op) do
    Op.divide(Op.sin(op), Op.cos(op))
  end

  @doc """
  Builds iota along axis.
  """
  def iota(builder, shape, nil) do
    total_elems = Nx.size(shape.dims)

    Op.reshape(
      Op.iota(builder, EXLA.Shape.make_shape(shape.dtype, {total_elems}), 0),
      shape.dims
    )
  end

  def iota(builder, shape, axis) do
    Op.iota(builder, shape, axis)
  end

  @doc """
  Computes the argmax of the given operation.

  ## Options

    * `:axis` - the axis to reduce on
    * `:tie_break` - how to break ties
  """
  def argmax(%Builder{} = builder, %Op{} = op, opts \\ []) do
    argmin_or_max(builder, op, false, opts)
  end

  @doc """
  Computes the argmin of the given operation.

  ## Options

    * `:axis` - the axis to reduce on
    * `:tie_break` - how to break ties
  """
  def argmin(%Builder{} = builder, %Op{} = op, opts \\ []) do
    argmin_or_max(builder, op, true, opts)
  end

  defp argmin_or_max(builder, op, is_min?, opts) do
    tie_break = opts[:tie_break] || :low
    op_shape = Op.get_shape(op)
    type = opts[:type] || op_shape.dtype

    init_value =
      if is_min?,
        do: max_value(builder, op_shape.dtype),
        else: min_value(builder, op_shape.dtype)

    axis = opts[:axis]
    index_init_value = Op.constant_r0(builder, 0, type)
    iota = iota(builder, Shape.make_shape(type, op_shape.dims), axis)
    reduction = create_min_max_computation(builder, op_shape.dtype, is_min?, tie_break)

    result =
      Op.variadic_reduce(
        builder,
        [op, iota],
        [init_value, index_init_value],
        reduction,
        if(axis, do: {axis}, else: List.to_tuple(Nx.axes(op_shape.dims)))
      )

    Op.get_tuple_element(result, 1)
  end

  defp create_min_max_computation(builder, type, is_min?, tie_break) do
    sub_builder = subbuilder(builder, "min-max")

    lhs_value = Op.parameter(sub_builder, 0, Shape.make_shape(type, {}), "lhs_value")
    lhs_index = Op.parameter(sub_builder, 1, Shape.make_shape({:s, 64}, {}), "lhs_index")
    rhs_value = Op.parameter(sub_builder, 2, Shape.make_shape(type, {}), "rhs_value")
    rhs_index = Op.parameter(sub_builder, 3, Shape.make_shape({:s, 64}, {}), "rhs_index")

    cmp =
      if is_min?,
        do: Op.less_equal(lhs_value, rhs_value),
        else: Op.greater_equal(lhs_value, rhs_value)

    max = Op.select(cmp, lhs_value, rhs_value)
    arg_max = Op.select(cmp, lhs_index, rhs_index)

    arg_max =
      case tie_break do
        :low ->
          eq? = Op.equal(lhs_value, rhs_value)
          id = Op.min(lhs_index, rhs_index)
          Op.select(eq?, id, arg_max)

        :high ->
          eq? = Op.equal(lhs_value, rhs_value)
          id = Op.max(lhs_index, rhs_index)
          Op.select(eq?, id, arg_max)
      end

    ast = Op.tuple(sub_builder, [max, arg_max])
    Builder.build(ast)
  end

  @doc """
  Returns a minimum value scalar operator for the given type.

  Minimum values are defined in `Nx.Type.min_value_binary/1`.
  """
  def min_value(%Builder{} = builder, type) do
    Op.constant_from_binary(builder, min_value_binary(type), Shape.make_shape(type, {}))
  end

  @doc """
  Returns a maximum value scalar operator for the given type.

  Maximum values are defined in `Nx.Type.max_value_binary/1`.
  """
  def max_value(builder, type) do
    Op.constant_from_binary(builder, max_value_binary(type), Shape.make_shape(type, {}))
  end

  defp subbuilder(%Builder{name: name} = builder, desc) do
    suffix = System.unique_integer([:positive])
    Builder.new(builder, name <> "-" <> desc <> "-" <> Integer.to_string(suffix))
  end

  defp min_value_binary({:pred, 8}), do: <<0>>
  defp min_value_binary(type), do: Nx.Type.min_value_binary(type)

  defp max_value_binary({:pred, 8}), do: <<1>>
  defp max_value_binary(type), do: Nx.Type.max_value_binary(type)

  @doc """
  Sorts a tensor and returns the corresponding indices in the new positions.
  """
  def argsort(builder, operand, dimension, comparator, iota_type) do
    shape = EXLA.Op.get_shape(operand)
    iota = iota(builder, Shape.make_shape(iota_type, shape.dims), dimension)

    builder
    |> Op.variadic_sort(
      [operand, iota],
      comparator,
      dimension
    )
    |> Op.get_tuple_element(1)
  end
end
