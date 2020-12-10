defmodule Exla.Lib do
  @moduledoc """
  High-level operations.
  """

  alias Exla.{Builder, Op, Shape}

  @doc """
  Computes the sum of the given operation.

  ## Options

    * `:axis` - the axis to reduce on

  """
  def sum(%Builder{} = builder, %Op{} = op, opts \\ []) do
    op_shape = Op.get_shape(op)
    reduction_shape = Shape.make_shape(op_shape.dtype, {})

    unique = System.unique_integer([:positive])
    sub_builder = Builder.new(builder, builder.name <> "-sum-" <> Integer.to_string(unique))
    a = Op.parameter(sub_builder, 0, reduction_shape, "a")
    b = Op.parameter(sub_builder, 1, reduction_shape, "b")
    add = Op.add(a, b)
    reduction = Builder.build(add)

    init_value = Op.constant_r0(builder, 0, reduction_shape.dtype)
    Op.reduce(op, init_value, reduction, reduce_dimensions(op_shape, opts))
  end

  @doc """
  Computes the argmax of the given operation.

  ## Options

    * `:axis` - the axis to reduce on
    * `:tie_break` - how to break ties
  """
  def argmax(%Builder{} = builder, %Op{} = op, opts \\ []), do: argmin_or_max(builder, op, false, opts)

  @doc """
  Computes the argmin of the given operation.

  ## Options

    * `:axis` - the axis to reduce on
    * `:tie_break` - how to break ties
  """
  def argmin(%Builder{} = builder, %Op{} = op, opts \\ []), do: argmin_or_max(builder, op, true, opts)

  defp argmin_or_max(builder, op, is_min?, opts) do
    tie_break = opts[:tie_break] || :low

    op_shape = Op.get_shape(op)

    init_value = if is_min?, do: max_value(builder, op_shape.dtype, {}), else: min_value(builder, op_shape.dtype, {})
    index_init_value = Op.constant_r0(builder, 0, op_shape.dtype)

    iota =
      if axis = opts[:axis] do
        Op.iota(builder, op_shape, axis)
      else
        flat = Op.iota(builder, Shape.make_shape(op_shape.dtype, {tuple_product(op_shape.dims)}), 0)
        Op.reshape(flat, op_shape.dims)
      end

    reduction = create_min_max_computation(builder, op_shape.dtype, is_min?, tie_break)

    result = Op.variadic_reduce(builder, [op, iota], [init_value, index_init_value], reduction, reduce_dimensions(op_shape, opts))

    Op.get_tuple_element(result, 1)
  end

  defp create_min_max_computation(builder, type, is_min?, tie_break) do
    sub_builder = Builder.new(builder, "min_max_sub_builder")

    lhs_value = Op.parameter(sub_builder, 0, Shape.make_shape(type, {}), "lhs_value")
    lhs_index = Op.parameter(sub_builder, 1, Shape.make_shape(type, {}), "lhs_index")
    rhs_value = Op.parameter(sub_builder, 2, Shape.make_shape(type, {}), "rhs_value")
    rhs_index = Op.parameter(sub_builder, 3, Shape.make_shape(type, {}), "rhs_index")

    cmp = if is_min?, do: Op.less_than_or_equal(lhs_value, rhs_value), else: Op.greater_than_or_equal(lhs_value, rhs_value)

    max = Op.select(cmp, lhs_value, rhs_value)
    arg_max = Op.select(cmp, lhs_index, rhs_index)

    # TODO: Random choice op
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

  # TODO: F16 and BF16

  def min_value(%Builder{} = builder, type, shape \\ {}),
    do: Op.constant_from_binary(builder, Nx.Type.min_value_binary(type), Shape.make_shape(type, shape))

  def max_value(builder, type, shape \\ {}),
    do: Op.constant_from_binary(builder, Nx.Type.max_value_binary(type), Shape.make_shape(type, shape))

  defp reduce_dimensions(op_shape, opts) do
    axis = opts[:axis]

    cond do
      axis == nil -> all_dimensions(op_shape.dims)
      axis >= 0 -> {axis}
      axis < 0 -> {tuple_size(op_shape.dims) + axis}
    end
  end

  # Converts {3, 255, 255} into {0, 1, 2}
  defp all_dimensions(dims), do: List.to_tuple(all_dimensions(0, tuple_size(dims)))
  defp all_dimensions(i, n) when i < n, do: [i | all_dimensions(i + 1, n)]
  defp all_dimensions(_, _), do: []

  defp tuple_product(tuple), do: tuple_product(tuple, tuple_size(tuple))
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)
end
