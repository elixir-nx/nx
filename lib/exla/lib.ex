defmodule Exla.Lib do
  @moduledoc """
  High-level operations.
  """

  alias Exla.{Builder, Op, Shape}

  @doc """
  Builds iota with optional axis and type.
  """
  def iota(builder, shape, opts) do
    if axis = opts[:axis] do
      Op.iota(builder, shape, axis)
    else
      total_elems = tuple_product(shape.dims)

      Op.reshape(
        Op.iota(builder, Exla.Shape.make_shape(shape.dtype, {total_elems}), 0),
        shape.dims
      )
    end
  end

  @doc """
  Computes the sum of the given operation.

  ## Options

    * `:axes` - the axes to reduce on

  """
  def sum(%Builder{} = builder, %Op{} = op, opts \\ []) do
    op_shape = Op.get_shape(op)
    reduction_shape = Shape.make_shape(op_shape.dtype, {})

    sub_builder = subbuilder(builder, "sum")
    a = Op.parameter(sub_builder, 0, reduction_shape, "a")
    b = Op.parameter(sub_builder, 1, reduction_shape, "b")
    add = Op.add(a, b)
    reduction = Builder.build(add)

    init_value = Op.constant_r0(builder, 0, reduction_shape.dtype)
    Op.reduce(op, init_value, reduction, reduce_axes(op_shape, opts[:axes]))
  end

  @doc """
  Computes the mean of the given operation.

  ## Options

    * `:axes` - the axes to reduce on
  """
  def mean(%Builder{} = builder, %Op{} = op, opts \\ []) do
    %Shape{dims: dims} = Op.get_shape(op)

    Op.divide(
      sum(builder, Op.convert_element_type(op, {:f, 64}), opts),
      Op.constant_r0(builder, mean_den(dims, opts[:axes]), {:f, 64})
    )
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

    init_value =
      if is_min?,
        do: max_value(builder, op_shape.dtype),
        else: min_value(builder, op_shape.dtype)

    index_init_value = Op.constant_r0(builder, 0, op_shape.dtype)
    iota = iota(builder, op_shape, opts)
    reduction = create_min_max_computation(builder, op_shape.dtype, is_min?, tie_break)
    axis = opts[:axis]

    result =
      Op.variadic_reduce(
        builder,
        [op, iota],
        [init_value, index_init_value],
        reduction,
        reduce_axes(op_shape, axis && [axis])
      )

    Op.get_tuple_element(result, 1)
  end

  defp create_min_max_computation(builder, type, is_min?, tie_break) do
    sub_builder = subbuilder(builder, "min-max")

    lhs_value = Op.parameter(sub_builder, 0, Shape.make_shape(type, {}), "lhs_value")
    lhs_index = Op.parameter(sub_builder, 1, Shape.make_shape(type, {}), "lhs_index")
    rhs_value = Op.parameter(sub_builder, 2, Shape.make_shape(type, {}), "rhs_value")
    rhs_index = Op.parameter(sub_builder, 3, Shape.make_shape(type, {}), "rhs_index")

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

  defp min_value(%Builder{} = builder, type) do
    Op.constant_from_binary(
      builder,
      Exla.Type.min_value_binary(type),
      Shape.make_shape(type, {})
    )
  end

  defp max_value(builder, type) do
    Op.constant_from_binary(
      builder,
      Exla.Type.max_value_binary(type),
      Shape.make_shape(type, {})
    )
  end

  defp subbuilder(%Builder{name: name} = builder, desc) do
    suffix = System.unique_integer([:positive])
    Builder.new(builder, name <> "-" <> desc <> "-" <> Integer.to_string(suffix))
  end

  defp reduce_axes(op_shape, axes) do
    if axes do
      axes
      |> Enum.sort()
      |> List.to_tuple()
    else
      List.to_tuple(all_dimensions(0, tuple_size(op_shape.dims)))
    end
  end

  defp all_dimensions(i, n) when i < n, do: [i | all_dimensions(i + 1, n)]
  defp all_dimensions(_, _), do: []

  defp tuple_product(tuple), do: tuple_product(tuple, tuple_size(tuple))
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)

  defp mean_den(dims, nil), do: tuple_product(dims)
  defp mean_den(_dims, []), do: 1
  defp mean_den(dims, [axis | axes]), do: elem(dims, axis) * mean_den(dims, axes)
end
