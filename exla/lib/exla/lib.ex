defmodule EXLA.Lib do
  @moduledoc """
  High-level operations built on top of `EXLA.Op`.
  """

  alias EXLA.Builder
  alias EXLA.Op
  alias EXLA.Shape

  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value

  @doc """
  Element-wise tangent function.
  """
  def tan(%Value{} = op) do
    Value.tan(op)
  end

  def tan(%Op{} = op) do
    Op.divide(Op.sin(op), Op.cos(op))
  end

  @doc """
  Builds iota along axis.
  """
  def iota(%Builder{} = builder, shape, nil) do
    total_elems = Nx.size(shape.dims)

    Op.reshape(
      Op.iota(builder, EXLA.Shape.make_shape(shape.dtype, {total_elems}), 0),
      shape.dims
    )
  end

  def iota(%Builder{} = builder, shape, axis) do
    Op.iota(builder, shape, axis)
  end

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

  def argmax(%Builder{} = builder, %Op{} = op, type, opts) do
    argmin_or_max(builder, op, false, type, opts)
  end

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

  def argmin(%Builder{} = builder, %Op{} = op, type, opts) do
    argmin_or_max(builder, op, true, type, opts)
  end

  def argmin(%Function{} = builder, %Value{} = op, type, opts) do
    argmin_or_max(builder, op, true, type, opts)
  end

  defp argmin_or_max(builder, %mod{} = op, is_min?, type, opts) do
    tie_break = opts[:tie_break] || :low
    keep_axis = opts[:keep_axis] || false
    op_shape = mod.get_shape(op)

    init_value =
      if is_min?,
        do: max_number(builder, op_shape.dtype),
        else: min_number(builder, op_shape.dtype)

    axis = opts[:axis]
    index_init_value = mod.constant_r0(builder, 0, type)
    iota = iota(builder, Shape.make_shape(type, op_shape.dims), axis)
    reduction = create_min_max_computation(builder, op_shape.dtype, type, is_min?, tie_break)

    dims =
      if axis do
        {axis}
      else
        List.to_tuple(Nx.axes(op_shape.dims))
      end

    result =
      case builder do
        %Function{} ->
          [_, result] =
            Value.reduce(reduction, [init_value, index_init_value], [op, iota], dims)

          result

        _ ->
          builder
          |> Op.variadic_reduce(
            [op, iota],
            [init_value, index_init_value],
            reduction,
            dims
          )
          |> Op.get_tuple_element(1)
      end

    if keep_axis do
      mod.reshape(result, put_elem(op_shape.dims, axis, 1))
    else
      result
    end
  end

  defp create_min_max_computation(%Function{} = builder, type, index_type, is_min?, tie_break) do
    %{module: module, name: name} = subbuilder(builder, "min-max")

    function =
      EXLA.Builder.new(
        {module, name},
        [
          {"p0", EXLA.Shape.make_shape(type, {})},
          {"p1", EXLA.Shape.make_shape(index_type, {})},
          {"p2", EXLA.Shape.make_shape(type, {})},
          {"p3", EXLA.Shape.make_shape(index_type, {})}
        ],
        {struct(Nx.Tensor, type: type, shape: {}),
         struct(Nx.Tensor, type: index_type, shape: {})},
        :mlir,
        false,
        true
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

    [%{function: function}, _] = Value.variadic_return([max, arg_max])
    function
  end

  defp create_min_max_computation(builder, type, index_type, is_min?, tie_break) do
    sub_builder = subbuilder(builder, "min-max")

    lhs_value = Op.parameter(sub_builder, 0, Shape.make_shape(type, {}), "lhs_value")
    lhs_index = Op.parameter(sub_builder, 1, Shape.make_shape(index_type, {}), "lhs_index")
    rhs_value = Op.parameter(sub_builder, 2, Shape.make_shape(type, {}), "rhs_value")
    rhs_index = Op.parameter(sub_builder, 3, Shape.make_shape(index_type, {}), "rhs_index")

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

  It will be negative infinity for floating point types.
  """
  def min_number(%Builder{} = builder, type) do
    Op.constant_from_binary(builder, min_binary(type), Shape.make_shape(type, {}))
  end

  def min_number(%Function{} = builder, type) do
    Value.constant_from_binary(builder, min_binary(type), Shape.make_shape(type, {}))
  end

  @doc """
  Returns a maximum value scalar operator for the given type.

  Maximum values are defined in `Nx.Type.max_finite_binary/1`.
  """
  def max_number(builder, type) do
    mod = if is_struct(builder, Function), do: Value, else: Op
    mod.constant_from_binary(builder, max_binary(type), Shape.make_shape(type, {}))
  end

  defp subbuilder(%Builder{name: name} = builder, desc) do
    suffix = System.unique_integer([:positive])
    Builder.new(builder, name <> "-" <> desc <> "-" <> Integer.to_string(suffix))
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

  def argsort(builder, operand, dimension, stable, comparator, iota_type) do
    shape = EXLA.Op.get_shape(operand)
    iota = iota(builder, Shape.make_shape(iota_type, shape.dims), dimension)

    builder
    |> Op.variadic_sort(
      [operand, iota],
      comparator,
      dimension,
      stable
    )
    |> Op.get_tuple_element(1)
  end
end
