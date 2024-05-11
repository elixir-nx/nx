defmodule EXLA.Lib do
  @moduledoc false
  # High-level operations built on top of `EXLA.MLIR.Value`.

  alias EXLA.Typespec
  alias EXLA.MLIR.Function
  alias EXLA.MLIR.Value

  @doc """
  Builds iota along axis.
  """
  def iota(%EXLA.MLIR.Function{} = function, nil, typespec) do
    total_elems = Nx.size(typespec.shape)

    Value.reshape(
      Value.iota(function, 0, Typespec.to_shape(typespec, {total_elems})),
      typespec
    )
  end

  def iota(%EXLA.MLIR.Function{} = function, axis, typespec) do
    Value.iota(function, axis, typespec)
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

    op_typespec = Value.get_typespec(op)

    init_value =
      if is_min?,
        do: max_number(builder, op_typespec.type),
        else: min_number(builder, op_typespec.type)

    axis = opts[:axis]
    index_init_value = Value.constant(builder, [0], Typespec.tensor(type, {}))
    iota = iota(builder, axis, Typespec.to_type(op_typespec, type))
    reduction = create_min_max_computation(builder, op_typespec.type, type, is_min?, tie_break)

    dims =
      if axis do
        [axis]
      else
        Nx.axes(op_typespec.shape)
      end

    shape = remove_axes(op_typespec.shape, dims)
    typespecs = [Typespec.tensor(op_typespec.type, shape), Typespec.tensor(type, shape)]

    [_, result] =
      Value.reduce(reduction, [init_value, index_init_value], [op, iota], dims, typespecs)

    if keep_axis do
      Value.reshape(result, Typespec.tensor(type, put_elem(op_typespec.shape, axis, 1)))
    else
      result
    end
  end

  defp remove_axes(shape, axes) do
    axes
    |> Enum.reverse()
    |> Enum.reduce(shape, &Tuple.delete_at(&2, &1))
  end

  defp create_min_max_computation(%Function{} = function, type, index_type, is_min?, tie_break) do
    arg_typespecs = [
      Typespec.tensor(type, {}),
      Typespec.tensor(index_type, {}),
      Typespec.tensor(type, {}),
      Typespec.tensor(index_type, {})
    ]

    {region, args} = Function.push_region(function, arg_typespecs)
    [lhs_value, lhs_index, rhs_value, rhs_index] = args

    pred_typespec = Typespec.tensor({:pred, 8}, {})
    value_typespec = Typespec.tensor(type, {})
    idx_typespec = Typespec.tensor(index_type, {})

    cmp =
      if is_min?,
        do: Value.less_equal(lhs_value, rhs_value, pred_typespec),
        else: Value.greater_equal(lhs_value, rhs_value, pred_typespec)

    max = Value.select(cmp, lhs_value, rhs_value, value_typespec)
    arg_max = Value.select(cmp, lhs_index, rhs_index, idx_typespec)

    arg_max =
      case tie_break do
        :low ->
          eq? = Value.equal(lhs_value, rhs_value, pred_typespec)
          id = Value.min(lhs_index, rhs_index, idx_typespec)
          Value.select(eq?, id, arg_max, idx_typespec)

        :high ->
          eq? = Value.equal(lhs_value, rhs_value, pred_typespec)
          id = Value.max(lhs_index, rhs_index, idx_typespec)
          Value.select(eq?, id, arg_max, idx_typespec)
      end

    Value.return(function, [max, arg_max])
    Function.pop_region(function)
    region
  end

  @doc """
  Returns a minimum value scalar operator for the given type.

  It will be negative infinity for floating point types.
  """
  def min_number(%Function{} = builder, type) do
    number =
      case type do
        {:pred, 8} ->
          0

        type ->
          type
          |> Nx.Constants.min(backend: Nx.BinaryBackend)
          |> Nx.to_number()
      end

    Value.constant(builder, [number], Typespec.tensor(type, {}))
  end

  @doc """
  Returns a maximum value scalar operator for the given type.

  Maximum values are defined in `Nx.Type.max_finite_binary/1`.
  """
  def max_number(builder, type) do
    number =
      case type do
        {:pred, 8} ->
          1

        type ->
          type
          |> Nx.Constants.max(backend: Nx.BinaryBackend)
          |> Nx.to_number()
      end

    Value.constant(builder, [number], Typespec.tensor(type, {}))
  end

  @doc """
  Sorts a tensor and returns the corresponding indices in the new positions.
  """
  def argsort(builder, %Value{} = operand, dimension, stable, comparator, iota_type) do
    typespec = Value.get_typespec(operand)
    iota_typespec = Typespec.to_type(typespec, iota_type)
    iota = iota(builder, dimension, iota_typespec)

    typespecs = [typespec, iota_typespec]
    [_, result] = Value.sort([operand, iota], comparator, dimension, stable, typespecs)

    result
  end
end
