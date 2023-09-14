defmodule EXLA.MLIR.Value do
  @moduledoc """
  Representation of an MLIR Value.

  MLIR Values are SSA and generally are either operations or
  block arguments. This module is used to construct most of the
  MLIR operations.
  """
  defstruct [:ref, :function]

  alias __MODULE__, as: Value
  alias EXLA.MLIR.Function

  @bin_ops [:add, :subtract, :multiply, :divide, :pow, :min] ++
             [:max, :remainder, :atan2, :equal, :less, :less_equal] ++
             [:greater, :greater_equal, :not_equal, :bitwise_and] ++
             [:bitwise_or, :bitwise_xor] ++
             [:left_shift, :right_shift_arithmetic, :right_shift_logical]

  for op <- @bin_ops do
    mlir_op = :"mlir_#{op}"

    def unquote(op)(
          %Value{ref: lhs, function: %Function{} = func},
          %Value{ref: rhs, function: %Function{} = func}
        ) do
      ref = EXLA.NIF.unquote(mlir_op)(func.ref, lhs, rhs) |> unwrap!()
      %Value{ref: ref, function: func}
    end
  end

  @unary_ops [:abs, :exp, :expm1, :floor, :ceil, :round] ++
               [:log, :log1p, :sigmoid, :sign, :cos] ++
               [:sin, :acos, :asin, :atan, :cosh, :sinh] ++
               [:tanh, :acosh, :asinh, :atanh, :sqrt, :cbrt] ++
               [:bitwise_not, :erf, :erfc, :erf_inv] ++
               [:is_infinity, :is_nan, :rsqrt, :negate, :count_leading_zeros] ++
               [:population_count]

  for op <- @unary_ops do
    mlir_op = :"mlir_#{op}"

    def unquote(op)(%Value{ref: operand, function: %Function{} = func}) do
      ref = EXLA.NIF.unquote(mlir_op)(func.ref, operand) |> unwrap!()
      %Value{ref: ref, function: func}
    end
  end

  def reshape(%Value{function: %Function{} = func} = op, shape_tuple) do
    ref = EXLA.NIF.mlir_reshape(func.ref, op.ref, shape_tuple) |> unwrap!()
    %Value{op | ref: ref}
  end

  def reverse(%Value{function: %Function{} = func} = op, dims) do
    ref = EXLA.NIF.mlir_reverse(func.ref, op.ref, dims) |> unwrap!()
    %Value{op | ref: ref}
  end

  def transpose(%Value{function: %Function{} = func} = op, axes) do
    ref = EXLA.NIF.mlir_transpose(func.ref, op.ref, axes) |> unwrap!()
    %Value{op | ref: ref}
  end

  def slice(%Value{function: %Function{} = func} = op, starts, limits, strides) do
    ref = EXLA.NIF.mlir_slice(func.ref, op.ref, starts, limits, strides) |> unwrap!()
    %Value{op | ref: ref}
  end

  def dynamic_slice(%Value{function: %Function{} = func} = op, starts, lengths) do
    starts = Enum.map(starts, fn %Value{ref: ref} -> ref end)
    ref = EXLA.NIF.mlir_dynamic_slice(func.ref, op.ref, starts, lengths) |> unwrap!()
    %Value{op | ref: ref}
  end

  def tuple([%Value{function: %Function{} = func} | _rest] = vals) do
    refs = Enum.map(vals, fn %Value{ref: ref, function: ^func} -> ref end)
    ref = EXLA.NIF.mlir_tuple(func.ref, refs) |> unwrap!()
    %Value{ref: ref, function: func}
  end

  def get_tuple_element(%Value{function: %Function{} = func, ref: ref}, index)
      when is_integer(index) do
    ref = EXLA.NIF.mlir_get_tuple_element(func.ref, ref, index) |> unwrap!()
    %Value{ref: ref, function: func}
  end

  def get_shape(%Value{ref: ref}) do
    shape_ref = EXLA.NIF.mlir_get_shape(ref) |> unwrap!()
    EXLA.Shape.get_shape_info(shape_ref)
  end

  def convert(%Value{ref: in_ref, function: %Function{} = func} = value, dtype) do
    out_ref =
      EXLA.NIF.mlir_convert(func.ref, in_ref, EXLA.Shape.dtype_to_charlist(dtype)) |> unwrap!()

    %Value{value | ref: out_ref}
  end

  def iota(%Function{} = func, shape, dim) do
    ref = EXLA.NIF.mlir_iota(func.ref, shape.ref, dim) |> unwrap!()
    %Value{ref: ref, function: func}
  end

  def constant_r0(%Function{} = func, value, type) do
    ref =
      EXLA.NIF.mlir_constant_r0(func.ref, value, EXLA.Shape.dtype_to_charlist(type)) |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def constant_from_binary(%Function{} = func, data, shape) do
    ref =
      EXLA.NIF.mlir_constant_from_binary(
        func.ref,
        data,
        EXLA.Shape.dtype_to_charlist(shape.dtype),
        shape.dims
      )
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def dot_general(
        output_shape,
        %Value{function: func} = lhs,
        %Value{function: func} = rhs,
        dnums,
        precision_config
      ) do
    config = get_precision_config_int(precision_config)

    ref =
      EXLA.NIF.mlir_dot_general(func.ref, output_shape.ref, lhs.ref, rhs.ref, dnums, config)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def broadcast_in_dim(%Value{function: func} = operand, output_shape, axes) do
    ref =
      EXLA.NIF.mlir_broadcast_in_dim(func.ref, output_shape.ref, operand.ref, axes)
      |> unwrap!()

    %Value{function: func, ref: ref}
  end

  def concatenate([%Value{function: func} | _rest] = operands, dimension) do
    refs = Enum.map(operands, & &1.ref)

    ref =
      EXLA.NIF.mlir_concatenate(func.ref, refs, dimension)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def optimization_barrier(%Value{function: func} = operand) do
    ref =
      EXLA.NIF.mlir_optimization_barrier(func.ref, operand.ref)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def clamp(
        %Value{function: func} = operand,
        %Value{function: func} = min,
        %Value{function: func} = max
      ) do
    ref =
      EXLA.NIF.mlir_clamp(func.ref, operand.ref, min.ref, max.ref)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  def select(
        %Value{function: func} = pred,
        %Value{function: func} = on_true,
        %Value{function: func} = on_false
      ) do
    ref =
      EXLA.NIF.mlir_select(func.ref, pred.ref, on_true.ref, on_false.ref)
      |> unwrap!()

    %Value{ref: ref, function: func}
  end

  defp get_precision_config_int(precision_config) do
    case precision_config do
      :default ->
        0

      :high ->
        1

      :highest ->
        2

      :packed_nibble ->
        3

      _ ->
        raise ArgumentError,
              "expected precision configuration to be one of" <>
                " :default, :high, :highest, or :packed_nibble," <>
                " got: #{inspect(precision_config)}"
    end
  end

  defp unwrap!({:ok, value}), do: value
  defp unwrap!(other), do: raise("#{inspect(other)}")
end
