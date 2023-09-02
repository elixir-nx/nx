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
             [:greater, :greater_equal, :not_equal]

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
               [:tanh, :acosh, :asinh, :atanh] ++
               [:sqrt, :cbrt]

  for op <- @unary_ops do
    mlir_op = :"mlir_#{op}"

    def unquote(op)(%Value{ref: operand, function: %Function{} = func}) do
      ref = EXLA.NIF.unquote(mlir_op)(func.ref, operand) |> unwrap!()
      %Value{ref: ref, function: func}
    end
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

  defp unwrap!({:ok, value}), do: value
  defp unwrap!(other), do: raise("#{inspect(other)}")
end
