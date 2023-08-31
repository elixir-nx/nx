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

  def add(
        %Value{ref: lhs, function: %Function{} = func},
        %Value{ref: rhs, function: %Function{} = func}
      ) do
    ref = EXLA.NIF.mlir_add(func.ref, lhs, rhs) |> unwrap!()
    %Value{ref: ref, function: func}
  end

  def subtract(
        %Value{ref: lhs, function: %Function{} = func},
        %Value{ref: rhs, function: %Function{} = func}
      ) do
    ref = EXLA.NIF.mlir_subtract(func.ref, lhs, rhs) |> unwrap!()
    %Value{ref: ref, function: func}
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

  defp unwrap!({:ok, value}), do: value
  defp unwrap!(other), do: raise("#{inspect(other)}")
end
