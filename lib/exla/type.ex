defmodule Exla.Type do
  # Equivalent to Nx.Type used throughout Exla
  @moduledoc false

  def to_nx({:pred, 1}), do: {:u, 8}
  def to_nx(type), do: type

  def merge({:pred, 1}, {:pred, 1}), do: {:pred, 1}
  def merge(left, right), do: Nx.Type.merge(to_nx(left), to_nx(right))

  def infer(infer), do: Nx.Type.infer(infer)
  def to_floating(type), do: Nx.Type.to_floating(to_nx(type))

  def min_value_binary(type), do: Nx.Type.min_value_binary(to_nx(type))
  def max_value_binary(type), do: Nx.Type.max_value_binary(to_nx(type))
  def validate!(type), do: Nx.Type.validate!(to_nx(type))

  def merge_scalar(type, scalar), do: Nx.Type.merge_scalar(to_nx(type), scalar)
  def cast_scalar!(type, scalar), do: Nx.Type.cast_scalar!(to_nx(type), scalar)

  # Version of merge_tensors
  def merge_ops(left, right) when is_number(left) and is_number(right), do: infer(left + right)
  def merge_ops(scalar, op) when is_number(scalar), do: merge_scalar(op_type(op), scalar)
  def merge_ops(op, scalar) when is_number(scalar), do: merge_scalar(op_type(op), scalar)
  def merge_ops(left, right), do: merge(op_type(left), op_type(right))
  defp op_type(op), do: Exla.Op.get_shape(op).dtype
end
