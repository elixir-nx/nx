defmodule Exla.Type do
  # Equivalent to Nx.Type used throughout Exla
  @moduledoc false

  def to_nx({:pred, 1}), do: {:u, 8}
  def to_nx(type), do: type

  def merge({:pred, 1}, {:pred, 1}), do: {:pred, 1}
  def merge(left, right), do: Nx.Type.merge(to_nx(left), to_nx(right))

  def infer(infer), do: Nx.Type.infer(infer)
  def to_floating(type), do: Nx.Type.to_floating(to_nx(type))
  def to_aggregate(type), do: Nx.Type.to_aggregate(to_nx(type))

  def min_value_binary(type), do: Nx.Type.min_value_binary(to_nx(type))
  def max_value_binary(type), do: Nx.Type.max_value_binary(to_nx(type))
  def normalize!(type), do: Nx.Type.normalize!(to_nx(type))

  def merge_scalar(type, scalar), do: Nx.Type.merge_scalar(to_nx(type), scalar)
  def cast_scalar!(type, scalar), do: Nx.Type.cast_scalar!(to_nx(type), scalar)
end
