defmodule Exla.Type do
  @moduledoc false
  def to_floating(type), do: Nx.Type.to_floating(to_nx(type))

  def merge({:pred, 1}, {:pred, 1}), do: {:pred, 1}
  def merge(left, right), do: Nx.Type.merge(to_nx(left), to_nx(right))

  def merge_scalar({:pred, 1}, scalar), do: Nx.Type.merge_scalar({:u, 8}, scalar)
  def merge_scalar(type, scalar), do: Nx.Type.merge_scalar(type, scalar)

  def infer(infer), do: Nx.Type.infer(infer)

  def to_nx({:pred, 1}), do: {:u, 8}
  def to_nx(type), do: type

  defdelegate min_value_binary(type), to: Nx.Type

  defdelegate max_value_binary(type), to: Nx.Type

  defdelegate validate!(type), to: Nx.Type

  defdelegate cast_scalar!(dtype, value), to: Nx.Type
end