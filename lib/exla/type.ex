defmodule Exla.Type do
  @moduledoc false
  def to_floating(type), do: Nx.Type.to_floating(to_nx(type))

  def merge(:pred, :pred), do: :pred
  def merge(left, right), do: Nx.Type.merge(to_nx(left), to_nx(right))

  def infer(infer), do: Nx.Type.infer(infer)

  def to_nx({:pred, 1}), do: {:u, 8}
  def to_nx(type), do: type
end