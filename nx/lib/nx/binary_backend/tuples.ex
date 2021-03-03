defmodule Nx.BinaryBackend.Tuples do

  def map(tup, fun) do
    map(tup, fun, 0, tuple_size(tup) - 1)
  end

  defp map(tup, i, len, fun) when i >= len do
    []
  end
  defp map(tup, i, len, fun) when i < len do
    [fun.(elem(tup, i), i) | map(tup, i + 1, len, fun)]
  end

end