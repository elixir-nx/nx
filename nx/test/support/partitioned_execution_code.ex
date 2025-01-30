defmodule TestFunctions do
  def f([x, y]), do: [x + y, x - y]
  def g([z]), do: [0, z - 1]
  def h([f0, f1, g1]), do: [f0, f1, g1]
end
