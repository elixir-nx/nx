defmodule Container do
  @derive {Nx.Container, containers: [:a, :b], keep: [:d]}
  defstruct [:a, :b, :c, :d]
end
