defmodule Container do
  @derive {Nx.Container, [:a, :b]}
  defstruct [:a, :b, :c]
end
