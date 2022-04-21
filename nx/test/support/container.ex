defmodule Container do
  @derive {Nx.Container, containers: [:a, :b], keep: [:d]}
  defstruct [:a, :b, c: %{}, d: %{}]
end

# Assert empty container emits no warnings
defmodule EmptyContainer do
  @derive {Nx.Container, containers: []}
  defstruct [:var, :fun, :acc]
end
