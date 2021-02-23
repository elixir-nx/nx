defmodule EXLA.Async do
  @moduledoc false

  @derive {Inspect, only: []}
  @enforce_keys [:executable, :holes]
  defstruct [:executable, :holes]

  defimpl Nx.Async do
    def await!(async), do: EXLA.Defn.__await__(async)
  end
end
