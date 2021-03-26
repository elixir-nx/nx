defmodule EXLA.Defn.Async do
  @moduledoc false

  @derive {Inspect, only: []}
  @enforce_keys [:executable, :outputs]
  defstruct [:executable, :outputs]

  defimpl Nx.Async do
    def await!(async), do: EXLA.Defn.__await__(async)
  end
end
