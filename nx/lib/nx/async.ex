defprotocol Nx.Async do
  @moduledoc """
  The protocol for asynchronous communication with async `defn` code.

  See `Nx.Defn.async/4` for more information.
  """

  @doc """
  Awaits for the asynchronous execution to terminate.
  """
  def await!(async)
end
