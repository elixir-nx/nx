defprotocol Nx.Async do
  @moduledoc """
  The protocol for asynchronous communication with async `defn` code.
  """

  @doc """
  Awaits for the asynchronous execution to terminate.
  """
  def await!(async)
end
