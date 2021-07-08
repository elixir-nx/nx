defprotocol Nx.Stream do
  @moduledoc """
  The protocol for streaming data in and out of backends.
  """

  @doc """
  Sends a tensor.

  Returns the given tensor.
  """
  def send(stream, tensor)

  @doc """
  Receives data from the stream.

  It may be a tensor, a tuple of tensors, or a map of tensors.
  """
  def recv(stream)

  @doc """
  Returns the output of the stream.

  It may be a tensor, a tuple of tensors, or a map of tensors.
  """
  def done(stream)
end
