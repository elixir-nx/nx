defmodule Nx.Tensor do
  @moduledoc """
  The tensor data structure.

  All of its fields are private. You can access tensor
  metadata via the functions in the Nx module.
  """
  # TODO: implement the inspect protocol.
  defstruct [:data, :type, :shape]
end
