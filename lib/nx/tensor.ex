defmodule Nx.Tensor do
  @moduledoc """
  The tensor data structure.

  All of its fields are private. You can access tensor
  metadata via the functions in the Nx module.
  """
  # TODO: implement the inspect protocol.

  @type data :: {module, term}
  @type type :: Nx.Type.t
  @type shape :: tuple()

  @enforce_keys [:data, :type, :shape]
  defstruct [:data, :type, :shape]
end
